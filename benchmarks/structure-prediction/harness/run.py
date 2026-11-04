# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Harness CLI.

    python -m harness.run inputs  --targets targets/targets.json --msa-dir MSA --out inputs/
    python -m harness.run predict --backend {neuron,gpu,openfold-trt,cpu} --target-id ID \
        --input inputs/ID.yaml --out-dir results/raw/... --seed {0,1} --ds {1,5} [...]
    python -m harness.run batch   --backend neuron --lnc 2 --workers 4 --inputs inputs/ ...

`predict` runs the model twice in fresh processes (cold: first run, compiles or loads caches;
warm: the same command again), then optionally a third, profiled run whose timings are
discarded. It writes `result.json` (schema `sp-result/1.0`) and `manifest.json` next to the
predictions. `--dry-run` builds and validates everything, writes a `dry-run` result with the
command it would run, and runs nothing; a dry-run result carries no measured numbers.

The CPU oracle (`--backend cpu`) runs once per (target, seed), not twice: its numbers are for
accuracy, not timing. Its results are cached under results/oracle/ keyed by target and seed and
are reused unless the target set or the Boltz version changes.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from harness import backends, pins, schema
from harness.manifest import capture_manifest, write_manifest

OOM_PATTERNS = (
    r"NRT_RESOURCE",
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"Failed to allocate",
)
ACCEL = {"neuron": "neuron", "gpu": "gpu", "openfold-trt": "gpu", "cpu": "cpu"}

Executor = Callable[[list[str], dict[str, str]], tuple[int, str, float]]


def execute(argv: list[str], env: dict[str, str]) -> tuple[int, str, float]:
    """Run argv with env overrides; return (returncode, combined tail of output, wall seconds)."""
    full = {**os.environ, **env}
    here = str(Path(__file__).resolve().parents[1])
    full["PYTHONPATH"] = here + os.pathsep + full.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    proc = subprocess.run(argv, env=full, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    return proc.returncode, (proc.stdout + proc.stderr)[-20000:], wall


def targets_digest(path: Path | None) -> str | None:
    """sha256 of the target manifest (part of the oracle cache key)."""
    return (
        hashlib.sha256(path.read_bytes()).hexdigest()
        if path and path.exists()
        else None
    )


def _oracle_cached(out_dir: Path, digest: str | None) -> dict[str, Any] | None:
    res = out_dir / "result.json"
    if not res.exists():
        return None
    doc: dict[str, Any] = json.loads(res.read_text())
    key = doc.get("oracle_key", {})
    if (
        doc.get("status") == "measured"
        and key.get("boltz") == pins.BOLTZ
        and key.get("targets_sha256") == digest
    ):
        return doc
    return None


def _predictions(inv: backends.Invocation) -> list[str]:
    if inv.predictions_dir is None or not inv.predictions_dir.exists():
        return []

    def rank(p: Path) -> int:
        m = re.search(r"_model_(\d+)", p.name)
        return int(m.group(1)) if m else 0

    return [
        str(p) for p in sorted(inv.predictions_dir.glob(inv.prediction_glob), key=rank)
    ]


def _read_timing(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def predict(
    cfg: backends.RunConfig,
    *,
    dry_run: bool,
    profile: bool,
    tokens: int | None,
    run_group: str,
    worker_index: int | None = None,
    workers: int | None = None,
    repo_paths: dict[str, str | None] | None = None,
    targets_json: Path | None = None,
    executor: Executor = execute,
    manifest_fn: Callable[..., dict[str, Any]] = capture_manifest,
    cold_wall: float | None = None,
) -> dict[str, Any]:
    """Run (or dry-run) one configuration and write result.json + manifest.json.

    `cold_wall` is set by batch mode, which runs every worker's cold pass first and then the warm
    pass concurrently (so device throughput is measured with all workers warm); predict then runs
    only the warm invocation.
    """
    inv = backends.build(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    digest = targets_digest(targets_json)
    if cfg.backend == "cpu" and not dry_run:
        cached = _oracle_cached(cfg.out_dir, digest)
        if cached is not None:
            return cached

    doc = schema.empty_result(
        status="dry-run",
        target_id=cfg.target_id,
        backend=cfg.backend,
        seed=cfg.seed,
        diffusion_samples=cfg.diffusion_samples,
        recycling_steps=cfg.recycling_steps,
        sampling_steps=cfg.sampling_steps,
        tokens=tokens,
        lnc=cfg.lnc,
        atom_graph_break=cfg.atom_graph_break,
        worker_index=worker_index,
        workers=workers,
        run_group=run_group,
        command=inv.argv,
    )
    doc["env"] = inv.env
    if cfg.backend == "cpu":
        doc["oracle_key"] = {"boltz": pins.BOLTZ, "targets_sha256": digest}

    manifest = manifest_fn(
        accelerator=ACCEL[cfg.backend],
        lnc=cfg.lnc,
        repo_paths=repo_paths or {},
        extra={
            "pins": {k: v for k, v in vars(pins).items() if k.isupper()},
            "dry_run": dry_run,
            "run_env": inv.env,
        },
    )
    mpath = write_manifest(manifest, cfg.out_dir / "manifest.json")
    doc["manifest_path"] = str(mpath)

    if not dry_run:
        runs = 1 if cfg.backend == "cpu" or cold_wall is not None else 2
        walls: list[float] = [cold_wall] if cold_wall is not None else []
        outputs: list[str] = []
        rc = 0
        for _ in range(runs):
            rc, out, wall = executor(inv.argv, inv.env)
            outputs.append(out)
            if rc != 0:
                break
            walls.append(wall)
        doc["returncode"] = rc
        timing = _read_timing(inv.timing_json) if rc == 0 else {}
        doc["precision"] = timing.get("precision")
        if rc == 0 and cfg.backend == "cpu" and timing.get("precision") != "32-true":
            # The oracle is only an oracle at fp32; stock Boltz-2 would silently run bf16-mixed.
            doc["status"] = "failed"
            doc["error"] = (
                f"oracle ran at precision {timing.get('precision')!r}, not '32-true'"
            )
        elif rc == 0:
            doc.update(
                status="measured",
                wall_cold_s=walls[0],
                wall_warm_s=walls[1] if len(walls) > 1 else None,
                model_s=timing.get("model_s"),
                peak_accel_mem_bytes=timing.get("peak_accel_mem_bytes"),
                peak_accel_mem_source=timing.get("peak_accel_mem_source"),
                predictions=_predictions(inv),
            )
            if profile and inv.profile_argv is not None:
                prc, pout, _ = executor(inv.profile_argv, inv.profile_env)
                doc["profile_path"] = str(inv.profile_path) if prc == 0 else None
                if prc != 0:
                    doc["profile_error"] = pout[-2000:]
        else:
            oom = any(re.search(p, outputs[-1]) for p in OOM_PATTERNS)
            doc["status"] = "did-not-run" if oom else "failed"
            doc["error"] = outputs[-1][-4000:]
    schema.validate_result(doc)
    (cfg.out_dir / "result.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def write_inputs(
    targets: dict[str, Any], msa_dir: Path | None, out: Path
) -> list[Path]:
    """Boltz YAML input per target; every protein chain points at its shared .a3m file."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for t in targets["targets"]:
        seqs = []
        for ch in t["chains"]:
            entity = {"id": ch["chain_id"], "sequence": ch["sequence"]}
            if ch["entity"] == "protein":
                a3m = msa_dir / f"{ch['msa_id']}.a3m" if msa_dir else None
                entity["msa"] = str(a3m) if a3m else "empty"
            seqs.append({ch["entity"]: entity})
        doc = {"version": 1, "sequences": seqs}
        path = out / f"{t['id']}.yaml"
        path.write_text(_yaml(doc))
        written.append(path)
    return written


def _yaml(doc: Any) -> str:
    import yaml

    text: str = yaml.safe_dump(doc, sort_keys=False)
    return text


def _repo_paths(pairs: Sequence[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for item in pairs:
        name, _, path = item.partition("=")
        out[name] = path or None
    return out


def _cfg(ns: argparse.Namespace, **over: Any) -> backends.RunConfig:
    single = ns.cmd == "predict"
    fields: dict[str, Any] = {
        "backend": ns.backend,
        "target_id": ns.target_id if single else None,
        "input_path": Path(ns.input) if single else None,
        "out_dir": Path(ns.out_dir) if single else None,
        "seed": ns.seed,
        "diffusion_samples": ns.ds,
        "sampling_steps": ns.sampling_steps,
        "lnc": ns.lnc,
        "visible_core": ns.core if single else None,
        "atom_graph_break": ns.atom_graph_break,
        "max_msa_seqs": ns.max_msa_seqs,
        "max_parallel_samples_cap": ns.cap_parallel_samples,
        "nim": ns.nim,
        "alignment_dir": Path(ns.alignment_dir) if ns.alignment_dir else None,
        "template_mmcif_dir": Path(ns.template_mmcif_dir)
        if ns.template_mmcif_dir
        else None,
        "jax_param_path": Path(ns.jax_param_path) if ns.jax_param_path else None,
        "trt_engine_dir": Path(ns.trt_engine_dir) if ns.trt_engine_dir else None,
        "openfold_dir": Path(ns.openfold_dir) if ns.openfold_dir else None,
    }
    fields.update(over)
    return backends.RunConfig(**fields)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=schema.BACKENDS, required=True)
    p.add_argument("--seed", type=int, choices=pins.SEEDS, required=True)
    p.add_argument("--ds", type=int, default=1, help="diffusion samples (1 or 5)")
    p.add_argument("--sampling-steps", type=int, default=pins.SAMPLING_STEPS)
    p.add_argument("--lnc", type=int, choices=(1, 2))
    p.add_argument("--atom-graph-break", type=int, choices=(0, 1))
    p.add_argument("--max-msa-seqs", type=int)
    p.add_argument(
        "--cap-parallel-samples",
        action="store_true",
        help="BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1",
    )
    p.add_argument(
        "--nim", action="store_true", help="Boltz-2 NIM column (off by default)"
    )
    p.add_argument(
        "--profile", action="store_true", help="extra profiled run (timings discarded)"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--targets-json", type=Path)
    p.add_argument("--repo", action="append", default=[], metavar="NAME=PATH")
    for opt in (
        "alignment-dir",
        "template-mmcif-dir",
        "jax-param-path",
        "trt-engine-dir",
        "openfold-dir",
    ):
        p.add_argument(f"--{opt}")


def main(argv: Sequence[str] | None = None, executor: Executor = execute) -> int:
    """CLI entry point; returns a process exit code."""
    ap = argparse.ArgumentParser(
        prog="harness.run", description=__doc__.split("\n\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser(
        "inputs", help="write Boltz YAML inputs from targets.json + MSAs"
    )
    pi.add_argument("--targets", type=Path, required=True)
    pi.add_argument("--msa-dir", type=Path)
    pi.add_argument("--out", type=Path, required=True)

    pp = sub.add_parser("predict", help="one target, one seed, one ds")
    _common(pp)
    pp.add_argument("--target-id", required=True)
    pp.add_argument("--input", required=True)
    pp.add_argument("--out-dir", required=True)
    pp.add_argument("--core", type=int, help="NEURON_RT_VISIBLE_CORES for this worker")
    pp.add_argument("--tokens", type=int)

    pb = sub.add_parser(
        "batch", help="per-logical-core workers over a set of inputs (neuron)"
    )
    _common(pb)
    pb.add_argument(
        "--inputs", type=Path, required=True, help="directory of <target>.yaml"
    )
    pb.add_argument("--out-root", type=Path, required=True)
    pb.add_argument("--workers", type=int, required=True)

    ns = ap.parse_args(argv)
    if ns.cmd == "inputs":
        targets = json.loads(ns.targets.read_text())
        for path in write_inputs(targets, ns.msa_dir, ns.out):
            print(path)
        return 0

    tokens_by_id: dict[str, int] = {}
    if ns.targets_json and ns.targets_json.exists():
        for t in json.loads(ns.targets_json.read_text())["targets"]:
            tokens_by_id[t["id"]] = t.get("tokens") or t.get("length")
    repos = _repo_paths(ns.repo)
    try:
        if ns.cmd == "predict":
            group = f"single-{uuid.uuid4().hex[:8]}"
            doc = predict(
                _cfg(ns),
                dry_run=ns.dry_run,
                profile=ns.profile,
                tokens=ns.tokens or tokens_by_id.get(ns.target_id),
                run_group=group,
                repo_paths=repos,
                targets_json=ns.targets_json,
                executor=executor,
            )
            print(
                json.dumps(
                    {
                        k: doc[k]
                        for k in ("status", "target_id", "backend", "wall_warm_s")
                    }
                )
            )
            return 0 if doc["status"] in ("measured", "dry-run", "did-not-run") else 1
        return _batch(ns, repos, tokens_by_id, executor)
    except backends.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


def _batch(
    ns: argparse.Namespace,
    repos: dict[str, str | None],
    tokens_by_id: dict[str, int],
    executor: Executor,
) -> int:
    if ns.backend != "neuron":
        raise backends.ConfigError("batch mode is for per-logical-core neuron workers")
    max_workers = 8 if ns.lnc == 1 else 4
    if not 1 <= ns.workers <= max_workers:
        raise backends.ConfigError(
            f"--workers must be 1..{max_workers} at LNC={ns.lnc}"
        )
    inputs = sorted(ns.inputs.glob("*.yaml"))
    shards = [inputs[i :: ns.workers] for i in range(ns.workers)]
    group = f"lnc{ns.lnc}-w{ns.workers}-ds{ns.ds}-s{ns.seed}-{uuid.uuid4().hex[:6]}"

    def cfg_for(i: int, inp: Path) -> backends.RunConfig:
        return _cfg(
            ns,
            target_id=inp.stem,
            input_path=inp,
            out_dir=ns.out_root / group / inp.stem,
            visible_core=i,
        )

    cold: dict[str, float | None] = {}

    def warmup(i: int) -> None:
        # Cold pass (compile or cache load); a failure here is recorded by the warm pass.
        for inp in shards[i]:
            if ns.dry_run:
                cold[inp.stem] = None
                continue
            inv = backends.build(cfg_for(i, inp))
            rc, _, wall = executor(inv.argv, inv.env)
            cold[inp.stem] = wall if rc == 0 else None

    def work(i: int) -> list[dict[str, Any]]:
        out = []
        for inp in shards[i]:
            tid = inp.stem
            cfg = cfg_for(i, inp)
            out.append(
                predict(
                    cfg,
                    dry_run=ns.dry_run,
                    profile=ns.profile,
                    tokens=tokens_by_id.get(tid),
                    run_group=group,
                    worker_index=i,
                    workers=ns.workers,
                    repo_paths=repos,
                    targets_json=ns.targets_json,
                    executor=executor,
                    cold_wall=cold.get(tid),
                )
            )
        return out

    with cf.ThreadPoolExecutor(max_workers=ns.workers) as pool:
        list(pool.map(warmup, range(ns.workers)))
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=ns.workers) as pool:
        docs = [d for res in pool.map(work, range(ns.workers)) for d in res]
    wall = time.perf_counter() - t0
    measured = [d for d in docs if d["status"] == "measured"]
    summary = {
        "schema": "sp-group/1.0",
        "run_group": group,
        "lnc": ns.lnc,
        "workers": ns.workers,
        "diffusion_samples": ns.ds,
        "seed": ns.seed,
        "dry_run": ns.dry_run,
        "inputs": len(inputs),
        "measured": len(measured),
        # Warm pass only, all workers concurrent. Device throughput is only meaningful for a
        # real run; a dry run records none.
        "group_wall_s": None if ns.dry_run else wall,
        "structures_per_s": (len(measured) * ns.ds / wall)
        if measured and not ns.dry_run
        else None,
    }
    (ns.out_root / group).mkdir(parents=True, exist_ok=True)
    (ns.out_root / group / "group.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
