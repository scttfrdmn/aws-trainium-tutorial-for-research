# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""In-process warm timing: one process per worker, every target listed twice.

    python -m harness.run warmbatch --backend neuron --lnc 2 --workers 4 --ds 1 --seed 0 \
        --inputs inputs --targets-json targets/targets.json --out-root results/raw/trn2-warm

Why: a fresh process per target makes `model_s` the first prediction in that process, which on
trn2 includes dynamo re-trace and NEFF load even with a warm cache (measured 21-31 s against
7.3-7.4 s warm, task 0) and on the GPU a much smaller first-call cost. That bias falls unevenly
on the two platforms. Here each worker runs one predict process over its shard, in which every
target appears twice (`<id>` and `<id>_w2`). Per target:

* the occurrence that ran FIRST is the canonical seed-s prediction (the RNG state right after
  seeding, as in a fresh process) and is the one scored for accuracy; its time is recorded as
  `wall_cold_s` (first call for that shape in the process);
* the occurrence that ran SECOND is timed: `model_s` and `wall_warm_s` are its predict-loop time.

Run order is read from the timing callback's per-record order (Boltz globs the directory, so
file order is not guaranteed). Targets are assigned to workers longest-first by token count.
Neuron workers get a thread cap and CPU pinning, so concurrent workers don't oversubscribe the
host. Group throughput (targets x ds / group wall) is written beside the per-target results.

Only for inputs that fit the device: a worker process that runs out of memory loses its whole
shard, so ceiling attempts stay single-target `predict` runs.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from harness import backends, schema

DUP = "_w2"
Executor = Callable[[list[str], dict[str, str]], tuple[int, str, float]]


def assign(tokens: dict[str, int], workers: int) -> list[list[str]]:
    """Longest-first assignment by tokens^2 (a rough cost proxy) to the least-loaded worker."""
    load = [0.0] * workers
    shards: list[list[str]] = [[] for _ in range(workers)]
    for tid in sorted(tokens, key=lambda t: (-tokens[t], t)):
        i = min(range(workers), key=lambda k: (load[k], k))
        shards[i].append(tid)
        load[i] += float(tokens[tid]) ** 2
    return shards


def cpu_sets(workers: int, vcpus: int, reserve: int = 0) -> list[str]:
    """Disjoint contiguous CPU sets, one per worker, after `reserve` CPUs for the OS."""
    per = max((vcpus - reserve) // workers, 1)
    return [
        f"{reserve + i * per}-{reserve + (i + 1) * per - 1}" for i in range(workers)
    ]


def order_from_timing(per_record: dict[str, float], tid: str) -> tuple[str, str] | None:
    """(first, second) record ids for a target, from the per-record execution order."""
    seen = [r for r in per_record if r in (tid, tid + DUP)]
    return (seen[0], seen[1]) if len(seen) == 2 else None


def run_warmbatch(
    *,
    backend: str,
    inputs: Path,
    targets: dict[str, Any],
    out_root: Path,
    seed: int,
    ds: int,
    lnc: int | None,
    workers: int,
    executor: Executor,
    manifest_fn: Callable[..., dict[str, Any]],
    only: Sequence[str] | None = None,
    atom_graph_break: int | None = None,
    variant: str | None = None,
    max_msa_seqs: int | None = None,
    profile: bool = False,
    vcpus: int | None = None,
    dry_run: bool = False,
    repo_paths: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Run the in-process warm batch and write one result.json per target plus group.json."""
    from harness.manifest import write_manifest
    from harness.run import ACCEL, OOM_PATTERNS

    if backend not in ("neuron", "gpu"):
        raise backends.ConfigError("warmbatch is for the neuron and gpu backends")
    if backend == "gpu" and workers != 1:
        raise backends.ConfigError("one GPU per process: --workers 1 on g7e.2xlarge")
    if backend == "neuron":
        cap = 8 if lnc == 1 else 4
        if not 1 <= workers <= cap:
            raise backends.ConfigError(f"--workers must be 1..{cap} at LNC={lnc}")
    by_id = {t["id"]: t for t in targets["targets"]}
    ids = [p.stem for p in sorted(inputs.glob("*.yaml")) if not p.stem.endswith(DUP)]
    if only:
        ids = [i for i in ids if i in set(only)]
    if not ids:
        raise backends.ConfigError("no inputs selected")
    tokens = {i: int(by_id.get(i, {}).get("tokens") or 0) for i in ids}
    shards = assign(tokens, min(workers, len(ids)))
    group = f"{backend}-lnc{lnc}-w{workers}-ds{ds}-s{seed}" + (
        f"-{variant}" if variant else ""
    )
    group += f"-{uuid.uuid4().hex[:6]}"
    gdir = out_root / group
    vc = vcpus or os.cpu_count() or 1
    cpus = cpu_sets(len(shards), vc) if backend == "neuron" else [None] * len(shards)
    threads = max((vc // len(shards)), 1)

    invs = []
    for i, shard in enumerate(shards):
        sdir = gdir / "shards" / f"w{i}"
        sdir.mkdir(parents=True, exist_ok=True)
        for tid in shard:
            shutil.copy(inputs / f"{tid}.yaml", sdir / f"{tid}.yaml")
            shutil.copy(inputs / f"{tid}.yaml", sdir / f"{tid}{DUP}.yaml")
        cfg = backends.RunConfig(
            backend=backend,
            target_id=f"shard{i}",
            input_path=sdir,
            out_dir=gdir / f"w{i}",
            seed=seed,
            diffusion_samples=ds,
            lnc=lnc,
            visible_core=i if backend == "neuron" else None,
            atom_graph_break=atom_graph_break,
            variant=variant,
            max_msa_seqs=max_msa_seqs,
        )
        inv = backends.build(cfg)
        env = dict(inv.env)
        argv = list(inv.argv)
        if backend == "neuron":
            env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
            if profile:
                env.update(inv.profile_env)
            if cpus[i] and shutil.which("taskset"):
                argv = ["taskset", "-c", str(cpus[i]), *argv]
        elif profile and inv.profile_argv:
            argv = list(inv.profile_argv)
        invs.append((cfg, inv, argv, env))

    manifest = manifest_fn(
        accelerator=ACCEL[backend],
        lnc=lnc,
        repo_paths=repo_paths or {},
        extra={
            "timing_method": "in-process-warm",
            "dry_run": dry_run,
            "profile": profile,
        },
    )
    mpath = write_manifest(manifest, gdir / "manifest.json")

    outs: list[dict[str, Any]] = [{} for _ in invs]

    def work(i: int) -> None:
        cfg, inv, argv, env = invs[i]
        if dry_run:
            outs[i] = {
                "rc": None,
                "wall": None,
                "per_record": {},
                "out": "",
                "argv": argv,
            }
            return
        rc, out, wall = executor(argv, env)
        tj = inv.timing_json
        timing = json.loads(tj.read_text()) if tj and tj.exists() else {}
        outs[i] = {"rc": rc, "wall": wall, "per_record": timing.get("per_record_s", {}), "out": out,
                   "timing": timing, "argv": argv}  # fmt: skip

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(invs)) as pool:
        list(pool.map(work, range(len(invs))))
    group_wall = time.perf_counter() - t0

    docs = []
    for i, shard in enumerate(shards):
        cfg, inv, argv, env = invs[i]
        o = outs[i]
        for tid in shard:
            doc = schema.empty_result(
                target_id=tid, backend=backend, seed=seed, diffusion_samples=ds,
                recycling_steps=cfg.recycling_steps, sampling_steps=cfg.sampling_steps,
                tokens=tokens[tid] or None, lnc=lnc, atom_graph_break=atom_graph_break,
                worker_index=i, workers=len(shards), run_group=group, command=argv,
                manifest_path=str(mpath),
            )  # fmt: skip
            doc.update(
                variant=variant,
                max_msa_seqs=max_msa_seqs,
                timing_method="in-process-warm",
            )
            doc["env"] = env
            if dry_run:
                doc["status"] = "dry-run"
            else:
                pair = order_from_timing(o["per_record"], tid)
                if o["rc"] == 0 and pair:
                    first, second = pair
                    pdir = (
                        cfg.out_dir
                        / f"boltz_results_{cfg.input_path.name}"
                        / "predictions"
                        / first
                    )

                    def rank(p: Path) -> int:
                        m = re.search(r"_model_(\d+)", p.name)
                        return int(m.group(1)) if m else 0

                    doc.update(
                        status="measured",
                        returncode=0,
                        wall_cold_s=o["per_record"][first],
                        wall_warm_s=o["per_record"][second],
                        model_s=o["per_record"][second],
                        peak_accel_mem_bytes=o["timing"].get("peak_accel_mem_bytes"),
                        peak_accel_mem_source=o["timing"].get("peak_accel_mem_source"),
                        predictions=[
                            str(p) for p in sorted(pdir.glob("*_model_*.cif"), key=rank)
                        ],
                        precision=o["timing"].get("precision"),
                    )
                    if profile:
                        doc["profile_path"] = str(inv.profile_path)
                else:
                    oom = any(re.search(p, o["out"]) for p in OOM_PATTERNS)
                    doc["status"] = "did-not-run" if oom else "failed"
                    doc["returncode"] = o["rc"]
                    doc["error"] = (o["out"] or "")[
                        -3000:
                    ] or "record missing from timing"
            schema.validate_result(doc)
            d = gdir / tid
            d.mkdir(parents=True, exist_ok=True)
            (d / "result.json").write_text(json.dumps(doc, indent=2) + "\n")
            docs.append(doc)

    measured = [d for d in docs if d["status"] == "measured"]
    summary = {
        "schema": "sp-group/1.1",
        "run_group": group,
        "timing_method": "in-process-warm",
        "backend": backend,
        "lnc": lnc,
        "workers": len(shards),
        "diffusion_samples": ds,
        "seed": seed,
        "threads_per_worker": threads if backend == "neuron" else None,
        "cpu_sets": cpus,
        "shards": shards,
        "dry_run": dry_run,
        "measured": len(measured),
        "of": len(docs),
        # Each process predicts every target twice, so this wall covers 2x the work.
        "group_wall_s": None if dry_run else group_wall,
        "worker_wall_s": [o.get("wall") for o in outs],
    }
    (gdir / "group.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
