# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""End-to-end batch run: the whole target directory through one predict process per worker.

    python -m harness.run directory --backend gpu --inputs inputs --out-dir results/raw/g7e-dir/ds1-s0 \
        --seed 0 --ds 1
    python -m harness.run directory --backend neuron --lnc 2 --workers 4 --inputs inputs \
        --out-dir results/raw/trn2-dir/lnc2-ds1-s0 --seed 0 --ds 1

The port's advice is to pass a directory so start-up (imports, weight load, cache load) is paid
once; the GPU backend runs the batch the same way. Each worker gets a shard of the inputs and
one process; the run is cold by design (start-up included). `directory.json` records the group
wall (all workers), each worker's wall and per-record model time. The cost table adds the
host's other billed stages (`harness.stages`: boot, cache restore, database preload, MSA) and
amortises the total over the targets in the directory: the "end-to-end batch" column.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness import backends

Executor = Callable[[list[str], dict[str, str]], tuple[int, str, float]]


def run_directory(
    *,
    backend: str,
    inputs: Path,
    out_dir: Path,
    seed: int,
    ds: int,
    lnc: int | None,
    workers: int,
    dry_run: bool,
    executor: Executor,
    manifest_fn: Callable[..., dict[str, Any]],
    repo_paths: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Shard the inputs, run one process per worker concurrently, write directory.json."""
    from harness.manifest import write_manifest
    from harness.run import ACCEL

    if backend not in ("neuron", "gpu"):
        raise backends.ConfigError("directory runs are for the neuron and gpu backends")
    if backend == "gpu" and workers != 1:
        raise backends.ConfigError("g7e.2xlarge has one GPU: --workers 1")
    if backend == "neuron":
        cap = 8 if lnc == 1 else 4
        if not 1 <= workers <= cap:
            raise backends.ConfigError(f"--workers must be 1..{cap} at LNC={lnc}")
    files = sorted(inputs.glob("*.yaml"))
    if not files:
        raise backends.ConfigError(f"no inputs in {inputs}")
    invs = []
    for i in range(workers):
        shard = out_dir / "shards" / f"w{i}"
        shard.mkdir(parents=True, exist_ok=True)
        for f in files[i::workers]:
            shutil.copy(f, shard / f.name)
        cfg = backends.RunConfig(
            backend=backend,
            target_id=f"shard{i}",
            input_path=shard,
            out_dir=out_dir / f"w{i}",
            seed=seed,
            diffusion_samples=ds,
            lnc=lnc,
            visible_core=i if backend == "neuron" else None,
        )
        invs.append(backends.build(cfg))
    manifest = manifest_fn(
        accelerator=ACCEL[backend],
        lnc=lnc,
        repo_paths=repo_paths or {},
        extra={"directory_run": True, "dry_run": dry_run},
    )
    mpath = write_manifest(manifest, out_dir / "manifest.json")

    results: list[dict[str, Any]] = [{} for _ in invs]

    def work(i: int) -> None:
        if dry_run:
            results[i] = {
                "worker": i,
                "returncode": None,
                "wall_s": None,
                "argv": invs[i].argv,
            }
            return
        rc, out, wall = executor(invs[i].argv, invs[i].env)
        tpath = invs[i].timing_json
        timing = json.loads(tpath.read_text()) if tpath and tpath.exists() else {}
        results[i] = {
            "worker": i,
            "returncode": rc,
            "wall_s": wall if rc == 0 else None,
            "per_record_model_s": timing.get("per_record_s", {}),
            "error": None if rc == 0 else out[-2000:],
        }

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(workers)))
    wall = time.perf_counter() - t0
    ok = not dry_run and all(r["returncode"] == 0 for r in results)
    doc = {
        "schema": "sp-directory/1.0",
        "backend": backend,
        "lnc": lnc,
        "workers": workers,
        "diffusion_samples": ds,
        "seed": seed,
        "n_targets": len(files),
        "targets": [f.stem for f in files],
        "status": "dry-run" if dry_run else ("measured" if ok else "failed"),
        "group_wall_s": wall if ok else None,
        "workers_detail": results,
        "manifest_path": str(mpath),
    }
    (out_dir / "directory.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc
