# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Time ColabFold-search per target, one query at a time (as in Didi et al. 2025).

    # g7e.2xlarge: MMseqs2-GPU, keeps the .a3m files (the shared inputs for every platform)
    python msa/run_search.py --mode gpu --targets targets/targets.json --db /data/colabfold \
        --out /data/msas --timing msa_gpu_timing.json
    # r7a.48xlarge: MMseqs2-CPU on the CPU-built index of the same databases; output discarded
    python msa/run_search.py --mode cpu --targets targets/targets.json --db /data/colabfold \
        --out /tmp/msa-cpu-scratch --timing msa_cpu_timing.json --discard

One search per unique protein chain (`msa_id`). GPU mode expects `mmseqs gpuserver` running
for both databases (msa_gpu.sh starts them); CPU mode expects the index resident in RAM
(msa_cpu_timing.sh preloads it with vmtouch). Both use `--db-load-mode 2`.

The timing file carries per-target seconds, versions, the host and its prices (from the run
manifest), so the cost table can bill the MSA term to the right host.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.manifest import capture_manifest, write_manifest  # noqa: E402

DATABASES = {"uniref30": "uniref30_2302", "colabfold_envdb": "colabfold_envdb_202108"}
COLABFOLD = "1.6.3"
MMSEQS = "18-8cc5c"

Runner = Callable[[list[str], dict[str, str]], tuple[int, float]]


def run_timed(argv: list[str], env: dict[str, str]) -> tuple[int, float]:
    """Run a command, return (returncode, wall seconds)."""
    t0 = time.perf_counter()
    rc = subprocess.run(argv, env={**os.environ, **env}).returncode
    return rc, time.perf_counter() - t0


def queries(targets: dict[str, Any]) -> dict[str, str]:
    """msa_id -> sequence for every protein chain (deduplicated)."""
    out: dict[str, str] = {}
    for t in targets["targets"]:
        for ch in t["chains"]:
            if ch["entity"] == "protein" and ch.get("msa_id"):
                out.setdefault(ch["msa_id"], ch["sequence"])
    return out


def search_argv(
    mode: str, fasta: Path, db: Path, out: Path, mmseqs: str, threads: int
) -> list[str]:
    """colabfold_search argv for one query."""
    argv = [
        "colabfold_search",
        "--mmseqs",
        mmseqs,
        "--db-load-mode",
        "2",
        "--threads",
        str(threads),
        str(fasta),
        str(db),
        str(out),
    ]
    if mode == "gpu":
        argv += ["--gpu", "1", "--gpu-server", "1"]
    return argv


def main(argv: list[str] | None = None, runner: Runner = run_timed) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="msa/run_search.py")
    ap.add_argument("--mode", choices=("gpu", "cpu"), required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument(
        "--discard", action="store_true", help="delete outputs after timing (CPU run)"
    )
    ap.add_argument("--only", nargs="*", help="restrict to these msa_ids")
    ns = ap.parse_args(argv)

    qs = queries(json.loads(ns.targets.read_text()))
    if ns.only:
        qs = {k: v for k, v in qs.items() if k in set(ns.only)}
    ns.out.mkdir(parents=True, exist_ok=True)
    per: dict[str, float | None] = {}
    errors: dict[str, int] = {}
    env = {"CUDA_VISIBLE_DEVICES": "0"} if ns.mode == "gpu" else {}
    for mid, seq in qs.items():
        with tempfile.TemporaryDirectory() as td:
            fasta = Path(td, f"{mid}.fasta")
            fasta.write_text(f">{mid}\n{seq}\n")
            dest = Path(
                td, "out"
            )  # one query per call; the .a3m is renamed to <msa_id>.a3m
            rc, wall = runner(
                search_argv(ns.mode, fasta, ns.db, dest, ns.mmseqs, ns.threads), env
            )
            per[mid] = wall if rc == 0 else None
            if rc != 0:
                errors[mid] = rc
            elif not ns.discard:
                a3ms = sorted(dest.glob("*.a3m"))
                if len(a3ms) != 1:
                    errors[mid] = -1
                    per[mid] = None
                else:
                    shutil.move(str(a3ms[0]), ns.out / f"{mid}.a3m")
        status = f"{wall:.2f}" if rc == 0 else f"FAILED rc={rc}"
        print(f"{mid}\t{status}", flush=True)

    manifest = capture_manifest(
        accelerator="gpu" if ns.mode == "gpu" else "cpu",
        extra={
            "colabfold": COLABFOLD,
            "mmseqs": MMSEQS,
            "databases": DATABASES,
            "mode": ns.mode,
        },
    )
    mpath = write_manifest(manifest, ns.timing.with_suffix(".manifest.json"))
    prices = manifest.get("prices") or {}
    doc = {
        "schema": "sp-msa-timing/1.0",
        "mode": ns.mode,
        "host_instance_type": (manifest.get("instance") or {}).get("instance_type"),
        "price_od_usd_per_h": (prices.get("on_demand") or {}).get("usd_per_hour"),
        "price_spot_usd_per_h": (prices.get("spot") or {}).get("usd_per_hour"),
        "colabfold": COLABFOLD,
        "mmseqs": MMSEQS,
        "databases": DATABASES,
        "threads": ns.threads,
        "per_target_s": per,
        "errors": errors,
        "manifest_path": str(mpath),
    }
    ns.timing.parent.mkdir(parents=True, exist_ok=True)
    ns.timing.write_text(json.dumps(doc, indent=2) + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
