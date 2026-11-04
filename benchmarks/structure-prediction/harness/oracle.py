# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""CPU fp32 oracle: parallel pinned workers, calibration, and a budget estimate.

    # 1. calibrate on one ~300-residue target: solo, then N concurrent copies
    python -m harness.run oracle-calibrate --target-id T1042 --input inputs/T1042.yaml \
        --out-root results/oracle-calibration --workers 8 --threads 24
    # 2. estimate the budget from the calibration (never from the toy run)
    python -m harness.oracle budget --calibration results/oracle-calibration/calibration.json \
        --targets targets/targets.json --workers 8 --extra T1061:msa2048
    # 3. run the oracle: every gated target x seeds 0,1, largest first, one job per worker
    python -m harness.run oracle --targets-json targets/targets.json --inputs inputs \
        --out-root results/oracle --workers 8 --threads 24

PyTorch CPU inference stops scaling well past a few dozen intra-op threads, so the oracle runs
several workers, each with `threads` intra-op threads pinned (taskset) to its own cores. The
per-worker slowdown under contention (memory bandwidth, caches) is measured by the calibration
before the layout is committed. The budget scales the calibration by (L/L0)^k for k = 2 and 3,
a range rather than a point: the diffusion token transformer is ~L^2 and the trunk's triangle
operations ~L^3, and their mix shifts with length.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import heapq
import json
import queue
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from harness import backends, pins

Predict = Callable[..., dict[str, Any]]


def cores_for(worker: int, threads: int) -> str:
    """Contiguous core list for a worker, e.g. worker 1 x 24 threads -> '24-47'."""
    return f"{worker * threads}-{(worker + 1) * threads - 1}"


def oracle_jobs(
    targets: dict[str, Any],
    only: Sequence[str] | None = None,
    seeds: Sequence[int] = pins.SEEDS,
) -> list[tuple[str, int, int]]:
    """(target, seed, tokens) for every gated target (or `only`), largest first."""
    jobs = []
    for t in targets["targets"]:
        if only:
            if t["id"] not in only:
                continue
        elif not t.get("gated"):
            continue
        for s in seeds:
            jobs.append((t["id"], s, int(t.get("tokens") or t["length"])))
    return sorted(jobs, key=lambda j: (-j[2], j[0], j[1]))


def oracle_dir(out_root: Path, tid: str, seed: int, variant: str | None) -> Path:
    """results/oracle/<target>/seed<s>[-<variant>]."""
    return out_root / tid / (f"seed{seed}" + (f"-{variant}" if variant else ""))


def run_oracle(
    jobs: list[tuple[str, int, int]],
    *,
    inputs: Path,
    out_root: Path,
    workers: int,
    threads: int,
    predict: Predict,
    variant: str | None = None,
    max_msa_seqs: int | None = None,
    dry_run: bool = False,
    pin: bool = True,
    **predict_kw: Any,
) -> dict[str, Any]:
    """Run oracle jobs on `workers` pinned workers pulling from one queue (largest first)."""
    q: queue.Queue[tuple[str, int, int]] = queue.Queue()
    for j in jobs:
        q.put(j)
    done: list[dict[str, Any]] = []

    def worker(i: int) -> None:
        while True:
            try:
                tid, seed, tokens = q.get_nowait()
            except queue.Empty:
                return
            cfg = backends.RunConfig(
                backend="cpu",
                target_id=tid,
                input_path=inputs / f"{tid}.yaml",
                out_dir=oracle_dir(out_root, tid, seed, variant),
                seed=seed,
                threads=threads,
                cpu_cores=cores_for(i, threads) if pin else None,
                variant=variant,
                max_msa_seqs=max_msa_seqs,
            )
            t0 = time.perf_counter()
            doc = predict(
                cfg,
                dry_run=dry_run,
                profile=False,
                tokens=tokens,
                run_group=f"oracle-w{workers}x{threads}",
                worker_index=i,
                workers=workers,
                **predict_kw,
            )
            done.append(
                {
                    "target_id": tid,
                    "seed": seed,
                    "tokens": tokens,
                    "worker": i,
                    "status": doc["status"],
                    "model_s": doc.get("model_s"),
                    "wall_s": time.perf_counter() - t0,
                }
            )

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    summary = {
        "schema": "sp-oracle-run/1.0",
        "workers": workers,
        "threads": threads,
        "variant": variant,
        "dry_run": dry_run,
        "wall_s": None if dry_run else time.perf_counter() - t0,
        "jobs": sorted(done, key=lambda d: (d["target_id"], d["seed"])),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"oracle_run{'-' + variant if variant else ''}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def calibrate(
    *,
    target_id: str,
    input_path: Path,
    tokens: int,
    out_root: Path,
    workers: int,
    threads: int,
    predict: Predict,
    pin: bool = True,
    **predict_kw: Any,
) -> dict[str, Any]:
    """Solo run, then `workers` concurrent copies of the same target; per-worker slowdown."""

    def one(i: int, tag: str) -> dict[str, Any]:
        cfg = backends.RunConfig(
            backend="cpu",
            target_id=target_id,
            input_path=input_path,
            out_dir=out_root / tag / f"w{i}",
            seed=0,
            threads=threads,
            cpu_cores=cores_for(i, threads) if pin else None,
        )
        return predict(
            cfg,
            dry_run=False,
            profile=False,
            tokens=tokens,
            run_group=f"calibration-{tag}",
            worker_index=i,
            workers=1 if tag == "solo" else workers,
            **predict_kw,
        )

    solo = one(0, "solo")
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        conc = list(pool.map(lambda i: one(i, "concurrent"), range(workers)))
    ok = solo["status"] == "measured" and all(d["status"] == "measured" for d in conc)
    res: dict[str, Any] = {
        "schema": "sp-oracle-calibration/1.0",
        "target_id": target_id,
        "tokens": tokens,
        "workers": workers,
        "threads": threads,
        "ok": ok,
        "solo": _calib_row(solo),
        "concurrent": [_calib_row(d) for d in conc],
    }
    if ok:
        cm = [d["model_s"] for d in conc]
        res["slowdown"] = statistics.median(cm) / solo["model_s"]
        res["startup_s"] = solo["wall_cold_s"] - solo["model_s"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "calibration.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def _calib_row(doc: dict[str, Any]) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    tpath = (
        Path(doc["manifest_path"]).parent / "timing.json"
        if doc.get("manifest_path")
        else None
    )
    if tpath and tpath.exists():
        timing = json.loads(tpath.read_text())
    return {
        "status": doc["status"],
        "model_s": doc.get("model_s"),
        "wall_s": doc.get("wall_cold_s"),
        "peak_host_rss_bytes": timing.get("peak_host_rss_bytes"),
    }


def lpt_makespan(durations: Sequence[float], workers: int) -> float:
    """Makespan of longest-processing-time-first list scheduling on `workers` machines."""
    loads = [0.0] * workers
    heapq.heapify(loads)
    for d in sorted(durations, reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + d)
    return max(loads)


def budget(
    cal: dict[str, Any],
    targets: dict[str, Any],
    workers: int,
    extra: Sequence[str] = (),
    host_ram_gib: float | None = None,
) -> dict[str, Any]:
    """Oracle wall-time estimate from a calibration, for scaling exponents k = 2 and 3."""
    if not cal.get("ok"):
        raise ValueError("calibration did not complete; rerun it before estimating")
    l0 = cal["tokens"]
    m0 = cal["solo"]["model_s"]
    slow = cal["slowdown"] if workers > 1 else 1.0
    start = max(cal.get("startup_s", 0.0), 0.0)
    jobs = oracle_jobs(targets)
    by_id = {t["id"]: t for t in targets["targets"]}
    for item in extra:  # e.g. "T1061:msa2048"
        tid = item.split(":")[0]
        tok = int(by_id[tid].get("tokens") or by_id[tid]["length"])
        jobs += [(item, s, tok) for s in pins.SEEDS]
    out: dict[str, Any] = {
        "calibration_target": cal["target_id"],
        "calibration_tokens": l0,
        "workers": workers,
        "slowdown_applied": slow,
        "jobs": len(jobs),
        "estimates": {},
    }
    for k in (2.0, 3.0):
        durs = [start + m0 * slow * (tok / l0) ** k for _, _, tok in jobs]
        span = lpt_makespan(durs, workers)
        out["estimates"][f"k={k:g}"] = {
            "makespan_h": span / 3600,
            "longest_job_h": max(durs) / 3600,
            "instance_hours": span / 3600,
        }
    rss = cal["solo"].get("peak_host_rss_bytes")
    if rss:
        lmax = max(tok for _, _, tok in jobs)
        proj = (
            rss * (lmax / l0) ** 2
        )  # pair representation ~ L^2; a heuristic, not a bound
        out["memory"] = {
            "calibration_rss_gib": rss / 2**30,
            "projected_largest_rss_gib_l2": proj / 2**30,
            "projected_workers_total_gib": workers * proj / 2**30,
            "host_ram_gib": host_ram_gib,
            "fits": None
            if host_ram_gib is None
            else workers * proj / 2**30 < 0.9 * host_ram_gib,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """`budget` CLI (the run and calibrate commands live in harness.run)."""
    ap = argparse.ArgumentParser(prog="harness.oracle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--calibration", type=Path, required=True)
    b.add_argument("--targets", type=Path, required=True)
    b.add_argument("--workers", type=int, required=True)
    b.add_argument("--extra", nargs="*", default=[], help="e.g. T1061:msa2048")
    b.add_argument("--host-ram-gib", type=float)
    ns = ap.parse_args(argv)
    res = budget(
        json.loads(ns.calibration.read_text()),
        json.loads(ns.targets.read_text()),
        ns.workers,
        ns.extra,
        ns.host_ram_gib,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
