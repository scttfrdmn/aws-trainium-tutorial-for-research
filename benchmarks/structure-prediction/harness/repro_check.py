# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Phase A task 0: compare a reproduction run with the port's published warm number.

    python -m harness.repro_check --timing ~/a0/timing.json --expect 7.5 --tol 0.10

The port's method (boltz-neuron README): warm model time per prediction, median of 3 after 1
warm-up, in one predict process. The timing JSON comes from `harness.entry` run on a directory
of 4 copies of the input; batch 1 is the warm-up and the median of batches 2-4 is compared.
Exit code 0 within tolerance, 3 outside it (the phase stops until the mismatch is explained).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def check(batch_s: list[float], expect: float, tol: float) -> dict[str, float | bool]:
    """Median of the warm batches vs the expected value."""
    if len(batch_s) < 4:
        raise ValueError(f"need 1 warm-up + 3 timed predictions, got {len(batch_s)}")
    median = statistics.median(batch_s[1:4])
    rel = (median - expect) / expect
    return {
        "median_s": median,
        "expected_s": expect,
        "relative_diff": rel,
        "within": abs(rel) <= tol,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="harness.repro_check")
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--expect", type=float, required=True)
    ap.add_argument("--tol", type=float, default=0.10)
    ns = ap.parse_args(argv)
    res = check(json.loads(ns.timing.read_text())["batch_s"], ns.expect, ns.tol)
    print(json.dumps(res))
    return 0 if res["within"] else 3


if __name__ == "__main__":
    sys.exit(main())
