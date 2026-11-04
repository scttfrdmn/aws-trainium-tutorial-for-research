# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Per-host stage timer for the end-to-end batch column.

    python -m harness.stages boot  --out ~/stages.json --role trn2     # first thing after login
    python -m harness.stages time setup         --out ~/stages.json -- bash scripts/setup.sh
    python -m harness.stages time cache_restore --out ~/stages.json -- bash scripts/restore_cache.sh X
    python -m harness.stages time msa           --out ~/stages.json -- bash msa/msa_gpu.sh ...

`boot` records seconds since the kernel started (/proc/uptime) at the moment it is called, a
proxy for the billed time from "running" to a usable host; it also stores the instance type
from IMDS so the cost table can match the file to a platform. `time` runs a command and records
its wall time and exit code.

Billed in the end-to-end batch column by default: boot, cache_restore, db_preload, msa (plus
the directory run itself). Measured and reported but not billed by default: setup, db_build,
cache_build. These are one-time per AMI or per database/cache build, which a deployment bakes
in; `harness.cost_table --bill-setup` bills them too.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "sp-stages/1.0"
BILLED = ("boot", "cache_restore", "db_preload", "msa")
SETUP = ("setup", "db_build", "cache_build")


def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    return {"schema": SCHEMA, "role": None, "instance_type": None, "stages": {}}


def _save(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def record_boot(
    path: Path,
    role: str,
    uptime: Callable[[], float] | None = None,
    instance_type: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    """Record boot seconds (first call wins) and the instance type."""
    doc = _load(path)
    if "boot" not in doc["stages"]:
        up = uptime() if uptime else float(Path("/proc/uptime").read_text().split()[0])
        doc["stages"]["boot"] = {
            "wall_s": up,
            "rc": 0,
            "how": "/proc/uptime at first call",
        }
    doc["role"] = role
    if doc.get("instance_type") is None:
        try:
            if instance_type:
                doc["instance_type"] = instance_type()
            else:
                from harness.manifest import instance_identity

                doc["instance_type"] = instance_identity()["instance_type"]
        except Exception as exc:  # off EC2: recorded, not guessed
            doc["instance_type_error"] = str(exc)
    _save(path, doc)
    return doc


def time_stage(
    path: Path,
    name: str,
    argv: Sequence[str],
    runner: Callable[[Sequence[str]], int] | None = None,
) -> dict[str, Any]:
    """Run argv, record its wall time and exit code under `name`."""
    doc = _load(path)
    t0 = time.perf_counter()
    rc = runner(argv) if runner else subprocess.run(list(argv)).returncode
    doc["stages"][name] = {
        "wall_s": time.perf_counter() - t0,
        "rc": rc,
        "argv": list(argv),
    }
    _save(path, doc)
    return doc


def billed_seconds(
    doc: dict[str, Any], bill_setup: bool = False, lnc: int | None = None
) -> tuple[float, list[str]]:
    """Sum of billed stage seconds and the names included (failed stages are excluded).

    Per-layout stages on one host are named `<stage>_lnc<n>` (e.g. `cache_restore_lnc2`); when
    `lnc` is given they replace the plain stage of the same name.
    """
    names = [*BILLED, *(SETUP if bill_setup else ())]
    total, used = 0.0, []
    stages = doc.get("stages", {})
    for base in names:
        n = (
            f"{base}_lnc{lnc}"
            if lnc is not None and f"{base}_lnc{lnc}" in stages
            else base
        )
        st = stages.get(n)
        if st and st.get("rc") == 0:
            total += float(st["wall_s"])
            used.append(n)
    return total, used


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in args:
        i = args.index("--")
        args, rest = args[:i], args[i + 1 :]
    ap = argparse.ArgumentParser(prog="harness.stages")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("boot")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--role", required=True)
    t = sub.add_parser("time")
    t.add_argument("name")
    t.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(args)
    if ns.cmd == "boot":
        doc = record_boot(ns.out, ns.role)
        print(json.dumps(doc["stages"]["boot"]))
        return 0
    if not rest:
        ap.error("time NAME --out F -- command ...")
    doc = time_stage(ns.out, ns.name, rest)
    st = doc["stages"][ns.name]
    print(json.dumps({ns.name: st["wall_s"], "rc": st["rc"]}))
    return int(st["rc"])


if __name__ == "__main__":
    sys.exit(main())
