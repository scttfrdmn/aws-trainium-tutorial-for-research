# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""$/structure tables from result JSONs (Phase A task 5).

    python -m harness.cost_table --results RESULTS_ROOT --targets targets/targets.json \
        --msa-gpu msa_gpu_timing.json --msa-cpu msa_cpu_timing.json \
        --out results/phaseA.csv --markdown results/phaseA.md

Definitions (fixed with the schema; later phases append rows, never redefine columns):

* One row per target × platform × ds. A platform is (instance type, backend, LNC, workers).
  Rows need both seeds; a row with a missing seed is written with `ran=false` and no numbers
  (no single-seed numbers in any table).
* "Structure" = one prediction call for one target (the top-ranked model is what a user
  receives), at the row's ds.
* DL seconds per structure = mean over the two seeds of `model_s` (predict-loop time, excluding
  process start-up, weight load and output writing) divided by `workers`: N concurrent workers
  on one device each hold 1/N of it. Rows without `model_s` get no DL numbers.
* DL $/structure = DL seconds × instance $/h ÷ 3600, at on-demand and at spot, both read from
  the run manifests (prices fetched at run time; never hard-coded here).
* End-to-end adds the MSA term: MSA seconds for the target × the MSA host's $/h. On g7e the MSA
  runs on the same GPU (MMseqs2-GPU timing file); for trn2 it is MMseqs2-CPU on a separate
  r7a.48xlarge host (CPU timing file), billed at that host's prices.
* ceiling_tokens = the largest token count that ran on that platform at that ds (over every
  target including the ceiling complexes); repeated on each of the platform's rows.
* Mean rows (`target_id = __mean__`) average the CASP14 targets that ran, T1047s1 included
  (timing-only: no ground truth, but its DL time is valid and it keeps the set identical to
  Didi et al.'s).
* 1.1 columns: `variant` / `max_msa_seqs` label secondary rows (T1061 at --max_msa_seqs 2048;
  no mean row). Per-sample columns divide the per-structure ones by ds, comparable with the
  port's throughput tables. "DL warm" is the `dl_*` columns (excludes start-up; Didi's DL
  column). The end-to-end batch columns, on mean rows, bill what EC2 bills (instance boot,
  model load inside the directory run, MSA, compile-cache restore; `harness.stages`) amortised
  over the directory run's targets, the same way on every platform; start-up is never billed
  per structure. One-time setup stages are reported but billed only with --bill-setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harness.schema import COST_COLUMNS, COST_SCHEMA

PHASE = "A"


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _manifest_path(result_path: Path, doc: dict[str, Any]) -> Path | None:
    """The run manifest: the recorded path, else beside the result, else one level up.

    warmbatch writes one manifest per worker group, above the per-target results.
    """
    rec = Path(doc["manifest_path"]) if doc.get("manifest_path") else None
    for cand in (
        rec,
        result_path.parent / "manifest.json",
        result_path.parent.parent / "manifest.json",
    ):
        if cand is not None and cand.is_absolute() and cand.exists():
            return cand
    return None


def _price_info(
    result_path: Path, doc: dict[str, Any], fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Instance, region, on-demand / spot / Capacity Block $/h and where they came from.

    The run manifest's prices come first. Where the manifest has none (the price lookup failed
    on the host), `fallback` (a `harness.prices fetch` document) fills them, and the row's
    price_source says so.
    """
    info: dict[str, Any] = dict.fromkeys(
        ("inst", "region", "od", "spot", "cb", "source")
    )
    mpath = _manifest_path(result_path, doc)
    if mpath is None:
        return info
    m = _load(mpath)
    ident = m.get("instance") or {}
    prices = m.get("prices") or {}
    info.update(inst=ident.get("instance_type"), region=ident.get("region"))
    keys = (("od", "on_demand"), ("spot", "spot"), ("cb", "capacity_block"))
    for k, key in keys:
        info[k] = (prices.get(key) or {}).get("usd_per_hour")
    src = ["manifest"] if any(info[k] is not None for k, _ in keys) else []
    if fallback and info["inst"]:
        fb = (fallback.get("prices") or {}).get(
            f"{info['inst']}@{info['region']}"
        ) or {}
        used = False
        for k, key in keys:
            v = (fb.get(key) or {}).get("usd_per_hour")
            if info[k] is None and v is not None:
                info[k], used = v, True
        if used:
            src.append(
                f"prices.json fetched {str(fallback.get('fetched_at', ''))[:10]}"
            )
    info["source"] = " + ".join(src) or None
    return info


def _prices(
    result_path: Path, doc: dict[str, Any], fallback: dict[str, Any] | None = None
) -> tuple[str | None, float | None, float | None]:
    i = _price_info(result_path, doc, fallback)
    return i["inst"], i["od"], i["spot"]


def _gate_for(
    gates: Sequence[dict[str, Any]], backend: str, lnc: Any, ds: Any
) -> dict[str, Any] | None:
    """The calibrated gate result whose accuracy-run filter is this platform (primary rows)."""
    for g in gates:
        acc = g.get("acc") or {}
        if (
            acc.get("backend") == backend
            and acc.get("diffusion_samples") == str(ds)
            and acc.get("lnc", "") == ("" if lnc is None else str(lnc))
            and not acc.get("variant")
        ):
            return g
    return None


def _gate_note(g: dict[str, Any] | None, tid: str | None) -> str | None:
    if g is None:
        return None
    if tid is None:  # the platform's mean row: the phase verdict
        verdict = f"{g.get('method')} {'PASS' if g.get('phase_pass') else 'FAIL'}"
        reasons = "; ".join(g.get("reasons") or [])
        return verdict + (f": {reasons[:240]}" if reasons else "")
    if tid in (g.get("incomplete") or []):
        return "incomplete"
    if tid in (g.get("cap_exceeded") or []):
        return "over per-target cap"
    if tid in (g.get("timing_only") or []):
        return None
    return "ok"


def _msa(path: Path | None) -> dict[str, Any]:
    return _load(path) if path and path.exists() else {}


def _mul(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a * b / 3600.0


def _add(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


def build_rows(
    results_root: Path,
    targets: dict[str, Any],
    msa_gpu: dict[str, Any],
    msa_cpu: dict[str, Any],
    stages: Sequence[dict[str, Any]] = (),
    bill_setup: bool = False,
    prices: dict[str, Any] | None = None,
    gates: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """All cost rows (per target, then per-platform mean rows)."""
    by_id = {t["id"]: t for t in targets["targets"]}
    groups: dict[tuple[Any, ...], dict[str, list[tuple[Path, dict[str, Any]]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for res in sorted(results_root.rglob("result.json")):
        doc = _load(res)
        if doc["backend"] == "cpu" or doc["status"] == "dry-run":
            continue
        inst, _, _ = _prices(res, doc)
        key = (
            inst,
            doc["backend"],
            doc["lnc"],
            doc["workers"] or 1,
            doc["diffusion_samples"],
            doc.get("variant") or "",
        )
        groups[key][doc["target_id"]].append((res, doc))
    directories = [(p, _load(p)) for p in sorted(results_root.rglob("directory.json"))]

    rows: list[dict[str, Any]] = []
    for (inst, backend, lnc, workers, ds, variant), per_target in sorted(
        groups.items(), key=str
    ):
        platform = f"{inst} {backend}" + (
            f" LNC={lnc} x{workers}"
            if lnc
            else (f" x{workers} (shared GPU)" if workers > 1 else "")
        )
        ran_tokens = [
            d["tokens"]
            for docs in per_target.values()
            for _, d in docs
            if d["status"] == "measured" and d["tokens"]
        ]
        ceiling = max(ran_tokens) if ran_tokens else None
        on_gpu_host = backend in ("gpu", "openfold-trt")
        msa = msa_gpu if on_gpu_host else msa_cpu
        gate = None if variant else _gate_for(gates, backend, lnc, ds)
        plat_rows = []
        for tid, docs in sorted(per_target.items()):
            t = by_id.get(tid, {})
            seeds = sorted(d["seed"] for _, d in docs)
            measured = [(p, d) for p, d in docs if d["status"] == "measured"]
            row: dict[str, Any] = dict.fromkeys(COST_COLUMNS)
            row.update(
                schema=COST_SCHEMA,
                phase=PHASE,
                target_id=tid,
                tokens=t.get("tokens") or docs[0][1]["tokens"],
                platform=platform,
                instance_type=inst,
                backend=backend,
                lnc=lnc,
                workers=workers,
                diffusion_samples=ds,
                seeds=",".join(map(str, seeds)),
                ceiling_tokens=ceiling,
                msa_host=msa.get("host_instance_type"),
                variant=variant or None,
                max_msa_seqs=docs[0][1].get("max_msa_seqs"),
                timing_only=bool(t.get("timing_only")),
                accuracy_gate=_gate_note(gate, tid),
            )
            label = f"secondary ({variant}); " if variant else ""
            label += "no ground truth, timing only; " if t.get("timing_only") else ""
            if sorted(d["seed"] for _, d in measured) != [0, 1]:
                status = sorted({d["status"] for _, d in docs})
                row.update(
                    ran=False,
                    note=label
                    + f"needs both seeds measured; got {status} for seeds {seeds}",
                )
                plat_rows.append(row)
                continue
            model = [d["model_s"] for _, d in measured]
            if any(m is None for m in model):
                row.update(
                    ran=True, note=label + "model_s missing; DL time not reported"
                )
                plat_rows.append(row)
                continue
            dl = statistics.mean(model) / workers
            pinfo = _price_info(*measured[0], prices)
            od, spot, cb = pinfo["od"], pinfo["spot"], pinfo["cb"]
            msa_s = (msa.get("per_target_s") or {}).get(tid)
            msa_od = od if on_gpu_host else msa.get("price_od_usd_per_h")
            msa_spot = spot if on_gpu_host else msa.get("price_spot_usd_per_h")
            row.update(
                ran=True,
                dl_s_per_structure=dl,
                msa_s=msa_s,
                price_od_usd_per_h=od,
                price_spot_usd_per_h=spot,
                msa_price_od_usd_per_h=msa_od,
                msa_price_spot_usd_per_h=msa_spot,
                dl_usd_per_structure_od=_mul(dl, od),
                dl_usd_per_structure_spot=_mul(dl, spot),
                e2e_usd_per_structure_od=_add(_mul(dl, od), _mul(msa_s, msa_od)),
                e2e_usd_per_structure_spot=_add(_mul(dl, spot), _mul(msa_s, msa_spot)),
                dl_s_per_sample=dl / ds,
                dl_usd_per_sample_od=_div(_mul(dl, od), ds),
                dl_usd_per_sample_spot=_div(_mul(dl, spot), ds),
                price_cb_usd_per_h=cb,
                dl_usd_per_structure_cb=_mul(dl, cb),
                dl_usd_per_sample_cb=_div(_mul(dl, cb), ds),
                price_source=pinfo["source"],
                note=label.rstrip("; "),
            )
            plat_rows.append(row)
        rows += plat_rows
        if not variant:  # secondary rows get no mean row
            mean = _mean_row(plat_rows, by_id, platform, ceiling)
            mean["accuracy_gate"] = _gate_note(gate, None)
            mean.update(
                _batch_columns(
                    directories,
                    stages,
                    msa_cpu,
                    inst,
                    backend,
                    lnc,
                    workers,
                    ds,
                    bill_setup,
                    prices,
                )
            )
            rows.append(mean)
    return rows


def _div(a: float | None, b: float) -> float | None:
    return None if a is None else a / b


def _stage_doc(
    stages: Sequence[dict[str, Any]], inst: str | None
) -> dict[str, Any] | None:
    for s in stages:
        if inst is not None and s.get("instance_type") == inst:
            return s
    return None


def _batch_columns(
    directories: Sequence[tuple[Path, dict[str, Any]]],
    stages: Sequence[dict[str, Any]],
    msa_cpu: dict[str, Any],
    inst: str | None,
    backend: str,
    lnc: int | None,
    workers: int,
    ds: int,
    bill_setup: bool,
    prices: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """End-to-end batch: billed host time amortised over the directory run's targets.

    Accelerator host: its billed stages (boot, cache restore, database preload, MSA when the MSA
    runs on it) plus the directory run's group wall (mean over seeds 0 and 1). trn2 also pays the
    separate MSA host's billed stages at that host's prices. Start-up is never billed per
    structure: it only enters here, divided by the number of targets in the directory.
    """
    from harness.stages import billed_seconds

    runs = []
    for p, d in directories:
        if (
            d.get("backend"),
            d.get("lnc"),
            d.get("workers"),
            d.get("diffusion_samples"),
        ) != (
            backend,
            lnc,
            workers,
            ds,
        ) or d.get("status") != "measured":
            continue
        dinst, od, spot = _prices(p, d, prices)
        if dinst == inst:
            runs.append((d, od, spot))
    seeds = sorted(d["seed"] for d, _, _ in runs)
    missing = []
    if seeds != [0, 1]:
        missing.append(f"directory runs for seeds 0 and 1 (have {seeds})")
    host = _stage_doc(stages, inst)
    if host is None:
        missing.append(f"stages.json for {inst}")
    msa_host = None
    if backend == "neuron":
        msa_host = _stage_doc(stages, msa_cpu.get("host_instance_type"))
        if msa_host is None:
            missing.append(
                f"stages.json for the MSA host {msa_cpu.get('host_instance_type')}"
            )
    if missing:
        return {
            "batch_targets": None,
            "note": "batch column needs " + ", ".join(missing),
        }
    assert host is not None
    wall = statistics.mean(d["group_wall_s"] for d, _, _ in runs)
    n = runs[0][0]["n_targets"]
    od, spot = runs[0][1], runs[0][2]
    host_s, used = billed_seconds(host, bill_setup, lnc)
    billed = host_s + wall
    cost_od, cost_spot = _mul(billed, od), _mul(billed, spot)
    note = f"batch: {n} targets; billed stages {used} + directory run"
    if msa_host is not None:
        m_s, m_used = billed_seconds(msa_host, bill_setup)
        cost_od = _add(cost_od, _mul(m_s, msa_cpu.get("price_od_usd_per_h")))
        cost_spot = _add(cost_spot, _mul(m_s, msa_cpu.get("price_spot_usd_per_h")))
        note += f"; MSA host stages {m_used}"
    return {
        "batch_targets": n,
        "batch_billed_s": billed,
        "batch_e2e_usd_per_structure_od": _div(cost_od, n),
        "batch_e2e_usd_per_structure_spot": _div(cost_spot, n),
        "note": note,
    }


def _mean_row(
    plat_rows: list[dict[str, Any]],
    by_id: dict[str, Any],
    platform: str,
    ceiling: int | None,
) -> dict[str, Any]:
    # Timing and cost cover the whole Didi et al. set, T1047s1 (timing only) included.
    casp = [
        r
        for r in plat_rows
        if r["ran"] and by_id.get(r["target_id"], {}).get("set") == "casp14"
    ]
    first = plat_rows[0]
    mean: dict[str, Any] = dict.fromkeys(COST_COLUMNS)
    mean.update(
        {
            k: first[k]
            for k in (
                "schema",
                "phase",
                "platform",
                "instance_type",
                "backend",
                "lnc",
                "workers",
                "diffusion_samples",
                "msa_host",
            )
        }
    )
    mean.update(
        target_id="__mean__",
        seeds="0,1",
        ceiling_tokens=ceiling,
        ran=bool(casp),
        timing_only=False,
    )
    for col in (
        "dl_s_per_structure", "msa_s", "dl_usd_per_structure_od", "dl_usd_per_structure_spot",
        "e2e_usd_per_structure_od", "e2e_usd_per_structure_spot",
        "dl_s_per_sample", "dl_usd_per_sample_od", "dl_usd_per_sample_spot",
        "dl_usd_per_structure_cb", "dl_usd_per_sample_cb",
    ):  # fmt: skip
        vals = [r[col] for r in casp if r[col] is not None]
        mean[col] = statistics.mean(vals) if vals and len(vals) == len(casp) else None
    mean["note"] = f"mean over {len(casp)} CASP14 targets that ran"
    return mean


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """CSV with the fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COST_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r[k] for k in COST_COLUMNS})


def to_markdown(rows: list[dict[str, Any]]) -> str:
    """Markdown table (one row per target × platform × ds, plus mean rows)."""

    def f(v: Any, nd: int) -> str:
        return "—" if v is None or v == "" else f"{float(v):.{nd}f}"

    out = [
        "| target | tokens | platform | ds | DL warm s/structure | DL warm $/structure OD "
        "| DL warm $/structure spot | DL warm $/structure CB | DL warm $/sample OD "
        "| E2E $/structure OD | E2E $/structure spot | E2E batch $/structure OD "
        "| E2E batch $/structure spot | ceiling (tokens) | accuracy gate | note |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        name = "**mean**" if r["target_id"] == "__mean__" else r["target_id"]
        out.append(
            f"| {name} | {r['tokens'] or ''} | {r['platform']} | {r['diffusion_samples']} | "
            f"{f(r['dl_s_per_structure'], 2)} | {f(r['dl_usd_per_structure_od'], 5)} | "
            f"{f(r['dl_usd_per_structure_spot'], 5)} | {f(r.get('dl_usd_per_structure_cb'), 5)} | "
            f"{f(r.get('dl_usd_per_sample_od'), 5)} | "
            f"{f(r['e2e_usd_per_structure_od'], 5)} | {f(r['e2e_usd_per_structure_spot'], 5)} | "
            f"{f(r.get('batch_e2e_usd_per_structure_od'), 5)} | "
            f"{f(r.get('batch_e2e_usd_per_structure_spot'), 5)} | "
            f"{r['ceiling_tokens'] or '—'} | {r.get('accuracy_gate') or ''} | {r['note'] or ''} |"
        )
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="harness.cost_table")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--msa-gpu", type=Path)
    ap.add_argument("--msa-cpu", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    ap.add_argument(
        "--stages", type=Path, nargs="*", default=[], help="per-host stages.json"
    )
    ap.add_argument(
        "--bill-setup", action="store_true", help="also bill setup/db_build/cache_build"
    )
    ap.add_argument(
        "--prices",
        type=Path,
        help="harness.prices fetch output, for manifests without prices",
    )
    ap.add_argument(
        "--gate", type=Path, action="append", default=[],
        help="calibrated gate result (harness.score gate-cal .json), one per platform",
    )  # fmt: skip
    ns = ap.parse_args(argv)
    rows = build_rows(
        ns.results,
        _load(ns.targets),
        _msa(ns.msa_gpu),
        _msa(ns.msa_cpu),
        [_load(s) for s in ns.stages],
        ns.bill_setup,
        _load(ns.prices) if ns.prices else None,
        [_load(g) for g in ns.gate],
    )
    write_csv(rows, ns.out)
    ns.markdown.parent.mkdir(parents=True, exist_ok=True)
    ns.markdown.write_text(to_markdown(rows))
    print(f"{len(rows)} rows -> {ns.out}, {ns.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
