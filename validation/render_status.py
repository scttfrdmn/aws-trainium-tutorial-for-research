#!/usr/bin/env python3
"""Render captured validation artifacts into VALIDATED.md.

Reads every validation/results/*.json and produces a single human-readable status table at the
repo root (VALIDATED.md). This is the honest, machine-generated source of truth for "what has
actually been proven on hardware" -- it never asserts more than the artifacts support.

Run after a validation pass:
    python -m validation.render_status
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "validation" / "results"
OUT = _REPO_ROOT / "VALIDATED.md"

from validation import registry  # noqa: E402


def _load_results() -> dict[str, list[dict]]:
    """Load all result artifacts, grouped by example module path.

    An example may have MORE than one artifact now -- one per instance it was validated on (e.g.
    trn1.2xlarge and trn2.48xlarge). Each example maps to a list of its per-instance results, sorted
    by instance type so the rendered rows are stable.
    """
    results: dict[str, list[dict]] = {}
    if RESULTS_DIR.is_dir():
        for path in sorted(RESULTS_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            results.setdefault(data.get("example", path.stem), []).append(data)
    for rows in results.values():
        # Seed-sweep artifacts (``seed`` set) order numerically within their instance; artifacts
        # without the field (all pre-sweep ones) keep their previous order.
        rows.sort(
            key=lambda d: (
                d.get("instance_type") or "",
                -1 if d.get("seed") is None else d["seed"],
            )
        )
    return results


def _rows_for(
    results: dict[str, list[dict]], ex, variant: str | None = None
) -> list[dict]:
    """Artifacts for one registry entry: same module AND same variant (None = the XLA default)."""
    return [r for r in results.get(ex.module, []) if r.get("variant") == variant]


_TABLE_HEADER = (
    "| Example | Status | Instance | Neuron SDK | torch-neuronx | Key metric | Wall clock | Commit | When |",
    "|---------|--------|----------|-----------|---------------|-----------|-----------|--------|------|",
)


def _group_rows(rows: list[dict]) -> list[list[dict]]:
    """Split one entry's artifacts into display groups, preserving their order.

    Every unseeded artifact is its own group (one row, exactly as before seed sweeps). Seeded
    artifacts (``seed`` set) on the same instance form a single group, placed where the first of
    them appears, so a sweep renders as one aggregated row per instance.
    """
    groups: list[list[dict]] = []
    seeded: dict[str | None, list[dict]] = {}
    for r in rows:
        if r.get("seed") is None:
            groups.append([r])
            continue
        inst = r.get("instance_type")
        if inst not in seeded:
            seeded[inst] = []
            groups.append(seeded[inst])
        seeded[inst].append(r)
    return groups


def _joined(values: list[str | None]) -> str:
    """Distinct non-empty values in first-seen order, joined with ``/`` (``—`` if none)."""
    seen = list(dict.fromkeys(v for v in values if v))
    return "/".join(seen) or "—"


def _sweep_row(ex, group: list[dict]) -> str:
    """One aggregated markdown row for a seed sweep (>1 seeded artifacts on one instance).

    The headline metric is ``name=mean±std (n=k)`` (sample std over the seeds that reported it),
    the wall clock is the mean, and the status is ✅ only if every seed passed.
    """
    n = len(group)
    n_failed = sum(1 for r in group if r.get("status") != "passed")
    icon = "✅ passed" if not n_failed else f"❌ failed ({n_failed}/{n} seeds)"
    metric_str = "—"
    if ex.thresholds:
        mk = next(iter(ex.thresholds))
        vals = [
            float((r.get("metrics") or {})[mk])
            for r in group
            if mk in (r.get("metrics") or {})
        ]
        if vals:
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            metric_str = f"{mk}={statistics.fmean(vals):.4f}±{std:.4f} (n={len(vals)})"
    walls = [r["wall_clock_s"] for r in group if r.get("wall_clock_s") is not None]
    wall = f"{round(statistics.fmean(walls), 2)}s" if walls else "—"
    versions = [r.get("versions") or {} for r in group]
    when = max((r.get("timestamp") or "")[:10] for r in group) or "—"
    return (
        f"| `{ex.key}` | {icon} | {group[0].get('instance_type') or '—'} | "
        f"{_joined([v.get('neuron_sdk') for v in versions])} | "
        f"{_joined([v.get('torch_neuronx') for v in versions])} | "
        f"{metric_str} | {wall} | {_joined([r.get('commit') for r in group])} | {when} |"
    )


def _table_rows(ex, rows: list[dict]) -> list[str]:
    """One markdown row per artifact (per instance), or a single unvalidated row.

    A seed sweep (several seeded artifacts on one instance) collapses into one aggregated row
    (see ``_sweep_row``); unseeded artifacts render one row each, unchanged.
    """
    if not rows:
        return [f"| `{ex.key}` | ⚠️ unvalidated | — | — | — | — | — | — | — |"]
    out = []
    for group in _group_rows(rows):
        if len(group) > 1:
            out.append(_sweep_row(ex, group))
            continue
        r = group[0]
        icon = {
            "passed": "✅ passed",
            "failed": "❌ failed",
            "skipped": "⏭ skipped",
        }.get(r.get("status", ""), "? ")
        versions = r.get("versions") or {}
        metrics = r.get("metrics") or {}
        # Show the first declared threshold metric as the headline number.
        metric_str = "—"
        if ex.thresholds:
            mk = next(iter(ex.thresholds))
            if mk in metrics:
                metric_str = f"{mk}={metrics[mk]:.4f}"
        wall = r.get("wall_clock_s")
        out.append(
            f"| `{ex.key}` | {icon} | {r.get('instance_type') or '—'} | "
            f"{versions.get('neuron_sdk') or '—'} | {versions.get('torch_neuronx') or '—'} | "
            f"{metric_str} | {f'{wall}s' if wall is not None else '—'} | "
            f"{r.get('commit') or '—'} | {(r.get('timestamp') or '—')[:10]} |"
        )
    return out


def _passed_on_any(rows: list[dict]) -> bool:
    """True if the example passed on at least one instance."""
    return any(r.get("status") == "passed" for r in rows)


def render(clock: str | None = None) -> str:
    """Build the VALIDATED.md content from registry + result artifacts."""
    results = _load_results()
    lines: list[str] = []
    lines.append("# Hardware Validation Status")
    lines.append("")
    lines.append(
        "This file is **generated** by `validation/render_status.py` from the provenance artifacts "
        "in `validation/results/`. Do not edit by hand. Each row reflects a real run on real Neuron "
        "hardware (or marks the example as not-yet-validated)."
    )
    lines.append("")
    if clock:
        lines.append(f"_Last rendered: {clock}_")
        lines.append("")

    total = len(registry.EXAMPLES)
    passed = sum(
        1 for ex in registry.EXAMPLES if _passed_on_any(_rows_for(results, ex))
    )
    # Distinct instances, not artifacts: a seed sweep on one instance is not "more than one instance".
    multi = sum(
        1
        for ex in registry.EXAMPLES
        if len({r.get("instance_type") for r in _rows_for(results, ex)}) > 1
    )
    coverage = f"**Coverage: {passed}/{total} examples validated on hardware.**"
    if multi:
        coverage += (
            f" ({multi} validated on more than one instance — one row each below.)"
        )
    lines.append(coverage)
    lines.append("")

    lines.extend(_TABLE_HEADER)
    for ex in registry.EXAMPLES:
        lines.extend(_table_rows(ex, _rows_for(results, ex)))

    # TorchNeuron variants (native backend, Trn2/Trn3 only) -- listed separately and NOT counted in
    # the coverage line above, which tracks the XLA path that runs on every Trainium generation.
    torchneuron = getattr(registry, "TORCHNEURON_EXAMPLES", ())
    if torchneuron:
        lines.append("")
        lines.append("## TorchNeuron (native backend, public beta; Trn2/Trn3 only)")
        lines.append("")
        lines.append(
            "The same examples run on AWS's native PyTorch backend (`torch-neuronx` 2.13+, "
            '`device="neuron"`). Not counted in the coverage line above.'
        )
        lines.append("")
        lines.extend(_TABLE_HEADER)
        for ex in torchneuron:
            lines.extend(_table_rows(ex, _rows_for(results, ex, variant=ex.key)))

    # Multi-process (torchrun) examples — validated by manual launch, not the single-device
    # auto-harness, so they carry no results/*.json. Listed separately so the count above stays
    # honest about what the harness auto-verifies, while still recording their hardware runs.
    torchrun = getattr(registry, "TORCHRUN_EXAMPLES", ())
    if torchrun:
        lines.append("")
        lines.append(
            "## Multi-process examples (torchrun — validated by manual launch)"
        )
        lines.append("")
        lines.append(
            "These need one process per NeuronCore (`torchrun`), which the single-device auto-harness "
            "doesn't orchestrate, so they're validated by a manual launch and recorded here rather "
            "than in the auto-table above."
        )
        lines.append("")
        lines.append("| Example | Status | Instance | Observed | Notes |")
        lines.append("|---------|--------|----------|----------|-------|")
        for ex in torchrun:
            note = (ex.description or "").replace("|", "·")
            # Only claim "validated" when a real hardware observation has been recorded. A missing or
            # placeholder validated_note (e.g. "TODO-AFTER-HW") means the run hasn't happened yet —
            # render it as pending so the table never overstates what was verified.
            observed = (ex.validated_note or "").strip()
            pending = (not observed) or observed.upper().startswith("TODO")
            status = "⏳ pending hardware" if pending else "✅ validated (manual)"
            lines.append(
                f"| `{ex.key}` | {status} | "
                f"{', '.join(ex.instances) or '—'} | {observed or '—'} | {note} |"
            )

    lines.append("")
    lines.append("### Legend")
    lines.append(
        "- ✅ **passed** — ran on the listed instance and met its registry thresholds."
    )
    lines.append(
        "- ❌ **failed** — ran but missed a threshold or errored (see the artifact's `error`)."
    )
    lines.append(
        "- ⚠️ **unvalidated** — no provenance artifact yet; not proven on hardware."
    )
    lines.append("")
    lines.append("Artifacts: `validation/results/*.json` · Logs: `validation/logs/`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write VALIDATED.md and report coverage."""
    import argparse

    p = argparse.ArgumentParser(
        description="Render VALIDATED.md from validation artifacts."
    )
    p.add_argument(
        "--clock", default=None, help="Fixed timestamp for deterministic output."
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any example is unvalidated.",
    )
    args = p.parse_args(argv)

    content = render(clock=args.clock)
    OUT.write_text(content)
    print(f"Wrote {OUT.relative_to(_REPO_ROOT)}")

    if args.check:
        results = _load_results()
        unvalidated = [
            e.key
            for e in registry.EXAMPLES
            if not _passed_on_any(_rows_for(results, e))
        ]
        if unvalidated:
            print(f"Unvalidated: {', '.join(unvalidated)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
