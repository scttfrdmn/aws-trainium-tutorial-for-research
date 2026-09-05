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
        rows.sort(key=lambda d: d.get("instance_type") or "")
    return results


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
        1 for ex in registry.EXAMPLES if _passed_on_any(results.get(ex.module, []))
    )
    multi = sum(1 for ex in registry.EXAMPLES if len(results.get(ex.module, [])) > 1)
    coverage = f"**Coverage: {passed}/{total} examples validated on hardware.**"
    if multi:
        coverage += (
            f" ({multi} validated on more than one instance — one row each below.)"
        )
    lines.append(coverage)
    lines.append("")

    lines.append(
        "| Example | Status | Instance | Neuron SDK | torch-neuronx | Key metric | Wall clock | Commit | When |"
    )
    lines.append(
        "|---------|--------|----------|-----------|---------------|-----------|-----------|--------|------|"
    )
    for ex in registry.EXAMPLES:
        rows = results.get(ex.module, [])
        if not rows:
            lines.append(f"| `{ex.key}` | ⚠️ unvalidated | — | — | — | — | — | — | — |")
            continue
        # One row per instance the example was validated on (sorted by instance in _load_results).
        for r in rows:
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
            lines.append(
                f"| `{ex.key}` | {icon} | {r.get('instance_type') or '—'} | "
                f"{versions.get('neuron_sdk') or '—'} | {versions.get('torch_neuronx') or '—'} | "
                f"{metric_str} | {f'{wall}s' if wall is not None else '—'} | "
                f"{r.get('commit') or '—'} | {(r.get('timestamp') or '—')[:10]} |"
            )

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
            if not _passed_on_any(results.get(e.module, []))
        ]
        if unvalidated:
            print(f"Unvalidated: {', '.join(unvalidated)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
