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


def _load_results() -> dict[str, dict]:
    """Load all result artifacts, keyed by example module path."""
    results: dict[str, dict] = {}
    if RESULTS_DIR.is_dir():
        for path in sorted(RESULTS_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            results[data.get("example", path.stem)] = data
    return results


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
        1
        for ex in registry.EXAMPLES
        if results.get(ex.module, {}).get("status") == "passed"
    )
    lines.append(f"**Coverage: {passed}/{total} examples validated on hardware.**")
    lines.append("")

    lines.append(
        "| Example | Status | Instance | Neuron SDK | torch-neuronx | Key metric | Wall clock | Commit | When |"
    )
    lines.append(
        "|---------|--------|----------|-----------|---------------|-----------|-----------|--------|------|"
    )
    for ex in registry.EXAMPLES:
        r = results.get(ex.module)
        if not r:
            lines.append(f"| `{ex.key}` | ⚠️ unvalidated | — | — | — | — | — | — | — |")
            continue
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
            if results.get(e.module, {}).get("status") != "passed"
        ]
        if unvalidated:
            print(f"Unvalidated: {', '.join(unvalidated)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
