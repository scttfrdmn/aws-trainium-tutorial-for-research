"""Unit tests for the multi-seed mean gate (issue #24).

An ``Example`` with ``gate="mean"`` passes in VALIDATED.md / ``--check`` iff the MEAN of its gated
metric over at least ``len(default_seeds)`` seeds meets the threshold; ``default_seeds`` are used by
``run_on_hardware`` for full runs when ``--seeds`` is absent. No AWS, Neuron, or torch needed.
"""

from __future__ import annotations

import statistics
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from validation import registry, render_status
from validation import run_on_hardware as roh
from validation.provenance import ValidationResult

pytestmark = pytest.mark.unit

_AB = "antibody_affinity_slm"


def _write(results_dir: Path, ex, vals: list[float], *, seeded: bool = True) -> None:
    """Write one artifact per value; each seed's own status uses the per-run threshold."""
    mk, minimum = next(iter(ex.thresholds.items()))
    variant = ex.key if ex in registry.TORCHNEURON_EXAMPLES else None
    for seed, v in enumerate(vals, start=1):
        ValidationResult(
            example=ex.module,
            status="passed" if v >= minimum else "failed",
            instance_type="trn2.3xlarge",
            variant=variant,
            seed=seed if seeded else None,
            wall_clock_s=100.0,
            metrics={mk: v},
            versions={"neuron_sdk": "2.33.38", "torch_neuronx": "2.9.0"},
            commit="abc1234",
            timestamp="2026-10-29T00:00:00Z",
        ).write(results_dir)


def _rows(out: str, key: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(f"| `{key}` |")]


@pytest.fixture
def results(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(render_status, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(render_status, "OUT", tmp_path / "VALIDATED.md")
    monkeypatch.setattr(render_status, "_REPO_ROOT", tmp_path)
    return tmp_path


def test_antibody_registry_entries_are_mean_gated():
    """Both the XLA entry and its TorchNeuron variant use a 5-seed mean gate at 0.40."""
    for key in (_AB, f"{_AB}_torchneuron"):
        ex = registry.get(key)
        assert ex.default_seeds == (1, 2, 3, 4, 5) and ex.gate == "mean"
        assert ex.thresholds == {"spearman": 0.40}
    assert all(
        e.gate == "each" and e.default_seeds == ()
        for e in (*registry.EXAMPLES, *registry.TORCHNEURON_EXAMPLES)
        if not e.key.startswith(_AB)
    )
    with pytest.raises(ValueError):
        replace(registry.get(_AB), gate="median")


def test_mean_gate_passes_despite_a_failing_seed(results: Path):
    """Mean >= threshold over 5 seeds passes even though one seed is far below it."""
    ex = registry.get(_AB)
    vals = [0.45, 0.50, 0.21, 0.44, 0.48]  # mean 0.416; seed 3 fails on its own
    _write(results, ex, vals)
    out = render_status.render()
    (row,) = _rows(out, _AB)
    mean, std = statistics.fmean(vals), statistics.stdev(vals)
    assert "✅ passed (mean)" in row
    assert f"spearman={mean:.4f}±{std:.4f} (n=5, gate: mean)" in row
    assert render_status._passed_on_any(
        render_status._rows_for(render_status._load_results(), ex), ex
    )
    assert "**Coverage: 1/" in out


def test_mean_gate_fails_when_mean_below_threshold(results: Path):
    """Mean < threshold fails, even when most individual seeds pass."""
    ex = registry.get(_AB)
    _write(results, ex, [0.41, 0.42, 0.10, 0.43, 0.20])  # mean 0.312
    out = render_status.render()
    (row,) = _rows(out, _AB)
    assert "❌ failed (mean)" in row and "(n=5, gate: mean)" in row
    assert "**Coverage: 0/" in out


def test_insufficient_seeds_never_pass(results: Path):
    """Fewer seeds than default_seeds -> insufficient, even with a high mean (sweep or lone run)."""
    ex = registry.get(_AB)
    _write(results, ex, [0.60, 0.70, 0.65])  # 3 seeds, all great
    out = render_status.render()
    (row,) = _rows(out, _AB)
    assert "⚠️ insufficient seeds (3/5)" in row and "(n=3, gate: mean)" in row
    assert "**Coverage: 0/" in out

    # A lone unseeded passing run is shown honestly but not counted either.
    for p in results.glob("*.json"):
        p.unlink()
    _write(results, ex, [0.70], seeded=False)
    out = render_status.render()
    (row,) = _rows(out, _AB)
    assert (
        "⚠️ insufficient seeds (1/5)" in row
        and "spearman=0.7000 (n=1, gate: mean)" in row
    )
    assert "**Coverage: 0/" in out


def test_torchneuron_variant_mean_gate(results: Path):
    """The TorchNeuron variant renders under the same mean gate in its own section."""
    ex = registry.get(f"{_AB}_torchneuron")
    _write(results, ex, [0.33, 0.35, 0.36, 0.37, 0.34])
    (row,) = _rows(render_status.render(), ex.key)
    assert "❌ failed (mean)" in row and "(n=5, gate: mean)" in row


def test_each_gate_unchanged(results: Path):
    """gate='each': a sweep with one failing seed is ❌ (k/n seeds) with no gate label, a lone run
    renders its plain row, and coverage counts any passing artifact -- exactly as before."""
    ex = registry.get("ner_biomedical")
    _write(results, ex, [0.84, 0.70, 0.86])
    _write(results, ex, [0.80], seeded=False)
    out = render_status.render()
    rows = _rows(out, ex.key)
    assert len(rows) == 2
    assert any("❌ failed (1/3 seeds)" in r and "(n=3) |" in r for r in rows)
    assert any("✅ passed" in r and "eval_f1=0.8000 |" in r for r in rows)
    assert not any("gate: mean" in r or "(mean)" in r for r in rows)
    assert "**Coverage: 1/" in out


def test_check_honors_mean_gate(results: Path, monkeypatch, capsys):
    """--check reports the mean-gated example unvalidated until its mean gate passes."""
    ab = registry.get(_AB)
    # Make every other XLA example pass so --check's verdict hinges on the antibody entry.
    for ex in registry.EXAMPLES:
        if ex.key != _AB:
            ValidationResult(
                example=ex.module,
                status="passed",
                instance_type="trn2.3xlarge",
                metrics=dict(ex.thresholds),
            ).write(results)
    _write(results, ab, [0.70], seeded=False)  # a lone passing run: not enough
    assert render_status.main(["--check"]) == 1
    assert _AB in capsys.readouterr().err

    _write(results, ab, [0.45, 0.50, 0.21, 0.44, 0.48])  # 5-seed mean 0.416
    assert render_status.main(["--check"]) == 0


@pytest.fixture(autouse=True)
def _scratch_dirs(tmp_path: Path, monkeypatch):
    """Keep run_in_instance's mkdirs out of the real validation/ tree."""
    monkeypatch.setattr(roh, "RESULTS_DIR", tmp_path / "roh-results")
    monkeypatch.setattr(roh, "LOGS_DIR", tmp_path / "roh-logs")


class _Done:
    def __init__(self, rc: int = 0):
        self.returncode = rc


def test_default_seeds_used_without_seeds_flag(monkeypatch):
    """A full run of a default_seeds example fans out one child per default seed."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, cwd=None, check=False: calls.append(cmd) or _Done(),
    )
    assert roh.main(["--in-instance", "--example", _AB]) == 0
    assert [c[c.index("--seeds") + 1 :] for c in calls] == [
        [str(s)] for s in (1, 2, 3, 4, 5)
    ]
    assert all(c[c.index("--example") + 1] == _AB and "--smoke" not in c for c in calls)

    # --all: the antibody entry expands, every other example still runs once, unseeded.
    calls.clear()
    assert roh.main(["--in-instance", "--all"]) == 0
    seeded = [c for c in calls if "--seeds" in c]
    assert len(seeded) == 5 and all(c[c.index("--example") + 1] == _AB for c in seeded)
    assert len(calls) == len(registry.EXAMPLES) - 1 + 5


def test_explicit_seeds_override_default_seeds(monkeypatch):
    """Explicit --seeds wins over default_seeds."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, cwd=None, check=False: calls.append(cmd) or _Done(),
    )
    assert roh.main(["--in-instance", "--example", _AB, "--seeds", "7", "8"]) == 0
    assert [c[c.index("--seeds") + 1 :] for c in calls] == [["7"], ["8"]]


def test_smoke_ignores_default_seeds(monkeypatch):
    """Smoke runs aren't gated, so a default_seeds example still runs once (in-process)."""
    seen: list[int | None] = []
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("must not spawn a child")
    )
    monkeypatch.setattr(
        roh,
        "_run_one",
        lambda ex, smoke, clock, **kw: (
            seen.append(kw.get("seed"))
            or ValidationResult(example=ex.module, status="passed")
        ),
    )
    monkeypatch.setattr(ValidationResult, "write", lambda self, d: Path(d) / "x.json")
    assert roh.run_in_instance([_AB], smoke=True, clock=None) == 0
    assert seen == [None]
