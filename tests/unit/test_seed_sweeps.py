"""Unit tests for seed sweeps (`run_on_hardware --seeds`, issue #27).

Covers artifact naming, JSON back-compat, VALIDATED.md aggregation of a sweep into one
``mean±std (n=k)`` row, threading of ``--seeds`` into isolated child processes, and the NER
example's ``warmup_steps`` config knob. No AWS, Neuron, or torch needed.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from validation import registry
from validation.provenance import ValidationResult

pytestmark = pytest.mark.unit

_NER = "examples.use_cases.biomedical_ner"


def test_seeded_filename_suffix_and_no_collision(tmp_path: Path):
    """Each seed of a sweep gets its own ``#s<seed>`` artifact; none overwrite another."""
    paths = [
        ValidationResult(
            example=_NER, status="passed", instance_type="trn1.2xlarge", seed=s
        ).write(tmp_path)
        for s in (0, 1, 42)
    ]
    assert [p.name for p in paths] == [
        f"{_NER}@trn1.2xlarge#s0.json",
        f"{_NER}@trn1.2xlarge#s1.json",
        f"{_NER}@trn1.2xlarge#s42.json",
    ]
    assert len(set(paths)) == 3 and all(p.exists() for p in paths)
    # seed 0 is a real seed, not "unset"
    assert json.loads(paths[0].read_text())["seed"] == 0


def test_unseeded_naming_unchanged(tmp_path: Path):
    """Without a seed the filename is exactly the pre-sweep one (existing artifacts keep names)."""
    r = ValidationResult(example=_NER, status="passed")
    assert r.write(tmp_path).name == f"{_NER}.json"
    r = ValidationResult(
        example=_NER,
        status="passed",
        instance_type="trn2.3xlarge",
        variant="ner_biomedical_torchneuron",
    )
    assert (
        r.write(tmp_path).name == f"{_NER}+ner_biomedical_torchneuron@trn2.3xlarge.json"
    )


def test_json_roundtrip_with_and_without_seed(tmp_path: Path):
    """``seed`` is serialized and reloads; an old artifact lacking the field still loads."""
    r = ValidationResult(
        example=_NER, status="passed", seed=7, metrics={"eval_f1": 0.8}
    )
    data = json.loads(r.write(tmp_path).read_text())
    assert data["seed"] == 7
    assert ValidationResult.from_dict(data) == r

    old = json.loads(r.to_json())
    del old["seed"]  # an artifact written before seed sweeps existed
    old["some_future_field"] = "ignored"
    loaded = ValidationResult.from_dict(old)
    assert loaded.seed is None and loaded.metrics == {"eval_f1": 0.8}


def _write_sweep(results_dir: Path, ex, f1s: list[float], statuses: list[str]):
    for seed, (f1, status) in enumerate(zip(f1s, statuses, strict=True)):
        ValidationResult(
            example=ex.module,
            status=status,
            instance_type="trn1.2xlarge",
            seed=seed,
            wall_clock_s=10.0 * (seed + 1),
            metrics={next(iter(ex.thresholds)): f1},
            versions={"neuron_sdk": "2.33.38", "torch_neuronx": "2.9.0"},
            commit="abc1234",
            timestamp="2026-10-26T00:00:00Z",
        ).write(results_dir)


def _ex_rows(out: str, key: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(f"| `{key}` |")]


def test_render_aggregates_sweep_all_pass(tmp_path: Path, monkeypatch):
    """Three passing seeds on one instance render ONE row with mean±std (n=3) and mean wall."""
    from validation import render_status

    ex = registry.get("ner_biomedical")
    mk = next(iter(ex.thresholds))
    f1s = [0.84, 0.85, 0.86]
    _write_sweep(tmp_path, ex, f1s, ["passed"] * 3)
    # An unseeded artifact on another instance still renders its own plain row.
    ValidationResult(
        example=ex.module,
        status="passed",
        instance_type="trn2.3xlarge",
        metrics={mk: 0.8465},
    ).write(tmp_path)

    monkeypatch.setattr(render_status, "RESULTS_DIR", tmp_path)
    out = render_status.render()
    rows = _ex_rows(out, ex.key)
    assert len(rows) == 2
    sweep = rows[0]
    mean, std = statistics.fmean(f1s), statistics.stdev(f1s)
    assert f"{mk}={mean:.4f}±{std:.4f} (n=3)" in sweep
    assert "✅ passed" in sweep and "trn1.2xlarge" in sweep
    assert "| 20.0s |" in sweep  # mean of 10/20/30
    assert "| 2.33.38 | 2.9.0 |" in sweep and "abc1234" in sweep
    assert f"{mk}=0.8465 |" in rows[1] and "trn2.3xlarge" in rows[1]
    # Coverage still counts the example once; two distinct instances -> multi-instance note.
    assert "**Coverage: 1/" in out
    assert "(1 validated on more than one instance" in out


def test_render_aggregates_sweep_one_failure(tmp_path: Path, monkeypatch):
    """One failing seed makes the aggregated row ❌ with the failure count."""
    from validation import render_status

    ex = registry.get("ner_biomedical")
    _write_sweep(tmp_path, ex, [0.84, 0.70, 0.86], ["passed", "failed", "passed"])
    monkeypatch.setattr(render_status, "RESULTS_DIR", tmp_path)
    out = render_status.render()
    rows = _ex_rows(out, ex.key)
    assert len(rows) == 1
    assert "❌ failed (1/3 seeds)" in rows[0] and "(n=3)" in rows[0]
    # A sweep on ONE instance is not "validated on more than one instance".
    assert "more than one instance" not in out


def test_seeds_thread_into_child_commands(monkeypatch):
    """--seeds fans out one child per (example, seed), each receiving its single seed."""
    import subprocess

    from validation import run_on_hardware as roh

    calls = []

    class Done:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, cwd=None, check=False):
        calls.append(cmd)
        return Done(1 if cmd[-2:] == ["--seeds", "2"] else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = roh.main(
        ["--in-instance", "--example", "ner_biomedical", "--seeds", "1", "2", "3"]
    )
    assert rc == 1  # the seed-2 child failed
    assert len(calls) == 3
    assert [c[c.index("--seeds") + 1 :] for c in calls] == [["1"], ["2"], ["3"]]
    assert all(c[c.index("--example") + 1] == "ner_biomedical" for c in calls)

    # Without --seeds the child command carries no seed flag (behavior unchanged).
    calls.clear()
    assert (
        roh.run_in_instance(
            ["ner_biomedical", "crystal_cif_slm"], smoke=True, clock=None
        )
        == 0
    )
    assert len(calls) == 2 and not any("--seeds" in c for c in calls)


def test_single_seed_runs_in_process_with_seed_in_config(tmp_path: Path, monkeypatch):
    """A single (example, seed) run stays in-process, overrides config['seed'], and records it."""
    import subprocess
    import types

    from validation import run_on_hardware as roh

    seen = {}

    def fake_entry(config):
        seen.update(config)
        return {"eval_f1": 0.9}

    fake_mod = types.SimpleNamespace(run=fake_entry)
    monkeypatch.setattr(roh.importlib, "import_module", lambda name: fake_mod)
    monkeypatch.setattr(roh, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(roh, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("must not spawn a child")
    )
    monkeypatch.setattr(roh, "capture_environment", lambda r, clock=None: r)
    assert (
        roh.run_in_instance(["ner_biomedical"], smoke=True, clock=None, seeds=[5]) == 0
    )
    assert seen["seed"] == 5
    artifact = tmp_path / "results" / f"{_NER}#s5.json"
    assert json.loads(artifact.read_text())["seed"] == 5


def test_ner_config_accepts_warmup_steps():
    """NerConfig.from_dict takes warmup_steps (default 3) and the harness's seed override."""
    from examples.use_cases.biomedical_ner import NerConfig

    assert NerConfig().warmup_steps == 3
    cfg = NerConfig.from_dict({"warmup_steps": 5, "seed": 11, "unknown": 1})
    assert cfg.warmup_steps == 5 and cfg.seed == 11
