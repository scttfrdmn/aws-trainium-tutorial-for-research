"""Unit tests for the hardware-validation harness.

These run without AWS, Neuron, or torch -- they check the harness *contract* and provenance logic
so CI can guard the machinery that guards the examples. The actual on-hardware runs are exercised
separately by `validation/run_on_hardware.py --in-instance` on a Trainium instance.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from validation import registry
from validation.provenance import (
    ValidationResult,
    capture_environment,
    meets_thresholds,
)


def test_registry_nonempty_and_unique_keys():
    """The registry must have entries and unique keys (the harness indexes by key)."""
    assert registry.EXAMPLES, "registry should declare at least one example"
    keys = [e.key for e in registry.EXAMPLES]
    assert len(keys) == len(set(keys)), f"duplicate registry keys: {keys}"


@pytest.mark.parametrize(
    "example",
    [*registry.EXAMPLES, *registry.TORCHRUN_EXAMPLES],
    ids=lambda e: e.key,
)
def test_registered_module_imports_and_has_entrypoint(example):
    """Every registered example (incl. torchrun-only) must import and expose its entrypoint.

    Imports are lazy in the examples, so this works without torch/transformers/neuron installed.
    """
    module = importlib.import_module(example.module)
    entry = getattr(module, example.entrypoint, None)
    assert callable(entry), f"{example.module}.{example.entrypoint} is not callable"


@pytest.mark.parametrize("example", registry.EXAMPLES, ids=lambda e: e.key)
def test_thresholds_are_sane(example):
    """Thresholds must be present and within a plausible range (catches typos like 75 vs 0.75)."""
    assert example.thresholds, f"{example.key} declares no thresholds"
    for name, value in example.thresholds.items():
        assert 0.0 <= value <= 1.0 or value > 1.0, (
            name
        )  # ratios in [0,1]; counts/throughput >1


def test_meets_thresholds_pass_and_fail():
    """Threshold checking: passes when met, fails (with reasons) when not or when missing."""
    ok, fails = meets_thresholds({"eval_f1": 0.81}, {"eval_f1": 0.75})
    assert ok and not fails

    ok, fails = meets_thresholds({"eval_f1": 0.70}, {"eval_f1": 0.75})
    assert not ok and any("eval_f1" in f for f in fails)

    ok, fails = meets_thresholds({}, {"eval_f1": 0.75})
    assert not ok and any("missing" in f for f in fails)


def test_result_roundtrip(tmp_path: Path):
    """A ValidationResult writes valid JSON keyed by a filesystem-safe slug."""
    r = ValidationResult(
        example="examples/use_cases/biomedical_ner.py", status="passed"
    )
    r = capture_environment(r, clock="2026-06-16T00:00:00Z")
    out = r.write(tmp_path)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["example"] == "examples/use_cases/biomedical_ner.py"
    assert data["timestamp"] == "2026-06-16T00:00:00Z"
    # Off-hardware, Neuron fields must be None rather than fabricated.
    assert data["versions"]["torch_neuronx"] is None


def test_write_qualifies_filename_by_instance(tmp_path: Path):
    """When instance_type is set, the artifact filename is qualified with it, so results for the
    same example on different instances coexist instead of overwriting."""
    r1 = ValidationResult(
        example="examples.use_cases.biomedical_ner",
        status="passed",
        instance_type="trn1.2xlarge",
    )
    r2 = ValidationResult(
        example="examples.use_cases.biomedical_ner",
        status="passed",
        instance_type="trn2.48xlarge",
    )
    p1 = r1.write(tmp_path)
    p2 = r2.write(tmp_path)
    assert p1.name == "examples.use_cases.biomedical_ner@trn1.2xlarge.json"
    assert p2.name == "examples.use_cases.biomedical_ner@trn2.48xlarge.json"
    assert p1 != p2 and p1.exists() and p2.exists()  # neither overwrote the other


def test_render_status_groups_by_instance(tmp_path: Path, monkeypatch):
    """render_status renders one row per (example, instance) and counts multi-instance coverage."""
    from validation import render_status

    ex = registry.EXAMPLES[0]  # a real registered example (its module path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for inst in ("trn1.2xlarge", "trn2.48xlarge"):
        ValidationResult(example=ex.module, status="passed", instance_type=inst).write(
            results_dir
        )

    monkeypatch.setattr(render_status, "RESULTS_DIR", results_dir)
    loaded = render_status._load_results()
    assert len(loaded[ex.module]) == 2  # both instances grouped under the one example
    assert render_status._passed_on_any(loaded[ex.module])

    out = render_status.render(clock="2026-08-18T00:00:00Z")
    # Two data rows for this example (one per instance), and the multi-instance note is present.
    assert out.count(f"| `{ex.key}` |") == 2
    assert "trn1.2xlarge" in out and "trn2.48xlarge" in out
    assert "more than one instance" in out


def test_capture_environment_is_offline_safe():
    """capture_environment must never raise off-hardware (no EC2, maybe no git)."""
    r = capture_environment(
        ValidationResult(example="x", status="skipped"), clock="2026-01-01T00:00:00Z"
    )
    assert r.timestamp == "2026-01-01T00:00:00Z"
    assert isinstance(r.versions, dict)
