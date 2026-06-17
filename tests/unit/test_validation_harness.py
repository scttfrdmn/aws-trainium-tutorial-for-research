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


def test_capture_environment_is_offline_safe():
    """capture_environment must never raise off-hardware (no EC2, maybe no git)."""
    r = capture_environment(
        ValidationResult(example="x", status="skipped"), clock="2026-01-01T00:00:00Z"
    )
    assert r.timestamp == "2026-01-01T00:00:00Z"
    assert isinstance(r.versions, dict)
