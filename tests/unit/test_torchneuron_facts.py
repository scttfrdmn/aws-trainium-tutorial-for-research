"""Unit tests for the TorchNeuron facts probe (``validation/torchneuron_facts.py``).

No hardware and no torch needed: probes are monkeypatched, and the off-hardware path is exercised
with ``import torch`` forced to fail. The point is the probe's honesty guarantees -- one failing
probe never hides the others, and without TorchNeuron nothing claims a pass.
"""

from __future__ import annotations

import json
import sys

import pytest

from validation import registry

pytestmark = pytest.mark.unit


def _module():
    from validation import torchneuron_facts as tf

    return tf


def test_module_imports_without_torch(monkeypatch):
    """The module must import with torch unavailable (torch is imported lazily)."""
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise
    monkeypatch.delitem(sys.modules, "validation.torchneuron_facts", raising=False)
    tf = _module()
    assert callable(tf.main)
    assert all(callable(getattr(tf, f"probe_{n}")) for n in tf.PROBE_NAMES)


def test_probes_are_isolated(monkeypatch):
    """A raising probe records its error; every other probe still runs."""
    tf = _module()
    ran = []
    for name in tf.PROBE_NAMES:

        def fake(name=name):
            ran.append(name)
            return {"data": {"probe": name}}

        monkeypatch.setattr(tf, f"probe_{name}", fake)

    def boom():
        ran.append("unlowerable_op")
        raise RuntimeError("device exploded")

    monkeypatch.setattr(tf, "probe_unlowerable_op", boom)
    obs = tf.run_probes()
    assert ran == list(tf.PROBE_NAMES)
    assert obs["unlowerable_op"]["error"] == "RuntimeError: device exploded"
    for name in tf.PROBE_NAMES:
        if name != "unlowerable_op":
            assert "error" not in obs[name]
            assert obs[name]["data"] == {"probe": name}


def test_main_on_hardware_writes_result_and_observations(tmp_path, monkeypatch):
    """With (faked) TorchNeuron, status reflects completion and observations go to a sibling."""
    tf = _module()
    monkeypatch.setattr(tf, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(tf, "neuron_status", lambda: (True, "ok"))
    for name in tf.PROBE_NAMES:
        monkeypatch.setattr(
            tf, f"probe_{name}", lambda name=name: {"data": {"n": name}}
        )

    def boom():
        raise ValueError("nope")

    monkeypatch.setattr(tf, "probe_sync_api", boom)
    assert tf.main([]) == 0
    result = json.loads((tmp_path / f"{tf.EXAMPLE}.json").read_text())
    assert result["status"] == "passed"
    assert result["metrics"]["probes_with_error"] == 1.0
    obs_file = tmp_path / f"{tf.EXAMPLE}.observations.json"
    assert result["log_path"].endswith(obs_file.name)
    obs = json.loads(obs_file.read_text())
    assert "example" not in obs  # never mistaken for an example's result artifact
    assert obs["probes"]["sync_api"]["error"] == "ValueError: nope"
    assert obs["probes_with_error"] == ["sync_api"]


def test_off_hardware_main_fails_and_writes_nothing_misleading(tmp_path, monkeypatch):
    """Without torch/TorchNeuron: non-zero exit, a failed result, no observations, no probes."""
    tf = _module()
    monkeypatch.setattr(tf, "RESULTS_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(
        tf, "run_probes", lambda *a: pytest.fail("probes must not run off-hardware")
    )
    assert tf.main([]) != 0
    files = list(tmp_path.glob("*.json"))
    assert [f.name for f in files] == [f"{tf.EXAMPLE}.json"]
    data = json.loads(files[0].read_text())
    assert data["status"] == "failed"
    assert data["error"].startswith("neuron unavailable")
    assert data["metrics"] == {}


def test_parse_child_output_counts_compile_lines_per_phase():
    tf = _module()
    text = "\n".join(
        [
            "init",
            "@@PHASE A_first",
            "neuronx-cc compile ...",
            "Compilation Successfully Completed for model.MODULE_1",
            "@@PHASE A_second",
            '@@JSON {"x": 1}',
        ]
    )
    parsed = tf.parse_child_output(text)
    assert parsed["data"] == {"x": 1}
    assert parsed["phases"]["A_first"]["Compilation Successfully Completed"] == 1
    assert parsed["phases"]["A_first"]["neuronx-cc"] == 1
    assert parsed["phases"]["A_second"]["Compilation Successfully Completed"] == 0
    assert parsed["phases"]["_setup"]["lines"] == 1


def test_run_child_reports_silent_crash():
    """A child that dies before emitting @@JSON is an error, not an empty success."""
    tf = _module()
    obs = tf.run_child("raise SystemExit(3)", timeout=60)
    assert obs["data"] is None
    # On torch-less CI the prelude's `import torch` fails first -- the same outcome.
    assert "error" in obs


def test_cv_compile_registry_entry():
    ex = registry.get("cv_utilization_spike_torchneuron_compile")
    base = registry.get("cv_utilization_spike_torchneuron")
    assert ex is not None and base is not None
    assert ex in registry.TORCHNEURON_EXAMPLES
    assert ex.key not in {e.key for e in registry.EXAMPLES}
    assert ex.full_config["compile"] is True
    assert ex.full_config == {**base.full_config, "compile": True}
    assert (ex.module, ex.thresholds, ex.smoke_config, ex.instances) == (
        base.module,
        base.thresholds,
        base.smoke_config,
        base.instances,
    )
