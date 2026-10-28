"""Unit tests for the debugging walkthrough (``examples/debugging``, issue #28).

The import/argparse tests run without torch (CI has none). The torch-dependent tests run the
CPU versions of demo 2 (recompile counting) and demo 3 (graph breaks) with stock PyTorch and
``torch.compile(backend="eager")``; they skip when torch is absent.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.unit


def _module():
    from debugging import diagnose_common_failures as dcf

    return dcf


def test_module_imports_without_torch(monkeypatch):
    """The module must import with torch unavailable (torch is imported lazily)."""
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise
    monkeypatch.delitem(
        sys.modules, "debugging.diagnose_common_failures", raising=False
    )
    dcf = _module()
    assert callable(dcf.main)


@pytest.mark.parametrize("device", ["cpu", "xla", "neuron"])
def test_argparse_accepts_devices(device):
    args = (
        _module().build_parser().parse_args(["--device", device, "--only", "recompile"])
    )
    assert args.device == device
    assert args.only == "recompile"


def test_argparse_rejects_unknown_device():
    with pytest.raises(SystemExit):
        _module().build_parser().parse_args(["--device", "tpu"])


def test_neuron_falls_back_to_cpu_without_torchneuron(capsys):
    torch = pytest.importorskip("torch")
    if getattr(torch, "neuron", None) is not None:
        pytest.skip("TorchNeuron installed; fallback path not exercised")
    device, backend = _module()._device("neuron")
    assert backend == "cpu" and device.type == "cpu"
    assert "TorchNeuron (beta) not available" in capsys.readouterr().out


def test_recompile_demo_ragged_builds_more_graphs_than_fixed():
    pytest.importorskip("torch")
    dcf = _module()
    res = dcf.demo_recompile_storm(None, "cpu")
    assert res["fixed"] == 1
    # dynamic=False: one graph per distinct shape (the XLA-like behavior).
    assert res["ragged_static"] == len(set(dcf.RAGGED_SIZES))
    # default (automatic dynamic shapes): still recompiles at least once, but no more than static.
    assert res["fixed"] < res["ragged_default"] <= res["ragged_static"]


def test_fallback_demo_graph_breaks_nonzero_but_not_where():
    pytest.importorskip("torch")
    res = _module().demo_cpu_fallback(None, "cpu")
    assert res["nonzero"]["graph_break_count"] >= 1
    assert res["nonzero"]["graph_count"] >= 2
    assert res["where"]["graph_break_count"] == 0
    assert res["where"]["graph_count"] == 1
