"""Unit tests for examples/use_cases/_device.py and the TorchNeuron registry entries (CPU only)."""

from __future__ import annotations

import importlib

import pytest

from validation import registry

pytestmark = pytest.mark.unit

PORTED = (
    "satellite_landcover",
    "cv_utilization_spike",
    "distill_ner_slm",
    "antibody_affinity_slm",
    "crystal_cif_slm",
)


def test_helper_imports_without_torch_side_effects():
    mod = importlib.import_module("examples.use_cases._device")
    for name in (
        "resolve_device",
        "wrap_model",
        "device_loader",
        "to_device",
        "optimizer_step",
        "sync",
    ):
        assert callable(getattr(mod, name))


def test_resolve_device_neuron_falls_back_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from examples.use_cases._device import resolve_device

    monkeypatch.delattr(torch, "neuron", raising=False)
    device, backend = resolve_device("neuron")
    assert backend == "cpu"
    assert device == torch.device("cpu")


def test_resolve_device_cpu():
    torch = pytest.importorskip("torch")
    from examples.use_cases._device import resolve_device

    assert resolve_device("cpu") == (torch.device("cpu"), "cpu")


def test_wrap_model_non_neuron_is_identity():
    torch = pytest.importorskip("torch")
    import contextlib

    from examples.use_cases._device import wrap_model

    model = torch.nn.Linear(4, 2)
    fwd, amp = wrap_model(model, "xla", compile=True)
    assert fwd is model
    assert amp is contextlib.nullcontext


def test_wrap_model_neuron_branch_on_cpu_matches_model():
    torch = pytest.importorskip("torch")
    from examples.use_cases._device import wrap_model

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU()).eval()
    x = torch.randn(3, 8)
    fwd, amp = wrap_model(
        model,
        "neuron",
        compile=True,
        autocast_bf16=False,
        amp_device="cpu",
        compile_backend="eager",
    )
    assert fwd is not model  # compiled wrapper
    with torch.no_grad(), amp():
        assert torch.allclose(fwd(x), model(x))

    # With bf16 autocast the output is bf16 and close to the fp32 reference.
    fwd, amp = wrap_model(
        model,
        "neuron",
        compile=True,
        autocast_bf16=True,
        amp_device="cpu",
        compile_backend="eager",
    )
    with torch.no_grad(), amp():
        out = fwd(x)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), model(x), atol=5e-2, rtol=5e-2)


def test_optimizer_step_on_cpu_calls_step():
    from examples.use_cases._device import optimizer_step

    calls = []

    class FakeOpt:
        def step(self):
            calls.append(1)

    optimizer_step(FakeOpt(), "cpu")
    optimizer_step(FakeOpt(), "neuron")
    assert calls == [1, 1]


def test_to_device_and_loader_passthrough_on_cpu():
    torch = pytest.importorskip("torch")
    from examples.use_cases._device import device_loader, to_device

    loader = [1, 2]
    assert device_loader(loader, torch.device("cpu"), "cpu") is loader
    batch = {"a": torch.zeros(2), "b": (torch.ones(1), torch.ones(1))}
    out = to_device(batch, torch.device("cpu"), "cpu")
    assert set(out) == {"a", "b"} and isinstance(out["b"], tuple)
    sentinel = object()
    assert (
        to_device(sentinel, None, "xla") is sentinel
    )  # MpDeviceLoader already placed it


def test_sync_forces_tensor_on_eager_backends():
    from examples.use_cases._device import sync

    forced = []

    class FakeTensor:
        def item(self):
            forced.append(1)
            return 0.0

    sync("cpu", FakeTensor())
    sync("neuron", FakeTensor())
    sync("cpu")  # no tensor: no-op
    assert forced == [1, 1]


@pytest.mark.parametrize("ex", registry.TORCHNEURON_EXAMPLES, ids=lambda e: e.key)
def test_torchneuron_entries_import_and_have_entrypoint(ex):
    mod = importlib.import_module(ex.module)
    assert callable(getattr(mod, ex.entrypoint))
    assert ex.full_config.get("device") == "neuron"
    assert ex.key not in {e.key for e in registry.EXAMPLES}
    assert registry.get(ex.key) is ex


@pytest.mark.parametrize("key", PORTED)
def test_ported_examples_have_torchneuron_variant(key):
    xla = registry.get(key)
    tn = registry.get(f"{key}_torchneuron")
    assert xla is not None and tn is not None
    assert tn.module == xla.module
    assert tn.thresholds == xla.thresholds
    assert tn.smoke_config == xla.smoke_config
    assert tn.full_config == {**xla.full_config, "device": "neuron"}
    assert tn.instances == ("trn2.3xlarge",)
    assert xla.full_config["device"] == "xla"  # XLA default untouched
