"""Unit tests for the opt-in native-PyTorch DDP path of examples/distributed/data_parallel_ner.py.

The backend-selection logic is a pure function, so it's tested here without torch. The real
2-process gloo run (``DDP_DEVICE=cpu DDP_PG_BACKEND=gloo DDP_SMOKE=1 torchrun --nproc_per_node=2
examples/distributed/data_parallel_ner.py``) is exercised manually -- it downloads a model/dataset.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

MODULE = "examples.distributed.data_parallel_ner"

pytestmark = pytest.mark.unit


def _mod():
    return importlib.import_module(MODULE)


def test_module_imports_without_torch():
    """Importing the example must not pull in torch (the harness imports every registered module)."""
    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'torch_xla', 'transformers', 'datasets'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"import {MODULE} as m\n"
        "assert callable(m.run) and callable(m.resolve_pg_backend)\n"
        "assert 'torch' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert proc.returncode == 0, proc.stderr


def test_default_config_is_xla():
    cfg = _mod().DDPConfig.from_dict({})
    assert cfg.device == "xla"
    assert cfg.pg_backend is None
    assert cfg.compile is False


def test_xla_default_unchanged():
    resolve = _mod().resolve_pg_backend
    assert resolve("xla") == "xla"
    assert resolve("xla", "xla") == "xla"


def test_xla_rejects_conflicting_backend():
    with pytest.raises(ValueError, match="conflicts"):
        _mod().resolve_pg_backend("xla", "gloo")


def test_cpu_defaults_to_gloo():
    assert _mod().resolve_pg_backend("cpu") == "gloo"


def test_cpu_explicit_backend_honored():
    assert _mod().resolve_pg_backend("cpu", "mpi") == "mpi"


def test_neuron_without_backend_raises_clear_error():
    with pytest.raises(ValueError) as exc:
        _mod().resolve_pg_backend("neuron")
    msg = str(exc.value)
    assert "DDP_PG_BACKEND" in msg
    assert "UNVERIFIED" in msg
    assert "validation/torchneuron_facts.py" in msg


def test_neuron_explicit_backend_honored():
    assert _mod().resolve_pg_backend("neuron", "some-backend") == "some-backend"


def test_neuron_candidates_probe_only_available():
    resolve = _mod().resolve_pg_backend
    avail = {"b"}.__contains__
    assert resolve("neuron", candidates=["a", " b ", "c"], is_available=avail) == "b"
    with pytest.raises(ValueError, match="Probed and unavailable"):
        resolve("neuron", candidates=["a", "c"], is_available=avail)
    # Candidates without an availability probe are never trusted.
    with pytest.raises(ValueError):
        resolve("neuron", candidates=["a"], is_available=None)


def test_unknown_device_rejected():
    with pytest.raises(ValueError, match="Unknown DDP device"):
        _mod().resolve_pg_backend("tpu")


def test_main_env_knobs(monkeypatch):
    m = _mod()
    seen = {}
    monkeypatch.setattr(m, "run", lambda cfg: seen.update(cfg) or {})
    for var in ("DDP_DEVICE", "DDP_PG_BACKEND", "DDP_COMPILE", "DDP_EPOCHS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DDP_SMOKE", "1")
    monkeypatch.setenv("DDP_PG_BACKEND", "gloo")
    m.main()
    assert seen["device"] == "cpu"
    assert seen["pg_backend"] == "gloo"
    assert seen["epochs"] == 1
    assert seen["max_train_samples"] == 64


def test_main_without_env_keeps_xla(monkeypatch):
    m = _mod()
    seen = {}
    monkeypatch.setattr(m, "run", lambda cfg: seen.update(cfg) or {})
    for var in (
        "DDP_SMOKE",
        "DDP_DEVICE",
        "DDP_PG_BACKEND",
        "DDP_COMPILE",
        "DDP_EPOCHS",
        "DDP_PER_CORE_BATCH",
        "DDP_MAX_TRAIN",
        "DDP_MAX_EVAL",
        "DDP_MAX_LENGTH",
    ):
        monkeypatch.delenv(var, raising=False)
    m.main()
    assert seen == {}
    assert m.DDPConfig.from_dict(seen).device == "xla"


def test_params_checksum_on_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    m = _mod()
    a = torch.nn.Linear(3, 2)
    b = torch.nn.Linear(3, 2)
    b.load_state_dict(a.state_dict())
    assert m._params_checksum(a) == m._params_checksum(b)
    with torch.no_grad():
        b.bias.add_(1.0)
    assert m._params_checksum(a) != m._params_checksum(b)
