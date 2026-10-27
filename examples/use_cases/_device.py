"""Tiny device/backend helpers shared by the use-case examples.

Two ways to run PyTorch on Trainium are supported:

* ``device="xla"`` -- **the default, documented path**: PyTorch/XLA (``torch_xla``). Tensors live
  on ``xm.xla_device()``, the optimizer steps through ``xm.optimizer_step`` and ``xm.mark_step``
  cuts the lazy graph, and ``MpDeviceLoader`` feeds batches. Runs on Trn1 and Trn2.
* ``device="neuron"`` -- **opt-in, beta**: TorchNeuron (torch-neuronx 2.13+), AWS's native PyTorch
  backend, Trn2/Trn3 only. Trainium is an ordinary eager device named ``"neuron"``; optionally
  wrap the model in ``torch.compile(backend="neuron")``, and get bf16 from ``torch.autocast``.

``"cpu"``/``"cuda"`` also work (laptop smoke tests). Every helper takes the resolved ``backend``
string so the example code reads the same on all of them. torch is imported lazily so the
examples stay importable (for registry listing) without torch installed.
"""

from __future__ import annotations

import contextlib


def resolve_device(requested: str):
    """Return ``(torch.device, backend)`` for the requested backend, falling back to CPU.

    ``backend`` is one of ``"xla"``, ``"neuron"``, ``"cuda"``, ``"cpu"`` -- what actually ran,
    so a laptop asking for ``"xla"`` gets ``"cpu"`` (with a warning) and still completes.
    """
    import torch

    if requested == "xla":
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device(), "xla"
        except ImportError:
            print(
                "⚠️  torch_xla not available; falling back to CPU (not a hardware run)."
            )
            return torch.device("cpu"), "cpu"
    if requested == "neuron":
        # TorchNeuron autoloads on `import torch`; Trainium is then a device named "neuron".
        neuron = getattr(torch, "neuron", None)
        if neuron is not None and neuron.is_available():
            return torch.device("neuron"), "neuron"
        print("⚠️  TorchNeuron not available; falling back to CPU (not a hardware run).")
        return torch.device("cpu"), "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def wrap_model(
    model,
    backend: str,
    *,
    compile: bool = False,
    autocast_bf16: bool = True,
    amp_device: str | None = None,
    compile_backend: str | None = None,
):
    """Return ``(fwd, amp)``: the callable to run forward with, and an autocast-context factory.

    Only the TorchNeuron backend changes anything: ``compile`` routes the model through
    ``torch.compile(backend="neuron")`` and ``autocast_bf16`` makes ``amp()`` a bf16
    ``torch.autocast``. On XLA/CPU/CUDA ``fwd`` is the model itself and ``amp()`` is a no-op (the
    XLA path gets bf16 from the Neuron compiler). Keep the original ``model`` for ``.train()``,
    ``.parameters()``, and CPU-side work -- ``fwd`` may be a compiled wrapper.

    ``amp_device``/``compile_backend`` override ``"neuron"`` so tests can exercise this branch on
    CPU (e.g. ``amp_device="cpu", compile_backend="eager"``).
    """
    import torch

    fwd = model
    amp = contextlib.nullcontext
    if backend == "neuron":
        if compile:
            fwd = torch.compile(model, backend=compile_backend or "neuron")
        if autocast_bf16:
            device_type = amp_device or "neuron"

            def amp():
                return torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    return fwd, amp


def device_loader(loader, device, backend: str):
    """Wrap a DataLoader for the backend (``MpDeviceLoader`` on XLA; unchanged otherwise)."""
    if backend == "xla":
        import torch_xla.distributed.parallel_loader as pl

        # Overlaps host->device transfer with compute and issues mark_step per batch.
        return pl.MpDeviceLoader(loader, device)
    return loader


def to_device(batch, device, backend: str):
    """Move a tensor, or a tuple/list/dict of tensors, to ``device``.

    A no-op on XLA, where ``MpDeviceLoader`` has already placed the batch.
    """
    if backend == "xla":
        return batch
    if isinstance(batch, dict):
        return {k: to_device(v, device, backend) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device, backend) for v in batch)
    return batch.to(device) if hasattr(batch, "to") else batch


def optimizer_step(optimizer, backend: str) -> None:
    """Step the optimizer: ``xm.optimizer_step`` + ``xm.mark_step`` on XLA, else ``.step()``."""
    if backend == "xla":
        import torch_xla.core.xla_model as xm

        xm.optimizer_step(optimizer)
        xm.mark_step()
    else:
        optimizer.step()


def sync(backend: str, tensor=None) -> None:
    """Block until queued device work finishes (a timing fence).

    XLA: ``xm.wait_device_ops()``. Otherwise ``tensor.item()`` if a tensor is given (forces the
    value, which works on any eager device incl. ``"neuron"``), plus ``cuda.synchronize`` on CUDA.
    """
    if backend == "xla":
        import torch_xla.core.xla_model as xm

        xm.wait_device_ops()
        return
    if tensor is not None:
        tensor.item()
    if backend == "cuda":
        import torch

        torch.cuda.synchronize()
