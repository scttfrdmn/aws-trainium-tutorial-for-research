# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""In-process wrapper around an unmodified Boltz-2 CLI.

    python -m harness.entry --target {boltz,boltz_neuron,boltz_fp32} --timing-out T.json \
        -- predict INPUT --out_dir ... (stock boltz arguments)

What it adds, without changing Boltz-2 or boltz-neuron:

* A Lightning callback that times each predict batch (`model_s`: the sum of
  on_predict_batch_start..end, which excludes data loading and the output writer, because the
  callback is inserted ahead of Boltz's writer) and reads peak accelerator memory at the end.
* For `boltz_fp32` (the CPU oracle): Lightning `precision="32-true"` on the CPU accelerator.
  Stock Boltz-2 2.2.1 hard-codes `precision="bf16-mixed"` for boltz2 (boltz/main.py), so
  `boltz predict --accelerator cpu` alone is not an fp32 run.

Both are installed by wrapping `pytorch_lightning.Trainer.__init__`, the same mechanism
boltz-neuron's preamble uses; the wrappers chain, so order does not matter.

`--target boltz_neuron` runs `python -m boltz_neuron` semantics exactly (runpy of the package's
__main__), so the port's own preamble and defaults apply unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any


def _peak_memory() -> tuple[int | None, str | None]:
    """Peak accelerator memory in bytes, and where it came from (None, None if unavailable)."""
    try:
        import torch
    except ImportError:
        return None, None
    try:
        if torch.cuda.is_available():
            return int(
                torch.cuda.max_memory_allocated()
            ), "torch.cuda.max_memory_allocated"
    except Exception:
        pass
    neuron = getattr(torch, "neuron", None)
    fn = getattr(neuron, "max_memory_allocated", None) if neuron is not None else None
    if fn is None and neuron is not None:
        fn = getattr(getattr(neuron, "memory", None), "max_memory_allocated", None)
    if fn is not None:
        try:
            return int(fn()), "torch.neuron.max_memory_allocated"
        except Exception:
            return None, None
    return None, None


# Top-level Boltz2 submodules the fp32 audit must see run (the diffusion decoder included).
AUDIT_REQUIRED = ("structure_module", "pairformer_module", "msa_module")


class DtypeAudit:
    """Records the floating dtype of every submodule's output and the autocast state.

    One forward hook per submodule; each hook records its module's first call and removes
    itself, so the cost is one call per module. Used for the CPU fp32 oracle: a Lightning
    precision override would not catch a `torch.autocast` (or a cast) inside the model, but the
    output dtypes do, including inside the diffusion module (`structure_module`).
    """

    def __init__(self) -> None:
        """Start empty; `attach` installs the hooks."""
        self.seen: dict[str, list[str]] = {}
        self.autocast: list[str] = []
        self._handles: dict[str, Any] = {}

    def attach(self, root: Any) -> None:
        """Hook every named submodule of `root` (a LightningModule)."""
        for name, mod in root.named_modules():
            if name:
                self._handles[name] = mod.register_forward_hook(self._hook(name))

    def _hook(self, name: str) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            import torch

            self.seen[name] = sorted(_float_dtypes(out))
            if any(torch.is_autocast_enabled(dev) for dev in ("cpu", "cuda")):
                self.autocast.append(name)
            handle = self._handles.pop(name, None)
            if handle is not None:
                handle.remove()

        return hook

    def report(self) -> dict[str, Any]:
        """Summary written into the timing JSON."""
        nonfp32 = sorted(
            n for n, d in self.seen.items() if d and d != ["torch.float32"]
        )
        required = {
            top: any(n == top or n.startswith(top + ".") for n in self.seen)
            for top in AUDIT_REQUIRED
        }
        ok = (
            bool(self.seen)
            and not nonfp32
            and not self.autocast
            and all(required.values())
        )
        by_top: dict[str, dict[str, int]] = {}
        for n, d in self.seen.items():
            c = by_top.setdefault(
                n.split(".")[0], {"modules": 0, "nonfp32": 0, "autocast": 0}
            )
            c["modules"] += 1
            c["nonfp32"] += int(bool(d) and d != ["torch.float32"])
        for n in self.autocast:
            c = by_top.setdefault(
                n.split(".")[0], {"modules": 0, "nonfp32": 0, "autocast": 0}
            )
            c["autocast"] += 1
        return {
            "modules_seen": len(self.seen),
            "by_top_module": dict(sorted(by_top.items())),
            "nonfp32_modules": nonfp32[:50],
            "nonfp32_count": len(nonfp32),
            "autocast_active_in": self.autocast[:50],
            "autocast_count": len(self.autocast),
            "required_seen": required,
            "fp32_ok": ok,
        }


def _float_dtypes(obj: Any) -> set[str]:
    """Floating dtypes of every tensor in a (nested) module output."""
    import torch

    if isinstance(obj, torch.Tensor):
        return {str(obj.dtype)} if obj.is_floating_point() else set()
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        out: set[str] = set()
        for item in obj:
            out |= _float_dtypes(item)
        return out
    return set()


class TimingRecorder:
    """Accumulates per-batch model time; written to JSON at the end of predict."""

    def __init__(self, out: Path, clock: Any = time.perf_counter) -> None:
        """Record to `out`; `clock` is injectable for tests."""
        self.out = out
        self.clock = clock
        self.batches: list[float] = []
        self.records: list[str | None] = []
        self._t0: float | None = None

    def start(self, record_id: str | None = None) -> None:
        """Mark a batch start (one batch = one Boltz input record)."""
        self._t0 = self.clock()
        self.records.append(record_id)

    def end(self) -> None:
        """Mark a batch end."""
        if self._t0 is not None:
            self.batches.append(self.clock() - self._t0)
            self._t0 = None

    def write(
        self,
        peak: tuple[int | None, str | None],
        precision: str | None = None,
        param_dtype: str | None = None,
        dtype_audit: dict[str, Any] | None = None,
    ) -> None:
        """Write the timing JSON (with the precision actually in force, as evidence)."""
        per_record = {
            r: s
            for r, s in zip(self.records, self.batches, strict=False)
            if r is not None
        }
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(
            json.dumps(
                {
                    "model_s": sum(self.batches) if self.batches else None,
                    "batch_s": self.batches,
                    "per_record_s": per_record,
                    "peak_accel_mem_bytes": peak[0],
                    "peak_accel_mem_source": peak[1],
                    "precision": precision,
                    "param_dtype": param_dtype,
                    "dtype_audit": dtype_audit,
                    "peak_host_rss_bytes": _peak_rss(),
                },
                indent=2,
            )
        )


def _peak_rss() -> int | None:
    """Peak resident set size of this process in bytes (host RAM, for worker layouts)."""
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    return int(rss if sys.platform == "darwin" else rss * 1024)  # Linux reports KiB


def _record_id(batch: Any) -> str | None:
    try:
        return str(batch["record"][0].id)
    except Exception:
        return None


def install(
    recorder: TimingRecorder, fp32_cpu: bool, dtype_audit: bool = False
) -> None:
    """Wrap Trainer.__init__ to add the timing callback (and fp32 CPU for the oracle).

    The dtype audit always runs for the oracle (`fp32_cpu`); `dtype_audit` also enables it for
    other targets, e.g. as a negative control on stock (bf16-mixed) Boltz-2.
    """
    import pytorch_lightning as pl

    audit = DtypeAudit() if fp32_cpu or dtype_audit else None

    class _Timing(pl.Callback):
        def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
            if audit is not None:
                audit.attach(pl_module)

        def on_predict_batch_start(
            self, trainer: Any, pl_module: Any, batch: Any, *a: Any, **k: Any
        ) -> None:
            recorder.start(_record_id(batch))

        def on_predict_batch_end(self, *a: Any, **k: Any) -> None:
            recorder.end()

        def on_predict_end(self, trainer: Any, pl_module: Any) -> None:
            dtype = next((str(p.dtype) for p in pl_module.parameters()), None)
            recorder.write(
                _peak_memory(),
                str(trainer.precision),
                dtype,
                audit.report() if audit is not None else None,
            )

    orig = pl.Trainer.__init__

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        cbs = list(kwargs.get("callbacks") or [])
        kwargs["callbacks"] = [
            _Timing(),
            *cbs,
        ]  # ahead of BoltzWriter: excludes writing
        if fp32_cpu:
            kwargs["accelerator"] = "cpu"
            kwargs["devices"] = 1
            kwargs["precision"] = "32-true"
        orig(self, *args, **kwargs)

    pl.Trainer.__init__ = _init


def _set_threads() -> None:
    """Honour SP_TORCH_THREADS (CPU oracle workers) for torch's intra-op pool."""
    raw = os.environ.get("SP_TORCH_THREADS")
    if raw and raw.isdigit():
        import torch

        torch.set_num_threads(int(raw))


def main(argv: list[str] | None = None) -> None:
    """Parse wrapper arguments, install the hooks and hand off to the target CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit(
            "usage: python -m harness.entry --target T --timing-out F -- predict ..."
        )
    split = argv.index("--")
    p = argparse.ArgumentParser(prog="harness.entry")
    p.add_argument(
        "--target", choices=("boltz", "boltz_neuron", "boltz_fp32"), required=True
    )
    p.add_argument("--timing-out", type=Path, required=True)
    p.add_argument(
        "--torch-profile",
        type=Path,
        default=None,
        help="write a torch.profiler Chrome trace (CPU+CUDA, module hierarchy) here",
    )
    p.add_argument(
        "--dtype-audit",
        action="store_true",
        help="record submodule output dtypes (always on for boltz_fp32)",
    )
    ns = p.parse_args(argv[:split])
    rest = argv[split + 1 :]

    _set_threads()
    install(
        TimingRecorder(ns.timing_out),
        fp32_cpu=ns.target == "boltz_fp32",
        dtype_audit=ns.dtype_audit,
    )

    def _run() -> None:
        if ns.target == "boltz_neuron":
            sys.argv = ["boltz_neuron", *rest]
            runpy.run_module("boltz_neuron", run_name="__main__", alter_sys=True)
        else:
            from boltz.main import cli

            sys.argv = ["boltz", *rest]
            cli(standalone_mode=False)

    if ns.torch_profile is None:
        _run()
        return
    import torch
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, with_modules=True) as prof:
        _run()
    ns.torch_profile.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(ns.torch_profile))


if __name__ == "__main__":
    main()
