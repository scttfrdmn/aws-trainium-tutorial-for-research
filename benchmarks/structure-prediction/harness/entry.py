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


class TimingRecorder:
    """Accumulates per-batch model time; written to JSON at the end of predict."""

    def __init__(self, out: Path, clock: Any = time.perf_counter) -> None:
        """Record to `out`; `clock` is injectable for tests."""
        self.out = out
        self.clock = clock
        self.batches: list[float] = []
        self._t0: float | None = None

    def start(self) -> None:
        """Mark a batch start."""
        self._t0 = self.clock()

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
    ) -> None:
        """Write the timing JSON (with the precision actually in force, as evidence)."""
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(
            json.dumps(
                {
                    "model_s": sum(self.batches) if self.batches else None,
                    "batch_s": self.batches,
                    "peak_accel_mem_bytes": peak[0],
                    "peak_accel_mem_source": peak[1],
                    "precision": precision,
                    "param_dtype": param_dtype,
                },
                indent=2,
            )
        )


def install(recorder: TimingRecorder, fp32_cpu: bool) -> None:
    """Wrap Trainer.__init__ to add the timing callback (and fp32 CPU for the oracle)."""
    import pytorch_lightning as pl

    class _Timing(pl.Callback):
        def on_predict_batch_start(self, *a: Any, **k: Any) -> None:
            recorder.start()

        def on_predict_batch_end(self, *a: Any, **k: Any) -> None:
            recorder.end()

        def on_predict_end(self, trainer: Any, pl_module: Any) -> None:
            dtype = next((str(p.dtype) for p in pl_module.parameters()), None)
            recorder.write(_peak_memory(), str(trainer.precision), dtype)

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
    ns = p.parse_args(argv[:split])
    rest = argv[split + 1 :]

    install(TimingRecorder(ns.timing_out), fp32_cpu=ns.target == "boltz_fp32")

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
