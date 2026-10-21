#!/usr/bin/env python3
"""TorchNeuron (native backend) smoke check -- the exact check from the tutorial's docs.

`docs/main_tutorial_doc.md#trying-torchneuron` shows a minimal TorchNeuron check adapted from AWS's
quickstart. This script runs that same check on real hardware and writes a provenance artifact
(`validation/results/torchneuron_smoke@<instance>.json`) so the section can point at a real run
instead of being labeled unvalidated.

Run it INSIDE a TorchNeuron environment on Trn2/Trn3, e.g. on the 2.33 multi-framework DLAMI:

    . /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
    python -m validation.torchneuron_smoke

It is NOT a registered example: TorchNeuron isn't this tutorial's path, so it isn't counted in
VALIDATED.md's coverage. torch is imported lazily so this module imports on a laptop.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.provenance import ValidationResult, capture_environment  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "validation" / "results"
EXAMPLE = "validation.torchneuron_smoke"


def run() -> dict[str, float]:
    """Run the documented check; raise on failure. Returns a few metrics."""
    import torch  # autoloads TorchNeuron when torch-neuronx 2.13+/2.14+ is installed

    assert torch.neuron.is_available(), "torch.neuron.is_available() is False"
    x = torch.randn(2, 3)
    expected = x @ x.T
    xn = x.to("neuron")
    torch.testing.assert_close((xn @ xn.T).cpu(), expected, rtol=1e-4, atol=1e-5)

    # Same check through torch.compile with the neuron backend (the docs' second mode).
    compiled = torch.compile(lambda t: t @ t.T, backend="neuron")
    torch.testing.assert_close(compiled(xn).cpu(), expected, rtol=1e-4, atol=1e-5)
    count = getattr(torch.neuron, "device_count", None)
    metrics = {"checks_passed": 2.0}
    if callable(count):
        metrics["neuron_device_count"] = float(count())
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Run the check and write a provenance artifact. Returns 0 on pass."""
    result = capture_environment(ValidationResult(example=EXAMPLE, status="failed"))
    start = time.time()
    try:
        result.metrics = run()
        result.status = "passed"
    except Exception as exc:  # record any failure as evidence, don't crash the run
        result.error = f"{type(exc).__name__}: {exc}"
    result.wall_clock_s = round(time.time() - start, 2)
    path = result.write(RESULTS_DIR)
    print(f"TorchNeuron smoke: {result.status} -> {path.name}")
    if result.error:
        print(f"  {result.error}")
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
