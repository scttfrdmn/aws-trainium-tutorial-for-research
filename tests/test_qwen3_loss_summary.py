"""Regression tests for qwen3_lora_finetune._summarize_training.

The point of these tests: a tutorial participant must NEVER see a misleading ``nan`` training loss.
On the XLA/Neuron path the trainer's aggregate ``training_loss`` comes back ``nan`` (the loss tensor
is only materialized on ``logging_steps``), so we derive the reported loss from ``log_history`` and
handle the no-loss-logged case explicitly. These tests pin that behavior without needing hardware.
"""

from __future__ import annotations

import math

from examples.use_cases.qwen3_lora_finetune import _summarize_training

NAN = float("nan")


class _FakeTrainer:
    def __init__(self, history):
        self.state = type("S", (), {"log_history": history})()


class _FakeResult:
    training_loss = NAN  # the poisoned aggregate the real trainer returns on XLA


class _FakeArgs:
    output_dir = "./out"
    logging_steps = 5


def test_extracts_real_losses_ignoring_nan_rows():
    """Healthy logged losses are picked out even when interleaved with nan rows."""
    history = [
        {"loss": NAN},
        {"loss": 1.93},
        {"loss": NAN},
        {"loss": 1.51},
        {"loss": 1.43},
        {"epoch": 1.0},
    ]
    metrics = _summarize_training(_FakeResult(), _FakeTrainer(history), _FakeArgs())
    assert metrics["train_loss"] == 1.43  # last materialized step loss
    assert metrics["logged_steps"] == 3.0
    assert math.isclose(metrics["mean_logged_loss"], (1.93 + 1.51 + 1.43) / 3)


def test_no_logged_loss_does_not_report_nan():
    """A run shorter than logging_steps must yield an explicit sentinel, never nan."""
    metrics = _summarize_training(
        _FakeResult(), _FakeTrainer([{"epoch": 1.0}]), _FakeArgs()
    )
    assert metrics["train_loss"] == -1.0
    assert metrics["logged_steps"] == 0.0


def test_no_metric_is_ever_nan():
    """Across both paths, no returned metric may be nan (the whole point of the fix)."""
    for history in ([{"loss": NAN}, {"loss": 2.0}], [{"epoch": 1.0}]):
        metrics = _summarize_training(_FakeResult(), _FakeTrainer(history), _FakeArgs())
        for key, value in metrics.items():
            assert not (isinstance(value, float) and math.isnan(value)), f"{key} is nan"
