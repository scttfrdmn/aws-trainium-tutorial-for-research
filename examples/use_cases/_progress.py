"""Periodic step-progress logging for long Trainium training loops.

Why this exists (a real teaching concern): on Trainium the **first step compiles the graph** and can
take seconds-to-minutes on a cold cache, and a loop that only prints once per epoch looks *hung* to a
learner watching the console. This helper:

  * announces up front that the first step is compiling (expected, not a hang), and
  * prints periodic ``step k/N  loss=…  (rate, elapsed)`` lines so progress is visible.

A note on the XLA cost: materializing the loss (``float(loss)``) forces a host sync, which
best-practices says NOT to do *every* step. We only do it every ``log_every`` steps, so the sync is
infrequent and the throughput hit is negligible — the right trade for a tutorial where seeing
movement matters. Pass ``log_every=0`` to disable and fall back to per-epoch logging.
"""

from __future__ import annotations

import time


class StepProgress:
    """Lightweight periodic logger for a training loop. Construct once per training phase."""

    def __init__(self, label: str, total_steps: int, log_every: int, backend: str):
        self.label = label
        self.total = total_steps
        self.every = log_every  # 0 disables periodic logging
        self.backend = backend
        self.start = time.time()
        self._announced_first = False

    def announce(self) -> None:
        """Print the one-time 'first step is compiling, be patient' heads-up (XLA only)."""
        if self.backend == "xla":
            print(
                f"   ⏳ {self.label}: the first step compiles the graph (cold cache: "
                f"seconds-to-minutes) — this is expected, NOT a hang. Later steps reuse it."
            )

    def step(self, idx: int, loss_tensor=None) -> None:
        """Call once per step with the 1-based global step index and (optionally) the loss tensor.

        Prints when the first step completes (so the compile wait has a visible end) and then every
        ``log_every`` steps. ``loss_tensor`` is only materialized at those cadence points.
        """
        if self.every <= 0:
            return
        if idx == 1 and not self._announced_first:
            self._announced_first = True
            print(
                f"   {self.label}: first step done in {time.time() - self.start:.0f}s "
                f"(compile + run) — now streaming progress every {self.every} steps."
            )
            return
        if idx % self.every == 0:
            elapsed = time.time() - self.start
            rate = idx / elapsed if elapsed else 0.0
            loss_str = ""
            if loss_tensor is not None:
                loss_str = f"loss={float(loss_tensor.detach()):.4f}  "
            total_str = f"/{self.total}" if self.total else ""
            print(
                f"   {self.label}: step {idx}{total_str}  {loss_str}"
                f"({rate:.1f} steps/s, {elapsed:.0f}s elapsed)"
            )
