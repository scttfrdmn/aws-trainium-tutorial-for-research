#!/usr/bin/env python3
"""Spike: *measure* Trainium under-utilization on a small CNN vs a ViT-shaped model.

The companion CV example (``satellite_landcover.py``) runs a small residual CNN that is correct,
static-shaped, and bf16-stable — it **runs** well on Trainium. But "runs well" is not the same as
"uses the hardware well." Trainium's tensor engine is a **128x128 systolic array** that wants *large*
matmuls filling its 128 partitions. A small CNN's early convs are tiny (contraction dim ~3-27 vs a
128-wide array), so per-core *utilization* is low.

This is a falsifiable hypothesis, so this script **tests it** rather than asserting it. It is a
controlled experiment — same device, same fixed 64x64x3 input, same batch — comparing two models:

  * **CNN**  — the small residual CNN from the CV example.
  * **ViT**  — a Trainium-shaped vision transformer: patch-embed -> wide (>=128) embedding ->
               large attention/MLP matmuls. (Attention is written with explicit matmuls + fp32
               softmax, NOT ``scaled_dot_product_attention`` — SDPA nans in bf16 on Neuron.)

For each it reports:
  * **FLOPs / forward**  — architecture-determined; counted on CPU with ``FlopCounterMode``.
  * **step time**        — measured steady-state (warmed up) on the real device.
  * **achieved TFLOP/s** = 3 x fwd-FLOPs / step_time  (3x ~= 1 fwd + 2 bwd, the standard convention).

**The point:** achieved TFLOP/s is a proxy for how full the systolic array is. The hypothesis is that
the ViT posts *much higher* achieved TFLOP/s than the CNN — i.e. the CNN leaves the array starved —
even though the ViT does more total work. If so, the lesson is concrete: the hardware isn't slow;
the small CNN just isn't the shape the hardware wants. (For the ground-truth MFU read, the Neuron
profiler is the authority -- see docs/neuron_tools_and_debugging.md; achieved TFLOP/s is the
portable, runnable proxy this spike uses.)

    # Laptop smoke (CPU; FLOP counts are exact, timings are not meaningful):
    CV_SMOKE=1 python examples/use_cases/cv_utilization_spike.py

    # On a Trainium instance (the real measurement):
    python examples/use_cases/cv_utilization_spike.py
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

IMAGE_SIZE = 64  # 64x64x3 patches (same shape as the CV example); fixed -> static shapes
IN_CHANS = 3


@dataclass
class SpikeConfig:
    """Knobs for the utilization spike."""

    device: str = "xla"  # "xla" (Trainium) | "cpu" (smoke) | "cuda"
    batch_size: int = 64
    warmup_steps: int = 5  # compile + reach steady state before timing
    timed_steps: int = 20  # steps to average over
    # ViT shape — sized to fill the 128-wide systolic array (embed_dim >= 128).
    patch: int = 8  # 64/8 = 8x8 = 64 patches/image
    embed_dim: int = 256
    depth: int = 6
    heads: int = 4
    mlp_ratio: int = 4
    num_classes: int = 10  # arbitrary head width; this is a throughput benchmark, not a real task


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------
def build_cnn(num_classes: int):
    """The small residual CNN from the CV example (re-used verbatim so this is an honest compare)."""
    from examples.use_cases.satellite_landcover import _build_model

    return _build_model(num_classes)


def build_vit(cfg: SpikeConfig):
    """A Trainium-shaped ViT: patch-embed -> wide embedding -> large attention/MLP matmuls.

    Attention uses explicit ``q @ k.T`` / ``attn @ v`` matmuls with an **fp32 softmax**, deliberately
    avoiding ``F.scaled_dot_product_attention`` — SDPA produces nan in bf16 on Neuron (the same bug
    the NER example hits). These big, regular matmuls are what the systolic array wants.
    """
    import torch
    import torch.nn as nn

    n_patches = (IMAGE_SIZE // cfg.patch) ** 2

    class Attention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.head_dim = dim // heads
            self.scale = self.head_dim**-0.5
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            b, n, d = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, b, heads, n, head_dim
            q, k, v = qkv[0], qkv[1], qkv[2]
            scores = (q @ k.transpose(-2, -1)) * self.scale  # big matmul
            attn = torch.softmax(scores.float(), dim=-1).to(
                v.dtype
            )  # fp32 softmax (bf16-safe)
            out = attn @ v  # big matmul
            out = out.transpose(1, 2).reshape(b, n, d)
            return self.proj(out)

    class Block(nn.Module):
        def __init__(self, dim, heads, mlp_ratio):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = Attention(dim, heads)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio),
                nn.GELU(),
                nn.Linear(dim * mlp_ratio, dim),
            )

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class ViT(nn.Module):
        def __init__(self, cfg: SpikeConfig):
            super().__init__()
            self.patch_embed = nn.Conv2d(
                IN_CHANS, cfg.embed_dim, kernel_size=cfg.patch, stride=cfg.patch
            )
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, cfg.embed_dim))
            self.blocks = nn.Sequential(
                *[
                    Block(cfg.embed_dim, cfg.heads, cfg.mlp_ratio)
                    for _ in range(cfg.depth)
                ]
            )
            self.norm = nn.LayerNorm(cfg.embed_dim)
            self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        def forward(self, x):
            b = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)  # b, n_patches, dim
            cls = self.cls.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1) + self.pos
            x = self.blocks(x)
            x = self.norm(x)
            return self.head(x[:, 0])  # cls token

    return ViT(cfg)


# --------------------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------------------
def count_forward_flops(model, sample) -> int:
    """Exact forward FLOPs via torch's FlopCounterMode (run on CPU; shape-determined, device-agnostic)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    model_cpu = model.to("cpu").eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model_cpu(sample.to("cpu"))
    return counter.get_total_flops()


def measure_step_time(model, cfg: SpikeConfig, device, backend) -> float:
    """Return median steady-state train step time (s) after warmup."""
    import torch

    model = model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    def one_step():
        x = torch.randn(cfg.batch_size, IN_CHANS, IMAGE_SIZE, IMAGE_SIZE, device=device)
        y = torch.randint(0, cfg.num_classes, (cfg.batch_size,), device=device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        if backend == "xla":
            import torch_xla.core.xla_model as xm

            xm.optimizer_step(opt, barrier=True)
        else:
            opt.step()

    for _ in range(cfg.warmup_steps):  # compile + warm
        one_step()
    if backend == "xla":
        import torch_xla.core.xla_model as xm

        xm.wait_device_ops()

    t0 = time.time()
    for _ in range(cfg.timed_steps):
        one_step()
    if backend == "xla":
        import torch_xla.core.xla_model as xm

        xm.wait_device_ops()
    return (time.time() - t0) / cfg.timed_steps


def _resolve_device(requested: str):
    import torch

    if requested == "xla":
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device(), "xla"
        except ImportError:
            print("⚠️  torch_xla not available; using CPU (timings NOT meaningful).")
            return torch.device("cpu"), "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def run(config: dict | None = None) -> dict[str, float]:
    """Run the CNN-vs-ViT utilization spike and return a flat metrics dict."""
    import torch

    cfg = SpikeConfig(
        **{k: v for k, v in (config or {}).items() if hasattr(SpikeConfig, k)}
    )
    device, backend = _resolve_device(cfg.device)
    sample = torch.randn(cfg.batch_size, IN_CHANS, IMAGE_SIZE, IMAGE_SIZE)

    print(f"🔬 CV utilization spike | device={backend} | batch={cfg.batch_size}")
    results = {}
    for name, build in (
        ("CNN", lambda: build_cnn(cfg.num_classes)),
        ("ViT", lambda: build_vit(cfg)),
    ):
        model = build()
        params = sum(p.numel() for p in model.parameters())
        fwd_flops = count_forward_flops(
            build(), sample
        )  # fresh model on CPU for counting
        step_s = measure_step_time(model, cfg, device, backend)
        # 3x fwd-FLOPs ~= 1 fwd + 2 bwd (standard training-FLOP convention). fwd_flops is already
        # for the full batch (counted on `sample`, which has batch_size rows).
        achieved_tflops = (3 * fwd_flops) / step_s / 1e12 if step_s else 0.0
        results[name] = {
            "params_M": round(params / 1e6, 2),
            "fwd_gflops": round(fwd_flops / 1e9, 2),
            "step_ms": round(step_s * 1e3, 2),
            "achieved_tflops": round(achieved_tflops, 3),
        }
        print(
            f"  {name:3s}  params={results[name]['params_M']:6.2f}M  "
            f"fwd={results[name]['fwd_gflops']:8.2f} GFLOP  "
            f"step={results[name]['step_ms']:8.2f} ms  "
            f"achieved={results[name]['achieved_tflops']:7.3f} TFLOP/s"
        )

    ratio = (
        results["ViT"]["achieved_tflops"] / results["CNN"]["achieved_tflops"]
        if results["CNN"]["achieved_tflops"]
        else float("nan")
    )
    print(
        f"\n  → ViT achieves {ratio:.1f}x the CNN's TFLOP/s on the SAME device.\n"
        f"    The hardware isn't slow — the small CNN just doesn't fill the 128x128 systolic array."
    )
    return {
        "cnn_achieved_tflops": results["CNN"]["achieved_tflops"],
        "vit_achieved_tflops": results["ViT"]["achieved_tflops"],
        "vit_over_cnn_tflops": round(ratio, 3) if not math.isnan(ratio) else 0.0,
        "cnn_step_ms": results["CNN"]["step_ms"],
        "vit_step_ms": results["ViT"]["step_ms"],
    }


def main() -> None:
    cfg = {}
    if os.environ.get("CV_SMOKE") or os.environ.get("NER_SMOKE"):
        cfg = {
            "device": "cpu",
            "warmup_steps": 1,
            "timed_steps": 2,
            "batch_size": 8,
            "depth": 2,
        }
    run(cfg)


if __name__ == "__main__":
    main()
