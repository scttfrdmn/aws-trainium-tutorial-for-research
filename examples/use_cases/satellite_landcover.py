#!/usr/bin/env python3
"""Satellite land-cover classification on AWS Trainium (real geospatial CV).

A genuine remote-sensing task: classify Sentinel-2 satellite image tiles into land-cover classes
(crops, forest, residential, river, ...) using the **EuroSAT** benchmark. This is the tutorial's
**vision / CNN** example — a different hardware story from the transformer examples: convolutions
are dense, regular, and map cleanly onto the systolic tensor engine, with **fixed-size image tiles**
giving static shapes for free.

Why it RUNS well on Trainium (the table-stakes from best-practices):
  * Fixed 64x64x3 tiles → static shapes → compiles once (no recompile storm).
  * bf16-native: plain convs are bf16-stable (no fragile SDPA), the tensor engine accumulates FP32.

**Is this the *form the hardware wants*? Honestly: only partly — and that's the lesson.**
Trainium's tensor engine is a **128x128 systolic array** that's hungry for *large* matmuls (the
contraction dimension filling those 128 partitions). Convolutions lower to matmul, but a small CNN's
early convs are tiny (the first has a contraction dim of ~3-27 vs a 128-wide array) and there's a
long tail of small conv/BN/pool ops — so per-core *utilization* is low even though it runs cleanly.
A utilization-optimal vision model on Trainium looks **more ViT-shaped**: patch-embed + large
attention/MLP matmuls, wide channels (>=128 to fill the partition dim), fused SBUF-resident blocks.
This example is the "it works, statically-shaped, bf16-stable" tier — NOT a claim that small-conv
CNNs maximize the systolic array. To *measure* the gap, read MFU/utilization in the profiler (see
docs/neuron_tools_and_debugging.md). See docs/novel_kernels_on_trainium.md and choose_your_path.md.

Note: data-parallel scaling (below) improves wall-clock *throughput* but NOT per-core utilization —
each core runs the same underfilled model; you just have more cores. Two different axes.

Harness contract: a module-level ``run(config) -> dict[str, float]`` returning ``eval_acc`` (the
gated metric). Runs on CPU for a smoke test and on Trainium (XLA) for the real run — exactly like
the validated NER example.

    # Laptop smoke test (CPU, tiny subset — proves the code path):
    NER_SMOKE=1 python examples/use_cases/satellite_landcover.py   # (any value works)

    # On a Trainium instance (real run):
    python examples/use_cases/satellite_landcover.py

Dataset: EuroSAT RGB via Hugging Face Datasets (parquet; no loader script). The dataset id is
configurable in case a mirror changes.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

# A parquet-native EuroSAT mirror (image + integer label). Overridable; the label set is read from
# the dataset's own schema at load time so a different mirror still works.
DEFAULT_DATASET = "blanchon/EuroSAT_RGB"
IMAGE_SIZE = 64  # EuroSAT tiles are 64x64; fixed -> static shapes on Trainium


@dataclass
class CVConfig:
    """Configuration for the land-cover CNN fine-tune."""

    dataset_name: str = DEFAULT_DATASET
    device: str = "xla"  # "xla" (Trainium) | "cpu" (smoke) | "cuda"
    epochs: int = 5
    train_batch_size: int = 64
    eval_batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> CVConfig:
        """Build from a plain dict, ignoring unknown keys."""
        cfg = dict(cfg or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(requested: str):
    """Return a torch device; fall back to CPU off-hardware (so the smoke path still runs)."""
    import torch

    if requested == "xla":
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device(), "xla"
        except ImportError:
            print("⚠️  torch_xla not available; using CPU (not a hardware run).")
            return torch.device("cpu"), "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def _build_loaders(cfg: CVConfig, world_size: int = 1, rank: int = 0):
    """Load EuroSAT, transform to fixed-size tensors, return (train, val, label_names).

    When world_size>1 the train set is sharded with DistributedSampler (one shard per NeuronCore)
    for data-parallel training; drop_last keeps every core's batch shape identical.
    """
    import numpy as np
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    ds = load_dataset(cfg.dataset_name)
    # Resolve splits (some mirrors only ship "train"); carve a val split if needed.
    train_split = ds["train"]
    if "test" in ds:
        val_split = ds["test"]
    elif "validation" in ds:
        val_split = ds["validation"]
    else:
        split = train_split.train_test_split(test_size=0.2, seed=cfg.seed)
        train_split, val_split = split["train"], split["test"]

    # Identify the image + label columns from the schema.
    feats = train_split.features
    label_col = next((c for c in ("label", "labels") if c in feats), None)
    image_col = next((c for c in ("image", "img") if c in feats), None)
    if label_col is None or image_col is None:
        raise ValueError(f"Could not find image/label columns in {list(feats)}")
    label_names = getattr(feats[label_col], "names", None) or [
        str(i) for i in range(max(train_split[label_col]) + 1)
    ]

    def cap(split, n):
        return split.select(range(min(n, len(split)))) if n else split

    def to_tensors(split):
        """Decode PIL images to a fixed 3x64x64 float tensor stack + label tensor."""
        imgs, labels = [], []
        for ex in split:
            img = ex[image_col].convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = (
                np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )  # CHW, [0,1]
            imgs.append(arr)
            labels.append(int(ex[label_col]))
        x = torch.tensor(np.stack(imgs))
        y = torch.tensor(labels, dtype=torch.long)
        return TensorDataset(x, y)

    train = to_tensors(cap(train_split, cfg.max_train_samples))
    val = to_tensors(cap(val_split, cfg.max_eval_samples))

    # drop_last keeps every batch shape identical -> the graph compiles once (best-practices §1).
    if world_size > 1:
        sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        train_loader = DataLoader(
            train, batch_size=cfg.train_batch_size, sampler=sampler, drop_last=True
        )
    else:
        sampler = None
        train_loader = DataLoader(
            train, batch_size=cfg.train_batch_size, shuffle=True, drop_last=True
        )
    val_loader = DataLoader(val, batch_size=cfg.eval_batch_size, drop_last=True)
    return train_loader, val_loader, label_names, sampler


def _build_model(num_classes: int):
    """A small residual CNN — dense, regular convolutions (systolic-engine-friendly) with skip
    connections so it actually reaches a respectable EuroSAT accuracy (a plain 3-block CNN plateaus
    ~0.65; residual blocks + a wider stem clear ~0.9). Still tiny and bf16-stable.
    """
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
            )
            self.skip = (
                nn.Sequential(
                    nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(cout),
                )
                if (stride != 1 or cin != cout)
                else nn.Identity()
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.conv(x) + self.skip(x))

    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1, bias=False),  # wider stem
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        ResBlock(64, 64, stride=1),
        ResBlock(64, 128, stride=2),  # 64 -> 32
        ResBlock(128, 256, stride=2),  # 32 -> 16
        ResBlock(256, 256, stride=2),  # 16 -> 8
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )


def _evaluate(model, loader, device, backend) -> float:
    """Return top-1 accuracy on the validation split."""
    import torch

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            if backend == "xla":
                import torch_xla.core.xla_model as xm

                xm.mark_step()
            pred = logits.argmax(dim=-1).cpu()
            correct += int((pred == y).sum())
            total += y.numel()
    return correct / total if total else 0.0


def run(config: dict | None = None) -> dict[str, float]:
    """Train the land-cover CNN and return metrics (the harness entrypoint).

    Single-process by default. If launched under torchrun (RANK in env) with device=xla, it runs
    **data-parallel across NeuronCores** — the same workload, sharded, with gradient all-reduce.
    Scaling cores ~linearly cuts per-epoch wall-clock (see the README's 1-core vs 2-core numbers).

        torchrun --nproc_per_node=2 examples/use_cases/satellite_landcover.py   # 2-core data parallel
    """
    import os
    import time

    import torch

    cfg = CVConfig.from_dict(config)
    _set_seed(cfg.seed)
    device, backend = _resolve_device(cfg.device)

    # Detect a torchrun launch on XLA -> data-parallel across cores.
    distributed = backend == "xla" and "RANK" in os.environ
    world_size, rank = 1, 0
    if distributed:
        import torch.distributed as dist
        import torch_xla.runtime as xr

        dist.init_process_group(backend="xla")
        rank, world_size = xr.global_ordinal(), xr.world_size()
    if rank == 0:
        mode = f"data-parallel x{world_size}" if distributed else "single-core"
        print(
            f"🛰️  Land-cover CV | dataset={cfg.dataset_name} | device={backend} | {mode}"
        )

    train_loader, val_loader, label_names, sampler = _build_loaders(
        cfg, world_size, rank
    )
    if rank == 0:
        print(f"   {len(label_names)} classes: {label_names}")

    model = _build_model(len(label_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    if backend == "xla":
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl

        device_loader = pl.MpDeviceLoader(train_loader, device)
    else:
        device_loader = train_loader

    wall = time.time()
    first_step_s = None
    for epoch in range(cfg.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = torch.zeros((), device=device)
        n = 0
        for x, y in device_loader:
            t0 = time.time()
            if backend != "xla":
                x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if backend == "xla":
                xm.optimizer_step(optimizer)  # applies grads + all-reduces across cores
                xm.mark_step()
            else:
                optimizer.step()
            running += loss.detach()
            n += 1
            if first_step_s is None:
                first_step_s = time.time() - t0
        if rank == 0:
            print(
                f"   epoch {epoch + 1}/{cfg.epochs}  avg_loss={(running / max(1, n)).item():.4f}"
            )

    # Eval on rank 0 only.
    if rank != 0:
        return {}
    acc = _evaluate(model, val_loader, device, backend)
    metrics = {
        "eval_acc": float(acc),
        "train_wall_s": round(time.time() - wall, 2),
        "first_step_compile_s": round(first_step_s or 0.0, 2),
    }
    print(f"✅ done. eval_acc={acc:.4f}  train_wall={metrics['train_wall_s']}s")
    return metrics


def main() -> None:
    """CLI entrypoint; NER_SMOKE-style env enables a tiny CPU smoke run."""
    cfg = {}
    if os.environ.get("CV_SMOKE") or os.environ.get("NER_SMOKE"):
        cfg = {
            "device": "cpu",
            "epochs": 1,
            "max_train_samples": 128,
            "max_eval_samples": 128,
        }
    run(cfg)


if __name__ == "__main__":
    main()
