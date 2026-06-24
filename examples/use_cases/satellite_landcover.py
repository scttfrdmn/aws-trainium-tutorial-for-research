#!/usr/bin/env python3
"""Satellite land-cover classification on AWS Trainium — from real RODA open data.

A genuine remote-sensing task built on the **AWS Registry of Open Data (RODA)**: classify
**Sentinel-2** RGB image patches into **ESA WorldCover** land-cover classes (tree cover, cropland,
built-up, water, ...). Both halves are public, anonymous-access S3 datasets on RODA — no Hugging
Face, no pre-tiled benchmark:
  * Imagery: **Sentinel-2 L2A cloud-optimized GeoTIFFs** — ``s3://sentinel-cogs`` (us-west-2).
  * Labels:  **ESA WorldCover 10 m v200** — ``s3://esa-worldcover`` (eu-central-1).

The example reads a window from a Sentinel-2 true-color COG, derives its geographic footprint,
reads the co-located WorldCover label raster, tiles both into fixed 64x64 patches, and labels each
patch by its **majority WorldCover class**. That gives real (image -> land-cover) training pairs,
assembled on the fly from open satellite data — the kind of pipeline a geospatial lab actually runs.

This is the tutorial's **vision / CNN** example. The hardware story is the same as before:

Why it RUNS well on Trainium (the table-stakes from best-practices):
  * Fixed 64x64x3 patches -> static shapes -> compiles once (no recompile storm).
  * bf16-native: plain convs are bf16-stable (no fragile SDPA), the tensor engine accumulates FP32.

**Is this the *form the hardware wants*? Honestly: only partly — and that's the lesson.**
Trainium's tensor engine is a **128x128 systolic array** hungry for *large* matmuls. A small CNN's
early convs are tiny (contraction dim ~3-27 vs a 128-wide array) with a long tail of small
conv/BN/pool ops, so per-core *utilization* is low even though it runs cleanly. A utilization-optimal
vision model on Trainium looks **more ViT-shaped**. We *measured* this: cv_utilization_spike.py pits
this CNN against a ViT and the ViT achieves ~5x the CNN's TFLOP/s. See cv_utilization_spike.md,
docs/novel_kernels_on_trainium.md, and choose_your_path.md.

Note: data-parallel scaling (torchrun) improves wall-clock *throughput* but NOT per-core utilization.

Harness contract: a module-level ``run(config) -> dict[str, float]`` returning ``eval_acc`` (the
gated metric). Runs on CPU for a smoke test and on Trainium (XLA) for the real run.

    # Laptop smoke test (CPU, few RODA patches — proves the code path; needs network + rasterio):
    CV_SMOKE=1 python examples/use_cases/satellite_landcover.py

    # On a Trainium instance (real run):
    python examples/use_cases/satellite_landcover.py

Dependencies (beyond the Neuron stack): ``rasterio`` (reads COGs from S3 anonymously) + ``numpy``.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

IMAGE_SIZE = 64  # patch size; fixed -> static shapes on Trainium

# --- RODA open-data sources (both anonymous / public) --------------------------------------------
SENTINEL_BUCKET = "sentinel-cogs"  # Element 84 Sentinel-2 L2A COGs
SENTINEL_REGION = "us-west-2"
WORLDCOVER_BUCKET = "esa-worldcover"  # ESA WorldCover 10 m v200 label rasters
WORLDCOVER_REGION = "eu-central-1"

# A curated list of Sentinel-2 L2A scenes (true-color COG paths) over varied geography, so the
# tiled patches span several land-cover classes. Cloud-light summer 2021 scenes. The loader skips
# any scene/patch it can't read, so a transient miss doesn't fail the run.
SENTINEL_SCENES = (
    # tile / date  ->  region                      dominant land cover
    "32/U/PU/2021/7/S2A_32UPU_20210705_0_L2A",  # S. Germany     crop/forest/urban
    "33/U/UP/2021/8/S2B_33UUP_20210812_0_L2A",  # Czech/Alps     forest/grass
    "31/U/DQ/2021/7/S2A_31UDQ_20210708_0_L2A",  # Belgium        crop/built/water
)

# ESA WorldCover class codes -> human-readable names (the published v100/v200 legend).
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse",
    70: "Snow / ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss / lichen",
}


@dataclass
class CVConfig:
    """Configuration for the land-cover CNN trained on RODA Sentinel-2 + WorldCover."""

    device: str = "xla"  # "xla" (Trainium) | "cpu" (smoke) | "cuda"
    epochs: int = 5
    train_batch_size: int = 64
    eval_batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    # Geospatial sampling knobs.
    scenes: tuple[str, ...] = SENTINEL_SCENES
    patches_per_side: int = (
        22  # NxN grid of 64px patches read per scene (22 -> up to 484/scene)
    )
    window_offset: int = (
        2500  # px offset into each 10980px scene (skips the black border)
    )
    min_valid_frac: float = (
        0.6  # drop patches with too much nodata/cloud (mostly-black RGB)
    )
    log_every: int = 25  # stream step progress (0 = per-epoch only); see _progress.py
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


def _worldcover_tile(center_lat: float, center_lon: float) -> str:
    """WorldCover tiles are 3°x3°, named by their SW corner, e.g. 'N48E009'."""
    tlat = math.floor(center_lat / 3) * 3
    tlon = math.floor(center_lon / 3) * 3
    ns, ew = ("N" if tlat >= 0 else "S"), ("E" if tlon >= 0 else "W")
    return f"{ns}{abs(tlat):02d}{ew}{abs(tlon):03d}"


def _load_roda_patches(cfg: CVConfig):
    """Assemble (image, label) patches from RODA Sentinel-2 + WorldCover. Returns (X, codes).

    For each scene: read an NxN-patch RGB window from the Sentinel-2 true-color COG, reproject its
    footprint to lon/lat, read the matching WorldCover label raster resampled onto the same grid,
    then tile both into 64x64 patches and label each by its majority WorldCover class. Patches that
    are mostly nodata (black RGB or WorldCover code 0) are dropped.

    Honest caveat: Sentinel-2 is in UTM and WorldCover in lon/lat; we align via the window's
    geographic bounds (a small-window approximation that ignores sub-pixel rotation). Fine for a
    teaching classifier; a production pipeline would warp to a common grid per-pixel.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    side = cfg.patches_per_side
    win_px = side * IMAGE_SIZE
    imgs: list[np.ndarray] = []
    codes: list[int] = []

    for scene in cfg.scenes:
        tci = f"s3://{SENTINEL_BUCKET}/sentinel-s2-l2a-cogs/{scene}/TCI.tif"
        try:
            with (
                rasterio.Env(
                    AWS_NO_SIGN_REQUEST=True, AWS_DEFAULT_REGION=SENTINEL_REGION
                ),
                rasterio.open(tci) as s2,
            ):
                off = min(cfg.window_offset, max(0, s2.width - win_px))
                win = Window(off, off, win_px, win_px)
                rgb = s2.read(window=win)  # (3, win_px, win_px), uint8
                b_utm = rasterio.windows.bounds(win, s2.transform)
                minlon, minlat, maxlon, maxlat = transform_bounds(
                    s2.crs, "EPSG:4326", *b_utm
                )
        except Exception as exc:  # noqa: BLE001 — skip an unreadable scene, don't fail the run
            print(f"   ⚠️  skipping scene {scene}: {type(exc).__name__} {str(exc)[:80]}")
            continue

        tile = _worldcover_tile((minlat + maxlat) / 2, (minlon + maxlon) / 2)
        wc_url = (
            f"s3://{WORLDCOVER_BUCKET}/v200/2021/map/"
            f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        )
        try:
            with (
                rasterio.Env(
                    AWS_NO_SIGN_REQUEST=True, AWS_DEFAULT_REGION=WORLDCOVER_REGION
                ),
                rasterio.open(wc_url) as lc,
            ):
                wc_win = from_bounds(
                    minlon, minlat, maxlon, maxlat, transform=lc.transform
                )
                # Resample labels onto the SAME pixel grid as the RGB window (nearest = labels).
                lab = lc.read(
                    1,
                    window=wc_win,
                    out_shape=(win_px, win_px),
                    resampling=Resampling.nearest,
                    boundless=True,
                    fill_value=0,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️  skipping labels {tile}: {type(exc).__name__} {str(exc)[:80]}")
            continue

        # WorldCover north-up rasters index row 0 at max latitude; the S2 RGB window we read is also
        # north-up (row 0 = max northing), so the two grids align after the bounds-based read above.
        kept = 0
        for i in range(side):
            for j in range(side):
                r0, c0 = i * IMAGE_SIZE, j * IMAGE_SIZE
                patch = rgb[:, r0 : r0 + IMAGE_SIZE, c0 : c0 + IMAGE_SIZE]
                lblk = lab[r0 : r0 + IMAGE_SIZE, c0 : c0 + IMAGE_SIZE]
                # Drop mostly-nodata patches (black S2 pixels or WorldCover 0).
                valid = (patch.sum(axis=0) > 0) & (lblk > 0)
                if valid.mean() < cfg.min_valid_frac:
                    continue
                # Majority WorldCover class over the patch (ignoring 0/nodata).
                flat = lblk[lblk > 0]
                code = int(np.bincount(flat).argmax())
                imgs.append(patch.astype("float32") / 255.0)  # CHW, [0,1]
                codes.append(code)
                kept += 1
        print(
            f"   scene {scene.split('/')[-1]}: kept {kept} patches (WorldCover tile {tile})"
        )

    if not imgs:
        raise RuntimeError(
            "No RODA patches assembled — check network/anonymous S3 access to "
            f"s3://{SENTINEL_BUCKET} (us-west-2) and s3://{WORLDCOVER_BUCKET} (eu-central-1)."
        )
    return np.stack(imgs), np.asarray(codes, dtype="int64")


def _build_loaders(cfg: CVConfig, world_size: int = 1, rank: int = 0):
    """Assemble RODA patches, map labels to contiguous indices, return (train, val, names, sampler).

    When world_size>1 the train set is sharded with DistributedSampler (one shard per NeuronCore);
    drop_last keeps every core's batch shape identical so the graph compiles once.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    x, codes = _load_roda_patches(cfg)

    # Map the WorldCover codes that actually appear to contiguous class indices [0..K).
    present = sorted({int(c) for c in codes})
    code_to_idx = {c: i for i, c in enumerate(present)}
    label_names = [WORLDCOVER_CLASSES.get(c, str(c)) for c in present]
    y = np.asarray([code_to_idx[int(c)] for c in codes], dtype="int64")

    # Deterministic train/val split (80/20).
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(len(x))
    n_val = max(cfg.eval_batch_size, int(0.2 * len(x)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if cfg.max_train_samples:
        train_idx = train_idx[: cfg.max_train_samples]
    if cfg.max_eval_samples:
        val_idx = val_idx[: cfg.max_eval_samples]

    def ds(idx):
        return TensorDataset(
            torch.tensor(x[idx]), torch.tensor(y[idx], dtype=torch.long)
        )

    train, val = ds(train_idx), ds(val_idx)
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
    connections. Tiny and bf16-stable. (See the docstring: it RUNS well but under-fills the array;
    cv_utilization_spike.py measures that.)
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
    """Train the land-cover CNN on RODA data and return metrics (the harness entrypoint).

    Single-process by default. If launched under torchrun (RANK in env) with device=xla, it runs
    **data-parallel across NeuronCores** — the same workload, sharded, with gradient all-reduce.

        torchrun --nproc_per_node=2 examples/use_cases/satellite_landcover.py
    """
    import time

    import torch

    cfg = CVConfig.from_dict(config)
    _set_seed(cfg.seed)
    device, backend = _resolve_device(cfg.device)

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
            f"🛰️  Land-cover CV (RODA: Sentinel-2 + WorldCover) | device={backend} | {mode}"
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

    from examples.use_cases._progress import StepProgress

    progress = StepProgress(
        "train",
        len(train_loader) * cfg.epochs,
        cfg.log_every if rank == 0 else 0,
        backend,
    )
    progress.announce()
    wall = time.time()
    first_step_s = None
    gstep = 0
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
            gstep += 1
            if first_step_s is None:
                first_step_s = time.time() - t0
            progress.step(gstep, loss)
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
        "num_classes": float(len(label_names)),
        "train_wall_s": round(time.time() - wall, 2),
        "first_step_compile_s": round(first_step_s or 0.0, 2),
    }
    print(f"✅ done. eval_acc={acc:.4f}  train_wall={metrics['train_wall_s']}s")
    return metrics


def main() -> None:
    """CLI entrypoint; CV_SMOKE-style env enables a tiny CPU smoke run (still reads real RODA)."""
    cfg: dict[str, Any] = {}
    if os.environ.get("CV_SMOKE") or os.environ.get("NER_SMOKE"):
        cfg = {
            "device": "cpu",
            "epochs": 1,
            "scenes": (SENTINEL_SCENES[0],),  # one scene
            "patches_per_side": 16,  # ~256 patches: enough for a train/val split after filtering
            "train_batch_size": 16,
            "eval_batch_size": 16,
        }
    run(cfg)


if __name__ == "__main__":
    main()
