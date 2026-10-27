#!/usr/bin/env python3
"""Antibody binding-affinity prediction with a small protein language model on AWS Trainium.

A real antibody-engineering task: given an antibody's **amino-acid sequence**, predict its
**binding affinity** to a target antigen. This is exactly the kind of in-silico screening that
therapeutic-antibody labs run to triage candidate designs before wet-lab work. We fine-tune a
small **ESM-2** protein language model (Lin et al., 2023 — `facebook/esm2_t6_8M_UR50D`, an 8M-param
SLM) with a regression head and score it by **Spearman correlation** against measured affinities.

Data: the public **AbBibench Antibody Binding Benchmark** (`AbBibench/Antibody_Binding_Benchmark_Dataset`),
which ships per-antigen CSVs of `heavy_chain_seq`, `light_chain_seq`, and a continuous `binding_score`.
We model the heavy chain (the dominant binding determinant) for one antigen complex at a time.

What this demonstrates for Trainium users:
    * A **protein language model** on the PyTorch/XLA path — proteins are just sequences over a
      ~25-symbol amino-acid vocabulary, so ESM-2 is an ordinary transformer encoder. Same Trainium
      lessons as the NLP examples apply directly.
    * **Build in the form the hardware wants** — `attn_implementation="eager"` (HF v5 SDPA → `nan`
      on the Neuron bf16 path; verified on the NER example) and a **fixed `max_length` + `drop_last`**
      so every batch has identical shape and the graph compiles once.
    * **Regression on top of a pretrained encoder**: mean-pooled embeddings → a linear head, MSE
      loss, Spearman/Pearson correlation as the honest scientific metric (ranking candidates matters
      more than absolute MSE in screening).
    * Device portability: identical code on CPU (laptop smoke) and Trainium.

Harness contract: a module-level ``run(config) -> dict[str, float]`` returning ``spearman`` (the
gated metric). Runs on CPU (smoke) and Trainium.

    # Laptop smoke test (CPU, tiny subset — proves the code path):
    ANTIBODY_SMOKE=1 python examples/use_cases/antibody_affinity_slm.py

    # On a Trainium instance (real run):
    python examples/use_cases/antibody_affinity_slm.py

Sizing: esm2_t6_8M (~8M params) + a regression head trains comfortably on a single `trn1.2xlarge`.
Bump `model_name` to `facebook/esm2_t12_35M_UR50D` for a stronger (still small) model.
"""

from __future__ import annotations

# Allow `python examples/use_cases/<this>.py` as well as `-m`/the harness: put the repo root on
# sys.path so the `examples.use_cases.*` imports below resolve when run as a plain script.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_ROOT = str(_Path(__file__).resolve().parents[2])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import os
from dataclasses import dataclass, field
from typing import Any

# ESM-2 small models are protein masked-LMs; we use the encoder + a fresh regression head.
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_DATASET = "AbBibench/Antibody_Binding_Benchmark_Dataset"
# One antigen complex's benchmarking CSV (2,223 sequence variants — the largest single file).
DEFAULT_DATA_FILE = "binding_affinity/2fjg_benchmarking_data.csv"


@dataclass
class AntibodyConfig:
    """Configuration for the antibody affinity regressor."""

    model_name: str = DEFAULT_MODEL
    dataset_name: str = DEFAULT_DATASET
    data_file: str = DEFAULT_DATA_FILE
    seq_column: str = (
        "heavy_chain_seq"  # the chain we model (dominant binding determinant)
    )
    label_column: str = "binding_score"
    # "xla" (Trainium, default) | "neuron" (TorchNeuron beta, Trn2/Trn3, opt-in) | "cpu" | "cuda"
    device: str = "xla"
    # TorchNeuron only (device="neuron"): torch.compile(backend="neuron") + bf16 autocast.
    compile: bool = False
    autocast_bf16: bool = True
    epochs: int = 8
    train_batch_size: int = 16
    eval_batch_size: int = 16
    max_length: int = (
        160  # antibody variable domains are ~110-130 aa; 160 covers them + specials
    )
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    test_fraction: float = 0.2
    freeze_encoder: bool = False  # set True to train only the head (faster, weaker)
    log_every: int = (
        25  # stream step progress every N steps (0 = per-epoch only). See _progress.py
    )
    seed: int = 42
    attn_implementation: str = (
        "eager"  # Neuron-friendly (SDPA → nan in bf16); see NER example
    )
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> AntibodyConfig:
        cfg = dict(cfg or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_dataframe(cfg: AntibodyConfig):
    """Load the chosen benchmark CSV into a pandas DataFrame (seq + label columns only)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(cfg.dataset_name, cfg.data_file, repo_type="dataset")
    df = pd.read_csv(path)
    missing = [c for c in (cfg.seq_column, cfg.label_column) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not in {cfg.data_file} (have {list(df.columns)[:10]})"
        )
    df = df[[cfg.seq_column, cfg.label_column]].dropna()
    return df.reset_index(drop=True)


def _build_loaders(cfg: AntibodyConfig, tokenizer):
    """Tokenize sequences to fixed length and return (train_loader, eval_loader, label_mean/std)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    df = _load_dataframe(cfg)
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(len(df))
    n_test = max(1, int(len(df) * cfg.test_fraction))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    if cfg.max_train_samples:
        train_idx = train_idx[: cfg.max_train_samples]
    if cfg.max_eval_samples:
        test_idx = test_idx[: cfg.max_eval_samples]

    seqs = df[cfg.seq_column].tolist()
    labels = df[cfg.label_column].astype("float32").to_numpy()
    # Standardize the regression target (z-score) using TRAIN statistics only — stabilizes MSE and
    # is the correct way to avoid leaking test-set scale into training.
    mu, sigma = float(labels[train_idx].mean()), float(labels[train_idx].std() or 1.0)

    def encode(indices):
        enc = tokenizer(
            [seqs[i] for i in indices],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        y = torch.tensor((labels[indices] - mu) / sigma, dtype=torch.float32)
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    train_ds, test_ds = encode(train_idx), encode(test_idx)
    # drop_last=True → every batch identical shape → graph compiles once (best-practices §1).
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train_batch_size, shuffle=True, drop_last=True
    )
    eval_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, drop_last=True)
    return train_loader, eval_loader, (mu, sigma)


def _build_model(cfg: AntibodyConfig):
    """ESM-2 encoder with a linear regression head over mean-pooled residue embeddings."""
    import torch.nn as nn
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(
        cfg.model_name, attn_implementation=cfg.attn_implementation
    )
    if cfg.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    class AffinityRegressor(nn.Module):
        def __init__(self, encoder, hidden):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(hidden, 1)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Mean-pool residue embeddings over real (non-pad) positions.
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            return self.head(pooled).squeeze(-1)

    return AffinityRegressor(encoder, encoder.config.hidden_size)


def _spearman(pred, gold) -> float:
    """Spearman rank correlation (ranking quality — what matters for candidate triage)."""
    import numpy as np

    def ranks(x):
        order = np.argsort(x)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        return r

    rp, rg = ranks(np.asarray(pred)), ranks(np.asarray(gold))
    rp -= rp.mean()
    rg -= rg.mean()
    denom = float(np.sqrt((rp**2).sum() * (rg**2).sum()))
    return float((rp * rg).sum() / denom) if denom else 0.0


def _evaluate(model, loader, device, backend, fwd=None, amp=None) -> dict[str, float]:
    """Return Spearman + Pearson + MSE on the held-out split.

    ``fwd``/``amp`` are the forward callable and autocast factory from ``wrap_model`` (default:
    the plain eager model, no autocast).
    """
    import contextlib

    import numpy as np
    import torch

    # `is None`, not `or`: truth-testing a torch.compile'd module raises.
    fwd = model if fwd is None else fwd
    amp = contextlib.nullcontext if amp is None else amp
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for input_ids, attention_mask, y in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with amp():
                out = fwd(input_ids=input_ids, attention_mask=attention_mask)
            if backend == "xla":
                import torch_xla.core.xla_model as xm

                xm.mark_step()
            preds.extend(out.float().cpu().numpy().tolist())
            golds.extend(y.numpy().tolist())
    preds_a, golds_a = np.asarray(preds), np.asarray(golds)
    mse = float(((preds_a - golds_a) ** 2).mean()) if len(preds_a) else 0.0
    # Pearson
    pc = 0.0
    if len(preds_a) > 1 and preds_a.std() and golds_a.std():
        pc = float(np.corrcoef(preds_a, golds_a)[0, 1])
    sp = _spearman(preds_a, golds_a) if len(preds_a) > 1 else 0.0
    print(
        f"   eval: spearman={sp:.4f} pearson={pc:.4f} mse={mse:.4f} (n={len(preds_a)})"
    )
    return {"spearman": sp, "pearson": pc, "mse": mse}


def run(config: dict | None = None) -> dict[str, float]:
    """Fine-tune ESM-2 to predict antibody binding affinity; return metrics (harness entrypoint).

    Gated metric: ``spearman`` (rank correlation between predicted and measured affinity on the
    held-out split). Also reports Pearson, MSE, params, and timings.
    """
    import time

    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from examples.use_cases._device import (
        device_loader,
        optimizer_step,
        resolve_device,
        to_device,
        wrap_model,
    )

    cfg = AntibodyConfig.from_dict(config)
    _set_seed(cfg.seed)
    device, backend = resolve_device(cfg.device)
    print(
        f"🧬 Antibody affinity SLM | model={cfg.model_name} | "
        f"data={cfg.data_file.split('/')[-1]} | device={backend}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader, eval_loader, (mu, sigma) = _build_loaders(cfg, tokenizer)
    print(f"   target standardized (train mean={mu:.3f}, std={sigma:.3f})")

    model = _build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   model params: {n_params / 1e6:.1f}M (encoder + regression head)")
    # Only device="neuron" (TorchNeuron, beta) changes fwd/amp; on XLA fwd is the model itself.
    fwd, amp = wrap_model(
        model, backend, compile=cfg.compile, autocast_bf16=cfg.autocast_bf16
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate
    )
    total = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total), total
    )
    loss_fn = torch.nn.MSELoss()

    train_batches = device_loader(train_loader, device, backend)

    from examples.use_cases._progress import StepProgress

    progress = StepProgress(
        "train", len(train_loader) * cfg.epochs, cfg.log_every, backend
    )
    progress.announce()
    wall_start = time.time()
    first_step_s = None
    gstep = 0
    for epoch in range(cfg.epochs):
        model.train()
        running = torch.zeros((), device=device)
        n = 0
        for batch in train_batches:
            step_start = time.time()
            input_ids, attention_mask, y = to_device(batch, device, backend)
            optimizer.zero_grad()
            with amp():
                pred = fwd(input_ids=input_ids, attention_mask=attention_mask)
            # Loss in fp32: under bf16 autocast the head's output is bf16 (a no-op cast otherwise).
            loss = loss_fn(pred.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer_step(optimizer, backend)
            scheduler.step()
            running += loss.detach()
            n += 1
            gstep += 1
            if first_step_s is None:
                first_step_s = time.time() - step_start
            progress.step(gstep, loss)
        print(
            f"   epoch {epoch + 1}/{cfg.epochs}  avg_mse={(running / max(1, n)).item():.4f}"
        )
    train_wall = time.time() - wall_start

    eval_metrics = _evaluate(model, eval_loader, device, backend, fwd, amp)
    metrics = {
        "spearman": float(eval_metrics["spearman"]),
        "pearson": float(eval_metrics["pearson"]),
        "mse": float(eval_metrics["mse"]),
        "params_m": float(round(n_params / 1e6, 2)),
        "train_wall_s": round(train_wall, 2),
        "first_step_compile_s": round(first_step_s or 0.0, 2),
        "torch_compile": 1.0 if (backend == "neuron" and cfg.compile) else 0.0,
    }
    print(
        f"✅ done. spearman={metrics['spearman']:.4f}  pearson={metrics['pearson']:.4f}  "
        f"train_wall={metrics['train_wall_s']}s"
    )
    return metrics


def main() -> None:
    """CLI entrypoint; ANTIBODY_SMOKE runs a tiny CPU configuration.

    ANTIBODY_DEVICE=neuron opts into TorchNeuron (beta, Trn2/Trn3); ANTIBODY_COMPILE=1 adds
    torch.compile. Both apply on top of ANTIBODY_SMOKE.
    """
    cfg: dict[str, Any] = {}
    if os.environ.get("ANTIBODY_SMOKE") or os.environ.get("NER_SMOKE"):
        cfg = {
            "device": "cpu",
            "epochs": 1,
            "max_train_samples": 48,
            "max_eval_samples": 32,
            "max_length": 160,
        }
    if os.environ.get("ANTIBODY_DEVICE"):
        cfg["device"] = os.environ["ANTIBODY_DEVICE"]
    if os.environ.get("ANTIBODY_COMPILE"):
        cfg["compile"] = os.environ["ANTIBODY_COMPILE"] not in ("", "0")
    run(cfg)


if __name__ == "__main__":
    main()
