#!/usr/bin/env python3
"""Generate crystal-structure CIFs from a composition with a small GPT on AWS Trainium.

A real materials-informatics task: train a **Small Language Model from scratch** to generate a
**crystallographic information file (CIF)** — the text description of a crystal structure — given a
chemical composition prompt (e.g. ``Na Cl`` → a CIF for rock salt). This is the core idea of
**CrystaLLM** (Antunes, Butler & Grau-Crespo, *Nature Communications* 2024): a crystal structure is
just *text*, so an autoregressive language model can learn to write valid structures, and sampling
the model is a fast generative prior for materials discovery.

We train a compact **character-level GPT** (a few transformer-decoder blocks) on the public
``yhjollin/CrystaLLM_use_Chemeleon_data_1`` corpus (36k composition→CIF pairs). Character-level
keeps the example self-contained (no external tokenizer) and is exactly how the original CrystaLLM
tokenizes structural text.

What this demonstrates for Trainium users:
    * **Pretraining a transformer decoder from scratch** on the PyTorch/XLA path — the generative
      counterpart to the encoder fine-tunes elsewhere in this repo. Causal self-attention, a fixed
      context window, next-token cross-entropy.
    * **Build in the form the hardware wants** — a **fixed block size** (every sequence padded/cropped
      to `block_size`) + `drop_last` so the graph compiles once; bf16-stable hand-written attention
      with an **fp32 softmax** (no `scaled_dot_product_attention`, which `nan`s in bf16 on Neuron —
      the same lesson as the NER and ViT examples).
    * A real, reportable outcome: **validation cross-entropy / perplexity** (the honest LM metric)
      plus a **sampled CIF** from a held-out composition so you can eyeball what the model writes.

> **Scope (honest):** v1 reports language-model loss/perplexity and prints a generated sample. It does
> NOT score structural *validity* (parseable CIF, charge balance, physical bond lengths) — that needs
> a domain validator (e.g. pymatgen) and is a deliberate follow-up. A low perplexity means the model
> learned CIF *syntax and statistics*, not that every sample is a synthesizable crystal.

Harness contract: a module-level ``run(config) -> dict[str, float]`` returning ``val_perplexity``
(lower is better). The harness gates on the derived ``inv_val_perplexity`` = 1/perplexity, which is
in (0, 1] and increases as the model improves. Runs on CPU (smoke) and Trainium.

    # Laptop smoke test (CPU, tiny subset — proves the code path):
    CRYSTAL_SMOKE=1 python examples/use_cases/crystal_cif_slm.py

    # On a Trainium instance (real run):
    python examples/use_cases/crystal_cif_slm.py

Sizing: a ~6-layer, 384-wide char GPT (~10-15M params) at block_size 512 trains comfortably on a
single `trn1.2xlarge`; scale layers/width/block_size up on `trn1.32xlarge` for a stronger model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_DATASET = "yhjollin/CrystaLLM_use_Chemeleon_data_1"


@dataclass
class CrystalConfig:
    """Configuration for the character-level crystal-CIF GPT."""

    dataset_name: str = DEFAULT_DATASET
    formula_column: str = "Reduced Formula"
    cif_column: str = "CIF"
    device: str = "xla"  # "xla" (Trainium) | "cpu" (smoke) | "cuda"

    # GPT architecture (small by design — this is an SLM).
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 512  # context window (chars). Fixed → static shapes.
    dropout: float = 0.1

    epochs: int = 1  # one pass over 36k CIFs is already a substantial run
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0
    log_every: int = (
        25  # stream step progress every N steps (0 = per-epoch only). See _progress.py
    )
    seed: int = 42
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    sample_tokens: int = 256  # how many chars to generate in the end-of-run demo
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> CrystalConfig:
        cfg = dict(cfg or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(requested: str):
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


# A tiny prompt/answer separator so the model learns "composition → CIF". Kept as printable chars so
# they live in the char vocab naturally.
PROMPT_SEP = "\n>>>\n"


def _build_corpus(cfg: CrystalConfig):
    """Load the dataset and build per-example 'formula >>> CIF' documents + a char vocabulary."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name)
    train_split = ds["train"]
    val_split = ds["validation"] if "validation" in ds else ds["test"]

    def cap(split, n):
        return split.select(range(min(n, len(split)))) if n else split

    def to_docs(split):
        docs = []
        for ex in split:
            formula = str(ex[cfg.formula_column])
            cif = str(ex[cfg.cif_column])
            docs.append(f"{formula}{PROMPT_SEP}{cif}")
        return docs

    train_docs = to_docs(cap(train_split, cfg.max_train_samples))
    val_docs = to_docs(cap(val_split, cfg.max_eval_samples))

    # Character vocabulary from the training docs (+ a pad char). Char-level keeps the example
    # dependency-free and matches CrystaLLM's structural-text tokenization.
    chars = sorted(set("".join(train_docs)))
    pad = "\x00"
    if pad not in chars:
        chars = [pad, *chars]
    stoi = {c: i for i, c in enumerate(chars)}
    return train_docs, val_docs, stoi, chars, stoi[pad]


def _encode_block(doc: str, stoi: dict, block_size: int, pad_id: int):
    """Encode one document to a fixed-length (block_size+1) int sequence; pad/crop. Unknown chars
    map to pad (keeps eval robust to a val-only character)."""
    ids = [stoi.get(c, pad_id) for c in doc][: block_size + 1]
    if len(ids) < block_size + 1:
        ids = ids + [pad_id] * (block_size + 1 - len(ids))
    return ids


def _build_loaders(cfg: CrystalConfig, train_docs, val_docs, stoi, pad_id):
    """Return (train_loader, val_loader). Each item is a fixed (block_size+1) int tensor; the model
    shifts it into (input, target). Fixed length + drop_last → one compiled graph."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def make(docs):
        rows = [_encode_block(d, stoi, cfg.block_size, pad_id) for d in docs]
        return TensorDataset(torch.tensor(rows, dtype=torch.long))

    train_loader = DataLoader(
        make(train_docs), batch_size=cfg.train_batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        make(val_docs), batch_size=cfg.eval_batch_size, drop_last=True
    )
    return train_loader, val_loader


def _build_model(cfg: CrystalConfig, vocab_size: int, pad_id: int):
    """A small GPT (decoder-only transformer). Attention is hand-written with an fp32 softmax and a
    causal mask — NOT F.scaled_dot_product_attention, which nans in bf16 on Neuron."""
    import math

    import torch
    import torch.nn as nn

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_head = cfg.n_head
            self.head_dim = cfg.n_embd // cfg.n_head
            self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
            self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.drop = nn.Dropout(cfg.dropout)
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            )
            self.register_buffer("mask", mask)

        def forward(self, x):
            b, t, c = x.shape
            qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Mask future positions with a large FINITE negative, NOT float("-inf"). On the bf16
            # Neuron path, -inf in the masked positions produces `0 * -inf = nan` in the softmax
            # BACKWARD pass (we hit loss=nan at step ~25 on a trn1.2xlarge). A large finite value
            # (-1e9) softmaxes to ~0 just the same but keeps gradients finite. (Same family of bf16
            # attention gotcha as the SDPA→nan lesson in the NER example.)
            att = att.masked_fill(self.mask[:, :, :t, :t] == 0, -1e9)
            att = torch.softmax(att.float(), dim=-1).to(
                v.dtype
            )  # fp32 softmax (bf16-safe)
            y = att @ v
            y = y.transpose(1, 2).reshape(b, t, c)
            return self.drop(self.proj(y))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.attn = CausalSelfAttention()
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layer)])
            self.ln_f = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
            self.pad_id = pad_id
            self.block_size = cfg.block_size

        def forward(self, idx, targets=None):
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device).unsqueeze(0)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
            for blk in self.blocks:
                x = blk(x)
            logits = self.head(self.ln_f(x))
            loss = None
            if targets is not None:
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.pad_id,  # don't train on padding
                )
            return logits, loss

    return GPT()


def _generate(model, prompt_ids, stoi, itos, cfg: CrystalConfig, device):
    """Greedy autoregressive sample for the end-of-run demo.

    Runs the decode loop on **CPU**: a composition prompt is short (~10 chars), and a negative
    slice like ``idx[:, -block_size:]`` on a sequence shorter than block_size raised "Value out of
    range" on the XLA backend (hit on a trn1.2xlarge). Generation is a tiny, one-off qualitative
    demo, so we move the model to CPU and use an explicit non-negative slice start — robust and
    backend-independent. (Training/eval stay on the device; only this demo decode is on CPU.)
    """
    import torch

    cpu_model = model.to("cpu")
    cpu_model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(cfg.sample_tokens):
            start = max(0, idx.shape[1] - cfg.block_size)  # explicit, non-negative
            idx_cond = idx[:, start:]
            logits, _ = cpu_model(idx_cond)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
    return "".join(itos[int(i)] for i in idx[0].tolist())


def run(config: dict | None = None) -> dict[str, float]:
    """Train the crystal-CIF GPT; return metrics (the harness entrypoint).

    Gated metric: ``neg_val_perplexity`` = -perplexity (higher is better, so the harness's
    ``metric >= threshold`` gate works). Also reports raw ``val_perplexity`` and ``val_loss``.
    """
    import math
    import time

    import torch
    from transformers import get_linear_schedule_with_warmup

    cfg = CrystalConfig.from_dict(config)
    _set_seed(cfg.seed)
    device, backend = _resolve_device(cfg.device)
    print(
        f"🔬 Crystal-CIF GPT (CrystaLLM-style) | {cfg.n_layer}L/{cfg.n_embd}d | "
        f"block={cfg.block_size} | device={backend}"
    )

    train_docs, val_docs, stoi, chars, pad_id = _build_corpus(cfg)
    itos = {i: c for c, i in stoi.items()}
    print(
        f"   corpus: {len(train_docs)} train / {len(val_docs)} val docs | vocab={len(chars)} chars"
    )
    train_loader, val_loader = _build_loaders(cfg, train_docs, val_docs, stoi, pad_id)

    model = _build_model(cfg, len(chars), pad_id).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   model params: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total), total
    )

    if backend == "xla":
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl

        device_loader = pl.MpDeviceLoader(train_loader, device)
    else:
        device_loader = train_loader

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
        for (block,) in device_loader:
            step_start = time.time()
            if backend != "xla":
                block = block.to(device)
            inputs, targets = block[:, :-1], block[:, 1:]  # next-char prediction
            optimizer.zero_grad()
            _logits, loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if backend == "xla":
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()
            scheduler.step()
            running += loss.detach()
            n += 1
            gstep += 1
            if first_step_s is None:
                first_step_s = time.time() - step_start
            progress.step(gstep, loss)
        print(
            f"   epoch {epoch + 1}/{cfg.epochs}  avg_loss={(running / max(1, n)).item():.4f}"
        )
    train_wall = time.time() - wall_start

    # --- Validation loss / perplexity --------------------------------------------------------
    model.eval()
    val_running = torch.zeros((), device=device)
    vn = 0
    with torch.no_grad():
        for (block,) in val_loader:
            block = block.to(device)
            inputs, targets = block[:, :-1], block[:, 1:]
            _logits, loss = model(inputs, targets)
            if backend == "xla":
                xm.mark_step()
            val_running += loss.detach()
            vn += 1
    val_loss = (val_running / max(1, vn)).item()
    val_ppl = math.exp(
        min(val_loss, 20.0)
    )  # clamp to avoid overflow on an untrained smoke run

    # --- Sample a CIF from a held-out composition (qualitative demo) -------------------------
    sample_formula = val_docs[0].split(PROMPT_SEP)[0] if val_docs else "Na Cl"
    prompt = f"{sample_formula}{PROMPT_SEP}"
    prompt_ids = [stoi.get(c, pad_id) for c in prompt][: cfg.block_size]
    sample = _generate(model, prompt_ids, stoi, itos, cfg, device)
    print(f"   sample generation for composition '{sample_formula}':")
    print("   " + sample.replace("\n", "\n   ")[:400])

    metrics = {
        # Gated metric: inverse perplexity in (0, 1], higher = better. A trained char-LM on CIF
        # text reaches low single-digit perplexity (inv >= 0.1); the untrained smoke run sits near 0.
        "inv_val_perplexity": float(1.0 / val_ppl) if val_ppl else 0.0,
        "val_perplexity": float(round(val_ppl, 3)),
        "val_loss": float(round(val_loss, 4)),
        "params_m": float(round(n_params / 1e6, 2)),
        "train_wall_s": round(train_wall, 2),
        "first_step_compile_s": round(first_step_s or 0.0, 2),
    }
    print(
        f"✅ done. val_perplexity={metrics['val_perplexity']:.3f}  "
        f"val_loss={metrics['val_loss']:.4f}  train_wall={metrics['train_wall_s']}s"
    )
    return metrics


def main() -> None:
    """CLI entrypoint; CRYSTAL_SMOKE runs a tiny CPU configuration."""
    cfg: dict[str, Any] = {}
    if os.environ.get("CRYSTAL_SMOKE") or os.environ.get("NER_SMOKE"):
        cfg = {
            "device": "cpu",
            "epochs": 1,
            "max_train_samples": 64,
            "max_eval_samples": 32,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 64,
            "block_size": 128,
            "sample_tokens": 64,
        }
    run(cfg)


if __name__ == "__main__":
    main()
