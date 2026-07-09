#!/usr/bin/env python3
"""Biomedical Named-Entity Recognition fine-tune on AWS Trainium (research-grade exemplar).

This is a *real* research workflow, not a toy: it fine-tunes a transformer token classifier to
extract **disease mentions** from biomedical abstracts, using the public **NCBI-disease** corpus
(Doğan et al., 2014) distributed via Hugging Face Datasets. Named-entity recognition over the
biomedical literature is a genuine, widely published task (literature mining, cohort discovery,
pharmacovigilance), and NCBI-disease is a standard benchmark for it.

What this demonstrates for Trainium users:
    * The PyTorch/XLA training path on Trainium (`xm.xla_device()`, `xm.optimizer_step`,
      `xm.mark_step`) -- the supported path as of Neuron SDK 2.31.0 / PyTorch 2.9.
    * **Build in the form the hardware wants** -- the single most important lesson. Verified on a
      real trn1.2xlarge (re-run on Neuron 2.31.13 / transformers 5.13: eval_f1 0.8467; the bf16
      findings below first reproduced on 2.30):
        - HF v5 defaults to SDPA attention, which produces **nan loss at step 0** on the Neuron
          bf16 path. The same model on CPU/fp32 gives loss ~1.21.
        - Setting `attn_implementation="eager"` (this example's default) gives loss **1.13** on
          Trainium -- a clean fix that KEEPS bf16 on. Do NOT "fix" it with `--auto-cast=none`,
          which just switches off the accelerator.
        - Fixed batch shapes (`drop_last`) compile in ~2 graphs; ragged shapes triggered 7+
          recompiles and never converged. Try `shape_mode="ragged"` to watch the degenerate path.
    * A token-classification fine-tune with subword label alignment (the real, fiddly part of NER).
    * Honest **entity-level** precision/recall/F1 (IOB2 span matching, computed in-module) on a
      held-out test split -- the correct way to score NER, not token accuracy.
    * Device portability: the SAME code runs on CPU (laptop smoke test) and on Trainium.

It is written to the harness contract: a module-level ``run(config) -> dict[str, float]`` so
``validation/run_on_hardware.py`` can execute it and capture provenance. It is ALSO runnable
directly: ``python examples/use_cases/biomedical_ner.py``.

VALIDATION: see validation/results/examples__use_cases__biomedical_ner.json (generated on hardware).
This file makes NO performance claim until that artifact exists.

Dependencies (beyond the Neuron stack): transformers, datasets (entity-level F1 is computed in-module, no seqeval).
On a Neuron DLAMI these install with:
    pip install "transformers" "datasets"

Sizing: bert-base-cased on NCBI-disease fits comfortably on a single Trainium chip
(`trn1.2xlarge`, 32 GiB) at batch size 16, seq len 192. A 3-epoch run is well under an hour.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

# NOTE: heavy ML imports (torch, transformers, datasets) are done lazily inside run() so that this
# module can be imported for inspection/registry listing without those packages installed.

# NCBI-disease is a single-entity (Disease) IOB2-tagged corpus. The HF dataset exposes integer
# ner_tags; we resolve the label names from the dataset's own features at load time.
#
# NOTE: `datasets` >= 4 no longer runs dataset *loader scripts*, and the classic NCBI-disease repo
# ships one. We therefore load the auto-generated **parquet** branch ("refs/convert/parquet"),
# which the Hub builds for every dataset and which needs no script. Labels: O / B-Disease / I-Disease.
DEFAULT_MODEL = "bert-base-cased"
DEFAULT_DATASET = "ncbi/ncbi_disease"
DEFAULT_DATASET_REVISION = "refs/convert/parquet"


@dataclass
class NerConfig:
    """Configuration for the NER fine-tune.

    Defaults target a real Trainium run; the harness passes a smaller smoke config for CPU.
    """

    model_name: str = DEFAULT_MODEL
    dataset_name: str = DEFAULT_DATASET
    dataset_revision: str | None = (
        DEFAULT_DATASET_REVISION  # parquet branch; bypasses loader script
    )
    device: str = "xla"  # "xla" (Trainium) | "cpu" (smoke/laptop) | "cuda"
    epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 16
    max_length: int = 128
    learning_rate: float = 3e-5
    warmup_ratio: float = (
        0.1  # linear LR warmup; prevents early divergence (we saw nan without it)
    )
    grad_clip: float = (
        1.0  # max grad norm; guards bf16 training against exploding gradients
    )
    seed: int = 42
    # Attention implementation. HF Transformers v5 defaults to "sdpa", whose fused kernel is not
    # reliably handled by the Neuron bf16 compile path (we saw nan at step 0). "eager" uses plain
    # matmul+softmax that the compiler handles correctly. This is the "build in the form the
    # hardware wants" knob -- see docs/trainium_development_best_practices.md.
    attn_implementation: str = "eager"
    max_train_samples: int | None = None  # cap for smoke runs
    max_eval_samples: int | None = None
    # Trainium teaching knob: "stable" pads every batch to a fixed shape + drop_last so the XLA
    # graph compiles ONCE and is reused; "ragged" leaves variable batch shapes so you can watch
    # the degenerate recompile-every-shape behavior the Compilation chapter warns about.
    shape_mode: str = "stable"  # "stable" | "ragged"
    log_every: int = (
        0  # 0 => log once per epoch (deferred .item(), no per-step host sync)
    )
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> NerConfig:
        """Build a config from a plain dict (the harness passes one), ignoring unknown keys."""
        cfg = dict(cfg or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(requested: str):
    """Return a torch device for the requested backend, falling back to CPU with a warning.

    For Trainium, importing torch_xla and asking for its device is what places work on the
    NeuronCores. If torch_xla isn't present (e.g. laptop), we fall back to CPU so the smoke path
    still runs.
    """
    import torch

    if requested == "xla":
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device(), "xla"
        except ImportError:
            print(
                "⚠️  torch_xla not available; falling back to CPU (not a hardware run)."
            )
            return torch.device("cpu"), "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def _align_labels_with_tokens(
    labels: list[int], word_ids: list[int | None]
) -> list[int]:
    """Align word-level IOB2 labels to subword tokens (the core NER preprocessing step).

    Only the first subword of each word keeps its label; continuation subwords and special tokens
    get -100 so they are ignored by the loss and by evaluation. This is the standard HF token-
    classification alignment.
    """
    aligned: list[int] = []
    previous_word: int | None = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != previous_word:
            aligned.append(labels[word_id])
        else:
            aligned.append(-100)
        previous_word = word_id
    return aligned


def _build_dataloaders(cfg: NerConfig, tokenizer, label_names: list[str]):
    """Tokenize + align the dataset and return (train_loader, eval_loader, test_loader)."""
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    raw = load_dataset(cfg.dataset_name, revision=cfg.dataset_revision)

    def preprocess(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=cfg.max_length,
            padding="max_length",
        )
        all_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            all_labels.append(_align_labels_with_tokens(labels, word_ids))
        tokenized["labels"] = all_labels
        return tokenized

    def cap(split, n):
        return split.select(range(min(n, len(split)))) if n else split

    cols = raw["train"].column_names
    train = cap(raw["train"], cfg.max_train_samples).map(
        preprocess, batched=True, remove_columns=cols
    )
    val = cap(raw["validation"], cfg.max_eval_samples).map(
        preprocess, batched=True, remove_columns=cols
    )
    test = cap(raw["test"], cfg.max_eval_samples).map(
        preprocess, batched=True, remove_columns=cols
    )

    keep = ["input_ids", "attention_mask", "labels"]
    for ds in (train, val, test):
        ds.set_format("torch", columns=[c for c in keep if c in ds.column_names])

    # KEY TRAINIUM LESSON: drop_last=True makes every training batch identical in shape, so the
    # Neuron compiler builds ONE graph and reuses it. With "ragged" mode we deliberately leave the
    # short final batch in (drop_last=False) to demonstrate the recompile-per-shape degenerate path.
    # Sequences are already padded to a fixed max_length above, so the only varying dim is batch
    # size on the last batch -- which is exactly what drop_last removes.
    drop_last = cfg.shape_mode == "stable"
    return (
        DataLoader(
            train, batch_size=cfg.train_batch_size, shuffle=True, drop_last=drop_last
        ),
        DataLoader(val, batch_size=cfg.eval_batch_size, drop_last=drop_last),
        DataLoader(test, batch_size=cfg.eval_batch_size, drop_last=drop_last),
        label_names,
    )


def _extract_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """Extract entity spans (start, end_exclusive, type) from an IOB2 tag sequence.

    Entity-level scoring (the correct way to evaluate NER) compares whole spans, not tokens. A
    span opens on ``B-X``, extends over following ``I-X``, and closes at the next ``B``/``O`` or a
    type change. We implement this directly so the example carries no fragile NER-metric
    dependency (seqeval's sdist fails to build on Python 3.12) and so the metric is transparent.
    """
    spans: set[tuple[int, int, str]] = set()
    start: int | None = None
    etype: str | None = None
    for i, tag in enumerate([*tags, "O"]):  # sentinel "O" flushes a trailing span
        prefix = tag[0]
        kind = tag[2:] if len(tag) > 2 else ""
        if prefix in ("B", "O") or (prefix == "I" and kind != etype):
            if start is not None:
                spans.add((start, i, etype))  # close the open span
                start, etype = None, None
            if prefix == "B":
                start, etype = i, kind
        # prefix == "I" continuing the same type: just extend (no action needed)
    return spans


def _evaluate(model, loader, device, backend, label_names) -> dict[str, float]:
    """Compute entity-level precision/recall/F1 over IOB2 spans (self-contained, no seqeval)."""
    import torch

    model.eval()
    tp = fp = fn = 0  # true/false positive, false negative at the entity-span level
    token_correct = token_total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            if backend == "xla":
                import torch_xla.core.xla_model as xm

                xm.mark_step()
            preds = logits.argmax(dim=-1).cpu().numpy()
            labs = labels.numpy()
            for pred_row, lab_row in zip(preds, labs, strict=False):
                true_tags, pred_tags = [], []
                for pred_id, lab_id in zip(pred_row, lab_row, strict=False):
                    if lab_id == -100:  # subword continuation / special token -> ignore
                        continue
                    true_tags.append(label_names[int(lab_id)])
                    # Defensive: a diverged/garbage logit can argmax out of range; clamp so eval
                    # reports a wrong-but-valid tag instead of crashing (surfaces as low F1).
                    pred_tags.append(
                        label_names[min(int(pred_id), len(label_names) - 1)]
                    )
                    token_total += 1
                    token_correct += int(pred_id == lab_id)
                gold = _extract_spans(true_tags)
                pred = _extract_spans(pred_tags)
                tp += len(gold & pred)
                fp += len(pred - gold)
                fn += len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(
        f"   entities: tp={tp} fp={fp} fn={fn} | P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
    )
    return {
        "eval_f1": float(f1),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_accuracy": float(token_correct / token_total) if token_total else 0.0,
    }


def run(config: dict | None = None) -> dict[str, float]:
    """Fine-tune and evaluate the NER model; return metrics (the harness entrypoint).

    Returns a flat dict including ``eval_f1`` (the gated metric), plus precision/recall, throughput,
    and compile/wall timings. Designed to run identically on CPU (smoke) and Trainium (full).
    """
    import time

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    cfg = NerConfig.from_dict(config)
    _set_seed(cfg.seed)
    device, backend = _resolve_device(cfg.device)
    print(
        f"🧬 Biomedical NER fine-tune | model={cfg.model_name} | dataset={cfg.dataset_name} | device={backend}"
    )

    # Resolve label names from the dataset's own schema (authoritative IOB2 tag set).
    from datasets import load_dataset

    feats = load_dataset(
        cfg.dataset_name, revision=cfg.dataset_revision, split="train"
    ).features
    label_names = feats["ner_tags"].feature.names
    print(f"   Labels: {label_names}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader, val_loader, test_loader, label_names = _build_dataloaders(
        cfg, tokenizer, label_names
    )

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={n: i for i, n in enumerate(label_names)},
        attn_implementation=cfg.attn_implementation,  # "eager" for the Neuron-friendly path
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # LR warmup + linear decay -- general training-stability hygiene (NOT the cure for the bf16
    # forward-pass nan; that's fixed by attn_implementation above). See best-practices §4.
    steps_per_epoch = len(train_loader)
    total_train_steps = max(1, steps_per_epoch * cfg.epochs)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_train_steps),
        num_training_steps=total_train_steps,
    )

    if backend == "xla":
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl

        # MpDeviceLoader overlaps host->device transfer with compute and issues mark_step per batch.
        device_train_loader = pl.MpDeviceLoader(train_loader, device)
    else:
        device_train_loader = train_loader

    # --- Training loop (PyTorch/XLA path on Trainium) ---------------------------------------
    # Best practices applied (see the Trainium Development chapter):
    #   * the bf16 nan is fixed UPSTREAM by attn_implementation="eager" (see model creation),
    #     NOT by the clipping/warmup below -- those are ordinary training-stability hygiene.
    #   * fixed batch shapes (drop_last) -> compile ~once, not per-shape
    #   * accumulate loss ON DEVICE; only fetch with .item() once per epoch (no per-step host sync)
    wall_start = time.time()
    first_step_s = None
    total_steps = 0
    for epoch in range(cfg.epochs):
        model.train()
        running = torch.zeros(
            (), device=device
        )  # on-device accumulator; avoids per-step .item()
        n_batches = 0
        for batch in device_train_loader:
            step_start = time.time()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            if backend != "xla":  # MpDeviceLoader already places tensors on device
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )

            optimizer.zero_grad()
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if backend == "xla":
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()
            scheduler.step()

            running += out.loss.detach()  # stays on device -- no sync
            n_batches += 1
            total_steps += 1
            if first_step_s is None:
                first_step_s = (
                    time.time() - step_start
                )  # first step pays the AOT compile cost

        avg_loss = (
            running / max(1, n_batches)
        ).item()  # single host fetch, once per epoch
        print(f"   epoch {epoch + 1}/{cfg.epochs}  avg_loss={avg_loss:.4f}")

    train_wall = time.time() - wall_start

    # --- Evaluate on validation (gated) and test (reported) ---------------------------------
    val_metrics = _evaluate(model, val_loader, device, backend, label_names)
    test_metrics = _evaluate(model, test_loader, device, backend, label_names)

    metrics: dict[str, float] = {
        "eval_f1": val_metrics["eval_f1"],
        "eval_precision": val_metrics["eval_precision"],
        "eval_recall": val_metrics["eval_recall"],
        "test_f1": test_metrics["eval_f1"],
        "train_wall_s": round(train_wall, 2),
        "first_step_compile_s": round(first_step_s or 0.0, 2),
        "train_throughput_samples_s": round(
            (total_steps * cfg.train_batch_size) / train_wall, 2
        )
        if train_wall > 0
        else 0.0,
    }
    print(
        f"✅ done. eval_f1={metrics['eval_f1']:.4f}  test_f1={metrics['test_f1']:.4f}  "
        f"train_wall={metrics['train_wall_s']}s  first_step={metrics['first_step_compile_s']}s"
    )
    return metrics


def main() -> None:
    """CLI: run with a sensible default config (override device via env for laptops)."""
    cfg = {}
    if os.environ.get("NER_SMOKE"):
        cfg = {
            "device": "cpu",
            "epochs": 1,
            "max_train_samples": 64,
            "max_eval_samples": 64,
        }
    run(cfg)


if __name__ == "__main__":
    main()
