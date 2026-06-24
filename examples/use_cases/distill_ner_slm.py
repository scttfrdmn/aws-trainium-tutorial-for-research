#!/usr/bin/env python3
"""Distill a biomedical-NER teacher into a Small Language Model (SLM) on AWS Trainium.

The other examples *use* models (fine-tune, LoRA). This one **makes a small one**: it takes the
fine-tuned BERT-base NER teacher from [`biomedical_ner.py`](biomedical_ner.py) and **distills** it
into a much smaller student (a 4-layer, narrow-hidden BERT) that keeps most of the entity-F1 at a
fraction of the parameters and latency. Knowledge distillation (Hinton et al., 2015) is the
workhorse behind nearly every deployable "small" model (DistilBERT, TinyBERT, the SLM wave of
2024-2026), and it is a *great* Trainium workload: teacher forward + student forward/backward are
dense transformer matmuls at fixed shapes — exactly the form the systolic array wants
(see [`cv_utilization_spike.py`](cv_utilization_spike.py) for why "shape the hardware wants" matters).

What this demonstrates for Trainium users:
    * **Distillation on the PyTorch/XLA path** — two models on device, a combined loss, one
      `xm.optimizer_step()`/`xm.mark_step()` per step. The teacher runs in `torch.no_grad()` (no
      backward), the student trains.
    * The standard distillation loss: a temperature-scaled **KL** term matching the student's logits
      to the teacher's soft labels, blended with the ordinary **CE** hard-label loss
      (`loss = α·CE + (1-α)·T²·KL`). Token classification ⇒ the KL is computed per (real) token.
    * **Build in the form the hardware wants** — same lessons as the teacher: `attn_implementation=
      "eager"` (HF v5 SDPA → `nan` on the Neuron bf16 path) and fixed shapes + `drop_last` so the
      graph compiles once. These carry over unchanged to the student.
    * A real, reportable outcome: student **entity-F1 vs teacher F1**, plus the **compression ratio**
      and **parameter counts** — distillation's actual selling point (small + fast, not just fast).

Reuses the validated NER data pipeline and metric from biomedical_ner.py (same corpus, same
entity-level span F1), so this example is about the *distillation technique*, not re-deriving NER.

Harness contract: a module-level ``run(config) -> dict[str, float]`` returning ``student_f1`` (the
gated metric) and ``f1_retention`` (student_f1 / teacher_f1). Runs on CPU (smoke) and Trainium.

    # Laptop smoke test (CPU, tiny subset — proves the code path):
    DISTILL_SMOKE=1 python examples/use_cases/distill_ner_slm.py

    # On a Trainium instance (real run):
    python examples/use_cases/distill_ner_slm.py

Sizing: teacher bert-base-cased (~108M) → student 4-layer/hidden-512 (~28M) fits easily on a single
`trn1.2xlarge`. The teacher is fine-tuned on-the-fly for a couple of epochs first (so the example is
self-contained); point `teacher_ckpt` at a saved teacher to skip that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# We reuse the validated NER building blocks verbatim: dataset/label loading, subword-label
# alignment, dataloaders (fixed shapes + drop_last), and entity-level span F1. This keeps the
# example focused on distillation and guarantees the metric matches the teacher's.
from examples.use_cases.biomedical_ner import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    _build_dataloaders,
    _evaluate,
    _resolve_device,
    _set_seed,
)

DEFAULT_TEACHER = "bert-base-cased"


@dataclass
class DistillConfig:
    """Configuration for NER distillation (teacher → small student)."""

    teacher_name: str = DEFAULT_TEACHER
    teacher_ckpt: str | None = (
        None  # path to a saved fine-tuned teacher; else fine-tune inline
    )
    dataset_name: str = DEFAULT_DATASET
    dataset_revision: str | None = DEFAULT_DATASET_REVISION
    device: str = "xla"  # "xla" (Trainium) | "cpu" (smoke) | "cuda"

    # Student architecture (a small BERT). These are the knobs that make it an SLM.
    student_layers: int = 4
    student_hidden: int = 512
    student_heads: int = 8
    student_intermediate: int = 2048

    # Teacher fine-tune (only if no teacher_ckpt given) + distillation schedule.
    teacher_epochs: int = 2
    distill_epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 16
    max_length: int = 128
    learning_rate: float = 5e-5
    temperature: float = 2.0  # softmax temperature for the soft-label KL term
    alpha_ce: float = (
        0.5  # weight on hard-label CE; (1-alpha) weights the soft-label KL
    )
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    attn_implementation: str = (
        "eager"  # Neuron-friendly (SDPA → nan in bf16); see NER example
    )
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> DistillConfig:
        cfg = dict(cfg or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})


def _build_student(cfg: DistillConfig, num_labels: int, label_names: list[str]):
    """Construct a small token-classification BERT from scratch (random init — the student learns
    from the teacher, not from pretrained weights). Small + fixed-shape ⇒ Trainium-friendly.
    """
    from transformers import BertConfig, BertForTokenClassification

    student_config = BertConfig(
        num_hidden_layers=cfg.student_layers,
        hidden_size=cfg.student_hidden,
        num_attention_heads=cfg.student_heads,
        intermediate_size=cfg.student_intermediate,
        num_labels=num_labels,
        id2label=dict(enumerate(label_names)),
        label2id={n: i for i, n in enumerate(label_names)},
        attn_implementation=cfg.attn_implementation,
    )
    return BertForTokenClassification(student_config)


def _train_teacher(cfg: DistillConfig, train_loader, device, backend, label_names):
    """Fine-tune the teacher inline (so the example is self-contained). Mirrors biomedical_ner.py."""
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        get_linear_schedule_with_warmup,
    )

    print(f"🎓 Fine-tuning teacher {cfg.teacher_name} ({cfg.teacher_epochs} epochs)...")
    teacher = AutoModelForTokenClassification.from_pretrained(
        cfg.teacher_name,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={n: i for i, n in enumerate(label_names)},
        attn_implementation=cfg.attn_implementation,
    ).to(device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=3e-5)
    total = max(1, len(train_loader) * cfg.teacher_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total), total
    )

    loader = _device_loader(train_loader, device, backend)
    for epoch in range(cfg.teacher_epochs):
        teacher.train()
        running = torch.zeros((), device=device)
        n = 0
        for batch in loader:
            ids, mask, labels = _batch_to_device(batch, device, backend)
            optimizer.zero_grad()
            out = teacher(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), cfg.grad_clip)
            _step(optimizer, backend)
            scheduler.step()
            running += out.loss.detach()
            n += 1
        print(
            f"   teacher epoch {epoch + 1}/{cfg.teacher_epochs}  avg_loss={(running / max(1, n)).item():.4f}"
        )
    return teacher


def _device_loader(train_loader, device, backend):
    """Wrap in MpDeviceLoader on XLA (overlap H2D copy + compute, mark_step per batch)."""
    if backend == "xla":
        import torch_xla.distributed.parallel_loader as pl

        return pl.MpDeviceLoader(train_loader, device)
    return train_loader


def _batch_to_device(batch, device, backend):
    """Return (input_ids, attention_mask, labels) on device. MpDeviceLoader pre-places on XLA."""
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    if backend != "xla":
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
    return ids, mask, labels


def _step(optimizer, backend):
    """One optimizer step on the right backend (XLA materializes the graph with mark_step)."""
    if backend == "xla":
        import torch_xla.core.xla_model as xm

        xm.optimizer_step(optimizer)
        xm.mark_step()
    else:
        optimizer.step()


def _distillation_loss(student_logits, teacher_logits, labels, cfg: DistillConfig):
    """Combined distillation loss: α·CE(hard) + (1-α)·T²·KL(soft) over REAL tokens only.

    `labels == -100` marks subword continuations / special tokens (ignored). The KL term matches
    the student's softened distribution to the teacher's; multiplying by T² keeps gradient
    magnitudes comparable to the CE term (the standard Hinton scaling).
    """
    import torch
    import torch.nn.functional as F

    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    # Soft-label KL on the real tokens (mask out -100 positions so padding doesn't dominate).
    mask = labels.view(-1) != -100
    s = student_logits.view(-1, student_logits.size(-1))[mask]
    t = teacher_logits.view(-1, teacher_logits.size(-1))[mask]
    if s.numel() == 0:
        return ce, ce.detach(), torch.zeros((), device=ce.device)
    T = cfg.temperature
    kl = F.kl_div(
        F.log_softmax(s / T, dim=-1),
        F.softmax(t / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    loss = cfg.alpha_ce * ce + (1.0 - cfg.alpha_ce) * kl
    return loss, ce.detach(), kl.detach()


def run(config: dict | None = None) -> dict[str, float]:
    """Distill the NER teacher into a small student; return metrics (the harness entrypoint).

    Gated metric: ``student_f1`` (entity-level span F1 of the student on the validation split).
    Also reports ``teacher_f1``, ``f1_retention`` (student/teacher), and the ``compression_ratio``.
    """
    import time

    import torch
    from transformers import AutoTokenizer

    cfg = DistillConfig.from_dict(config)
    _set_seed(cfg.seed)
    device, backend = _resolve_device(cfg.device)
    print(
        f"🧪 NER distillation | teacher={cfg.teacher_name} | "
        f"student={cfg.student_layers}L/{cfg.student_hidden}H | device={backend}"
    )

    # Reuse the validated label resolution + dataloaders from the NER example.
    from datasets import load_dataset

    feats = load_dataset(
        cfg.dataset_name, revision=cfg.dataset_revision, split="train"
    ).features
    label_names = feats["ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_name)
    # _build_dataloaders takes an NER-style config; build a tiny shim with the fields it reads.
    ner_like = _NerLike(cfg)
    train_loader, val_loader, _test_loader, label_names = _build_dataloaders(
        ner_like, tokenizer, label_names
    )

    # --- Teacher: load a checkpoint or fine-tune inline -------------------------------------
    if cfg.teacher_ckpt:
        from transformers import AutoModelForTokenClassification

        print(f"🎓 Loading teacher checkpoint: {cfg.teacher_ckpt}")
        teacher = AutoModelForTokenClassification.from_pretrained(
            cfg.teacher_ckpt, attn_implementation=cfg.attn_implementation
        ).to(device)
    else:
        teacher = _train_teacher(cfg, train_loader, device, backend, label_names)
    teacher.eval()
    teacher_metrics = _evaluate(teacher, val_loader, device, backend, label_names)
    teacher_f1 = teacher_metrics["eval_f1"]
    print(f"   teacher eval_f1={teacher_f1:.4f}")

    # --- Student: distill from the (frozen) teacher -----------------------------------------
    student = _build_student(cfg, len(label_names), label_names).to(device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    compression = teacher_params / student_params if student_params else 0.0
    print(
        f"📦 teacher={teacher_params / 1e6:.1f}M params → student={student_params / 1e6:.1f}M "
        f"({compression:.1f}× smaller)"
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate)
    from transformers import get_linear_schedule_with_warmup

    total = max(1, len(train_loader) * cfg.distill_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total), total
    )

    loader = _device_loader(train_loader, device, backend)
    wall_start = time.time()
    first_step_s = None
    for epoch in range(cfg.distill_epochs):
        student.train()
        running = torch.zeros((), device=device)
        n = 0
        for batch in loader:
            step_start = time.time()
            ids, mask, labels = _batch_to_device(batch, device, backend)
            optimizer.zero_grad()
            with torch.no_grad():  # teacher provides soft labels only — no backward
                teacher_logits = teacher(input_ids=ids, attention_mask=mask).logits
            student_logits = student(input_ids=ids, attention_mask=mask).logits
            loss, _ce, _kl = _distillation_loss(
                student_logits, teacher_logits, labels, cfg
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            _step(optimizer, backend)
            scheduler.step()
            running += loss.detach()
            n += 1
            if first_step_s is None:
                first_step_s = time.time() - step_start
        print(
            f"   distill epoch {epoch + 1}/{cfg.distill_epochs}  avg_loss={(running / max(1, n)).item():.4f}"
        )
    train_wall = time.time() - wall_start

    student_metrics = _evaluate(student, val_loader, device, backend, label_names)
    student_f1 = student_metrics["eval_f1"]
    retention = (student_f1 / teacher_f1) if teacher_f1 else 0.0

    metrics = {
        "student_f1": float(student_f1),
        "teacher_f1": float(teacher_f1),
        "f1_retention": float(retention),
        "compression_ratio": float(round(compression, 2)),
        "student_params_m": float(round(student_params / 1e6, 2)),
        "teacher_params_m": float(round(teacher_params / 1e6, 2)),
        "train_wall_s": round(train_wall, 2),
        "first_step_compile_s": round(first_step_s or 0.0, 2),
    }
    print(
        f"✅ done. student_f1={student_f1:.4f} (teacher {teacher_f1:.4f}, "
        f"retention {retention:.1%}) | {compression:.1f}× smaller"
    )
    return metrics


class _NerLike:
    """Adapter exposing the fields biomedical_ner._build_dataloaders reads, from a DistillConfig.

    Lets us reuse the validated NER data pipeline without subclassing or duplicating it. Shape mode
    is fixed to "stable" (drop_last) — the Trainium-correct default; distillation has no reason to
    demonstrate the ragged degenerate path.
    """

    def __init__(self, cfg: DistillConfig):
        self.dataset_name = cfg.dataset_name
        self.dataset_revision = cfg.dataset_revision
        self.train_batch_size = cfg.train_batch_size
        self.eval_batch_size = cfg.eval_batch_size
        self.max_length = cfg.max_length
        self.max_train_samples = cfg.max_train_samples
        self.max_eval_samples = cfg.max_eval_samples
        self.shape_mode = "stable"


def main() -> None:
    """CLI entrypoint; DISTILL_SMOKE runs a tiny CPU configuration."""
    cfg: dict[str, Any] = {}
    if os.environ.get("DISTILL_SMOKE") or os.environ.get("NER_SMOKE"):
        cfg = {
            "device": "cpu",
            "teacher_epochs": 1,
            "distill_epochs": 1,
            "max_train_samples": 64,
            "max_eval_samples": 64,
            "student_layers": 2,
            "student_hidden": 128,
            "student_heads": 2,
            "student_intermediate": 256,
        }
    run(cfg)


if __name__ == "__main__":
    main()
