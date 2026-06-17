#!/usr/bin/env python3
"""Data-parallel training across NeuronCores (PyTorch/XLA DDP on Trainium).

This fills the most conspicuous gap in the tutorial: a **real multi-NeuronCore** training example.
It's the data-parallel sibling of the validated single-core NER example — same task (disease NER on
NCBI-disease), same Trainium-native rules (eager attention for bf16, fixed shapes), but sharded
across the cores of one instance with PyTorch/XLA's distributed data parallel.

Why this matters: `trn1.2xlarge` has 2 NeuronCores; `trn1.32xlarge` has 32. Single-core training
leaves most of the chip idle. DDP replicates the model per core and averages gradients each step.

Launch mechanism (the AWS-recommended one for torch-neuronx / XLA):

    torchrun --nproc_per_node=2 examples/distributed/data_parallel_ner.py     # 2 cores (trn1.2xlarge)
    torchrun --nproc_per_node=32 examples/distributed/data_parallel_ner.py    # 32 cores (trn1.32xlarge)

`--nproc_per_node` = number of NeuronCores to use. Run `neuron-ls` to see how many you have.

CPU/off-hardware: this script is import-safe and exposes `run(config)` for the validation harness,
but DDP itself needs the XLA runtime — run the real thing with torchrun on a Neuron instance.

Key XLA-DDP pieces (vs. single-core training):
  * `torch.distributed.init_process_group(backend="xla")` — the Neuron/XLA collective backend.
  * `DistributedSampler` — each core sees a disjoint shard of the data (with drop_last for fixed shapes).
  * `xm.optimizer_step(optimizer)` — applies grads AND all-reduces them across cores in one call.
  * `pl.MpDeviceLoader` — overlaps host→device copy and calls mark_step per batch.
  * Checkpoint from rank 0 only, via `xm._maybe_convert_to_cpu` then `torch.save` (the
    Neuron-correct way to serialize XLA tensors).

Sources (public): the torch-neuronx DDP BERT tutorial (torchrun + `_mp_fn`, MpDeviceLoader,
`xm._maybe_convert_to_cpu` + torch.save) and PyTorch/XLA distributed docs. See
docs/trainium_development_best_practices.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Reuse the validated example's data + metric code so the two stay consistent.
from examples.use_cases.biomedical_ner import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    DEFAULT_MODEL,
    _align_labels_with_tokens,  # noqa: F401 (re-exported for parity / docs)
    _evaluate,
    _extract_spans,  # noqa: F401
)


@dataclass
class DDPConfig:
    """Config for the data-parallel NER fine-tune."""

    model_name: str = DEFAULT_MODEL
    dataset_name: str = DEFAULT_DATASET
    dataset_revision: str | None = DEFAULT_DATASET_REVISION
    attn_implementation: str = "eager"  # bf16-stable on Neuron (see best-practices §4)
    epochs: int = 2
    per_core_batch_size: int = 16
    max_length: int = 64
    learning_rate: float = 3e-5
    grad_clip: float = 1.0
    seed: int = 42
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> DDPConfig:
        """Build from a plain dict, ignoring unknown keys."""
        cfg = dict(cfg or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})


def _build_sharded_loaders(cfg: DDPConfig, tokenizer, label_names, world_size, rank):
    """Tokenize NCBI-disease and shard the train set across cores with DistributedSampler."""
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    raw = load_dataset(cfg.dataset_name, revision=cfg.dataset_revision)

    def preprocess(batch):
        tok = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=cfg.max_length,
            padding="max_length",
        )
        tok["labels"] = [
            _align_labels_with_tokens(batch["ner_tags"][i], tok.word_ids(batch_index=i))
            for i in range(len(batch["ner_tags"]))
        ]
        return tok

    def cap(split, n):
        return split.select(range(min(n, len(split)))) if n else split

    cols = raw["train"].column_names
    train = cap(raw["train"], cfg.max_train_samples).map(
        preprocess, batched=True, remove_columns=cols
    )
    val = cap(raw["validation"], cfg.max_eval_samples).map(
        preprocess, batched=True, remove_columns=cols
    )
    for ds in (train, val):
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Each rank (NeuronCore) sees a disjoint shard. drop_last keeps shapes fixed -> compile once.
    train_sampler = DistributedSampler(
        train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(
        train, batch_size=cfg.per_core_batch_size, sampler=train_sampler, drop_last=True
    )
    # Eval only on rank 0, unsharded.
    eval_loader = DataLoader(val, batch_size=cfg.per_core_batch_size, drop_last=True)
    return train_loader, eval_loader, train_sampler


def train_ddp(cfg: DDPConfig) -> dict[str, float]:
    """The per-process training function (one process per NeuronCore). Returns rank-0 metrics."""
    import torch
    import torch.distributed as dist
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(cfg.seed)
    device = xm.xla_device()
    # XLA is the collective backend for Neuron. torchrun sets RANK/WORLD_SIZE in the environment.
    dist.init_process_group(backend="xla")
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    if rank == 0:
        print(
            f"🧬 DDP NER | world_size={world_size} cores | model={cfg.model_name} | attn={cfg.attn_implementation}"
        )

    from datasets import load_dataset

    label_names = (
        load_dataset(cfg.dataset_name, revision=cfg.dataset_revision, split="train")
        .features["ner_tags"]
        .feature.names
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader, eval_loader, sampler = _build_sharded_loaders(
        cfg, tokenizer, label_names, world_size, rank
    )

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={n: i for i, n in enumerate(label_names)},
        attn_implementation=cfg.attn_implementation,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )

    device_loader = pl.MpDeviceLoader(train_loader, device)
    for epoch in range(cfg.epochs):
        model.train()
        sampler.set_epoch(epoch)  # reshuffle shards each epoch
        running = torch.zeros((), device=device)
        n = 0
        for batch in device_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            # optimizer_step applies grads AND all-reduces them across cores (the DDP heart).
            xm.optimizer_step(optimizer)
            scheduler.step()
            running += out.loss.detach()
            n += 1
        if rank == 0:
            print(
                f"   epoch {epoch + 1}/{cfg.epochs}  rank0 avg_loss={(running / max(1, n)).item():.4f}"
            )

    # Evaluate + checkpoint on rank 0 only.
    metrics: dict[str, float] = {}
    if rank == 0:
        metrics = _evaluate(model, eval_loader, device, "xla", label_names)
        metrics["world_size"] = float(world_size)
        # Neuron-correct checkpointing: move XLA tensors to CPU, then torch.save.
        os.makedirs("/tmp/ddp_ner_ckpt", exist_ok=True)
        cpu_state = xm._maybe_convert_to_cpu(model.state_dict())
        torch.save(cpu_state, "/tmp/ddp_ner_ckpt/model.pt")
        print(
            f"✅ DDP done. eval_f1={metrics.get('eval_f1', float('nan')):.4f} on {world_size} cores; checkpoint saved."
        )

    xm.rendezvous("train_ddp_finished")
    return metrics


def run(config: dict | None = None) -> dict[str, float]:
    """Harness entrypoint.

    NOTE: real DDP must be launched with `torchrun --nproc_per_node=<cores>` so each NeuronCore gets
    its own process. When this `run()` is called inside an existing torchrun context (RANK set), it
    drives one process. Outside torchrun there's nothing to parallelize, so it explains how to launch.
    """
    cfg = DDPConfig.from_dict(config)
    if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
        print(
            "This is a distributed example. Launch it with torchrun on a Neuron instance:\n"
            "    torchrun --nproc_per_node=2 examples/distributed/data_parallel_ner.py\n"
            "(--nproc_per_node = number of NeuronCores; see `neuron-ls`)."
        )
        return {}
    return train_ddp(cfg)


def main() -> None:
    """CLI entrypoint (under torchrun)."""
    run({})


if __name__ == "__main__":
    main()
