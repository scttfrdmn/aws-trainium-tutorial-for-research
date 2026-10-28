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

Native PyTorch DDP (opt-in; TorchNeuron is BETA). PyTorch/XLA above is the default, documented
path. Setting ``DDP_DEVICE`` to ``"cpu"`` or ``"neuron"`` switches to plain PyTorch distributed:
``dist.init_process_group(backend=DDP_PG_BACKEND)``, ``torch.nn.parallel.DistributedDataParallel``
(gradient all-reduce happens in ``backward``), plain ``optimizer.step()``, ``dist.barrier()``.

    # Laptop, 2 CPU processes over gloo (proves the DDP code path end to end):
    DDP_DEVICE=cpu DDP_PG_BACKEND=gloo DDP_SMOKE=1 \\
        torchrun --nproc_per_node=2 examples/distributed/data_parallel_ner.py

    # TorchNeuron (Trn2/Trn3, beta). The process-group backend name is UNVERIFIED -- discover it on
    # hardware (validation/torchneuron_facts.py records the available backends) and pass it in:
    DDP_DEVICE=neuron DDP_PG_BACKEND=<name> \\
        torchrun --nproc_per_node=2 examples/distributed/data_parallel_ner.py

Sources (public): the torch-neuronx DDP BERT tutorial (torchrun + `_mp_fn`, MpDeviceLoader,
`xm._maybe_convert_to_cpu` + torch.save) and PyTorch/XLA distributed docs. See
docs/trainium_development_best_practices.md.
"""

from __future__ import annotations

# Allow `torchrun examples/distributed/<this>.py` (a plain script) as well as `-m`/the harness: put
# the repo root on sys.path so the `examples.use_cases.*` imports below resolve.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_ROOT = str(_Path(__file__).resolve().parents[2])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import os
from collections.abc import Callable, Sequence
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
    # "xla" (default, documented: PyTorch/XLA DDP) | "neuron" (TorchNeuron native DDP, BETA,
    # Trn2/Trn3) | "cpu" (native DDP over gloo -- laptop proof of the code path).
    device: str = "xla"
    # Process-group backend for the native path (the "xla" path always uses "xla").
    # None => "gloo" on cpu; REQUIRED for "neuron" (the TorchNeuron name is unverified).
    pg_backend: str | None = None
    # Native path, device="neuron" only: route the forward through torch.compile(backend="neuron").
    compile: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> DDPConfig:
        """Build from a plain dict, ignoring unknown keys."""
        cfg = dict(cfg or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})


NATIVE_DEVICES = ("cpu", "neuron")
# Comma-separated candidate names to probe for device="neuron" when DDP_PG_BACKEND isn't set. There
# is deliberately NO built-in default list: the TorchNeuron process-group backend name has not been
# verified on hardware, and silently picking a guessed name would turn a config error into a hang
# or a confusing c10d failure. Only names the running torch reports as available are accepted.
PG_BACKEND_CANDIDATES_ENV = "DDP_PG_BACKEND_CANDIDATES"


def resolve_pg_backend(
    device: str,
    pg_backend: str | None = None,
    candidates: Sequence[str] | None = None,
    is_available: Callable[[str], bool] | None = None,
) -> str:
    """Pick the ``torch.distributed`` process-group backend for ``device``. Pure (no torch import).

    * ``"xla"`` -> ``"xla"`` (the default, documented path; unchanged).
    * ``"cpu"`` -> ``pg_backend`` if given, else ``"gloo"``.
    * ``"neuron"`` -> ``pg_backend`` if given (honored verbatim). Otherwise the first of
      ``candidates`` that ``is_available`` accepts; with no explicit name and no available
      candidate, raise -- the name must be discovered on hardware, never guessed.
    """
    if device == "xla":
        if pg_backend and pg_backend != "xla":
            raise ValueError(
                f"DDP_PG_BACKEND={pg_backend!r} conflicts with device='xla' (which always uses "
                "the 'xla' backend). Set DDP_DEVICE=cpu|neuron for the native path."
            )
        return "xla"
    if device not in NATIVE_DEVICES:
        raise ValueError(
            f"Unknown DDP device {device!r}; expected 'xla' (default), 'neuron' (beta), or 'cpu'."
        )
    if pg_backend:
        return pg_backend
    if device == "cpu":
        return "gloo"
    # device == "neuron" with no explicit name: probe candidates, never guess.
    tried = [c.strip() for c in (candidates or []) if c and c.strip()]
    if is_available is not None:
        for name in tried:
            if is_available(name):
                return name
    raise ValueError(
        "DDP_DEVICE=neuron needs an explicit process-group backend: set DDP_PG_BACKEND=<name>. "
        "The TorchNeuron (native backend, beta) collective backend name is UNVERIFIED in this "
        "tutorial and is not guessed. Discover it on a Trn2/Trn3 instance -- "
        "validation/torchneuron_facts.py records the process-group backends torch reports there -- "
        f"or list names to probe in {PG_BACKEND_CANDIDATES_ENV} (comma-separated; only names "
        "torch.distributed reports as available are used)."
        + (f" Probed and unavailable: {tried}." if tried else "")
    )


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
    """The per-process training function (one process per core). Returns rank-0 metrics.

    ``cfg.device == "xla"`` (default) runs the PyTorch/XLA path; ``"cpu"``/``"neuron"`` run the
    opt-in native-PyTorch DDP path.
    """
    if cfg.device == "xla":
        # Reject a conflicting DDP_PG_BACKEND early; the XLA path itself is unchanged.
        resolve_pg_backend("xla", cfg.pg_backend)
        return _train_ddp_xla(cfg)
    return _train_ddp_native(cfg)


def _train_ddp_xla(cfg: DDPConfig) -> dict[str, float]:
    """PyTorch/XLA DDP (the default, documented path)."""
    import time

    import torch
    import torch.distributed as dist
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(cfg.seed)
    device = xm.xla_device()
    # XLA is the collective backend for Neuron. torchrun sets RANK/WORLD_SIZE in the environment.
    dist.init_process_group(backend="xla")
    # torch-xla 2.x: rank/world come from torch_xla.runtime (the older xm.get_ordinal /
    # xm.xrt_world_size were removed). dist.get_rank()/get_world_size() also work post-init.
    rank = xr.global_ordinal()
    world_size = xr.world_size()
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
    xm.rendezvous(
        "train_start"
    )  # align all cores before timing so wall-clock is comparable
    wall_start = time.time()
    steps_per_core = 0
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
            steps_per_core += 1
        if rank == 0:
            print(
                f"   epoch {epoch + 1}/{cfg.epochs}  rank0 avg_loss={(running / max(1, n)).item():.4f}"
            )
    xm.mark_step()
    train_wall = time.time() - wall_start
    # Aggregate throughput = samples processed across ALL cores / wall-clock. Each core did
    # steps_per_core steps of per_core_batch_size samples; world_size cores ran concurrently.
    total_samples = steps_per_core * cfg.per_core_batch_size * world_size

    # Evaluate + checkpoint on rank 0 only.
    metrics: dict[str, float] = {}
    if rank == 0:
        metrics = _evaluate(model, eval_loader, device, "xla", label_names)
        metrics["world_size"] = float(world_size)
        metrics["train_wall_s"] = round(train_wall, 2)
        metrics["train_throughput_samples_s"] = (
            round(total_samples / train_wall, 2) if train_wall > 0 else 0.0
        )
        # Neuron-correct checkpointing: move XLA tensors to CPU, then torch.save.
        os.makedirs("/tmp/ddp_ner_ckpt", exist_ok=True)
        cpu_state = xm._maybe_convert_to_cpu(model.state_dict())
        torch.save(cpu_state, "/tmp/ddp_ner_ckpt/model.pt")
        print(
            f"✅ DDP done. eval_f1={metrics.get('eval_f1', float('nan')):.4f} on {world_size} cores; "
            f"throughput={metrics['train_throughput_samples_s']} samples/s; checkpoint saved."
        )

    xm.rendezvous("train_ddp_finished")
    return metrics


def _params_checksum(model) -> float:
    """Sum of all parameters in float64 -- identical across ranks iff DDP kept replicas in sync."""
    import torch

    with torch.no_grad():
        return float(
            sum(p.detach().to("cpu", torch.float64).sum() for p in model.parameters())
        )


def _native_device(device: str, local_rank: int):
    """The per-process torch.device for the native path."""
    import torch

    if device == "cpu":
        return torch.device("cpu")
    # device == "neuron" (BETA, UNVERIFIED on hardware): one process per NeuronCore, indexed by
    # LOCAL_RANK. If NEURON_RT_VISIBLE_CORES scopes each process to a single core, that process
    # sees only index 0 -- so fall back to the bare "neuron" device when the index is out of range.
    neuron = getattr(torch, "neuron", None)
    if neuron is None or not neuron.is_available():
        raise RuntimeError(
            "DDP_DEVICE=neuron but TorchNeuron isn't available (torch.neuron missing or no "
            "devices). It's a beta backend for Trn2/Trn3; use the default XLA path otherwise."
        )
    count = getattr(neuron, "device_count", lambda: 0)()
    if count and local_rank < count:
        return torch.device("neuron", local_rank)
    return torch.device("neuron")


def _train_ddp_native(cfg: DDPConfig) -> dict[str, float]:
    """Native PyTorch DDP: cpu (gloo, laptop) or TorchNeuron (beta). Opt-in only."""
    import time

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from examples.use_cases._device import wrap_model

    candidates = os.environ.get(PG_BACKEND_CANDIDATES_ENV, "").split(",")
    pg_backend = resolve_pg_backend(
        cfg.device,
        cfg.pg_backend,
        candidates=candidates,
        is_available=getattr(dist, "is_backend_available", None),
    )

    torch.manual_seed(cfg.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _native_device(cfg.device, local_rank)
    backend = cfg.device  # "cpu" | "neuron" -- the _device/_evaluate backend string
    dist.init_process_group(backend=pg_backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print(
            f"🧬 DDP NER (native) | device={cfg.device} pg_backend={pg_backend} | "
            f"world_size={world_size} | model={cfg.model_name} | attn={cfg.attn_implementation}"
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
    # device_ids is only meaningful for single-device CUDA modules: None for cpu and (unverified)
    # neuron. DDP broadcasts rank 0's weights at construction and all-reduces grads in backward().
    ddp_model = DDP(model, device_ids=None)
    # bf16 autocast (+ optional torch.compile) on neuron; a no-op pair on cpu.
    fwd, amp = wrap_model(ddp_model, backend, compile=cfg.compile)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.learning_rate)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )

    dist.barrier()  # align all ranks before timing (native stand-in for xm.rendezvous)
    wall_start = time.time()
    steps_per_core = 0
    running = torch.zeros((), device=device)
    for epoch in range(cfg.epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)
        running = torch.zeros((), device=device)
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with amp():
                out = fwd(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            out.loss.backward()  # DDP all-reduces (averages) gradients here
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            running += out.loss.detach()
            n += 1
            steps_per_core += 1
        if rank == 0:
            print(
                f"   epoch {epoch + 1}/{cfg.epochs}  rank0 avg_loss={(running / max(1, n)).item():.4f}"
            )
    running.item()  # fence: force queued device work before stopping the clock
    train_wall = time.time() - wall_start
    total_samples = steps_per_core * cfg.per_core_batch_size * world_size

    # Replica-sync check: every rank trained on a different shard, so identical parameter
    # checksums after training mean the gradient all-reduce kept the replicas in lockstep.
    checksum = _params_checksum(model)
    print(f"   rank {rank}: steps={steps_per_core} param_checksum={checksum:.10f}")
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, checksum)

    metrics: dict[str, float] = {}
    if rank == 0:
        tol = 1e-6 * max(1.0, abs(gathered[0]))
        in_sync = all(abs(c - gathered[0]) <= tol for c in gathered)
        print(f"   replicas in sync across {world_size} ranks: {in_sync}")
        # Evaluate the unwrapped module on rank 0 alone (DDP's forward may issue collectives the
        # other ranks wouldn't join).
        eval_fwd, eval_amp = wrap_model(model, backend, compile=cfg.compile)
        metrics = _evaluate(
            model, eval_loader, device, backend, label_names, eval_fwd, eval_amp
        )
        metrics["world_size"] = float(world_size)
        metrics["replicas_in_sync"] = float(in_sync)
        metrics["train_wall_s"] = round(train_wall, 2)
        metrics["train_throughput_samples_s"] = (
            round(total_samples / train_wall, 2) if train_wall > 0 else 0.0
        )
        os.makedirs("/tmp/ddp_ner_ckpt", exist_ok=True)
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, "/tmp/ddp_ner_ckpt/model.pt")
        print(
            f"✅ DDP (native, {cfg.device}/{pg_backend}) done. "
            f"eval_f1={metrics.get('eval_f1', float('nan')):.4f} on {world_size} ranks; "
            f"throughput={metrics['train_throughput_samples_s']} samples/s; checkpoint saved."
        )

    dist.barrier()
    dist.destroy_process_group()
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
            "(--nproc_per_node = number of NeuronCores; see `neuron-ls`).\n"
            "Laptop proof of the native-DDP path (2 CPU processes over gloo):\n"
            "    DDP_DEVICE=cpu DDP_PG_BACKEND=gloo DDP_SMOKE=1 torchrun --nproc_per_node=2 "
            "examples/distributed/data_parallel_ner.py"
        )
        return {}
    return train_ddp(cfg)


def main() -> None:
    """CLI entrypoint (under torchrun).

    A few knobs are read from the environment so a benchmark harness can match a single-core run's
    config without editing code: DDP_EPOCHS, DDP_PER_CORE_BATCH, DDP_MAX_TRAIN, DDP_MAX_EVAL,
    DDP_MAX_LENGTH.

    Native-DDP knobs (opt-in; XLA stays the default): DDP_DEVICE ("xla" | "neuron" | "cpu"),
    DDP_PG_BACKEND (process-group backend; default "gloo" on cpu, required for neuron),
    DDP_PG_BACKEND_CANDIDATES (neuron only: comma-separated names to probe), DDP_COMPILE=1
    (neuron only: torch.compile). DDP_SMOKE=1 shrinks the run like NER_SMOKE (1 epoch, 64
    train/eval samples) and defaults the device to cpu; explicit DDP_* values still win.
    """
    cfg: dict[str, Any] = {}
    if os.environ.get("DDP_SMOKE") not in (None, "", "0"):
        cfg.update(
            {
                "device": "cpu",
                "epochs": 1,
                "max_train_samples": 64,
                "max_eval_samples": 64,
            }
        )
    if os.environ.get("DDP_DEVICE"):
        cfg["device"] = os.environ["DDP_DEVICE"]
    if os.environ.get("DDP_PG_BACKEND"):
        cfg["pg_backend"] = os.environ["DDP_PG_BACKEND"]
    if os.environ.get("DDP_COMPILE"):
        cfg["compile"] = os.environ["DDP_COMPILE"] not in ("", "0")
    _env = {
        "epochs": "DDP_EPOCHS",
        "per_core_batch_size": "DDP_PER_CORE_BATCH",
        "max_train_samples": "DDP_MAX_TRAIN",
        "max_eval_samples": "DDP_MAX_EVAL",
        "max_length": "DDP_MAX_LENGTH",
    }
    for key, envvar in _env.items():
        if os.environ.get(envvar):
            cfg[key] = int(os.environ[envvar])
    run(cfg)


if __name__ == "__main__":
    main()
