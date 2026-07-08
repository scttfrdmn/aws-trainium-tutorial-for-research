#!/usr/bin/env python3
"""Tensor parallelism vs. one core: a full fine-tune that won't fit, and what TP does about it.

The data-parallel sibling (`data_parallel_ner.py`) *replicates* the whole model on every NeuronCore —
which only works if the model already fits on one core. This example is the case where it does
**not**: a **full** (non-LoRA) fine-tune of a ~1–2B LLM, whose weights + AdamW optimizer state
exceed a single NeuronCore's HBM. Run it two ways and watch what happens.

**The measured lesson (validated on a real trn1.2xlarge — see the companion README):**
tensor parallelism is *necessary but, on the smallest box, not sufficient* for a full fine-tune.

  * **One core** — always out-of-memories. The compiler/runtime reports a hard **16.00 GB per-core
    HBM ceiling**; a full 1–2B fine-tune needs more. (Qwen3-1.7B: 17.87 GB at compile time.
    Llama-3.2-1B: exhausts HBM at runtime.)
  * **TP=2** — *splits* the model across both cores, which is real progress: Llama-3.2-1B actually
    trains a couple of steps this way. But full fine-tuning is genuinely **marginal** on 2 cores —
    Qwen3-1.7B is still too big to compile (19.59 GB/core), and Llama-3.2-1B runs a few steps then
    tips over the 16 GB ceiling (15.958 GB — 32 MB short).

So the real-world takeaways, with numbers behind them:
  * Full fine-tuning a 1–2B model does **not** fit on one NeuronCore — TP (or LoRA) is mandatory.
  * TP=2 halves the *weight* footprint but the smallest box stays tight; that's exactly why the
    [Qwen3 example](../use_cases/qwen3_lora_finetune.py) uses **LoRA** on trn1.2xlarge and reserves
    **full** fine-tuning for trn1.32xlarge / Trn2 (more cores → more aggregate HBM and a
    data-parallel dimension for ZeRO-1 to shard optimizer state).

Data parallel vs. tensor parallel, in one line each:
  * **Data parallel** — every core holds a full copy of the model; cores split the *data* and
    all-reduce gradients. Needs the model to fit on one core. (See `data_parallel_ner.py`.)
  * **Tensor parallel** — one model is *sharded* across cores (each holds a slice of every big
    matmul); the cores cooperate on a single forward/backward. The only single-instance option once
    the model is too big for one core — but not a free pass past the per-core HBM ceiling.

The memory arithmetic (full fine-tune, AdamW, bf16):
    per parameter = 2 (bf16 weight) + 4 (fp32 master) + 4 (Adam m) + 4 (Adam v) = 14 bytes
    a 1.2–1.7B model  ->  ~17–24 GiB  of weights + optimizer state (before activations).
Each NeuronCore of a trn1.2xlarge exposes a hard **16.00 GB** HBM ceiling (reported verbatim by
neuronx-cc). TP=2 shards weights across the two cores, but activations + the fp32/Adam optimizer
state in the update step keep the peak near — and, for these models, over — that ceiling.

> **HARDWARE-ONLY.** Like the Qwen3 example, this needs optimum-neuron + the Neuron runtime; there is
> no CPU path. Off-Neuron (or un-launched) it prints the launch commands and exits cleanly.

## Install (use the pinned [training] extra — this matters)

    # On a Neuron DLAMI's NxD-training venv. Installing trl/peft UNPINNED breaks optimum-neuron's SFT
    # trainer; the [training] extra pins the versions it needs (see the Qwen3 example's note).
    pip install "optimum-neuron[training]"

## Run it BOTH ways (the contrast IS the lesson)

    export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
    export NEURON_FUSE_SOFTMAX=1

    # 1 core — EXPECTED TO OOM (this is the lesson, not a bug):
    NEURON_RT_NUM_CORES=1 torchrun --nproc_per_node=1 \
        examples/distributed/tensor_parallel_full_finetune.py --tensor_parallel_size 1

    # TP=2 — shards the model across both cores (gets further; still tight on the smallest box):
    torchrun --nproc_per_node=2 \
        examples/distributed/tensor_parallel_full_finetune.py --tensor_parallel_size 2

Reuses the optimum-neuron path and dataset from the Qwen3 LoRA example
(`NeuronModelForCausalLM.from_pretrained(model_id, trn_config)` + `NeuronSFTTrainer`), dropping LoRA
(`peft_config=None`) so the *full* model is trainable — which is what makes it not fit.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

# Both are import-safe on CPU (their heavy imports are lazy inside the functions), so importing them
# here does not break the registry's import test.
from examples.use_cases.qwen3_lora_finetune import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_MODEL_ID,
    _neuron_available,
    build_dataset,
)

# Substrings that indicate the Neuron runtime / compiler ran out of device (HBM) memory. The exact
# wording is version-dependent, so we match defensively on the lowercased error text. The
# `ncc_eoom001` / `peak hbm usage` / `hbm limit` markers are the exact ones neuronx-cc 2.25 raised on
# hardware for this example (a compile-time HBM-ceiling error wrapped in a ValueError).
_OOM_MARKERS = (
    "ncc_eoom001",  # neuronx-cc compile-time HBM-ceiling error code (observed on trn1)
    "peak hbm usage",
    "hbm limit",
    "not enough neuron memory",  # NRT runtime allocation failure (observed on trn1)
    "nrt_resource",
    "allocbuffer",
    "out of memory",
    "oom",
    "allocation failure",
    "not enough hbm",
    "failed to allocate",
    "resource_exhausted",
    "nrt_tensor_allocate",
    "insufficient device memory",
    "exceeds",
)


@dataclass
class TPConfig:
    """Config for the tensor-parallel full fine-tune.

    ``tensor_parallel_size`` is the knob the whole example turns on: **1** reproduces the
    single-core OOM (the lesson); **2** shards the model across both cores of a trn1.2xlarge so it
    fits. The run is deliberately *bounded* by ``max_steps`` — the point is "does it fit and train?",
    not convergence.
    """

    model_id: str = DEFAULT_MODEL_ID
    dataset_id: str = DEFAULT_DATASET
    output_dir: str = "./tp-fullft-out"
    tensor_parallel_size: int = 2  # 1 => expected-OOM demo; 2 => fits on trn1.2xlarge
    epochs: int = 1
    max_steps: int = 20  # bounded: prove it trains, don't chase convergence
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5  # full-FT LR (lower than LoRA's 8e-4)
    max_length: int = 1024  # margin knob: shorter seq => smaller activations
    gradient_checkpointing: bool = False  # margin knob: trade compute for activation memory
    zero_1: bool = True  # ZeRO-1 shards optimizer state further (orthogonal to TP)
    logging_steps: int = 1  # log every step so a short run still reports a real loss
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> TPConfig:
        """Build from a plain dict, ignoring unknown keys."""
        cfg = dict(cfg or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})


def _is_probable_oom(exc: Exception) -> bool:
    """Heuristic: did this exception come from running out of device (HBM) memory?"""
    text = str(exc).lower()
    return any(marker in text for marker in _OOM_MARKERS)


def _read_hbm_gib_per_core() -> float | None:
    """Best-effort per-core HBM usage in GiB, or None if the runtime doesn't expose it.

    Availability of ``get_memory_info`` varies by torch-xla version, so this is guarded — when it's
    missing, use ``neuron-monitor`` externally to read HBM (see the README).
    """
    try:
        import torch_xla.core.xla_model as xm

        info = xm.get_memory_info(xm.xla_device())
        used = info.get("bytes_used") or info.get("kb_used", 0) * 1024
        return round(used / (1024**3), 2) if used else None
    except Exception:
        return None


def train(cfg: TPConfig) -> dict[str, float]:
    """Full fine-tune under tensor parallelism. Returns rank-0 metrics.

    Structure: build the trainer, then load the model + run a bounded number of steps inside a
    try/except. A full fine-tune of a 1–2B model is expected to hit the 16 GB/core HBM ceiling on
    this hardware — on **one core** always, and on **TP=2** it's marginal (see the module docstring).
    We catch a *memory* failure at any TP, explain what it means, and return a structured "didn't fit"
    result instead of a raw stack trace. Any **non-OOM** exception is re-raised, so a genuine bug
    never hides behind the OOM handler.
    """
    import torch
    import torch_xla.runtime as xr
    from optimum.neuron import (
        NeuronSFTConfig,
        NeuronSFTTrainer,
        NeuronTrainingArguments,
    )
    from optimum.neuron.models.training import NeuronModelForCausalLM
    from transformers import AutoTokenizer

    torch.manual_seed(cfg.seed)
    rank0 = xr.global_ordinal() == 0
    tp = cfg.tensor_parallel_size
    if rank0:
        approx_gib = round(_param_billions(cfg.model_id) * 14, 1)
        print(
            f"🧩 TP full fine-tune | tensor_parallel_size={tp} | model={cfg.model_id} | FULL FT (no LoRA)\n"
            f"   full-FT AdamW needs ~14 B/param => ~{approx_gib} GiB weights+optimizer; "
            f"each NeuronCore has a hard 16.00 GB HBM ceiling. TP shards this across {tp} core(s)."
        )

    training_args = NeuronTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        tensor_parallel_size=tp,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        bf16=True,  # Trainium is bf16-native (PSUM accumulates FP32)
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        zero_1=cfg.zero_1,
        lr_scheduler_type="constant",
        overwrite_output_dir=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    dataset = build_dataset(tokenizer, cfg.dataset_id)

    def formatting_func(examples):
        return tokenizer.apply_chat_template(
            examples["messages"], tokenize=False, add_generation_prompt=False
        )

    try:
        # The tensor-parallel loader: trn_config (derived from training_args) shards every big matmul
        # across `tensor_parallel_size` cores. peft_config=None => the FULL model is trainable.
        model = NeuronModelForCausalLM.from_pretrained(
            cfg.model_id,
            training_args.trn_config,
            torch_dtype=torch.bfloat16,
        )
        sft_config = NeuronSFTConfig(
            max_length=cfg.max_length,
            packing=True,
            **training_args.to_dict(),
        )
        trainer = NeuronSFTTrainer(
            model=model,
            args=sft_config,
            peft_config=None,  # <-- full fine-tune (this is what forces the OOM on one core)
            train_dataset=dataset,
            processing_class=tokenizer,
            formatting_func=formatting_func,
        )
        result = trainer.train()
    except Exception as exc:  # noqa: BLE001 — intentionally broad; re-raised unless it's a memory (OOM) error
        if _is_probable_oom(exc):
            if rank0:
                if tp == 1:
                    print(
                        f"\n📉 THIS IS THE LESSON — a full fine-tune of {cfg.model_id} does NOT fit "
                        "on ONE NeuronCore.\n"
                        "   Full-FT AdamW needs ~14 B/param, over the hard 16.00 GB per-core HBM "
                        "ceiling. Shard the model with tensor parallelism across both cores:\n"
                        "     torchrun --nproc_per_node=2 "
                        "examples/distributed/tensor_parallel_full_finetune.py --tensor_parallel_size 2"
                    )
                else:
                    print(
                        f"\n📉 THE HONEST NUANCE — TP={tp} SHARDS the model (real progress: it gets "
                        "further than one core), but a full fine-tune is still MARGINAL on the "
                        "smallest box.\n"
                        "   The weight-update step's fp32 master + Adam moments push the peak over "
                        "the 16.00 GB/core ceiling. With only 2 cores there's no data-parallel "
                        "dimension for ZeRO-1 to shard optimizer state.\n"
                        "   This is exactly why full fine-tuning belongs on trn1.32xlarge / Trn2 "
                        "(more cores → more aggregate HBM), and why LoRA is the trn1.2xlarge play "
                        "(see qwen3_lora_finetune.py). Shrinking --max_length trims activations but "
                        "not the optimizer-state peak."
                    )
                print(f"   (caught: {type(exc).__name__}: {str(exc)[:200]})")
            return {
                "oom": 1.0,
                "tensor_parallel_size": float(tp),
                "fit": 0.0,
            }
        raise  # a genuine, non-memory error — surface it loudly

    metrics = _summarize(result, trainer, cfg)
    if rank0:
        hbm = _read_hbm_gib_per_core()
        if hbm is not None:
            metrics["hbm_gib_per_core"] = hbm
        print(
            f"✅ TP={tp} full fine-tune FIT and trained. "
            f"final_loss={metrics.get('train_loss', float('nan')):.4f} over "
            f"{int(metrics.get('logged_steps', 0))} logged steps"
            + (f"; ~{hbm} GiB/core HBM" if hbm is not None else "")
        )
    return metrics


def _param_billions(model_id: str) -> float:
    """Rough parameter count (billions) parsed from the model id, for the memory heads-up print.

    Falls back to 1.7 (the default Qwen3-1.7B) when the id has no recognizable size token.
    """
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    return float(m.group(1)) if m else 1.7


def _summarize(result, trainer, cfg: TPConfig) -> dict[str, float]:
    """Metrics dict that never reports a misleading ``nan`` loss (same reasoning as the Qwen3 example).

    On the XLA/Neuron path only ``logging_steps`` losses are materialized, so ``result.training_loss``
    is polluted by ``nan`` from un-synced steps. We derive the reported loss from the real logged
    values in ``trainer.state.log_history`` instead.
    """

    def _is_real(x) -> bool:
        return isinstance(x, (int, float)) and x == x  # x==x is False only for nan

    logged = [
        h["loss"]
        for h in getattr(trainer.state, "log_history", [])
        if _is_real(h.get("loss"))
    ]
    metrics: dict[str, float] = {
        "tensor_parallel_size": float(cfg.tensor_parallel_size),
        "fit": 1.0,
    }
    if logged:
        metrics["train_loss"] = float(logged[-1])
        metrics["mean_logged_loss"] = float(sum(logged) / len(logged))
        metrics["logged_steps"] = float(len(logged))
    else:
        # Too short to log a real loss — report the "-1.0 = not measured" sentinel, never bare nan.
        metrics["train_loss"] = -1.0
        metrics["mean_logged_loss"] = -1.0
        metrics["logged_steps"] = 0.0
    return metrics


def _parser() -> argparse.ArgumentParser:
    """CLI. ``--tensor_parallel_size 1`` reproduces the single-core OOM; ``2`` shards across cores."""
    p = argparse.ArgumentParser(
        description="Tensor-parallel FULL fine-tune on Trainium (1 core OOMs; TP=2 shards but stays tight)."
    )
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p


def run(config: dict | None = None) -> dict[str, float]:
    """Harness entrypoint. Hardware-only; explains the two-way launch when off-Neuron or un-launched."""
    hint = (
        "This is a HARDWARE-ONLY tensor-parallel example (needs optimum-neuron on a Trainium "
        "instance). Run it BOTH ways to see the lesson:\n"
        "    # 1 core — EXPECTED TO OOM (the lesson):\n"
        "    NEURON_RT_NUM_CORES=1 torchrun --nproc_per_node=1 "
        "examples/distributed/tensor_parallel_full_finetune.py --tensor_parallel_size 1\n"
        "    # TP=2 — shards the model across both cores (gets further; still tight on this box):\n"
        "    torchrun --nproc_per_node=2 "
        "examples/distributed/tensor_parallel_full_finetune.py --tensor_parallel_size 2"
    )
    if not _neuron_available():
        print(hint)
        return {}
    cfg = TPConfig.from_dict(config)
    if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
        print(hint)
        return {}
    return train(cfg)


def main() -> None:
    """CLI entrypoint (under torchrun, on a Neuron instance).

    A few knobs are read from the environment so a benchmark harness can adjust the run without
    editing code: TP_TENSOR_PARALLEL_SIZE, TP_MAX_STEPS, TP_MAX_LENGTH, TP_GRAD_CKPT, TP_MODEL_ID.
    """
    if not _neuron_available():
        run({})
        return
    args = _parser().parse_args()
    cfg: dict[str, Any] = {
        "model_id": os.environ.get("TP_MODEL_ID", args.model_id),
        "tensor_parallel_size": int(
            os.environ.get("TP_TENSOR_PARALLEL_SIZE", args.tensor_parallel_size)
        ),
        "max_steps": int(os.environ.get("TP_MAX_STEPS", args.max_steps)),
        "max_length": int(os.environ.get("TP_MAX_LENGTH", args.max_length)),
        "gradient_checkpointing": os.environ.get("TP_GRAD_CKPT", "") not in ("", "0")
        or args.gradient_checkpointing,
    }
    run(cfg)


if __name__ == "__main__":
    main()
