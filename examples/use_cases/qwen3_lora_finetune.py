#!/usr/bin/env python3
"""LoRA fine-tune of Qwen3 on AWS Trainium (the headline 2026 LLM workflow).

This is the most common real-world Trainium task in 2026: parameter-efficient (LoRA) supervised
fine-tuning of a current-gen open LLM. It uses Hugging Face **optimum-neuron**, which is AWS +
HF's supported, documented path for LLM training on Trainium — `NeuronModelForCausalLM` +
`NeuronSFTTrainer` + PEFT `LoraConfig`.

It closely follows the public optimum-neuron Qwen3 example and tutorial:
  * Example: https://github.com/huggingface/optimum-neuron/blob/main/examples/training/qwen3/finetune_qwen3.py
  * Tutorial: https://huggingface.co/docs/optimum-neuron/training_tutorials/finetune_qwen3
(Apache-2.0; adapted here to this repo's run(config) harness contract and honesty conventions.)

> **HARDWARE-ONLY (unlike the NER example).** There is **no CPU smoke path** here: optimum-neuron's
> `NeuronModelForCausalLM`, the `trn_config` tensor-parallel loader, and `NeuronSFTTrainer` are
> Trainium-specific and import the Neuron runtime. On a non-Neuron box this script explains how to
> launch and exits cleanly — it does not pretend to run.

## Launch (torchrun — one process per NeuronCore)

    # Recommended env (from the public example):
    export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
    export NEURON_FUSE_SOFTMAX=1

    # trn1.2xlarge (2 NeuronCores) — use a small Qwen3 that fits:
    torchrun --nproc_per_node=2 examples/use_cases/qwen3_lora_finetune.py \
        --model_id Qwen/Qwen3-1.7B --tensor_parallel_size 2

    # trn1.32xlarge (32 cores) — larger model:
    torchrun --nproc_per_node=32 examples/use_cases/qwen3_lora_finetune.py \
        --model_id Qwen/Qwen3-8B --tensor_parallel_size 8

`--nproc_per_node` = number of NeuronCores (see `neuron-ls`). On a single chip,
**LoRA + a ~1.7B model is the realistic ceiling** for trn1.2xlarge (32 GiB); 7-8B needs more cores
/ memory (e.g. trn1.32xlarge or Trn2).

## Status

⚠️ **Not yet hardware-validated** through this repo's harness (the harness doesn't yet orchestrate
a torchrun multi-process launch). It's built to the public, supported API — treat it as a correct,
runnable pattern until a provenance artifact exists. See the [best-practices chapter](../../docs/trainium_development_best_practices.md).
"""

from __future__ import annotations

import argparse
import os

# Defaults sized for a single small instance. The model id is overridable so the same script serves
# trn1.2xlarge (Qwen3-1.7B) and trn1.32xlarge (Qwen3-8B), matching the public example's two sizes.
DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_DATASET = (
    "tengomucho/simple_recipes"  # the dataset used by the public Qwen3 tutorial
)


def _neuron_available() -> bool:
    """True only if the optimum-neuron training stack is importable (i.e. on a Neuron box)."""
    from importlib.util import find_spec

    return find_spec("optimum") is not None and find_spec("torch_neuronx") is not None


def build_dataset(tokenizer, dataset_id: str):
    """Load + chat-format the SFT dataset (mirrors the public Qwen3 tutorial)."""
    from datasets import load_dataset

    raw = load_dataset(dataset_id, split="train")
    eos = tokenizer.eos_token

    def to_chat(examples):
        chats = []
        for recipe, name in zip(examples["recipes"], examples["names"], strict=False):
            chats.append(
                [
                    {"role": "user", "content": f"How can I make {name}?"},
                    {"role": "assistant", "content": recipe + eos},
                ]
            )
        return {"messages": chats}

    return raw.map(to_chat, batched=True, remove_columns=raw.column_names)


def train(args) -> dict[str, float]:
    """Run the LoRA SFT on Trainium. Returns a small metrics dict (rank 0)."""
    import torch
    from datasets import load_dataset  # noqa: F401  (load happens in build_dataset)
    from optimum.neuron import (
        NeuronSFTConfig,
        NeuronSFTTrainer,
        NeuronTrainingArguments,
    )
    from optimum.neuron.models.training import NeuronModelForCausalLM
    from peft import LoraConfig
    from transformers import AutoTokenizer

    training_args = NeuronTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        tensor_parallel_size=args.tensor_parallel_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,  # Trainium is bf16-native (PSUM accumulates FP32); see best-practices §4
        logging_steps=args.logging_steps,
        zero_1=True,  # ZeRO-1: shard optimizer state across cores
        lr_scheduler_type="cosine",
        overwrite_output_dir=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dataset = build_dataset(tokenizer, args.dataset_id)

    # Custom-architecture models (Qwen3/Llama/Granite) load via NeuronModelForCausalLM with the
    # tensor-parallel trn_config derived from the training args.
    model = NeuronModelForCausalLM.from_pretrained(
        args.model_id,
        training_args.trn_config,
        torch_dtype=torch.bfloat16,
    )

    # Standard PEFT LoRA config — passed to the trainer, which calls get_peft_model internally.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = NeuronSFTConfig(
        max_length=args.max_length,
        packing=True,
        **training_args.to_dict(),
    )

    def formatting_func(examples):
        return tokenizer.apply_chat_template(
            examples["messages"], tokenize=False, add_generation_prompt=False
        )

    trainer = NeuronSFTTrainer(
        model=model,
        args=sft_config,
        peft_config=lora_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    result = trainer.train()
    trainer.save_model(args.output_dir)  # saves the LoRA adapter

    # train() returns a TrainOutput with training_loss; expose it as the harness metric.
    metrics = {"train_loss": float(getattr(result, "training_loss", float("nan")))}
    print(
        f"✅ Qwen3 LoRA SFT done. train_loss={metrics['train_loss']:.4f}; adapter -> {args.output_dir}"
    )
    return metrics


def _parser() -> argparse.ArgumentParser:
    """CLI mirroring the public example's flags."""
    p = argparse.ArgumentParser(
        description="Qwen3 LoRA SFT on Trainium (optimum-neuron)."
    )
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--dataset_id", default=DEFAULT_DATASET)
    p.add_argument("--output_dir", default="./qwen3-lora-out")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=8e-4)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--logging_steps", type=int, default=2)
    return p


def run(config: dict | None = None) -> dict[str, float]:
    """Harness entrypoint. Hardware-only; explains the launch when run off-Neuron or un-launched."""
    if not _neuron_available():
        print(
            "Qwen3 LoRA fine-tune is a HARDWARE-ONLY example (needs optimum-neuron on a Trainium "
            "instance — no CPU path).\nLaunch on a Neuron instance with torchrun:\n"
            "    torchrun --nproc_per_node=2 examples/use_cases/qwen3_lora_finetune.py "
            "--model_id Qwen/Qwen3-1.7B --tensor_parallel_size 2"
        )
        return {}
    # Build args from defaults, overlaid with any harness-provided config keys.
    args = _parser().parse_args([])
    for k, v in (config or {}).items():
        if hasattr(args, k):
            setattr(args, k, v)
    if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
        print(
            "Launch with torchrun (one process per NeuronCore); see the module docstring."
        )
        return {}
    return train(args)


def main() -> None:
    """CLI entrypoint (under torchrun, on a Neuron instance)."""
    if not _neuron_available():
        run({})
        return
    train(_parser().parse_args())


if __name__ == "__main__":
    main()
