# Qwen3 LoRA fine-tune on Trainium

> **Assumed knowledge:** you've fine-tuned (or understand) an LLM with LoRA/PEFT, and you've read
> the [best-practices chapter](../../docs/trainium_development_best_practices.md).
> **What you'll get:** the real, supported way to LoRA-fine-tune a current-gen open LLM on Trainium
> via Hugging Face **optimum-neuron** — the most common 2026 researcher workflow.

This mirrors the public optimum-neuron Qwen3 example
([code](https://github.com/huggingface/optimum-neuron/blob/main/examples/training/qwen3/finetune_qwen3.py) ·
[tutorial](https://huggingface.co/docs/optimum-neuron/training_tutorials/finetune_qwen3)), adapted to
this repo's conventions. The stack is the documented, supported one: `NeuronModelForCausalLM` +
`NeuronSFTTrainer` + PEFT `LoraConfig`.

## ⚠️ Hardware-only — no CPU smoke path

Unlike the [NER example](biomedical_ner.py), this **cannot run on CPU**: `NeuronModelForCausalLM`,
the tensor-parallel `trn_config`, and `NeuronSFTTrainer` are Trainium-specific. Off a Neuron box the
script prints launch instructions and exits.

## Install (use the pinned `[training]` extra)

```bash
# In the DLAMI's NxD-training venv. The [training] extra pins the exact trl/peft the SFT
# trainer needs:
pip install "optimum-neuron[training]"
```

> **Gotcha (verified on hardware):** installing `trl`/`peft` *unpinned* gives newer versions and
> breaks with `ImportError: cannot import name 'clone_chat_template' from 'trl.models'`.
> `optimum-neuron 0.4.3` needs `trl==0.24.0`, `peft==0.17.0`, `transformers~=4.57` — which the
> `[training]` extra enforces.

## Run it (on a Neuron instance, via torchrun)

```bash
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX=1
neuron-ls    # how many NeuronCores?

# trn1.2xlarge (2 cores) — small Qwen3 fits:
torchrun --nproc_per_node=2 examples/use_cases/qwen3_lora_finetune.py \
    --model_id Qwen/Qwen3-1.7B --tensor_parallel_size 2

# trn1.32xlarge (32 cores) — larger model:
torchrun --nproc_per_node=32 examples/use_cases/qwen3_lora_finetune.py \
    --model_id Qwen/Qwen3-8B --tensor_parallel_size 8
```

## What it demonstrates

| Concept | How |
|---|---|
| Supported LLM-training path | `optimum-neuron` `NeuronSFTTrainer` (not a hand-rolled loop) |
| Tensor parallelism | `--tensor_parallel_size` + `trn_config` shard the model across cores |
| Parameter-efficient FT | PEFT `LoraConfig` passed via `peft_config=` |
| bf16-native | `bf16=True` — Trainium accumulates FP32 in PSUM (best-practices §4) |
| Optimizer-state sharding | `zero_1=True` |
| SFT data | chat-formatted instruction dataset via `formatting_func` + chat template |

## Sizing (be realistic on one chip)

LoRA + a **~1.7B** model is the practical ceiling on `trn1.2xlarge` (32 GiB). 7–8B needs more cores
/ memory — `trn1.32xlarge` (32 cores) or Trn2. See the
[choose-your-path guide](../../docs/choose_your_path.md).

## Status

**Not yet hardware-validated** through this repo's harness — the harness doesn't yet orchestrate a
torchrun multi-process launch (tracked). It's built to the public, supported API; validate by
launching torchrun manually and recording `train_loss`. Treat it as a correct pattern until a
provenance artifact exists.
