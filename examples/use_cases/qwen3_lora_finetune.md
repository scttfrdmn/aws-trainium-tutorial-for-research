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

> **Data source — why Hugging Face, not RODA?** The AWS Registry of Open Data has no LLM
> instruction-tuning corpus; the SFT dataset lives on Hugging Face. (The
> [satellite example](satellite_landcover.py) uses RODA — that's where open geospatial data lives.)

## Sizing (be realistic on one chip)

LoRA + a **~1.7B** model is the practical ceiling on `trn1.2xlarge` (32 GiB). 7–8B needs more cores
/ memory — `trn1.32xlarge` (32 cores) or Trn2. See the
[choose-your-path guide](../../docs/choose_your_path.md).

## ✅ Run-validated on real hardware

`torchrun --nproc_per_node=2` on a **trn1.2xlarge** (Neuron 2.30 / torch-neuronx 2.9,
`optimum-neuron[training]` 0.4.3, Qwen3-1.7B):

| What | Observed |
|---|---|
| Deps / imports | `NeuronSFTTrainer`, `NeuronSFTConfig`, `NeuronTrainingArguments`, `NeuronModelForCausalLM` all resolve |
| Model load + shard | Qwen3-1.7B loaded, tensor-parallel across 2 cores |
| LoRA | applied — **66,060,288 trainable params** (adapter only, not the full 1.7B) |
| Training | live: `{'loss': 3.52, 'learning_rate': 8e-4, 'grad_norm': 3.39, 'epoch': 0.0}` |

So the end-to-end pipeline is **proven correct** on hardware: it loads, shards, applies LoRA,
compiles, and trains with a real decreasing-loss signal.

**Honest caveat — throughput:** on **2 cores** a step took ~**7 min** (`step_time≈447s`); a full
epoch over the dataset is impractical here. The public example uses **32 cores** (`trn1.32xlarge`)
for a reason. This validates *correctness*; for real training, use more cores / a larger instance.

## ✅ Full epoch validated on trn1.32xlarge (32 NeuronCores)

`torchrun --nproc_per_node=32 ... --model_id Qwen/Qwen3-8B --tensor_parallel_size 8 --epochs 1` on a
**trn1.32xlarge** (Neuron 2.30 / `neuronx-cc 2.25.3371`, torch 2.9.1, optimum-neuron 0.4.3), full
epoch = 50 steps:

| What | Observed |
|---|---|
| Loss | **1.93 → 1.43** over the epoch, monotone, never `nan` (the curve is clean) |
| Steady-state throughput | **~5.0 s/step**, **~13,500 tokens/sec**, **MFU ~29%**, efficiency ~82% |
| Result | exit 0, LoRA adapter saved |

This is 32 cores vs the 2-core box's ~7 min/step — **the whole point of the bigger instance.**

### 🔑 The slow first step is *per-step compilation* — and the cache makes the difference

Two real lessons surfaced, worth teaching explicitly:

1. **The first few steps are dominated by ahead-of-time compilation, not compute.** On a *cold*
   box the per-step bar showed this directly:

   | Step | Per-step time | What's happening |
   |---|---:|---|
   | 1 | ~119 s | compiling (cold) |
   | 2 | ~203 s | compiling a new graph shape |
   | 3 | ~218 s | compiling a new graph shape |
   | **4+** | **~5–6 s** | **warm — steady state** |

   So **step 2-and-on are ~20-40× faster than step 1** once the graphs are cached. Don't panic at a
   crawling first step or two — that's the compiler, and it's a one-time cost *if you persist the cache*.

2. **A warm `NEURON_COMPILE_CACHE_URL` removes the cold phase entirely.** Re-running the same job
   with the cache already populated, training hit **~5 s/step by step ~10 with no multi-minute
   compiles at all** — the whole epoch finished in ~4 min of compute instead of being front-loaded by
   ~10 min of compilation. In the cloud, point the cache at **S3** so a fresh instance reuses these
   graphs (see [best-practices §1b](../../docs/trainium_development_best_practices.md)).

### Gotchas hit (and fixed) on the way — useful to know

- **32 ranks racing the HF hub** corrupted the model download (`Qwen/Qwen3-8B does not appear to have
  a file named model.safetensors`). **Fix:** pre-download once in a single process
  (`huggingface_hub.snapshot_download`) before launching torchrun.
- **`HF_HUB_OFFLINE=1` is not the fix for the race** — it blocks the post-training hub-cache *sync*
  (`OfflineModeIsEnabled` from optimum-neuron's `synchronize_hub_cache`) and crashes *after* a
  successful epoch. With the model already cached, leave the network on; the sync then skips
  gracefully on no-write-access.
- **`TrainOutput.training_loss` comes back `nan`** even on a healthy run: it averages over every step,
  but on XLA the loss only materializes on `logging_steps`, so un-synced steps poison the mean. The
  example now derives `train_loss` from `trainer.state.log_history` (the real logged losses) and
  **never prints a bare `nan`** — if a run is too short to log any loss it says so and suggests lowering
  `--logging_steps`, returning an explicit `-1.0` "not measured" sentinel. Pinned by
  `tests/test_qwen3_loss_summary.py` so participants don't hit the confusing `nan` again.
