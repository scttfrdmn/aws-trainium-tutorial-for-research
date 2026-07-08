# Distributed training across NeuronCores

> **Assumed knowledge:** you've run the single-core [NER example](../use_cases/biomedical_ner.py) and
> read the [best-practices chapter](../../docs/trainium_development_best_practices.md).
> **What you'll get:** a real **data-parallel** fine-tune sharded across the NeuronCores of one
> instance — the thing single-core training leaves on the table.

A `trn1.2xlarge` has **2 NeuronCores**; a `trn1.32xlarge` has **32**. Training on one core wastes the
rest. There are two ways to use the others, and this directory has an example of each:

- **Data parallel** — [`data_parallel_ner.py`](data_parallel_ner.py): *replicate* the model on every
  core, split the data, all-reduce gradients. Speeds up training when the model already fits on one
  core. (First half of this README.)
- **Tensor parallel** — [`tensor_parallel_full_finetune.py`](tensor_parallel_full_finetune.py):
  *split* one model across cores — the only single-instance option when the model is **too big for
  one core** (and an honest, measured look at how far that gets you on 2 cores).
  (["Tensor parallelism"](#tensor-parallelism-when-the-model-doesnt-fit-on-one-core) section below.)

The data-parallel example replicates the model per core and averages gradients each step (PyTorch/XLA
distributed data parallel), using the same task and Trainium-native rules as the validated
single-core NER example.

## Run it

```bash
neuron-ls                                  # how many cores do I have?
torchrun --nproc_per_node=2  examples/distributed/data_parallel_ner.py   # trn1.2xlarge
torchrun --nproc_per_node=32 examples/distributed/data_parallel_ner.py   # trn1.32xlarge
```

`--nproc_per_node` is the number of NeuronCores (one process per core) — this is the AWS-recommended
launcher for torch-neuronx/XLA training.

## What it demonstrates (beyond single-core)

| Concept | How |
|---|---|
| **XLA collective backend** | `dist.init_process_group(backend="xla")` |
| **Data sharding** | `DistributedSampler(num_replicas, rank, drop_last=True)` — disjoint shard per core, fixed shapes |
| **Gradient all-reduce** | `xm.optimizer_step(optimizer)` applies grads *and* averages them across cores in one call |
| **Overlap + mark_step** | `pl.MpDeviceLoader` |
| **Neuron-correct checkpointing** | rank 0 only: `xm._maybe_convert_to_cpu(state_dict)` → `torch.save` (don't `torch.save` raw XLA tensors) |

## Does the second core pay off? (measured)

The point of DDP is throughput: two cores should train faster than one. Here's the **same job**
(NCBI-disease, `bert-base-cased`, 1600 train samples, batch 16/core, 2 epochs, `max_length=64`) run
single-core (`NEURON_RT_NUM_CORES=1`) vs. 2-core (`torchrun --nproc_per_node=2`) on one
**trn1.2xlarge**, sharing one S3 compile cache:

| Run | Cores | Train throughput | `eval_f1` |
|---|---|---|---|
| Single-core | 1 | **90.1 samples/s** | 0.809 |
| Data-parallel (DDP) | 2 | **119.8 samples/s** | 0.774 |

**~1.33× faster on 2 cores — not 2×, and that's the honest lesson.** Two things eat the ideal
speedup at this scale: (1) each core now sees only 800 samples/epoch, so per-step fixed overhead and
the extra **gradient all-reduce** are a larger slice of a shorter run; (2) rank-0 eval is serial. DDP
scaling improves as the per-core batch and step count grow (it pays off far more on `trn1.32xlarge`
with 32 cores and a real dataset). The small `eval_f1` dip is expected too: sharding halves each
core's data per epoch while gradient averaging acts like a larger effective batch with fewer updates
per shard — tune epochs/LR if you need to close it.

> **Measure warm, not cold.** These are **warm-cache** numbers — the *first* run of each pays an
> ahead-of-time compile (the single-core cold run above spent **201 s of its 844 s** wall-clock on
> the first-step compile alone). Comparing cold wall-clocks would measure the compiler, not the
> cores. With the [S3 compile cache](../../docs/trainium_development_best_practices.md) warm, the
> single-core run drops from **844 s → 36 s**. Always benchmark the second run.

## Notes

- **Fixed shapes still matter.** `drop_last=True` on the sampler *and* loaders keeps every core's
  batch shape identical, so the graph compiles ~once (see best-practices §1).
- **Eager attention still matters.** bf16 + HF-default SDPA still `nan`s here; the example keeps
  `attn_implementation="eager"`.

## ✅ Hardware-validated

Run on a real **trn1.2xlarge** (2 NeuronCores, Neuron 2.30 / torch-neuronx 2.9) via
`torchrun --nproc_per_node=2` on the **full** NCBI-disease train set (the throughput table above
instead caps to 1600 samples for a fair single-vs-2-core A/B, so its `eval_f1` is lower):

| Metric | Value |
|---|---|
| `world_size` | 2 cores (gradient all-reduce working) |
| epoch 1 → 2 loss | 0.179 → 0.032 (clean convergence, no nan) |
| `eval_f1` | **0.826** (P 0.793 / R 0.862) |
| checkpoint | saved via `xm._maybe_convert_to_cpu` + `torch.save` (rank 0) |

> Note: it's validated by a **manual torchrun launch**, not the single-process harness (which
> doesn't orchestrate multi-process runs yet) — so it doesn't appear in `VALIDATED.md`'s auto-table.
> The captured result above is the proof. One API fix was needed and made along the way:
> torch-xla 2.x replaced `xm.get_ordinal()`/`xm.xrt_world_size()` with
> `torch_xla.runtime.global_ordinal()`/`world_size()`.

---

# Tensor parallelism: when the model doesn't fit on one core

> **Assumed knowledge:** you've run the data-parallel example above and the
> [Qwen3 LoRA fine-tune](../use_cases/qwen3_lora_finetune.py).
> **What you'll get:** the case data parallelism *can't* solve — a **full** fine-tune that
> **out-of-memories on one NeuronCore** — and an honest, measured picture of how far tensor
> parallelism gets you on the smallest box (further, but not all the way).

[`tensor_parallel_full_finetune.py`](tensor_parallel_full_finetune.py) fine-tunes **all** of a ~1–2B
LLM (no LoRA) on one `trn1.2xlarge`. It's the sharpest demonstration of the difference between the two
kinds of parallelism — and, validated on real hardware, a more honest lesson than "TP makes it fit":
**TP is *necessary but, on 2 cores, not sufficient* for a full fine-tune.**

## Why TP, not DP?

Data parallelism *replicates* the whole model on each core — so it only helps if the model already
fits on one core. When it doesn't, DP can't help: there's nowhere to put the copy. Tensor parallelism
*splits* every large matmul across cores, so each core holds only a **slice** of the model. That's
the only way to attempt a model bigger than one core on a single instance.

The [Qwen3 LoRA example](../use_cases/qwen3_lora_finetune.py) already passes `tensor_parallel_size=2`,
but LoRA's trainable footprint is tiny — the model fits on one core anyway, so TP looks optional.
Full fine-tuning the *same* model makes TP **mandatory**. That contrast is the lesson.

## The memory arithmetic

Full fine-tune with AdamW, bf16 weights:

| Per parameter | Bytes |
|---|---|
| bf16 weight | 2 |
| fp32 master weight | 4 |
| Adam `m` | 4 |
| Adam `v` | 4 |
| **Total** | **14 B/param** (before gradients + activations) |

A 1.2–1.7B model → **~17–24 GiB** of weights + optimizer state. Each NeuronCore of a `trn1.2xlarge`
has a hard **16.00 GB** HBM ceiling (reported verbatim by `neuronx-cc`). One core can't hold it. TP=2
shards the *weights* across both cores — but activations plus the fp32-master + Adam-moment optimizer
state in the weight-update step keep the peak near, and for these models over, that 16 GB ceiling.

## Run it BOTH ways (the contrast is the lesson)

```bash
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX=1

# 1 core — EXPECTED TO OOM (this is the lesson, not a bug):
NEURON_RT_NUM_CORES=1 torchrun --nproc_per_node=1 \
    examples/distributed/tensor_parallel_full_finetune.py --tensor_parallel_size 1

# TP=2 — shards the model across both cores (gets further; still tight on this box):
torchrun --nproc_per_node=2 \
    examples/distributed/tensor_parallel_full_finetune.py --tensor_parallel_size 2
```

Both runs catch the Neuron out-of-memory error and print the lesson (they do **not** crash with a raw
stack trace) — with a message tailored to the single-core vs. TP=2 case.

## What it demonstrates (beyond data parallel)

| Concept | How |
|---|---|
| **Model *sharding* (not replication)** | `NeuronModelForCausalLM.from_pretrained(model_id, training_args.trn_config)` splits each big matmul across cores |
| **TP is the knob** | `NeuronTrainingArguments(tensor_parallel_size=…)` — `1` vs `2` cores |
| **Full FT is the load-bearing footprint** | `NeuronSFTTrainer(..., peft_config=None)` — the whole model is trainable (LoRA would hide the OOM) |
| **The 16 GB/core ceiling is real** | both a compile-time (`NCC_EOOM001`) and a runtime (`NRT_RESOURCE`) OOM are caught and explained |
| **TP ≠ unlimited memory** | 2 cores shard weights but not enough to fit full-FT optimizer state — the honest nuance |

## ✅ Hardware-validated (manual torchrun launch)

Measured on a real **trn1.2xlarge** (2 NeuronCores, Neuron 2.30 / torch-neuronx 2.9 / optimum-neuron
0.4.3). Each core has a hard **16.00 GB** HBM ceiling. The circle-closing comparison — **same model,
same 2 cores, only `peft_config` differs:**

| Qwen3-1.7B, TP=2 on trn1.2xlarge | Trainable params | Result |
|---|---|---|
| **Full fine-tune** (this example, `peft_config=None`) | 1,720,574,976 (100%) | ❌ OOM — **19.59 GB/core** > 16 GB ceiling |
| **LoRA** ([`qwen3_lora_finetune.py`](../use_cases/qwen3_lora_finetune.py), `peft_config=LoraConfig(...)`) | 66,060,288 (~3.8%) | ✅ **fits and trains** on the same 2 cores |

And the full fine-tune across two models and both core counts (this example, `max_length` 512, AdamW):

| Full FT | 1 core (`tp=1`) | TP=2 (`--nproc_per_node=2`) |
|---|---|---|
| **Qwen3-1.7B** | ❌ OOM — **17.87 GB** needed at compile (`NCC_EOOM001`) | ❌ OOM — **19.59 GB/core** at compile (too big to lay out) |
| **Llama-3.2-1B** | ❌ OOM — exhausts HBM at runtime (`NRT_RESOURCE`) | ⚠️ **trains steps 1–2**, then OOMs on step 3 at **15.958 GB** (32 MB over) |

**Reading of the result.** One core never fits a full 1–2B fine-tune. TP=2 is real progress — it
shards the model far enough that Llama-3.2-1B actually *runs training steps* — but full fine-tuning is
genuinely marginal on 2 cores: the weight-update step's fp32-master + Adam moments tip it over the
ceiling. With only 2 cores there's no data-parallel dimension for ZeRO-1 to shard optimizer state.

**Squaring the circle — full FT vs LoRA on the *same* model and box.** The top table is the punchline:
on one `trn1.2xlarge`, TP=2, Qwen3-1.7B, the only thing that changes is `peft_config`. Full FT trains
all 1.72B params and OOMs; LoRA trains a **66M-param adapter (~3.8%)** and fits comfortably — same
model, same 2 cores. That's *why* the [Qwen3 example](../use_cases/qwen3_lora_finetune.py) uses LoRA
here, and why **full** fine-tuning belongs on `trn1.32xlarge` / Trn2 (32 cores → far more aggregate
HBM *and* a data-parallel dimension for optimizer-state sharding). This example makes the "just use
LoRA / get a bigger box" advice concrete with numbers.

## Notes

- **bf16 + eager rules still apply** — same Trainium-native constraints as the other examples.
- **Margin knobs (and their limits).** `--max_length` shrinks *activations*, but the full-FT peak here
  is driven by *optimizer state* in the update step, so shorter sequences don't rescue it.
  `--gradient_checkpointing` is exposed but trips an optimum-neuron API incompatibility on this stack
  (`Unexpected keyword arguments: use_cache,reduction`) — left off by default. The honest fix is more
  cores, not a knob.
- **ZeRO-1 is orthogonal to TP.** TP shards the *model*; ZeRO-1 shards the *optimizer state* across
  data-parallel replicas — of which there are none at TP=2 on a 2-core box. This is the crux of why
  TP alone doesn't close the gap here.
- Like the DDP and Qwen3 examples, this is validated by a **manual torchrun launch**, not the
  single-process auto-harness, so it appears in `VALIDATED.md`'s manual table rather than the auto one.
