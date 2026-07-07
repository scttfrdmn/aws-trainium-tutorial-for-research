# Distributed training across NeuronCores

> **Assumed knowledge:** you've run the single-core [NER example](../use_cases/biomedical_ner.py) and
> read the [best-practices chapter](../../docs/trainium_development_best_practices.md).
> **What you'll get:** a real **data-parallel** fine-tune sharded across the NeuronCores of one
> instance — the thing single-core training leaves on the table.

A `trn1.2xlarge` has **2 NeuronCores**; a `trn1.32xlarge` has **32**. Training on one core wastes the
rest. [`data_parallel_ner.py`](data_parallel_ner.py) replicates the model per core and averages
gradients each step (PyTorch/XLA distributed data parallel), using the same task and Trainium-native
rules as the validated single-core example.

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
