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

## Notes

- **Fixed shapes still matter.** `drop_last=True` on the sampler *and* loaders keeps every core's
  batch shape identical, so the graph compiles ~once (see best-practices §1).
- **Eager attention still matters.** bf16 + HF-default SDPA still `nan`s here; the example keeps
  `attn_implementation="eager"`.

## ✅ Hardware-validated

Run on a real **trn1.2xlarge** (2 NeuronCores, Neuron 2.30 / torch-neuronx 2.9) via
`torchrun --nproc_per_node=2`:

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
