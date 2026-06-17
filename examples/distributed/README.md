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
- This example is **not yet hardware-validated** through the harness (it needs a multi-core launch
  the harness doesn't yet orchestrate). It's built to the same standards as the validated example;
  validating it is tracked work. Until then, treat it as a correct *pattern*, not a measured result.
