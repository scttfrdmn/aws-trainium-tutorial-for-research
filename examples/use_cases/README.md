# Real-World Use Cases

Research-domain examples for AWS Trainium (PyTorch 2.9 / XLA path). These favor **real datasets and
real tasks** over synthetic demos. Earlier synthetic-data examples (a `np.random` "genomics" demo
and a synthetic "computer vision" demo) were **removed** — they taught the wrong thing on real
hardware. Each remaining example is meant to be runnable and honest about what it does.

## Suggested order

Work through these roughly in order — each builds on the concepts of the last:

| # | Example | Domain | Teaches | Status |
|---|---|---|---|---|
| 1 | [Biomedical NER](biomedical_ner.py) | Biomedical NLP | the reference workflow: real corpus, XLA path, bf16/SDPA→`nan` fix, `drop_last` | ⭐ validated (trn1.2xlarge) |
| 2 | [Satellite land-cover](satellite_landcover.py) | Geospatial / vision | a CNN on Trainium; static-shape image tiles; "runs well" vs "uses the chip well" | ✅ validated (trn1.2xlarge) |
| 3 | [CV utilization spike](cv_utilization_spike.py) | (concept) | *measuring* under-utilization — CNN vs ViT achieved TFLOP/s; "build in the form the hardware wants" | ✅ validated (trn1.2xlarge) |
| 4 | [Distributed NER](../distributed/) | scaling | data-parallel across NeuronCores (torchrun + XLA DDP), gradient all-reduce | ✅ validated |
| 5 | [Qwen3 LoRA fine-tune](qwen3_lora_finetune.py) | LLM fine-tuning | the headline 2026 workflow: optimum-neuron, tensor parallelism, LoRA, the compile-cache lesson | ✅ validated (trn1.32xlarge, full epoch) |

### Building small / specialized models (SLM track)

A second thread: *making* a small, domain-specific model — distillation and from-scratch SLM
pretraining. These are excellent Trainium workloads (dense transformer matmul, fixed shapes) and
show off "the shape the hardware wants" from the [utilization spike](cv_utilization_spike.py).

| # | Example | Domain | Teaches | Status |
|---|---|---|---|---|
| 6 | [Distill NER → SLM](distill_ner_slm.py) | NLP / distillation | knowledge distillation (KL+CE, temperature); student F1 vs teacher at N× compression | ✅ validated (student F1 0.57, 71% retention, 3.8× smaller) |
| 7 | [Antibody affinity SLM](antibody_affinity_slm.py) | Protein / drug discovery | ESM-2 protein LM → binding-affinity regression on real antibody sequences (Spearman) | ✅ validated (Spearman 0.54) |
| 8 | [Crystal-CIF SLM](crystal_cif_slm.py) | Materials science | CrystaLLM-style char GPT generating crystal structures from a composition (perplexity + pymatgen validity) | ✅ validated (perplexity 1.74 → gated `inv_val_perplexity` 0.58) |

Each has a companion `.md` with prerequisites, "what you'll learn," run instructions, and an honest
validation record. Most expose a CPU **smoke path** (e.g. `NER_SMOKE=1` / `CV_SMOKE=1`) so you can
prove the code runs before paying for hardware:

```bash
# Laptop smoke test (CPU, tiny subset — proves the code path; near-zero accuracy is expected):
NER_SMOKE=1 python examples/use_cases/biomedical_ner.py

# On a Trainium instance (the real run that reaches the validated metric):
python examples/use_cases/biomedical_ner.py
```

> **Note — these examples use one NeuronCore.** A `trn1.2xlarge` has **2 NeuronCores**, but the
> examples here run single-core on purpose (simplest, most reproducible; the validation scripts even
> set `NEURON_RT_NUM_CORES=1`). So a default run leaves the second core idle. To put *both* cores to
> work, see the [distributed training examples](../distributed/) — **data parallel**
> (`torchrun --nproc_per_node=2`), which **measures** the payoff (~1.33× throughput on 2 cores for
> this NER job, not a clean 2× — the write-up explains why), and **tensor parallel**, for the case a
> single core *can't* hold the model (a full fine-tune that OOMs on one core; TP=2 shards it across
> both and gets further, with a measured, honest look at why the smallest box stays tight). Leaving
> silicon idle is itself worth noticing: it's the same "use what the hardware gives you" theme as the
> [utilization spike](cv_utilization_spike.py).

> The Qwen3 LoRA example is **hardware-only** (no CPU smoke path) — it needs the Neuron runtime. Its
> companion `.md` explains how to launch it with `torchrun`.

See also: the [train → serve pipeline](../complete_workflow/) (illustrative end-to-end template) and
[enterprise security patterns](../enterprise/) (reference boto3 snippets — read, don't run blindly).

## 🎯 Common patterns

- **FinOps:** spot instances + auto-termination + cost tracking (see `scripts/` and the
  [validation harness](../../validation/README.md)).
- **Real data:** prefer real public datasets (RODA, Hugging Face) over synthetic.
- **Trainium-native:** static shapes, bf16-stable models, `xm.mark_step()` — see the
  [Trainium development best practices](../../docs/trainium_development_best_practices.md).

## ✅ Validation

Performance and cost claims in this repo are only "validated" when backed by a provenance artifact
the [harness](../../validation/README.md) captured on real hardware. `biomedical_ner` is validated;
others are marked accordingly. No hand-typed benchmark tables.

## 🤝 Contributing new use cases

See [CONTRIBUTING.md](../../CONTRIBUTING.md). A new use case should bring: a real-world problem, a
real (or clearly-labeled) dataset, a Trainium-friendly model (static shapes, bf16-stable), a
`run(config) -> dict[str, float]` entrypoint so the harness can validate it, and honest metrics.
