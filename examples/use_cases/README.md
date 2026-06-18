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

Each has a companion `.md` with prerequisites, "what you'll learn," run instructions, and an honest
validation record. Most expose a CPU **smoke path** (e.g. `NER_SMOKE=1` / `CV_SMOKE=1`) so you can
prove the code runs before paying for hardware:

```bash
# Laptop smoke test (CPU, tiny subset — proves the code path; near-zero accuracy is expected):
NER_SMOKE=1 python examples/use_cases/biomedical_ner.py

# On a Trainium instance (the real run that reaches the validated metric):
python examples/use_cases/biomedical_ner.py
```

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
