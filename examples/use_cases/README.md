# Real-World Use Cases

Research-domain examples for AWS Trainium (PyTorch 2.9 / XLA path). These favor **real datasets and
real tasks** over synthetic demos. Earlier synthetic-data examples (a `np.random` "genomics" demo
and a synthetic "computer vision" demo) were **removed** — they taught the wrong thing on real
hardware. Each remaining example is meant to be runnable and honest about what it does.

## 🧬 Available Use Cases

### [Biomedical NER](biomedical_ner.py) — ⭐ hardware-validated
**Research Domain**: Biomedical NLP / literature mining

Fine-tunes a transformer token classifier to extract **disease mentions** from biomedical abstracts,
using the real **NCBI-disease** corpus. This is the reference example for the whole repo: it runs on
the PyTorch/XLA Trainium path, scores entity-level F1, and is **validated on real `trn1.2xlarge`**
(see [`/VALIDATED.md`](../../VALIDATED.md) and [`biomedical_ner.md`](biomedical_ner.md)).

```bash
# Laptop smoke test (CPU, tiny subset — proves the code path):
NER_SMOKE=1 python examples/use_cases/biomedical_ner.py

# On a Trainium instance (real run):
python examples/use_cases/biomedical_ner.py
```

### [Financial Modeling](financial_modeling.py)
**Research Domain**: Quantitative Finance

Portfolio optimization and risk modeling on **real market data from Yahoo Finance** (`yfinance`),
with a synthetic fallback only when the network/data is unavailable.

```bash
python examples/use_cases/financial_modeling.py --portfolio tech_portfolio --simulations 50000
```

> **Status:** Not yet hardware-validated through the harness. Treat its cost/throughput statements
> as estimates until it carries a `validation/results/` artifact.

## 🎯 Common patterns

- **FinOps:** spot instances + auto-termination + cost tracking (see `scripts/` and the
  [validation harness](../../validation/README.md)).
- **Real data:** prefer real public datasets (RODA, Hugging Face, Yahoo Finance) over synthetic.
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
