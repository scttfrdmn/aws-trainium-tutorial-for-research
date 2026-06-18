# Quick Start Guide

## Who this is for & what you'll get

**Before you start, you should have:**
- An AWS account with access to Trainium/Inferentia instances (and permission to launch EC2).
- Working Python knowledge and basic familiarity with PyTorch + Hugging Face `transformers`.
- Comfort on the command line. No prior Neuron/Trainium experience required.

**By the end of this quick start, you'll be able to:**
- Set up the repo and cost guardrails.
- Run the validated biomedical-NER example on CPU (smoke) and understand what a real Trainium run does.
- Know where to go next (the full tutorial, the best-practices chapter, the validation harness).

**Versions:** this tutorial targets **Neuron SDK 2.30**, **PyTorch 2.9** on the **PyTorch/XLA** path,
and **Python 3.12** via **uv** (the one supported version, pinned in `.python-version`).

## Installation

```bash
# Clone the repository
git clone https://github.com/scttfrdmn/aws-trainium-tutorial-for-research
cd aws-trainium-tutorial-for-research

# Install with uv (this repo standardizes on uv + Python 3.12)
make install-dev

# Neuron SDK (run this ON a Neuron instance / DLAMI, not your laptop)
make install-neuron
```

## First Steps

### 1. Set up cost controls (do this first)

```bash
python scripts/setup_budget.py --limit 500 --email your-email@university.edu
```

### 2. Configure AWS

```bash
aws configure
python scripts/cost_monitor.py     # sanity-check credentials + spend
```

### 3. Run the reference example on CPU (free)

Before spending anything on hardware, run the **validated** example's smoke path on your laptop —
it proves the code works end to end:

```bash
NER_SMOKE=1 python examples/use_cases/biomedical_ner.py
```

F1 will be near zero (tiny subset, 1 epoch) — that's expected; this step tests plumbing, not
accuracy. See [`examples/use_cases/biomedical_ner.md`](../examples/use_cases/biomedical_ner.md).

### 4. Launch a real, auto-terminating instance (optional, costs money)

```bash
# Auto-terminates after max-hours so a forgotten instance can't run up a bill
python scripts/ephemeral_instance.py --name "test-experiment" --max-hours 2
```

### 5. Validate on real hardware (optional)

The harness launches an instance, runs the example, captures provenance, and self-terminates:

```bash
python -m validation.run_on_hardware --instance trn1.2xlarge --region us-east-2 \
    --example ner_biomedical          # dry run (prints the plan, launches nothing)
# add --yes to actually provision (spot + auto-terminate + cost ceiling)
```

## Cost monitoring

```bash
make monitor-costs        # cost report
make emergency-shutdown   # terminate all ML instances if something runs away
```

## Next steps

1. Read the [complete tutorial](main_tutorial_doc.md).
2. Read [Trainium development best practices](trainium_development_best_practices.md) — the
   compile/bf16/static-shape lessons that save you hours.
3. Explore the [use-case examples](../examples/use_cases/) and the
   [validation harness](../validation/README.md).

## Getting help

- [Neuron tools & debugging](neuron_tools_and_debugging.md) — the symptom→tool table for
  "it nans / won't compile / is slow / lands ops on CPU".
- The runnable [debugging walkthrough](../examples/debugging/) — reproduces the three classic
  failures (bf16 `nan`, recompile storm, silent CPU fallback) with diagnosis + fix.
- Open an issue on [GitHub](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/issues).
