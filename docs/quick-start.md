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

**Versions:** this tutorial targets **Neuron SDK 2.33** (latest) with **PyTorch 2.9** on the
**PyTorch/XLA** path (`torch-neuronx 2.9.0.2.15`, unchanged since 2.31; runs on Trn1 and Trn2), and
**Python 3.12** via **uv** (the one supported version, pinned in `.python-version`). AWS's new native
backend, **TorchNeuron**, is in public beta on Trn2/Trn3 only; see
[Trying TorchNeuron](main_tutorial_doc.md#trying-torchneuron).

## Installation

```bash
# Clone the repository
git clone https://github.com/scttfrdmn/aws-trainium-tutorial-for-research
cd aws-trainium-tutorial-for-research

# Install with uv (this repo standardizes on uv + Python 3.12)
make install-dev

# Neuron SDK (run this ON a Neuron instance / DLAMI — the AWS Deep Learning AMI with Neuron
# preinstalled — not your laptop)
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

<a id="using-trn2"></a>
### Using Trn2 (optional)

**Start on `trn1.2xlarge`.** It's cheap, usually available, and runs the PyTorch/XLA path this
tutorial teaches. On Trainium2 the examples use the same XLA code, unchanged:
[`VALIDATED.md`](../VALIDATED.md) shows 4 of the 6 single-device examples passing on `trn2.3xlarge`
and which 2 didn't. What to know before you try Trn2:

- **Sizes:** `trn2.3xlarge` (1 Trainium2 chip, 4 logical NeuronCores at LNC=2, 96 GB HBM, 12 vCPU)
  and `trn2.48xlarge` (16 chips).
- **Regions are sparse.** For this project's account, `trn2.48xlarge` was offered only in
  `us-east-2` and `trn2.3xlarge` only in `sa-east-1` and `ap-southeast-4`; `us-east-1` offered no
  Trn2. Check what your account sees before planning.
- **Spot and on-demand may not get you an instance.** In Sept–Oct 2026, every spot and on-demand Trn2
  request we made, in every offered AZ, returned `InsufficientInstanceCapacity`. That includes a
  spot-retry watcher that ran for 12 hours. Spot prices for `trn2.3xlarge` were *observed* at about
  $6–8/hr when listed, but the capacity wasn't there. Check
  [current pricing](https://aws.amazon.com/ec2/pricing/).
- **EC2 Capacity Blocks for ML worked.** You buy a block ahead of time for a fixed window. For
  example, a ~14 h `trn2.3xlarge` block in `ap-southeast-4` cost **$32.07 prepaid
  (non-refundable)**, and a `trn2.48xlarge` block in `us-east-2` for the same window was quoted at
  about $514. **AWS starts reclaiming the instance before the block ends** (we saw it ~30 min
  early), so finish your work and copy results off well before the end time.

To run the validation harness inside a block, pass its reservation ID and AZ:

```bash
python -m validation.run_on_hardware --instance trn2.3xlarge --region ap-southeast-4 \
    --az <block-az> --capacity-reservation-id cr-... --launcher awscli --bootstrap \
    --example ner_biomedical          # dry run; add --yes to launch
```

See `python -m validation.run_on_hardware --help` for the other flags. TorchNeuron, AWS's native
backend, runs only on Trn2/Trn3 and is still **beta**, so this tutorial stays on XLA; see
[Trying TorchNeuron](main_tutorial_doc.md#trying-torchneuron) if you want to experiment.

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
