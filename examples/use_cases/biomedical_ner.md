# Biomedical NER on Trainium — fine-tuning a disease tagger

**Status:** ✅ **hardware-validated** on `trn1.2xlarge` (Neuron SDK 2.30.10 / torch-neuronx 2.9),
`eval_f1 ≈ 0.85`. See [`/VALIDATED.md`](../../VALIDATED.md) for the captured provenance record.

## What you'll build

A transformer that reads biomedical text and tags **disease mentions** — the kind of
information-extraction model used in literature mining, cohort discovery, and pharmacovigilance.
You fine-tune `bert-base-cased` on the **NCBI-disease** corpus (a standard, peer-reviewed NER
benchmark) on a single Trainium chip, and evaluate it with **entity-level** precision/recall/F1.

This is deliberately a *real* task on *real* data — not a synthetic demo. If you swap in your own
IOB2-tagged corpus, the same script trains your model.

## Before you start / what you'll be able to do

**Assumed knowledge:** basic PyTorch + Hugging Face `transformers`; what NER is. **No** prior
Trainium/Neuron experience needed.

**After this example you can:**

1. Run a genuine PyTorch fine-tune on Trainium via the **PyTorch/XLA** path (`xm.xla_device()`,
   `xm.optimizer_step`, `xm.mark_step`) — the supported path on Neuron SDK 2.30 / PyTorch 2.9.
2. Handle the one genuinely tricky part of NER: **subword↔label alignment** (why `-100` labels
   exist and where they come from).
3. Evaluate honestly with **entity-level** precision/recall/F1 — computed in-module from IOB2 spans
   (no `seqeval` dependency; its sdist fails to build on Python 3.12) — not token accuracy, which
   lies on NER.
4. Recognize and fix the two Trainium gotchas this example surfaces: the **bf16 SDPA → `nan`** trap
   (fixed with `attn_implementation="eager"`) and the **first-step compile cost** (reported as
   `first_step_compile_s`). See the [best-practices chapter](../../docs/trainium_development_best_practices.md).

## Prerequisites

- A Neuron instance (`trn1.2xlarge` is plenty) or any CPU box for a smoke run.
- `pip install transformers datasets` (the Neuron stack comes from the DLAMI). Entity-level F1 is
  computed in-module — no `seqeval`/`evaluate` needed.

## Run it

```bash
# On a Trainium instance (real run, ~tens of minutes on trn1.2xlarge):
python examples/use_cases/biomedical_ner.py

# Laptop smoke test (CPU, tiny subset — proves the code path, not accuracy):
NER_SMOKE=1 python examples/use_cases/biomedical_ner.py
```

Or through the validation harness (captures provenance + checks the F1 threshold):

```bash
# From your workstation — dry run shows the launch plan, no cost:
python -m validation.run_on_hardware --instance trn1.2xlarge --region us-east-2 --example ner_biomedical
# Add --yes to actually launch (spot + auto-terminate + cost ceiling).
```

## What to observe

- **`first_step_compile_s` ≫ later steps.** The first batch triggers ahead-of-time compilation;
  subsequent steps reuse the compiled graph. Fixed shapes (padding to `max_length`) keep it to one
  compile instead of one-per-step.
- **Entity-level F1, not accuracy.** Most tokens are `O` (outside any entity), so token accuracy is
  misleadingly high; the in-module **entity-span** F1 is the metric that matters.
- **eval vs test F1.** We tune nothing on test; it's reported once as an unbiased estimate.

## Now try this with your own data

Replace the dataset with any IOB2 token-classification corpus exposing `tokens` and `ner_tags`
(e.g. your own annotations loaded via `datasets`). The label set is read from the dataset schema,
so multi-entity corpora (genes, chemicals, drugs) work without code changes.

## Hardware verification log (real trn1.2xlarge, us-east-2, Neuron 2.30.10 / torch-neuronx 2.9 / transformers 5.12.1)

**Validated.** A full run on real hardware passed the gate:

| Metric | Value |
|---|---|
| `eval_f1` (validation split) | **0.846** (gate: ≥ 0.75) |
| `test_f1` | 0.840 |
| `first_step_compile_s` | ~285 s (the AOT compile cost) |
| `train_wall_s` | ~1150 s (3 epochs, single NeuronCore) |

The findings that got it there, each reproduced on hardware:

| Question | Result | Evidence |
|---|---|---|
| Does HF-default SDPA attention work on the Neuron bf16 path? | **No — `nan` loss at step 0** | diagnostic matrix |
| Does `attn_implementation="eager"` fix it (keeping bf16)? | **Yes — loss 1.13** (matches CPU fp32 ~1.21) | diagnostic matrix |
| Do fixed batch shapes (`drop_last`) tame compilation? | **Yes — ~2 graphs** vs **7+** for ragged | run logs |

The captured provenance artifact is `validation/results/examples.use_cases.biomedical_ner.json`,
rendered into [`/VALIDATED.md`](../../VALIDATED.md).

## Honesty note

Every number above is from a captured provenance artifact on the listed instance + SDK — not typed
by hand. Re-run the harness after an SDK bump to refresh it.
