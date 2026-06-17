# Biomedical NER on Trainium — fine-tuning a disease tagger

**Status:** see [`/VALIDATED.md`](../../VALIDATED.md) for the hardware-validation record of this example.

## What you'll build

A transformer that reads biomedical text and tags **disease mentions** — the kind of
information-extraction model used in literature mining, cohort discovery, and pharmacovigilance.
You fine-tune `bert-base-cased` on the **NCBI-disease** corpus (a standard, peer-reviewed NER
benchmark) on a single Trainium chip, and evaluate it with entity-level **seqeval** F1.

This is deliberately a *real* task on *real* data — not a synthetic demo. If you swap in your own
IOB2-tagged corpus, the same script trains your model.

## Learning objectives

1. Run a genuine PyTorch fine-tune on Trainium via the **PyTorch/XLA** path (`xm.xla_device()`,
   `xm.optimizer_step`, `xm.mark_step`) — the supported path on Neuron SDK 2.30 / PyTorch 2.9.
2. Handle the one genuinely tricky part of NER: **subword↔label alignment** (why `-100` labels
   exist and where they come from).
3. Evaluate honestly with entity-level precision/recall/F1 (not token accuracy, which lies on NER).
4. See the **first-step compilation cost** firsthand (the run reports `first_step_compile_s`) — the
   phenomenon the [Compilation chapter](../../docs/main_tutorial_doc.md#compilation) explains.

## Prerequisites

- A Neuron instance (`trn1.2xlarge` is plenty) or any CPU box for a smoke run.
- `pip install transformers datasets seqeval evaluate` (the Neuron stack comes from the DLAMI).

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
  ~misleadingly high; seqeval's entity F1 is the metric that matters.
- **eval vs test F1.** We tune nothing on test; it's reported once as an unbiased estimate.

## Now try this with your own data

Replace the dataset with any IOB2 token-classification corpus exposing `tokens` and `ner_tags`
(e.g. your own annotations loaded via `datasets`). The label set is read from the dataset schema,
so multi-entity corpora (genes, chemicals, drugs) work without code changes.

## Hardware verification log (real trn1.2xlarge, us-east-2, Neuron 2.30.10 / torch-neuronx 2.9 / transformers 5.12.1)

What has been **proven on hardware** so far:

| Question | Result | Evidence |
|---|---|---|
| Does HF-default SDPA attention work on the Neuron bf16 path? | **No — `nan` loss at step 0** | diagnostic matrix |
| Does `attn_implementation="eager"` fix it (keeping bf16)? | **Yes — loss 1.13** (matches CPU fp32 ~1.21) | diagnostic matrix |
| Do fixed batch shapes (`drop_last`) tame compilation? | **Yes — ~2 graphs** vs **7+** for ragged | run logs |

What is **not yet captured**: a full multi-epoch run to a green `eval_f1 >= 0.75` artifact. On a
single NeuronCore with eager attention the per-epoch wall-clock is long, and the larger-batch graph
is slow to compile; reaching the F1 threshold artifact is tracked in
[issue #5](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/issues/5). Until that
artifact exists, `/VALIDATED.md` correctly shows this example as not-yet-validated.

## Honesty note

This README makes **no accuracy claim** beyond what's in the verification log above. Final numbers
live in `/VALIDATED.md`, written by the harness from a full run. The bug-and-fix findings, however,
are real and reproduced on hardware.
