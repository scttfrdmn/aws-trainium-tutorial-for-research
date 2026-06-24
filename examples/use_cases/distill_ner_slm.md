# Distill a NER teacher into a Small Language Model (SLM)

> **Assumed knowledge:** you've run the [biomedical NER example](biomedical_ner.py) and understand
> fine-tuning + the bf16/`eager`-attention lesson; basic familiarity with softmax/cross-entropy.
> **What you'll get:** the standard knowledge-distillation workflow on Trainium — train a small
> student from a larger teacher's *soft labels* — and a real outcome: student F1 vs teacher F1, the
> compression ratio, and why distillation is "the shape the hardware wants."

The other use-cases *use* models. This one **makes a small one**. It takes the fine-tuned BERT-base
disease-NER teacher from [`biomedical_ner.py`](biomedical_ner.py) and distills it into a ~4-layer,
narrow student that keeps most of the entity-F1 at a fraction of the parameters — the technique
behind DistilBERT, TinyBERT, and the modern SLM wave.

## Why distillation is a strong Trainium workload

| Property | This example |
|---|---|
| Dense matmul, fixed shapes | teacher forward + student forward/backward are transformer matmuls at a fixed `max_length` + `drop_last` → compiles once |
| bf16-stable | inherits `attn_implementation="eager"` from the teacher (HF v5 SDPA → `nan` on Neuron bf16) |
| Fills the array | unlike a small CNN, a transformer student keeps the 128×128 systolic array busy (see the [utilization spike](cv_utilization_spike.py)) |

## The distillation loss

```
loss = α · CE(student_logits, hard_labels)          # ordinary supervised term
     + (1−α) · T² · KL(student_softmax/T ‖ teacher_softmax/T)   # soft-label matching
```

- The **teacher runs in `torch.no_grad()`** — it only provides soft labels; no backward pass.
- **Temperature `T`** softens both distributions so the student learns the teacher's *relative*
  class confidences (the "dark knowledge"), not just the argmax. Multiplying the KL by `T²` keeps
  its gradient scale comparable to the CE term (Hinton et al., 2015).
- For token classification the KL is computed over **real tokens only** (`labels != -100` masks
  subword continuations and padding).

## Run it

```bash
# Laptop smoke test (CPU, tiny subset — proves the code path; near-zero F1 expected):
DISTILL_SMOKE=1 python examples/use_cases/distill_ner_slm.py

# On a Trainium instance (real run — fine-tunes the teacher, then distills):
python examples/use_cases/distill_ner_slm.py
```

The teacher is fine-tuned inline so the example is self-contained; point `teacher_ckpt` at a saved
teacher to skip that and distill directly.

## Prerequisites

- A Neuron instance (`trn1.2xlarge`) or any CPU box for the smoke run.
- `pip install transformers datasets` (the Neuron stack comes from the DLAMI).

## What "good" looks like

A successful run reports the student keeping a high fraction of the teacher's F1 (**`f1_retention`**)
at a real **`compression_ratio`** (~25× fewer params for the smoke-sized student; tune
`student_layers`/`student_hidden` for your size/accuracy trade-off). That retention-vs-size number is
distillation's actual selling point — small *and* accurate, not just fast.

## Now try this with your own task

Swap the teacher (`teacher_name` / `teacher_ckpt`) and dataset for your own fine-tuned model and
corpus. The distillation loop is task-agnostic; only the data pipeline (reused here from the NER
example) is NER-specific.

## Status

✅ **Hardware-validated on trn1.2xlarge** (Neuron 2.30, torch 2.9.1): teacher eval_f1 **0.8023**,
distilled student eval_f1 **0.5732** — **71.4% of the teacher's F1 at 3.8× fewer parameters**.

Two lessons surfaced *because* we ran it on hardware (and are now baked into the code):
- **A from-scratch student under-learns** in a short run (we measured student_f1≈0.24). Real
  distillation initializes the student from **pretrained** weights (`prajjwal1/bert-small`); that
  alone lifted it to 0.57. The from-scratch baseline is still available via
  `student_from_pretrained=None`.
- **Don't select real tokens with boolean mask indexing** in the loss — `logits[labels != -100]`
  has a data-dependent shape that **recompiles the XLA graph every step** (we watched the compile
  count climb). The loss now computes a dense masked KL (static shape, compiles once).
