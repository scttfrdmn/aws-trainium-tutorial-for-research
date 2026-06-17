# Video Tutorial 3: Real-World Application — Biomedical NER on Trainium

**Duration**: ~20 minutes
**Difficulty**: Intermediate
**Prerequisites**: Tutorials 1 & 2; basic PyTorch + transformers

> **Why this example?** It's the repo's **hardware-validated** reference workflow — a real
> fine-tune of a transformer on the real **NCBI-disease** corpus, proven on a real `trn1.2xlarge`
> (see [`/VALIDATED.md`](../../VALIDATED.md)). An earlier synthetic "genomics" demo was removed
> because it generated DNA with `np.random` — it looked real but taught nothing. This one is real.

## 🎯 Learning objectives

By the end, viewers can:
1. Fine-tune a token-classification model (disease NER) on Trainium via the PyTorch/XLA path.
2. Handle the genuinely fiddly part of NER — subword↔label alignment (`-100` masking).
3. Evaluate honestly with **entity-level** precision/recall/F1 (not token accuracy).
4. Recognize the two Trainium gotchas this example surfaces: the **bf16 SDPA→`nan`** trap (fixed
   with `attn_implementation="eager"`) and the **compile storm** from variable shapes (fixed with
   `drop_last` / static shapes).
5. Capture a provenance artifact through the validation harness.

## 📋 Tutorial flow

### Opening (0:00–2:00)
Frame the task: extracting disease mentions from biomedical abstracts is a real literature-mining
problem. Show [`examples/use_cases/biomedical_ner.py`](../../examples/use_cases/biomedical_ner.py)
and its companion [`biomedical_ner.md`](../../examples/use_cases/biomedical_ner.md).

### Smoke test on CPU first (2:00–5:00)
```bash
NER_SMOKE=1 python examples/use_cases/biomedical_ner.py
```
Explain: the smoke run proves the code path on a laptop before spending a cent on hardware. F1 is
~0 here (64 samples, 1 epoch) — that's expected; we're testing plumbing, not accuracy.

### The Trainium-native concerns (5:00–12:00)
Walk through the parts that matter on real hardware (all in the example + the
[best-practices chapter](../trainium_development_best_practices.md)):
- **`attn_implementation="eager"`** — why HF's default SDPA produces `nan` in bf16 on Neuron, and
  why eager (not `--auto-cast=none`) is the right fix. *Build in the form the hardware wants.*
- **Static shapes** — `drop_last=True` + fixed `max_length` so the graph compiles ~once, not
  per-shape.
- **`xm.mark_step()`** and not calling `.item()` every step.

### Run on Trainium (12:00–17:00)
```bash
python examples/use_cases/biomedical_ner.py        # on a trn1.2xlarge
# or via the harness, which captures provenance:
python -m validation.run_on_hardware --instance trn1.2xlarge --region us-east-2 \
    --example ner_biomedical --yes
```
Point out the first-step compile time in the output (the AOT cost) and the steady-state throughput.

### Read the result honestly (17:00–20:00)
Show the captured artifact and `/VALIDATED.md`: `eval_f1 ≈ 0.85` on the validation split, with the
exact instance + SDK + commit recorded. Emphasize: the number is meaningful **because** it's tied
to a real run, not typed into a slide.

## 🎬 Production notes
See [`video_production_guide.md`](video_production_guide.md). Record the real run; if the compile
wait is long on camera, cut to a warm-cache run and say so.

## Now try this with your own data
Any IOB2 token-classification dataset with `tokens` + `ner_tags` works (genes, chemicals, drugs).
Swap the dataset name; the label set is read from the dataset schema.
