# Generate crystal-structure CIFs from a composition (a CrystaLLM-style SLM)

> **Assumed knowledge:** you've run an earlier example (so the XLA training path + the bf16/`eager`
> attention lesson are familiar) and know what an autoregressive language model does. No
> crystallography background needed — a CIF is just structured text.
> **What you'll get:** pretrain a **Small Language Model from scratch** on Trainium that writes a
> crystal structure (CIF) given a chemical composition — the generative counterpart to the encoder
> fine-tunes elsewhere in this repo — plus the honest LM metric (perplexity) and a sampled structure.

A crystal structure can be written as a **CIF** (crystallographic information file) — plain text. So,
as **CrystaLLM** (Antunes, Butler & Grau-Crespo, *Nature Communications* 2024) showed, an
autoregressive language model can learn to *generate* structures: prompt it with a composition
(`Na Cl`) and let it write the CIF. Sampling the model becomes a fast generative prior for materials
discovery. This example trains a compact **character-level GPT** on the public
`yhjollin/CrystaLLM_use_Chemeleon_data_1` corpus (36k composition→CIF pairs).

## Why it fits Trainium well

| Property | This example |
|---|---|
| Dense decoder matmul | causal self-attention + MLP blocks — the systolic array's home turf |
| Static shapes | every document padded/cropped to a fixed `block_size` + `drop_last` → compiles once |
| bf16-stable | hand-written attention with an **fp32 softmax** + causal mask (no `scaled_dot_product_attention`, which `nan`s in bf16 on Neuron — same lesson as the NER/ViT examples) |
| From-scratch pretraining | the generative counterpart to the repo's encoder fine-tunes |

## How it's framed

Each training document is `"<formula>\n>>>\n<CIF>"`, so the model learns the mapping
**composition → structure**. Character-level tokenization (a vocab built from the training text)
keeps the example dependency-free and matches CrystaLLM's structural-text tokenization.

## Run it

```bash
# Laptop smoke test (CPU, tiny subset — proves the code path; output will be gibberish):
CRYSTAL_SMOKE=1 python examples/use_cases/crystal_cif_slm.py

# On a Trainium instance (real run — one pass over 36k CIFs):
python examples/use_cases/crystal_cif_slm.py
```

At the end it prints a **sampled CIF** for a held-out composition so you can eyeball what the model
writes.

## ⚠️ Scope: what the metric does and doesn't tell you

v1 reports **validation cross-entropy / perplexity** (the honest LM metric) and prints a generated
sample. It does **not** score structural *validity* — whether the CIF parses, charges balance, or
bond lengths are physical. That needs a domain validator (e.g. `pymatgen`) and is a deliberate
follow-up. **A low perplexity means the model learned CIF syntax and statistics, not that every
sample is a synthesizable crystal.** Treat the generated structures as candidates to validate
downstream, exactly as you would CrystaLLM's raw samples.

## Prerequisites

- A Neuron instance (`trn1.2xlarge`) or any CPU box for the smoke run.
- `pip install transformers datasets` (the Neuron stack comes from the DLAMI; the GPT is plain
  `torch.nn`).

## Now try this

- Scale `n_layer` / `n_embd` / `block_size` up on a `trn1.32xlarge` for a stronger model.
- Swap the corpus for a larger CIF dump (Materials Project, OQMD) to broaden chemical coverage.
- Add a `pymatgen`-based validity check on samples to turn perplexity into a structure-validity rate
  (the natural v2).

## Status

✅ **Hardware-validated on trn1.2xlarge** (Neuron 2.30, torch 2.9.1; 6L/384d char GPT, 8000 CIFs,
1 epoch): **val_perplexity = 1.735** (low single-digit — the LM learned CIF syntax/statistics well)
and it generated a CIF sample from a held-out composition. *Reminder: this gates LM quality, not
structural validity — see the scope note above.*

A bf16 lesson surfaced on hardware (now fixed): masking future positions with `float("-inf")`
produced `loss=nan` at step ~25 (the `0 * -inf = nan` softmax-backward trap). The causal mask uses a
large **finite** negative (`-1e9`) instead — same family of bf16-attention gotcha as the NER SDPA→nan
lesson.
