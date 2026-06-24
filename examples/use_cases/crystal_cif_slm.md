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

## Scope: two metrics, only one gated

| Metric | What it tells you | Role |
|---|---|---|
| `val_perplexity` (gated via `inv_val_perplexity`) | did the model learn CIF **syntax and statistics** | pass/fail |
| `validity_rate` (**reported**) | fraction of generated CIFs that **parse into a real structure** (pymatgen) | reported only |

The example generates a batch of CIFs from held-out compositions and parses each with **pymatgen**;
`validity_rate` is the fraction that yield a valid structure. It's **reported, not gated** — validity
on a small/short-trained model is noisy, so gating on it would be flaky. **A parseable CIF still
isn't a guarantee of a *synthesizable* crystal** (charge balance, formation energy, etc. are further
checks) — but it's a real, honest step beyond perplexity. If pymatgen isn't installed the run still
works and reports `validity_rate = -1.0` (not measured).

> **Data source — why Hugging Face, not RODA?** The AWS Registry of Open Data has no materials /
> crystal-structure dataset; the CrystaLLM CIF corpus lives on Hugging Face. (The
> [satellite example](satellite_landcover.py) uses RODA — that's where open geospatial data lives.)

## Prerequisites

- A Neuron instance (`trn1.2xlarge`) or any CPU box for the smoke run.
- `pip install transformers datasets` (the Neuron stack comes from the DLAMI; the GPT is plain
  `torch.nn`). Add `pip install pymatgen` to measure the structural `validity_rate` (optional).

## Now try this

- Scale `n_layer` / `n_embd` / `block_size` up on a `trn1.32xlarge` for a stronger model.
- Swap the corpus for a larger CIF dump (Materials Project, OQMD) to broaden chemical coverage.
- Go beyond parse-validity: filter generated structures by charge balance or a formation-energy
  surrogate to rank synthesizability — the natural next step for a discovery pipeline.

## Status

✅ **Hardware-validated on trn1.2xlarge** (Neuron 2.30, torch 2.9.1; 6L/384d char GPT, 8000 CIFs,
1 epoch): **val_perplexity = 1.735** (low single-digit — the LM learned CIF syntax/statistics well)
and it generated a CIF sample from a held-out composition. *Reminder: perplexity gates LM quality, not
structural validity — see the scope note above.*

**Structural `validity_rate` (measured on hardware): 0.0%** of 20 generated CIFs parsed via pymatgen.
That's honest and instructive — a 1-epoch char-GPT on 8k CIFs learns CIF *syntax* well (perplexity
1.7) but not enough to emit fully parseable structures. **This is exactly why validity is reported,
not gated**, and why low perplexity ≠ valid structures: the two metrics measure different things, and
the gap between them is the real lesson. Train longer / on more data (the 36k-CIF full corpus) to lift
the validity rate.

A bf16 lesson surfaced on hardware (now fixed): masking future positions with `float("-inf")`
produced `loss=nan` at step ~25 (the `0 * -inf = nan` softmax-backward trap). The causal mask uses a
large **finite** negative (`-1e9`) instead — same family of bf16-attention gotcha as the NER SDPA→nan
lesson.
