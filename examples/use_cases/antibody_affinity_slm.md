# Antibody binding-affinity prediction with a small protein LM

> **Assumed knowledge:** you've run the [biomedical NER example](biomedical_ner.py) (so the XLA path
> + `eager`-attention lesson are familiar). No protein-biology background needed — a sequence is just
> a string over a ~25-letter amino-acid alphabet.
> **What you'll get:** a real antibody-engineering workflow on Trainium — fine-tune a small **ESM-2**
> protein language model to predict **binding affinity from sequence**, scored by rank correlation —
> and the recognition that protein LMs are "just transformers" that fit Trainium like any NLP model.

Therapeutic-antibody labs screen candidate designs *in silico* before committing to wet-lab assays.
A core primitive of that screening: given an antibody's amino-acid sequence, predict how strongly it
binds its target antigen. This example fine-tunes **`facebook/esm2_t6_8M_UR50D`** (an 8M-param
protein SLM) with a regression head on the public **AbBibench Antibody Binding Benchmark** and reports
**Spearman correlation** — because in screening, *ranking* candidates correctly matters more than
absolute error.

## Why it fits Trainium well

| Property | This example |
|---|---|
| Dense transformer matmul | ESM-2 is an ordinary encoder; affinity head is mean-pool + linear |
| Static shapes | fixed `max_length=160` (antibody variable domains are ~110–130 aa) + `drop_last` → compiles once |
| bf16-stable | `attn_implementation="eager"` — same HF-v5-SDPA-→-`nan` fix as the NER example |
| Small + amortizable | 8M params, many epochs over a few thousand sequences → compile cost amortizes immediately |

## The task, concretely

- **Input:** `heavy_chain_seq` (the dominant binding determinant) from one antigen complex's
  benchmark file (e.g. `2fjg_benchmarking_data.csv`, ~2,200 sequence variants).
- **Target:** the continuous `binding_score`, **z-standardized using train statistics only** (no
  test-set scale leakage).
- **Model:** ESM-2 encoder → mean-pool over real residues → linear head → scalar affinity.
- **Metric:** Spearman (rank) + Pearson correlation and MSE on a held-out 20% split.

## Run it

```bash
# Laptop smoke test (CPU, tiny subset — proves the code path):
ANTIBODY_SMOKE=1 python examples/use_cases/antibody_affinity_slm.py

# On a Trainium instance (real run):
python examples/use_cases/antibody_affinity_slm.py
```

## Prerequisites

- A Neuron instance (`trn1.2xlarge`) or any CPU box for the smoke run.
- `pip install transformers datasets pandas huggingface_hub` (the Neuron stack comes from the DLAMI).

## Now try this with your own data

- Swap `data_file` for a different antigen complex (17 are shipped), or `seq_column` to model the
  **light chain** or a concatenated heavy+light input.
- Bump `model_name` to `facebook/esm2_t12_35M_UR50D` for a stronger (still small) protein LM.
- Set `freeze_encoder=True` to train only the head — a fast baseline (probing the frozen embeddings).
- Point it at your own CSV of `sequence,label` pairs for any sequence→property regression task
  (developability, expression, thermostability, …).

## Status

✅ **Hardware-validated on trn1.2xlarge** (Neuron 2.30, torch 2.9.1, ESM-2 t6 8M, 8 epochs on the
`2fjg` complex): **Spearman = 0.542**, Pearson = 0.638 on the held-out split — a real, useful
ranking signal for candidate triage on a genuinely hard task.
