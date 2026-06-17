# Satellite land-cover classification on Trainium

> **Assumed knowledge:** basic PyTorch + CNNs. No Neuron experience needed.
> **What you'll get:** a real geospatial computer-vision workflow on Trainium — and a feel for why
> dense convolutions are a *great* fit for the systolic tensor engine.

Classify **Sentinel-2** satellite tiles into land-cover classes (AnnualCrop, Forest, Residential,
River, SeaLake, ...) with a small CNN on the **EuroSAT** benchmark. This is the tutorial's vision
example — a different hardware story from the transformer/LLM ones.

## Why it RUNS well on Trainium

| Property | This example |
|---|---|
| Static shapes | fixed **64×64×3** tiles + `drop_last` → graph compiles once, no recompile storm |
| bf16-native | plain convs are bf16-stable; the tensor engine accumulates FP32 in PSUM (best-practices §4) |

Contrast with the [NER](biomedical_ner.py) example: there, HF-default SDPA attention had to be
swapped to `eager` to survive bf16. A plain CNN has no such fragile op — it's bf16-stable as-is.

## ⚠️ Is this the *form the hardware wants*? Only partly — and that's the lesson

It **runs** well (static, bf16-stable), but a small CNN does **not** maximize the **128×128 systolic
array**, which is hungry for *large* matmuls that fill its 128 partitions. A small CNN's early convs
are tiny (contraction dim ~3–27 vs a 128-wide array) with a long tail of small conv/BN/pool ops, so
per-core **utilization** is low even though it's correct and fast-to-compile.

A *utilization-optimal* Trainium vision model looks **more ViT-shaped**: patch-embed → large
attention/MLP matmuls, wide channels (≥128 to fill the partition dimension), fused SBUF-resident
blocks (see [novel kernels](../../docs/novel_kernels_on_trainium.md)). To **measure** the gap, read
MFU/utilization in the profiler ([tools & debugging](../../docs/neuron_tools_and_debugging.md)).

So treat this as the **"it works — statically-shaped, bf16-stable"** tier, not as a claim that
small-conv CNNs are the ideal Trainium CV workload.

## Scaling across cores (throughput, a *different* axis)

Launch data-parallel across NeuronCores with torchrun:

```bash
torchrun --nproc_per_node=2 examples/use_cases/satellite_landcover.py   # 2-core data parallel
```

This cuts per-epoch **wall-clock** (more cores chewing through the data) — but it does **not** raise
per-core utilization: each core runs the same underfilled model. "Scaling helps throughput" and
"the model fits the array well" are two separate things; this example improves the first, not the second.

## Run it

```bash
# Laptop smoke test (CPU, tiny subset — proves the code path):
CV_SMOKE=1 python examples/use_cases/satellite_landcover.py

# On a Trainium instance (real run):
python examples/use_cases/satellite_landcover.py

# Or via the harness (captures provenance, single-device so it joins VALIDATED.md):
python -m validation.run_on_hardware --instance trn1.2xlarge --region us-east-2 \
    --example satellite_landcover --yes
```

## Prerequisites

- A Neuron instance (`trn1.2xlarge`) or any CPU box for the smoke run.
- `pip install datasets pillow` (the Neuron stack comes from the DLAMI; torchvision-style CNN uses
  plain `torch.nn`).

## Now try this with your own data

Swap `dataset_name` for any HF image-classification dataset exposing an `image` + `label` column
(the label set is read from the schema). Keep tiles a **fixed size** to preserve static shapes.

## Status

See [`/VALIDATED.md`](../../VALIDATED.md) for the hardware-validation record (it's a single-device
example, so the harness validates it automatically).
