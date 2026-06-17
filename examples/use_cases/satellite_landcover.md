# Satellite land-cover classification on Trainium

> **Assumed knowledge:** basic PyTorch + CNNs. No Neuron experience needed.
> **What you'll get:** a real geospatial computer-vision workflow on Trainium — and a feel for why
> dense convolutions are a *great* fit for the systolic tensor engine.

Classify **Sentinel-2** satellite tiles into land-cover classes (AnnualCrop, Forest, Residential,
River, SeaLake, ...) with a small CNN on the **EuroSAT** benchmark. This is the tutorial's vision
example — a different hardware story from the transformer/LLM ones.

## Why it fits Trainium well

| Property | This example |
|---|---|
| Dense, regular compute | convolutions → the 128×128 systolic tensor engine's home turf |
| Static shapes | fixed **64×64×3** tiles + `drop_last` → graph compiles once, no recompile storm |
| bf16-native | the tensor engine accumulates FP32 in PSUM (best-practices §4) |

Contrast with the [NER](biomedical_ner.py) example: there, HF-default SDPA attention had to be
swapped to `eager` to survive bf16. A plain CNN has no such fragile op — it's bf16-stable as-is.

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
