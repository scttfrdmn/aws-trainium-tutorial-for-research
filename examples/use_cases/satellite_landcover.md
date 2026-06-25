# Satellite land-cover classification on Trainium (from AWS Open Data)

> **Assumed knowledge:** basic PyTorch + CNNs. No Neuron or geospatial experience needed.
> **What you'll get:** a real geospatial computer-vision workflow on Trainium, built on **live AWS
> Registry of Open Data** — assemble labeled training patches on the fly from two public S3 datasets,
> then classify them with a CNN.

Classify **Sentinel-2** satellite imagery into **ESA WorldCover** land-cover classes (tree cover,
cropland, built-up, water, ...) with a small CNN. Unlike a pre-tiled benchmark, this example builds
its training set directly from the **AWS Registry of Open Data (RODA)**:

| Half | RODA dataset | Bucket (anonymous) |
|---|---|---|
| Imagery | Sentinel-2 L2A cloud-optimized GeoTIFFs | `s3://sentinel-cogs` (us-west-2) |
| Labels | ESA WorldCover 10 m v200 | `s3://esa-worldcover` (eu-central-1) |

It reads a window from a Sentinel-2 true-color COG, reprojects its footprint to lon/lat, reads the
co-located WorldCover label raster onto the same grid, tiles both into fixed 64×64 patches, and
labels each patch by its **majority WorldCover class** — real `(image → land-cover)` pairs from open
satellite data. This is the tutorial's vision example — a different hardware story from the LLM ones.

> **Honest note on the task:** majority-class labeling of mixed patches carries inherent label noise,
> so this is *harder* than the curated EuroSAT benchmark and accuracy is lower (gated at 0.60, not
> 0.80). That's the price of using raw open data instead of a pre-cleaned dataset — and it's the
> realistic version of the task.

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
blocks (see [novel kernels](../../docs/novel_kernels_on_trainium.md)).

**We measured this** — it's not a hunch. The [utilization spike](cv_utilization_spike.md) runs this
exact CNN against a ViT on the same trn1.2xlarge: the **ViT achieves 5.1× the CNN's TFLOP/s** while
doing *less* total work. The CNN does 2.7× more FLOPs yet takes 13.7× longer per step — the array sits
idle on its small convs. (On CPU the ordering *flips* — proof the gap is the systolic array, not the
model.) That's the whole lesson, with numbers.

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

- A Neuron instance (`trn1.2xlarge`) or any CPU box for the smoke run; **network access** (it reads
  from S3 at runtime).
- `pip install rasterio numpy` — `rasterio` reads the cloud-optimized GeoTIFFs from S3 anonymously
  (the Neuron stack comes from the DLAMI; the CNN is plain `torch.nn`). No AWS credentials needed —
  both buckets are public.

## Now try this with your own area

- Add Sentinel-2 scene paths to `scenes` (browse `s3://sentinel-cogs/sentinel-s2-l2a-cogs/`) to cover
  new geography / more land-cover classes.
- Raise `patches_per_side` for a bigger training set per scene; `window_offset` moves the sampled
  window within a scene.
- The same pattern generalizes to any RODA raster pair — swap WorldCover for another labeled product
  (e.g. a crop-type map) to build a different classifier.

## Status

✅ **Hardware-validated on `trn1.2xlarge`** (us-west-2, Neuron 2.30 / torch 2.9.1): `eval_acc = 0.75`
on the held-out RODA patches (4 classes: Tree/Grassland/Cropland/Built-up), trained end-to-end on
real Sentinel-2 + WorldCover open data. Warm-cache train wall-clock **~80 s**.

**The interesting part — the compile story (a real teaching artifact).** This small-conv residual CNN
is *slow to compile* on the trn1.2xlarge's 8 vCPUs: a cold compile of the training graph took **~44
minutes**. That is not a hardware fault — compilation is a CPU-bound, ahead-of-time step
(`neuronx-cc` lowers the whole graph to a NEFF before the first step runs), and this particular
multi-ResBlock + BatchNorm graph is expensive. The fix is the standard Neuron pattern, **not** a
bigger accelerator:

- **`neuron_parallel_compile`** compiles all graphs up front, and a **persistent S3 compile cache**
  (`NEURON_COMPILE_CACHE_URL=s3://…`) makes that a one-time cost — the validated **warm re-run finished
  in ~1.5 min** (`Using a cached neff`, 0 recompiles). Compilation needs only CPU, so you can even
  pre-compile on a cheap compute instance and let the trn1.2xlarge consume the cache
  (see [best-practices §1](../../docs/trainium_development_best_practices.md)).

It's also a live example of this repo's thesis: a small-conv CNN is *not* the shape the array wants —
the [utilization spike](cv_utilization_spike.md) measured this same CNN at ~5× *lower* TFLOP/s than a
ViT, and the heavy compile is another face of that mismatch. See [`/VALIDATED.md`](../../VALIDATED.md).
