# Spike: *measuring* Trainium under-utilization (CNN vs ViT)

> **Assumed knowledge:** you've read the [CV example](satellite_landcover.md) and know what a CNN and
> a ViT are.
> **What you'll get:** a runnable experiment — on real hardware — that *proves* a small CNN starves
> the systolic array, and that a ViT-shaped model gives the hardware what it wants. With numbers, not
> adjectives.

The [CV example](satellite_landcover.md) makes a claim: a small CNN **runs** fine on Trainium
(static shapes, bf16-stable) but doesn't **use** the 128×128 systolic array well — its early convs
are too small to fill 128 partitions. That's a *falsifiable* claim, so this spike tests it instead of
asserting it.

## The experiment

Same device, same fixed 64×64×3 input, same batch (64). Two models:

- **CNN** — the small residual CNN from the CV example (re-used verbatim).
- **ViT** — a Trainium-shaped vision transformer: patch-embed → 256-wide embedding → 6 blocks of
  *large* attention/MLP matmuls. Attention is hand-written with explicit `q@kᵀ` / `attn@v` matmuls and
  an **fp32 softmax** — deliberately *not* `scaled_dot_product_attention`, which `nan`s in bf16 on
  Neuron (the [NER example](biomedical_ner.py) hits the same bug).

For each model we report exact forward FLOPs (`torch.utils.flop_counter`), steady-state device step
time, and **achieved TFLOP/s** = `3 × fwd_FLOPs ÷ step_time` — a portable proxy for *how full the
systolic array is*. (The Neuron profiler's MFU is the ground-truth read; achieved TFLOP/s is what you
can run anywhere — see [tools & debugging](../../docs/neuron_tools_and_debugging.md).)

```bash
# On a Trainium instance:
python examples/use_cases/cv_utilization_spike.py

# Laptop smoke (CPU; FLOP counts exact, timings not meaningful):
CV_SMOKE=1 python examples/use_cases/cv_utilization_spike.py
```

## ✅ Result (measured on real hardware)

**trn1.2xlarge**, 1 NeuronCore, Neuron 2.30 (`neuronx-cc 2.25.3371`, torch 2.9.1), batch=64:

| Model | Params | FWD FLOPs | Step time | **Achieved TFLOP/s** |
|---|---:|---:|---:|---:|
| **CNN** (small residual) | 2.47 M | **109.9 GFLOP** | **307.3 ms** | **1.07** |
| **ViT** (patch-embed, 256-wide, 6 blocks) | 4.81 M | 41.3 GFLOP | 22.5 ms | **5.51** |

**→ The ViT achieves 5.1× the CNN's TFLOP/s on the same chip.**

Look closer — it's even sharper than a 5× headline:

- The CNN does **2.7× *more* total FLOPs** (110 vs 41 GFLOP) yet takes **13.7× *longer* per step**
  (307 vs 22 ms). It is doing more math and getting less done.
- That's the signature of **under-utilization**: the CNN's FLOPs live in small early convs (contraction
  dim ~3–64) that can't fill a 128-wide array, plus a long tail of conv/BN/pool ops. The array spends
  most of its cycles idle. The ViT's FLOPs live in big patch/attention/MLP matmuls that *do* fill it.
- **The CPU sanity check flips the ordering.** Run the same spike with `CV_SMOKE=1` on a laptop and the
  CNN looks *fine* — a CPU has no 128×128 systolic array to starve. The reversal between CPU and
  Trainium is the proof: the gap is the *hardware's shape*, not the model being "bad" in the abstract.

## The lesson

The hardware is not slow. **We were feeding it work that doesn't fit its shape.** "It runs" (static,
bf16-stable) and "it uses the hardware" (fills the systolic array) are two different bars. The CV
example clears the first; this spike shows what clearing the second looks like — and that you can
*measure* the difference in ten minutes, before committing to a full training run.

> This is the general move for Trainium: when something feels slow, don't assume the chip is the
> limit — measure achieved FLOP/s (or read MFU in the profiler) and ask whether your *matmuls are big
> enough to fill 128 partitions*. If they're not, reshape the model, don't blame the hardware.

## A note on the compile tax (and how to stop paying it)

This spike spent ~**10 minutes compiling** on a cold box (CNN ~5 min, ViT ~2 min) and seconds actually
measuring. That's the ahead-of-time-compilation cost, and on a fresh cloud instance you pay it *every
provision* unless you persist the cache. Point `NEURON_COMPILE_CACHE_URL` at **S3** (not local disk,
which is empty on each new instance) so compiled graphs survive reprovisioning:

```bash
export NEURON_COMPILE_CACHE_URL=s3://my-bucket/neuron-cache
python examples/use_cases/cv_utilization_spike.py     # first box compiles; every later box fetches
```

The validation harness exposes this directly: `... --cache-url s3://my-bucket/neuron-cache`. See
[best-practices §1b](../../docs/trainium_development_best_practices.md) for the full caching guidance
(compiler-version pinning, cache-key gotchas).

See [novel kernels on Trainium](../../docs/novel_kernels_on_trainium.md) for *why* the array wants
large contractions, and [choose your path](../../docs/choose_your_path.md) for which workloads have
that shape naturally.
