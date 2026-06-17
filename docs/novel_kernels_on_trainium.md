# Developing Novel Kernels on Trainium

> **Assumed knowledge:** you've written or read a CUDA kernel (or at least know the GPU SIMT model),
> and you've run something on Trainium (the [NER example](../examples/use_cases/biomedical_ner.py)).
> **What you'll be able to do:** explain what Trainium's architecture does that's genuinely
> different, decide whether *your* problem maps to it, and design a kernel whose payoff is a **better
> result — not just more speed**.

Most "write a custom kernel" guides chase throughput. The more interesting question for a researcher
is: **what can this architecture compute that's awkward or lossy to express as a stack of stock
GPU/TPU library calls?** On Trainium there's a concrete, public answer — and it's about *accuracy*,
not just FLOPs.

> **Sourcing.** Every architectural number below is quoted from the public
> [NKI Trainium/Inferentia2 Architecture Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html).
> Where AWS hasn't published a figure, this chapter says **"(not published)"** rather than guess.
> This is NeuronCore-v2 (Trn1/Inf2); v3 (Trn2) differs and has its own guide.

---

## 1. The architecture, briefly (and how it differs from a GPU)

A GPU is thousands of threads (SIMT) with a cache hierarchy and registers; you think in threads and
warps. A NeuronCore is **a few specialized engines feeding off an explicitly-managed on-chip
buffer**; you think in **tiles** moving through engines. Public facts:

- **PE (Tensor) engine** — a **128×128 systolic array** of processing elements. Dense matmul is its
  native act. It does **mixed-precision with accumulation at FP32 — output is always FP32**, even
  when inputs are BF16/FP16/TF32/FP8.
- **Vector engine** — **128 parallel lanes** streaming from the buffers; arithmetic in FP32.
- **Scalar engine** — **128 lanes**; activations / non-linearities (GELU, Sqrt), internally FP32.
- **GPSIMD engine** — **eight fully programmable processors that can run arbitrary C/C++** — the
  escape hatch for things that aren't pure tensor ops.
- **SBUF (state buffer)** — **24 MiB**, **128 partitions × 192 KiB**. The on-chip workspace; you
  *explicitly* stage data here. (Bandwidth: not published.)
- **PSUM** — **2 MiB**, **128 partitions × 8 banks × 512 FP32 values**. The matmul accumulator;
  **"PSUM accumulation is always done in FP32."**

The mental model that matters: **the tile is the unit**, the **partition axis (≤128) is parallel**,
the **free axis is time**, and **you manage the memory** (`nl.load`/`nl.store` between HBM and SBUF)
rather than relying on a cache. There are far fewer "knobs" than CUDA — and far less that's implicit.

---

## 2. The unique lever: FP32 accumulation you can *keep*

Here's the part that enables a better *result*, and it's straight from the public spec:

> The tensor engine accumulates in **FP32 in PSUM**, always — regardless of input dtype.

On a GPU you get this too *inside* a single fused library call. The difference on Trainium is that
**NKI lets you write a kernel that keeps a whole multi-step computation resident in SBUF/PSUM at
FP32**, instead of doing what a framework does by default: run op A in BF16 → **round-trip to HBM in
BF16** → run op B in BF16 → round-trip again. Every one of those BF16 round-trips between composed
library ops **sheds mantissa bits**.

So a custom fused kernel can return a **more numerically accurate** answer than the same math
expressed as a chain of stock BF16 ops — not because the hardware is magic, but because **you
control where precision is kept and where data crosses HBM.** That's hard to replicate by gluing
together cuBLAS/cuDNN or XLA library calls, where the op boundaries (and their precision drops) are
fixed for you.

**This is the novel-kernel thesis for a researcher:** pick a computation that is *numerically
fragile when composed from BF16 library ops*, and fuse it so the sensitive accumulation stays in
FP32 on-chip. Win = accuracy.

### Candidate computations where this pays off
- **Long reductions / running statistics** (large sums, variance, norms) — BF16 partial sums lose
  precision fast; an FP32-resident accumulation is markedly more accurate.
- **Layer/RMS norm or softmax over long axes** done in one fused pass instead of compose-and-spill.
- **Compensated / split-precision matmul** (Ozaki-style or Kahan-style correction) — keep the
  residual in FP32 on-chip to recover accuracy lost in BF16 matmul.
- **Iterative refinement** where each step's error feeds the next — round-tripping through BF16
  poisons the refinement; FP32-resident state preserves it.

> A time-boxable exercise (fits an afternoon): take a **long-axis reduction or a fused normalization**
> and show your NKI kernel's output is closer to the FP64 reference than the stock BF16 composed
> path — *at similar or better speed*. The headline is the **lower error**, not the wall-clock.

---

## 3. Does your problem map? <a id="does-your-problem-map"></a>

Score it against the architecture (this is the detailed version of the
[choose-your-path](choose_your_path.md) questions):

| Property | Good fit (do it) | Poor fit (don't) |
|---|---|---|
| **Core op** | dense matmul / convolution / regular tensor contractions | pointer-chasing, graph traversal, sparse scatter/gather |
| **Shapes** | static, tileable to ≤128 partitions | dynamic, data-dependent |
| **Contraction** | maps cleanly onto the 128×128 PE array (K in the partition dim) | irregular, tiny, or non-matmul-shaped |
| **Data movement** | bulk, predictable loads into SBUF; reuse on-chip | random access, huge working sets that can't tile into 24 MiB |
| **Control flow** | uniform across tiles | per-element data-dependent branching (though GPSIMD is the escape hatch) |
| **Precision** | benefits from FP32-resident accumulation | already fine in BF16 end-to-end (then why bother) |

**A quick litmus test:** can you write the kernel's inner loop with *no data-dependent branches* and
tiles that fit 128 partitions? If yes, it likely maps. If you keep wanting `if (x[i] > t)` or random
indexing, the PE/vector/scalar engines will idle — that's the GPSIMD-or-GPU signal.

**Be honest about the misfits:** irregular sparsity, dynamic shapes, and pointer-heavy algorithms
map poorly. The systolic array wants regular, dense, statically-tiled work. Forcing a misfit kernel
onto Trainium is the inverse of "build in the form the hardware wants."

---

## 4. How you'd actually build it

The NKI workflow (see the public [NKI docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/)
and [nki-samples](https://github.com/aws-neuron/nki-samples)):

1. **Express it in tiles.** Decompose into ≤128-partition tiles; map the contraction dim to the
   partition axis for the PE engine.
2. **Stage explicitly.** `nl.load` HBM→SBUF, compute keeping intermediates in SBUF/PSUM (FP32),
   `nl.store` back only the final result. The whole point is to *not* spill intermediates.
3. **Develop on CPU first.** Use [NKI simulation](neuron_tools_and_debugging.md#6-develop-kernels-on-cpu-first-nki-simulation)
   to verify numerics for free — compare against an FP64 reference.
4. **Profile on hardware.** Only real hardware tells you the speed; check it's not memory-bound (see
   [tools & debugging](neuron_tools_and_debugging.md)).
5. **Report the win honestly.** Error vs FP64 reference *and* time vs the composed baseline — so the
   accuracy claim is measured, not asserted.

> The illustrative kernels under [`advanced/`](../advanced/) and
> [`examples/advanced/`](../examples/advanced/) show NKI *shape* but are not drop-in runnable. Start
> from the official nki-samples for working code.

---

## 5. The "Build on Trainium" angle

AWS runs research/credit programs that give academics access to Trainium for exactly this kind of
kernel and systems research (the NKI interface exists to enable custom-kernel work, and open
academic results like the HLAT LLaMA-on-Trainium training exist publicly). **Program names, funding,
and eligibility change — verify the current program and terms on the official AWS site** rather than
trusting a number quoted in a tutorial. What's durable: NKI is the public door to novel-kernel
research on Trainium, and the accuracy-via-FP32-resident-fusion angle above is a genuinely
publishable research direction, not just an optimization.

---

## Honesty box

- All architecture numbers are quoted from the public NKI Trainium/Inferentia2 architecture guide.
- SBUF/engine **bandwidths are not published**; this chapter doesn't state them.
- This is **NeuronCore-v2 (Trn1/Inf2)**. Trn2 (v3) has its own guide with different numbers.
- The "better result via FP32-resident fusion" thesis follows from the published FP32-accumulation
  fact; the *magnitude* of any accuracy win is workload-specific — **measure it, don't assume it**.
