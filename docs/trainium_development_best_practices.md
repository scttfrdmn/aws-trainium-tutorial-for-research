# Trainium Development Best Practices

> **The one mindset that matters: build models in the form the hardware *wants*.**
>
> The most common way people fail on Trainium is assuming they're smarter than the hardware —
> taking a model shaped for GPU eager execution, throwing it at Trainium, and then fighting the
> compiler when it misbehaves. **The hardware is the hardware.** It is an ahead-of-time-compiled,
> bf16-native, static-shape accelerator. Work *with* those properties and it flies; work against
> them and you'll spend your day waiting on recompiles and chasing `nan`s. Every practice in this
> chapter is a specific instance of "fit the hardware's grain instead of forcing your habits onto
> it."

> **Field notes, not folklore.** Everything here was learned building the
> [biomedical NER example](../examples/use_cases/biomedical_ner.py) on a real `trn1.2xlarge`. Where
> we say "we saw X," it literally happened during validation — including the mistakes. Official AWS
> Neuron / PyTorch-XLA docs are linked throughout.

This tutorial runs PyTorch on Trainium through **PyTorch/XLA** (`torch-neuronx` 2.9): your Python code *traces* a graph,
which the Neuron compiler (`neuronx-cc`) compiles **ahead of time** into a hardware binary, which
then executes. It is **bf16-native** (the compiler casts matmuls to bf16 by default) and wants
**static shapes**. This is fundamentally different from eager CUDA, and almost every Trainium
surprise traces back to ignoring one of those three properties (AOT, bf16, static).

## A cautionary tale: the day we tried to outsmart the hardware

Our first NER exemplar was a stock Hugging Face `bert-base-cased` training loop — the exact code
you'd run on a GPU. On Trainium it produced **`nan` loss from step 0**, before a single weight
update. The instinct is to "fix" it with training tricks (lower LR, warmup, gradient clipping). We
tried. **None of it worked**, because none of it addressed the cause.

What the hardware was telling us:
- The **same model and data on CPU in fp32 gave a clean loss of 1.21.** So the model wasn't wrong.
- On Trainium the failure was in the **forward pass, in bf16** — the canonical "ported GPU model
  meets bf16" trap: BERT adds a large-negative attention-mask term before softmax, which **overflows
  to `nan` in bf16's narrow range**.
- The tempting "fix" — `NEURON_CC_FLAGS=--auto-cast=none` to force fp32 — is *defeating the entire
  point of the chip.* It's the smartest-guy-in-the-room move, and it's wrong: you bought a
  bf16-native accelerator and then turned bf16 off.

The right lesson isn't a flag. It's: **choose and build models that are bf16-stable and
static-shaped by design** — the form the hardware wants — instead of porting a GPU model and
patching symptoms. (The example carries both paths: the degenerate "fight the hardware" version,
clearly labeled, and the native version, so you can see the contrast.)

This is the whole chapter in one story. The rest is the specifics.

---

## 1. The #1 issue: compilation, and why iteration feels slow

**First, the execution model.** PyTorch/XLA does **not** run your ops the moment you write them.
It *records* them lazily into a graph, and nothing actually executes until something forces it —
that something is `xm.mark_step()` (or reading a tensor's value). `mark_step()` means "compile and
run everything I've recorded since the last one." This is why the terms below (`mark_step`,
"the graph") keep appearing: you're really programming a graph that gets compiled, not eager ops.

> **TorchNeuron (beta) note.** This chapter describes PyTorch/XLA, the lead path. The opt-in
> TorchNeuron backend (`device="neuron"`, Trn2/Trn3) has no lazy graph and no `mark_step`: ops run
> eagerly, and graphs only exist where you wrap code in `torch.compile(backend="neuron")`. In
> `torch.compile` terms, a "new graph per new shape" is a **recompile** (`TORCH_LOGS=recompiles`),
> and a data-dependent op splits the graph with a **graph break** (`TORCH_LOGS=graph_breaks`,
> `torch._dynamo.explain`). The shape-stability advice below still applies to any compiled region.
> Measured on a `trn2.3xlarge` (TorchNeuron 2.14.3, Neuron 2.33) with `python -m validation.torchneuron_facts`:
> eager mode **compiles per op and per shape**. The first call of a matmul took 55 s, the repeat
> 0.0008 s, and the same op at a new shape 1.3 s. `NEURON_COMPILE_CACHE_URL` had **no effect**: no
> files were written. The bf16 SDPA `nan` (§4) **does not reproduce**: SDPA and plain attention both
> stay finite under `torch.autocast(device_type="neuron", dtype=torch.bfloat16)`.

A new graph is compiled **every time the compiler sees a shape (or computation) it hasn't seen
before.** Compilation is expensive — seconds to *minutes* per graph for big models. If your program
keeps presenting new shapes, you recompile constantly and never make progress.

### The degenerate path (we triggered this for real)

Our first NER training run on `trn1.2xlarge` looked reasonable but behaved terribly:

- **7+ separate compilations in a single short run**, because the last batch of each epoch had a
  different batch size (5433 samples ÷ 16 ≠ integer → a ragged final batch → a *new shape* → a new
  graph).
- It also printed `loss.item()` **every step**, forcing a host↔device sync each iteration and
  stalling the pipeline.
- Net result: ~30+ minutes mostly spent compiling, plus `avg_loss=nan` (see §4).

That run is preserved as a teaching artifact. You can reproduce it with `shape_mode="ragged"` in the
example and watch the recompiles pile up.

### The fixes (in priority order)

**a) Make shapes static.** This is the single highest-leverage fix.
- Pad sequences to a fixed `max_length` (we pad to 128).
- Use `drop_last=True` in your `DataLoader` so the final short batch never creates a new shape.
- For genuinely variable-length data, **bucket** into a small set of fixed lengths, or pad to a
  fixed max — trading a little memory for *vastly* fewer compiles.
- Avoid data-dependent control flow (`if tensor.sum() > k:`); it changes the graph.
  Source: [PyTorch/XLA — source of recompilation](https://docs.pytorch.org/xla/master/notes/source_of_recompilation.html).

**b) Persist the compile cache** so you only pay compilation *once*, ever:
```bash
export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache   # local dir (survives only on THIS box)
```
A warm cache turned our repeated runs from minutes-of-compiling into seconds.

> **In the cloud, point the cache at S3 — not local disk.** This is the single biggest iteration-speed
> win for cloud Trainium, and it's easy to miss. Every time you provision a fresh `trn1`/`trn2`
> instance you get an *empty* local cache, so a local `NEURON_COMPILE_CACHE_URL` recompiles from
> scratch on every new box — exactly the tax you're trying to avoid. An **S3** cache URL persists
> across instance churn and is shared by every node:
> ```bash
> export NEURON_COMPILE_CACHE_URL=s3://my-bucket/neuron-cache   # survives reprovision; shared cluster-wide
> ```
> The Neuron runtime keys cache entries on a hash of the graph + compiler version + flags, so a hit is
> only reused when it's genuinely the same compile — safe to share broadly. **Measured cost of *not*
> doing this:** our CV utilization spike spent ~**10 min** compiling on a cold box (CNN ~5 min, ViT
> ~2 min) and seconds actually running; a warm S3 cache drops that startup to the S3 fetch (seconds).
> Pin the compiler so the cache key is stable across instances:
> ```bash
> export NEURON_CC_FLAGS="--model-type transformer"   # keep flags identical run-to-run, or you miss the cache
> pip show neuronx-cc | grep Version                   # cache is keyed on this; a version bump = cold cache
> ```
> Caveat: changing *any* of {graph shape, compiler version, `NEURON_CC_FLAGS`} produces a new key and a
> fresh compile — that's correct (the old artifact wouldn't be valid), just budget for it after an SDK
> upgrade. Give the bucket lifecycle rules / the instance role `s3:GetObject`+`PutObject` on the prefix.

**c) Pre-compile instead of compiling lazily during training:**
```bash
neuron_parallel_compile python train.py   # trace + compile ALL graphs up front, in parallel
python train.py                           # real run hits a warm cache
```
`neuron_parallel_compile` runs your script with execution stubbed, collects every graph, and
compiles them in parallel — so epoch 1 isn't where you discover a 20-minute compile.

> **Compilation is a CPU job — it does NOT need a NeuronCore.** `neuronx-cc` lowers HLO → NEFF
> entirely on the host CPU; the accelerator sits idle during compile. Two consequences worth
> internalizing:
> - **Don't reach for a bigger Trainium instance just to compile faster.** A `trn1.32xlarge` compiles
>   quicker than a `trn1.2xlarge` only because it has more *vCPUs* — you'd be renting 16 Trainium chips
>   to do a CPU job. That's wasteful and pushes the tutorial off the cheap single-device box.
> - **Compile on a cheap CPU instance, then run warm on Trainium.** The supported pattern: run
>   `neuron_parallel_compile` on a compute-optimized box (e.g. a `c7g`/`c6i` with many vCPUs), point
>   `NEURON_COMPILE_CACHE_URL` at S3, and let the `trn1.2xlarge` *consume* the warm cache. This
>   decouples compile horsepower from accelerator cost and keeps your actual Trainium time spent on
>   training, not compiling. If a model is so compile-heavy that even this is painful, that's a signal
>   to reshape it toward the form the array wants (see §1's "build in the form the hardware wants").

**Expect a slow first step (or first few), then a cliff.** Because compilation happens lazily the
first time each graph shape is seen, the opening steps are dominated by the *compiler*, not compute.
Measured on a cold trn1.32xlarge (Qwen3-8B LoRA, 32 cores):

| Step | Per-step time | |
|---|---:|---|
| 1 | ~119 s | compiling |
| 2–3 | ~200 s | compiling new shapes |
| **4+** | **~5 s** | **warm — steady state** |

That's a **~20-40× drop** once graphs are cached. So don't read a crawling first step as "Trainium is
slow" — it's the one-time compile, and with a **persistent (S3) cache** a *re-run* skips it entirely
(the same job, cache warm, hit ~5 s/step almost immediately with no multi-minute compiles). When you
benchmark, **always discard the compile-bound warmup steps** and measure steady state.

### Clever ways to keep compiles small and fast

This is the heart of "big unrolls vs. small pieces":

- **Don't unroll loops into the graph.** A Python `for t in range(seq_len): step(t)` that builds one
  giant graph compiles slowly and can blow up compiler memory. Prefer a single batched/vectorized op,
  or a bounded loop with a *fixed* trip count so the graph is small and reused.
- **Cut the graph at natural seams with `xm.mark_step()`.** Each `mark_step()` ends the current
  graph and executes it. One `mark_step` per training step gives the compiler many *small, identical*
  graphs (compile once, reuse) instead of one enormous one. Too few mark_steps → giant graphs, slow
  compiles, high memory; too many → execution overhead. One per step is the default sweet spot.
- **Keep train vs. eval shapes consistent.** Different batch sizes for training and eval are two
  graph families. We use `drop_last` on both so each compiles once.
- **Gradient accumulation** lets you keep a small fixed micro-batch shape (cheap to compile) while
  reaching a large effective batch — instead of one huge batch that compiles slowly.
- **Warm up tiny.** When iterating on code, run 1 epoch on a few hundred samples with a *persistent
  cache*; the shapes match the full run, so the full run reuses those compiles.

---

## 2. Never sync the device every step

Calling `.item()`, `.cpu()`, `print(loss)`, or `loss.cpu().numpy()` **inside** the training loop
forces the device to finish and hand a value back to Python — destroying the overlap between tracing
the next step and executing the current one.

**Anti-pattern (what we did first):**
```python
for batch in loader:
    ...
    running += loss.item()  # ❌ host sync EVERY step
```

**Better — accumulate on device, fetch once:**
```python
running = torch.zeros((), device=device)
for batch in loader:
    ...
    running += loss.detach()  # ✅ stays on device
avg = (running / n).item()  # ✅ one sync per epoch
```
For periodic logging without a hard sync, use
[`xm.add_step_closure`](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html).

---

## 3. The recommended single-device training skeleton

```python
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

device = xm.xla_device()
model = Model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

# MpDeviceLoader overlaps host->device copy with compute and issues mark_step per batch.
loader = pl.MpDeviceLoader(
    torch.utils.data.DataLoader(ds, batch_size=16, drop_last=True), device
)

for epoch in range(epochs):
    model.train()
    for batch in loader:  # tensors already on device
        opt.zero_grad()
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # see §4
        xm.optimizer_step(opt)  # applies grads
        xm.mark_step()  # end + execute this step's graph
```
Source: [PyTorch on XLA devices](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html).

---

## 4. BF16 is the target, not a hazard — design for it

`neuronx-cc` casts matmuls to **BF16** by default because that's what the hardware is fast at. BF16
has the same exponent range as fp32 but far fewer mantissa bits. The mistake is treating bf16 as a
problem to be worked around; the right move is to **build models that are numerically happy in
bf16.**

**Important reassurance:** the matmul *inputs* are bf16, but the systolic array **accumulates the
running sum in FP32** (in a dedicated FP32 register bank). So you keep fp32-range dynamics on the
accumulation — the precision risk is in *representing* individual values, not in summing them.
That's why bf16 on Trainium is far less lossy than "everything is 16-bit" would suggest, and why
the fix below is never "turn bf16 off."

**What actually bites you (and what doesn't):**
- ❌ **Wrong diagnosis (ours, at first):** "loss is `nan`, so training is diverging — add LR warmup
  and gradient clipping." We added both. **It didn't help**, because the `nan` appeared at *step 0,
  in the forward pass*, before any gradient step. Warmup/clipping treat training dynamics; this
  wasn't a training-dynamics problem.
- ✅ **Right diagnosis:** it's the **attention implementation in the forward pass**, not training
  dynamics. Hugging Face Transformers v5 defaults to **SDPA** (the fused
  `scaled_dot_product_attention` kernel). On the Neuron bf16 compile path that produced `nan` at step 0,
  while the same model on CPU/fp32 gave a normal loss (~1.21). We didn't pin down the exact numerical
  mechanism inside the compiled SDPA graph; what we measured is which implementation works.
- ✅ **Right fix:** `attn_implementation="eager"` (plain matmul + softmax). That gave a stable loss of
  **1.13 on Trainium with bf16 still on**. It's the default in the
  [NER example](../examples/use_cases/biomedical_ner.py), and
  [the debugging walkthrough](../examples/debugging/diagnose_common_failures.py) reproduces both outcomes.
  **Don't reach for `--auto-cast=none`.** Forcing fp32 "fixes" the symptom by switching off the
  accelerator's whole reason for existing.

**Still good practice once the model is bf16-sane:** LR warmup and gradient clipping help *training
stability* generally, and **stochastic rounding** improves bf16 accuracy. They're just not a cure
for a forward-pass overflow — fix the model's form first.

See the [Neuron compiler data-types guidance](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/index.html).

---

## 5. Pitfalls checklist

| Pitfall | What we observed / it causes | Fix |
|--------|------------------------------|-----|
| Ragged final batch | 7+ recompiles in one run | `drop_last=True`, fixed `max_length` |
| `.item()`/print every step | host sync, stalled device | accumulate on device; fetch per epoch |
| Ported GPU model `nan` in bf16 | forward-pass overflow (big additive mask) | use a bf16-native model/attention — **not** `--auto-cast=none` |
| "Fix" bf16 by forcing fp32 | throws away the accelerator's speed | redesign the model for bf16; keep auto-cast on |
| Cold compile cache | minutes of compiling every run | `NEURON_COMPILE_CACHE_URL` (+ S3 to share) |
| Lazy first-epoch compile | epoch 1 mysteriously takes forever | `neuron_parallel_compile` up front |
| Giant unrolled graph | slow compile, high memory | smaller graphs via `mark_step`; vectorize; grad-accum |
| Plain DataLoader | no data/compute overlap | `pl.MpDeviceLoader` |
| Timing without sync | you measure tracing, not compute | warm-up run + `xm.wait_device_ops()` before timing |

---

## 6. How to debug compilation

```bash
export NEURON_RT_LOG_LEVEL=INFO        # see runtime / compile activity
grep -c "Compilation Successfully Completed" run.log   # count how many graphs you compiled
```
If that count is more than "a handful," you have a shape-stability problem — go back to §1.

You can see the same lesson on a laptop. `python examples/debugging/diagnose_common_failures.py
--only recompile` runs `torch.compile(backend="eager")` on CPU and counts graphs: fixed shapes give 1,
ragged shapes with `dynamic=False` give 4 (by default Dynamo switches to dynamic shapes after the
2nd shape, giving 2). These are torch.compile counts, not Neuron compiles.

> **TorchNeuron (beta) note.** Timing works differently too. Under XLA you call
> `xm.wait_device_ops()` before stopping the clock. On TorchNeuron, `torch.neuron.synchronize()` (or
> `torch.accelerator.synchronize()`) blocks until queued work finishes; without one, a 4096 matmul
> "took" 0.1 ms instead of about 4 ms. The `Compilation Successfully Completed` line **never appears**
> under TorchNeuron, so the `grep` count above doesn't apply there. Use `TORCH_LOGS=recompiles`
> instead. (Measured on a `trn2.3xlarge` (TorchNeuron 2.14.3, Neuron 2.33) with `python -m validation.torchneuron_facts`.)

For CPU-side kernel development without burning Trainium time, use NKI's simulation mode (see the
[Simulator chapter](main_tutorial_doc.md#simulator)).

---

> **Where this fits:** read this **after your first NER run, before you scale**. Came from
> [choose your path](choose_your_path.md). Next: [Neuron tools & debugging](neuron_tools_and_debugging.md)
> for the symptom→tool map and the profiler; [novel kernels](novel_kernels_on_trainium.md) if you need
> a custom operator.
