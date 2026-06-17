# Neuron Tools, Debugging, Tracing & the Simulator

> **Assumed knowledge:** you can run a PyTorch/XLA training script on Trainium (see the
> [NER example](../examples/use_cases/biomedical_ner.py)). No prior Neuron-tooling experience.
> **What you'll be able to do:** name each Neuron tool, know what it inspects, and — given a symptom
> (`nan`, slow training, won't compile, OOM) — reach for the right one.

The Neuron docs tend to assume you already know this toolbox. You don't, and that's fine. This
chapter is the map. Every tool here is from the **public** AWS Neuron documentation; where the
public docs are thin, this chapter says so rather than inventing behavior.

> **Sourcing & honesty:** exact metric schemas, some flags, and some UI details are **not fully
> published** by AWS. Treat command names and purposes as reliable; treat exact field lists as
> "check `--help` on your DLAMI." Primary sources:
> [Neuron tools docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/index.html),
> [dev-tools release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/dev-tools.html),
> [PyTorch/XLA](https://github.com/pytorch/xla).

---

## The toolbox at a glance

| Tool | Layer | One-liner |
|------|-------|-----------|
| `neuron-ls` | system | What Neuron devices/cores exist, and who's using them. |
| `neuron-top` | system | Live, `top`-style per-core utilization. |
| `neuron-monitor` | system | Machine-readable metrics stream (JSON → CloudWatch/dashboards). |
| **Neuron Explorer** | profiling | The profiler/visualizer (timeline, memory, tensors). Supersedes `neuron-profile` in SDK 2.30. |
| `torch_xla` profiler | framework | Op-level trace of your PyTorch/XLA graph. |
| **NKI simulation** | kernel dev | Run an NKI kernel on CPU to check *correctness* (not speed). |
| compile cache + flags | compiler | Avoid recompiles; dump artifacts when compilation fails. |

---

## 1. See your hardware: `neuron-ls`

The first command to run on any Neuron instance.

```bash
neuron-ls
```
Shows the Neuron devices, NeuronCores per device, device memory, and which **PID** currently owns a
core. Use it to: confirm the hardware is present, see how many cores you have to parallelize across,
and find out *what process is holding a core* when a launch fails with "device busy."

## 2. Watch it live: `neuron-top`

```bash
neuron-top
```
A `top`-like live view of per-NeuronCore utilization, plus host CPU/memory and device-memory use.
Use it to answer "is the device actually busy, or am I bottlenecked on the host / data loading?"
If cores sit near-idle during "training," you're likely host-bound (data pipeline) or stuck
compiling — not computing.

## 3. Export metrics: `neuron-monitor`

```bash
neuron-monitor                       # emits JSON metrics to stdout
```
The programmatic sibling of `neuron-top`: a JSON metric stream you can pipe into CloudWatch or a
dashboard for long runs. Reach for it when you need *history* (a 6-hour training run) rather than a
live glance. (The exact metric schema isn't fully published — inspect the JSON on your instance.)

---

## 4. Profile a run: Neuron Explorer

For "it runs but it's slow," you need a profile — *where* time and memory actually go.

As of **SDK 2.30, Neuron Explorer is the profiler** (it supersedes the older `neuron-profile` and
"neuron-profiler 2.0"). It offers a CLI and a UI (VS Code extension / standalone) with, per the
public release notes, a **timeline** view (op execution + latencies), a **memory** view, and a
**tensor** view (shapes/dtypes/layout).

Use it to find: the ops eating wall-clock, memory/SBUF pressure, and recompilation. This is the
primary "why is it slow?" tool. (The detailed user guide is thin publicly — explore the UI and the
[dev-tools release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/dev-tools.html).)

> If you're on an older SDK, you may still have `neuron-profile`; the concepts (capture a profile,
> read a timeline) carry over.

## 5. Profile the framework graph: `torch_xla` profiler

A complementary, PyTorch-side view — what *your XLA graph* looks like and which ops are expensive,
independent of the hardware profile:

```python
from torch_xla.debug import profiler as xp

# Wrap a few steps; writes a trace you can open in a profiler/timeline viewer.
with xp.trace("./xla_profile"):
    for batch in a_few_batches:
        train_step(batch)
```
Use it to spot **recompilation** (new graphs appearing for new shapes) and to see op timings at the
framework level. **Caveat:** this reflects XLA's *view*, not cycle-accurate hardware timing — for
real device timing use Neuron Explorer. (`xp.trace` is from the public PyTorch/XLA project; the
exact API can shift between torch-xla versions — check the version on your DLAMI.)

---

## 6. Develop kernels on CPU first: NKI simulation

If you're writing a custom **NKI** kernel (see [novel kernels](novel_kernels_on_trainium.md)), you
don't need to burn Trainium time to check correctness. NKI can run a kernel on CPU (NumPy-backed)
so you can validate numerics before compiling for hardware.

- ✅ **Verifies:** numerical correctness, shapes/dtypes, indexing/tiling logic.
- ❌ **Does NOT tell you:** real performance, timing, or hardware bottlenecks (CPU timing is
  meaningless here).

Workflow: develop + unit-test in simulation (cheap, CI-friendly) → then compile and **profile on
real hardware** for speed. Simulation answers *"is it correct?"*; only hardware answers *"is it
fast?"* (Public coverage of the exact simulation entry points is uneven; see the public
[NKI docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/) and
[nki-samples](https://github.com/aws-neuron/nki-samples).)

---

## 7. Compilation: cache it, and dump artifacts when it breaks

Compilation is the #1 source of Trainium surprises (see the
[Compilation chapter](main_tutorial_doc.md#compilation)).

**Make repeat runs fast — persistent cache:**
```bash
export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache   # or an s3:// path to share
```

**Pre-compile instead of compiling lazily during epoch 1:**
```bash
neuron_parallel_compile python train.py   # collect graphs, compile in parallel → warm cache
python train.py                           # real run hits the cache
```

**When compilation *fails*, capture artifacts for a bug report** (these env vars are documented in
public AWS Neuron repos):
```bash
export NEURONX_DUMP_TO=./compiler_debug    # dumps IR/logs/artifacts the Neuron team asks for
```

**Count your recompiles** — the fastest slow-training diagnosis:
```bash
grep -c "Compilation Successfully Completed" run.log
```
More than a handful for a fixed-shape training loop ⇒ you have a shape-stability problem (go fix
shapes; see best-practices §1).

> Some additional flags (`NEURON_CC_FLAGS` verbosity, `NEURON_RT_LOG_LEVEL`, `XLA_IR_DEBUG`,
> `XLA_HLO_DEBUG`) exist; their exact accepted values aren't all in the public docs. Check
> `neuronx-cc --help` on your DLAMI before relying on a specific flag.

---

## Symptom → tool

| Symptom | Reach for | What you're looking for |
|---------|-----------|-------------------------|
| **`nan`/`inf` loss** | print/inspect on CPU first; Neuron Explorer tensor view | Is it forward (step 0) or training divergence? (bf16 SDPA → see best-practices §4) |
| **Training is slow** | recompile count (`grep`), `neuron-top`, Neuron Explorer timeline | Recompiling every step? Host-bound? Or genuinely compute-bound? |
| **"Stuck" with no progress** | `neuron-top`, recompile count | Almost always recompilation from changing shapes. |
| **Device busy / won't launch** | `neuron-ls` | Which PID owns the core; kill the stale process. |
| **Out of memory / SBUF pressure** | Neuron Explorer memory view | Batch/seq too large; tile/layout pressure. |
| **Kernel won't compile** | `NEURONX_DUMP_TO`, compiler error | Capture artifacts; file an issue with the dump. |
| **Correct but slow custom kernel** | NKI simulation (correctness) → Neuron Explorer (speed) | Validate math on CPU, then profile on hardware. |

---

## A worked debugging example

The repo ships a runnable debugging walkthrough that *deliberately* triggers the two classic
failures and shows how to diagnose them with these tools:
[`examples/debugging/diagnose_common_failures.py`](../examples/debugging/diagnose_common_failures.py).
It reproduces (1) the bf16 SDPA→`nan` forward-pass failure and the eager fix, and (2) the
variable-shape recompile storm vs. the `drop_last` fix — counting compilations so you can *see* the
difference. Read its companion notes in
[`examples/debugging/README.md`](../examples/debugging/README.md).
