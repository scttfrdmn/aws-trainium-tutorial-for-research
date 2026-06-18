# Debugging walkthrough

A runnable companion to [docs/neuron_tools_and_debugging.md](../../docs/neuron_tools_and_debugging.md).
It reproduces the three classic Trainium failures — on purpose — and shows the diagnosis and fix for
each. All were hit for real while building the [validated NER example](../use_cases/biomedical_ner.py).

> **Assumed knowledge:** you can run a Python script and have read the
> [best-practices chapter](../../docs/trainium_development_best_practices.md).
> Running the `nan` demo locally needs `pip install torch transformers` (the CPU "free" mode still
> imports them); the recompile and fallback demos short-circuit before any hardware call.
> **What you'll get:** hands-on recognition of (1) the bf16 SDPA→`nan` forward-pass failure,
> (2) the variable-shape recompile storm, and (3) the silent CPU fallback — and the fix for each.

## Run it

```bash
# CPU walkthrough (free; explains what you'd see on hardware):
python examples/debugging/diagnose_common_failures.py

# On a Neuron instance — also counts real compilations:
python examples/debugging/diagnose_common_failures.py --device xla

# Just one demo:
python examples/debugging/diagnose_common_failures.py --only nan
python examples/debugging/diagnose_common_failures.py --only recompile
python examples/debugging/diagnose_common_failures.py --only fallback
```

## What it teaches

| Failure | Symptom | Diagnosis | Fix |
|---------|---------|-----------|-----|
| **bf16 SDPA → nan** | `nan` loss at **step 0** (forward) | Same model+data on CPU/fp32 is finite ⇒ it's the bf16 path, not your LR | `attn_implementation="eager"` (keeps bf16; **not** `--auto-cast=none`) |
| **Recompile storm** | "stuck", host CPU busy, cores idle, no step progress | `grep -c "Compilation Successfully Completed"` ≫ a handful | fixed shapes: `drop_last=True` + fixed `max_length` |
| **Silent CPU fallback** | slow; no error; device underused | `torch_xla.debug.metrics.metrics_report()` → any **`aten::`** counter = an op ran on CPU | vectorize to a fixed-shape op (`torch.where`, masking); avoid `.item()`/`.nonzero()` in the hot loop |

The first two aren't toys: on real `trn1.2xlarge` we measured SDPA→`nan` vs eager→loss 1.13, and
ragged shapes → 7+ compiles vs fixed → ~2; the validated example bakes in both fixes.

### The CPU-fallback case in detail (discover → fix → accept)

This is the one you can hit without any crash: the Neuron compiler has no device lowering for an op,
so PyTorch/XLA runs it on the **CPU** ("aten fallback") and copies tensors host↔device every step.
The demo walks three outcomes:

1. **Discover** — a mask built with `.nonzero()`/host-reads makes `metrics_report()` list
   `aten::nonzero`, `aten::_local_scalar_dense`. Any `aten::` line = that op left the device.
2. **Fix** — the same intent via `torch.where(...)` / boolean masking keeps a **fixed shape**, stays
   on the device, and the `aten::` counters disappear.
3. **Accept (when it can't be fixed)** — `aten::nonzero` / `_local_scalar_dense` on genuinely
   *data-dependent shapes* are inherent: the result *size* depends on the data, which a static-shape
   accelerator can't produce on-device. The PyTorch/XLA docs treat these as expected. **The call:**
   if such an op runs once at setup, accept it; if it's per-step in the hot loop, it must be removed
   or the model rethought — no flag makes a dynamic-size op fast on a static-shape device.

Detection is documented in the PyTorch/XLA troubleshooting guide; deeper signals: `PT_XLA_DEBUG_LEVEL=2`,
and `NEURONX_DUMP_TO=./dir` for compiler logs. See [tools & debugging](../../docs/neuron_tools_and_debugging.md).

> **Confirmed on real hardware** (trn1.2xlarge, Neuron 2.30, torch-neuronx 2.9): the `--only fallback`
> demo showed `Counter: aten::nonzero` in `metrics_report()` for the `.nonzero()` path, and **0
> `aten::` counters** for the `torch.where` path — i.e. detection and fix both work as described.
