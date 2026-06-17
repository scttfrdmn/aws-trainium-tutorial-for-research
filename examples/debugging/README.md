# Debugging walkthrough

A runnable companion to [docs/neuron_tools_and_debugging.md](../../docs/neuron_tools_and_debugging.md).
It reproduces the two classic Trainium failures — on purpose — and shows the diagnosis and fix for
each. Both were hit for real while building the [validated NER example](../use_cases/biomedical_ner.py).

> **Assumed knowledge:** you can run a Python script and have read the
> [best-practices chapter](../../docs/trainium_development_best_practices.md).
> **What you'll get:** hands-on recognition of (1) the bf16 SDPA→`nan` forward-pass failure and
> (2) the variable-shape recompile storm — and the one-line fix for each.

## Run it

```bash
# CPU walkthrough (free; explains what you'd see on hardware):
python examples/debugging/diagnose_common_failures.py

# On a Neuron instance — also counts real compilations:
python examples/debugging/diagnose_common_failures.py --device xla

# Just one demo:
python examples/debugging/diagnose_common_failures.py --only nan
python examples/debugging/diagnose_common_failures.py --only recompile
```

## What it teaches

| Failure | Symptom | Diagnosis | Fix |
|---------|---------|-----------|-----|
| **bf16 SDPA → nan** | `nan` loss at **step 0** (forward) | Same model+data on CPU/fp32 is finite ⇒ it's the bf16 path, not your LR | `attn_implementation="eager"` (keeps bf16; **not** `--auto-cast=none`) |
| **Recompile storm** | "stuck", host CPU busy, cores idle, no step progress | `grep -c "Compilation Successfully Completed"` ≫ a handful | fixed shapes: `drop_last=True` + fixed `max_length` |

Neither is a toy: on real `trn1.2xlarge` we measured SDPA→`nan` vs eager→loss 1.13, and ragged
shapes → 7+ compiles vs fixed → ~2. The validated example bakes in both fixes.
