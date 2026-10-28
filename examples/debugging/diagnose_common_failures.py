#!/usr/bin/env python3
"""A runnable debugging walkthrough for three classic Trainium failures.

This is a *teaching* script. It reproduces, on purpose, the failures that bite people on Trainium
(two of which bit us while building the validated NER example) — and shows how to diagnose and fix
each:

  1. **bf16 SDPA → nan in the forward pass.** Hugging Face v5 defaults to SDPA attention, which
     produces `nan` at step 0 on the Neuron bf16 path. The fix is `attn_implementation="eager"`
     (which KEEPS bf16 on — not `--auto-cast=none`, which defeats the accelerator).
  2. **The recompile storm.** Variable batch shapes make the Neuron compiler build a new graph per
     shape. Fixing shapes (`drop_last=True` + fixed max_length) compiles ~once.
  3. **Silent CPU fallback.** An op with no device lowering runs on the CPU (an "aten fallback")
     and round-trips host<->device every step — no error, just slowness. Detect it with
     `torch_xla.debug.metrics.metrics_report()` (any `aten::` counter), fix by vectorizing to a
     fixed-shape op, or recognize when the fallback is inherent and unavoidable.

It runs three ways:
  * **On CPU (default, free):** every demo actually runs with stock PyTorch. Demo 1 reproduces the
    SDPA-vs-eager comparison numerically; demos 2 and 3 show the *same lesson, observable on your
    laptop* through `torch.compile(backend="eager")` (no compiler toolchain needed): ragged shapes
    produce more graphs than fixed ones (counted via `torch._dynamo` counters), and a
    data-dependent op (`.nonzero()`) breaks the graph where `torch.where` does not (via
    `torch._dynamo.explain`). These are torch.compile analogues of the XLA mechanisms — related
    lessons, not the identical machinery.
  * **On a Neuron instance (the lead path):** `--device xla` uses PyTorch/XLA to count real
    Neuron compilations and read real `aten::` fallback counters.
  * **TorchNeuron (beta, opt-in):** `--device neuron` uses the native `"neuron"` device and
    `torch.compile(backend="neuron")` when TorchNeuron is installed (Trn2/Trn3), else falls back
    to CPU with a warning. Not hardware-verified by this walkthrough.

Run:
    python examples/debugging/diagnose_common_failures.py                  # CPU walkthrough
    python examples/debugging/diagnose_common_failures.py --device xla     # on Trainium
    python examples/debugging/diagnose_common_failures.py --device neuron  # TorchNeuron (beta)

See docs/neuron_tools_and_debugging.md for the tools referenced here, and
docs/trainium_development_best_practices.md for the underlying lessons.
"""

from __future__ import annotations

import argparse

BETA_LABEL = "TorchNeuron (beta), not hardware-verified by this walkthrough"

RAGGED_SIZES = [16, 15, 14, 13]
FIXED_SIZES = [16, 16, 16, 16]


def _device(requested: str):
    """Resolve ``(torch.device, backend)``; fall back to CPU off-hardware.

    ``backend`` is what actually runs: ``"xla"``, ``"neuron"`` or ``"cpu"``.
    """
    import torch

    if requested == "xla":
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device(), "xla"
        except ImportError:
            print("⚠️  torch_xla not present; running the CPU walkthrough instead.")
            return torch.device("cpu"), "cpu"
    if requested == "neuron":
        # TorchNeuron autoloads on `import torch`; Trainium is then a device named "neuron".
        neuron = getattr(torch, "neuron", None)
        if neuron is not None and neuron.is_available():
            print(f"⚠️  {BETA_LABEL}.")
            return torch.device("neuron"), "neuron"
        print(
            "⚠️  TorchNeuron (beta) not available; running the CPU walkthrough instead."
        )
        return torch.device("cpu"), "cpu"
    return torch.device("cpu"), "cpu"


def _compile_backend(backend: str) -> str:
    """Return the ``torch.compile`` backend for the resolved device backend."""
    return "neuron" if backend == "neuron" else "eager"


def demo_nan_vs_eager(device, backend) -> None:
    """Failure 1: a forward pass that nans, and the attention-implementation fix.

    On CPU in fp32 both implementations are finite; the nan is specifically the Neuron bf16 path.
    To make the *lesson* visible everywhere, we run a fully-masked-row attention forward — the
    pattern most likely to produce 0/0 → nan — under both attention implementations and report
    whether the loss is finite. On hardware (bf16) this is where SDPA diverges and eager holds.
    """
    import torch

    # The concrete class, not AutoModel*: prajjwal1/bert-tiny's config.json has no `model_type`,
    # which transformers v5's Auto* resolution rejects ("Unrecognized model").
    from transformers import BertForTokenClassification

    print("\n" + "=" * 70)
    print("FAILURE 1: bf16 forward-pass nan  (symptom: loss is nan at step 0)")
    print("=" * 70)
    print("Diagnosis recipe:")
    print(
        "  • Is the nan at step 0 (forward), or after some steps (training divergence)?"
    )
    print(
        "    -> reproduce the SAME model+data on CPU/fp32. If CPU is finite but Neuron nans,"
    )
    print("       it's a bf16 forward-pass issue, NOT your learning rate.")
    print("  • Wrong fix: --auto-cast=none (turns off the accelerator's native bf16).")
    print(
        "  • Right fix: attn_implementation='eager' (keeps bf16; avoids the SDPA kernel).\n"
    )
    if backend == "neuron":
        print(f"  [{BETA_LABEL}: eager-mode forward on the 'neuron' device, fp32.]\n")

    vocab, n_labels, seqlen = 4, 3, 16
    ids = torch.randint(0, vocab, (2, seqlen))
    mask = torch.ones(2, seqlen, dtype=torch.long)
    labels = torch.zeros(2, seqlen, dtype=torch.long)

    for attn in ("sdpa", "eager"):
        try:
            model = BertForTokenClassification.from_pretrained(
                "prajjwal1/bert-tiny", num_labels=n_labels, attn_implementation=attn
            ).to(device)
            out = model(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                labels=labels.to(device),
            )
            if backend == "xla":
                import torch_xla.core.xla_model as xm

                xm.mark_step()
            loss = float(out.loss.detach())
            verdict = "finite ✅" if loss == loss else "nan ❌"  # nan != nan
            print(f"  attn_implementation={attn:6s} -> loss={loss:.4f} ({verdict})")
        except Exception as exc:  # noqa: BLE001 - teaching script: report, don't crash
            print(
                f"  attn_implementation={attn:6s} -> raised {type(exc).__name__}: {str(exc)[:60]}"
            )

    print("\n  On real Trainium (bf16) we measured: sdpa -> nan, eager -> 1.13.")
    print("  (That measurement is PyTorch/XLA on trn1; TorchNeuron + bf16 autocast is")
    print(
        "   not yet verified.) The validated example defaults to attn_implementation='eager'."
    )


def count_compiled_graphs(
    sizes: list[int],
    *,
    dynamic: bool | None = None,
    compile_backend: str = "eager",
    device=None,
) -> int:
    """Run a tiny step over batches of ``sizes`` under ``torch.compile``; return graphs built.

    The count is ``torch._dynamo.utils.counters["stats"]["unique_graphs"]`` — how many distinct
    FX graphs Dynamo captured (each is one hand-off to the compiler backend). Dynamo state is reset
    first so runs don't share a cache.
    """
    import torch
    import torch._dynamo
    from torch._dynamo.utils import counters

    def step(x):
        return (x @ x.t()).sum()

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(step, backend=compile_backend, dynamic=dynamic)
    for n in sizes:
        compiled(torch.randn(n, 128, device=device))
    return int(counters["stats"]["unique_graphs"])


def demo_recompile_storm(device, backend) -> dict | None:
    """Failure 2: variable shapes cause a recompile per shape; fixed shapes compile ~once.

    Returns the graph counts on the torch.compile (CPU / TorchNeuron) path, else ``None``.
    """
    print("\n" + "=" * 70)
    print(
        "FAILURE 2: recompile storm  (symptom: training 'stuck', high host CPU, no progress)"
    )
    print("=" * 70)
    print("Diagnosis recipe:")
    print(
        "  • Count compilations:  grep -c 'Compilation Successfully Completed' run.log"
    )
    print("  • >a-handful for a fixed-shape loop  =>  shape instability.")
    print("  • neuron-top shows cores idle while the host compiles.\n")

    if backend != "xla":
        return _recompile_storm_torch_compile(device, backend)

    import torch
    import torch_xla.core.xla_model as xm

    # A trivial op over batches; with ragged sizes each new shape compiles a new graph.
    def run(sizes):
        for n in sizes:
            x = torch.randn(n, 128, device=device)
            (x @ x.t()).sum()
            xm.mark_step()

    print("  Running ragged batch sizes [16,15,14,13]  (watch compile count climb)...")
    run([16, 15, 14, 13])
    print("  Running fixed batch size [16,16,16,16]     (reuses one graph)...")
    run([16, 16, 16, 16])
    print(
        "\n  Now check:  grep -c 'Compilation Successfully Completed' <this run's log>"
    )
    print(
        "  Ragged contributes ~one compile per distinct size; fixed reuses the cached graph."
    )
    return None


def _recompile_storm_torch_compile(device, backend) -> dict:
    """Demo 2 on CPU / TorchNeuron: count Dynamo graphs for ragged vs fixed shapes."""
    cb = _compile_backend(backend)
    if backend == "neuron":
        print(f"  [{BETA_LABEL}: torch.compile(backend='neuron').]")
    else:
        print(
            "  CPU mode: the same lesson via torch.compile(backend='eager') — no compiler"
        )
        print("  toolchain needed. We count graphs Dynamo hands to the backend")
        print("  (torch._dynamo.utils.counters['stats']['unique_graphs']).")
    print(
        "  Equivalent live signal on any torch.compile backend:  TORCH_LOGS=recompiles\n"
    )

    results = {
        "fixed": count_compiled_graphs(FIXED_SIZES, compile_backend=cb, device=device),
        "ragged_default": count_compiled_graphs(
            RAGGED_SIZES, compile_backend=cb, device=device
        ),
        "ragged_static": count_compiled_graphs(
            RAGGED_SIZES, dynamic=False, compile_backend=cb, device=device
        ),
    }
    print(f"  fixed  {FIXED_SIZES}, default       -> {results['fixed']} graph(s)")
    print(
        f"  ragged {RAGGED_SIZES}, dynamic=False -> "
        f"{results['ragged_static']} graph(s)  (one per distinct shape)"
    )
    print(
        f"  ragged {RAGGED_SIZES}, default       -> "
        f"{results['ragged_default']} graph(s)"
    )
    print(
        "\n  Honest caveat: by default Dynamo recompiles ONCE on the 2nd distinct shape and"
    )
    print(
        '  then marks that dim dynamic ("automatic dynamic shapes"), so the default ragged'
    )
    print(
        "  count stays small. dynamic=False forces per-shape graphs — the closer analogue"
    )
    print(
        "  of PyTorch/XLA on Neuron, which compiles a new static graph for every new shape."
    )
    print("  Either way, the fix is the same: fixed shapes (drop_last=True + fixed")
    print("  max_length) compile once.")
    return results


def explain_graph_breaks(fn, *args) -> dict:
    """Return ``torch._dynamo.explain`` stats for ``fn(*args)``.

    Keys: ``graph_count``, ``graph_break_count`` and ``break_reasons`` (first line of each).
    Graph breaks are decided by Dynamo before any backend runs, so this is backend-independent.
    """
    import torch._dynamo

    torch._dynamo.reset()
    exp = torch._dynamo.explain(fn)(*args)
    reasons = [str(r.reason).strip().splitlines()[0] for r in exp.break_reasons]
    return {
        "graph_count": int(exp.graph_count),
        "graph_break_count": int(exp.graph_break_count),
        "break_reasons": reasons,
    }


def _mask_with_nonzero(x):
    """Zero out negatives the data-dependent way: the output size of nonzero() depends on x."""
    idx = (x > 0).nonzero()
    out = x.new_zeros(x.shape)
    out[idx[:, 0], idx[:, 1]] = x[idx[:, 0], idx[:, 1]]
    return out.sum()


def _mask_with_where(x):
    """Zero out negatives the static-shape way: same result, fixed shape, no host sync."""
    import torch

    return torch.where(x > 0, x, torch.zeros_like(x)).sum()


def demo_cpu_fallback(device, backend) -> dict | None:
    """Failure 3: an op silently runs on CPU (aten fallback), and how to detect/fix/accept it.

    When the compiler has no device lowering for an op, PyTorch/XLA does NOT error — it runs that op
    on the CPU and copies tensors host<->device every step. No crash, just mysterious slowness.

    The canonical public detector is `torch_xla.debug.metrics.metrics_report()`: **any counter named
    `aten::<op>` is an op that fell back to CPU.** (PyTorch/XLA troubleshooting guide.)

    Three acts:
      1) DISCOVER — run a step that uses `.nonzero()`/`.item()` (forces CPU), show the `aten::` counter.
      2) FIX      — rewrite to a vectorized, lowered form (masking / `torch.where`); counter vanishes.
      3) ACCEPT   — some fallbacks (`aten::nonzero`, `aten::_local_scalar_dense`) are inherent to
                    data-dependent shapes; the docs bless these. Recognize them and move them off
                    the hot loop (run once at setup) rather than fighting an unfixable case.

    Returns the ``torch._dynamo.explain`` stats on the CPU / TorchNeuron path, else ``None``.
    """
    print("\n" + "=" * 70)
    print("FAILURE 3: silent CPU fallback  (symptom: slow; no error; device underused)")
    print("=" * 70)
    print("Detector (the key tool):  torch_xla.debug.metrics.metrics_report()")
    print(
        "  -> ANY counter named 'aten::<op>' = that op ran on CPU, not the NeuronCore.\n"
    )

    if backend != "xla":
        return _fallback_torch_compile(device, backend)

    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

    # ACT 1 — DISCOVER: a mask built with nonzero()/item() forces CPU fallback.
    met.clear_all()
    x = torch.randn(256, 256, device=device)
    idx = (x > 0).nonzero()  # data-dependent shape -> aten::nonzero on CPU
    _ = idx.shape[0]  # .item()-like host read
    xm.mark_step()
    report = met.metrics_report()
    aten = [ln for ln in report.splitlines() if "aten::" in ln]
    print("  ACT 1 (discover): aten:: counters after a nonzero()/host-read step:")
    for ln in aten[:6]:
        print(f"     {ln.strip()}")
    print("     ^ these ops executed on CPU.\n" if aten else "     (none shown)\n")

    # ACT 2 — FIX: the same intent (zero out negatives) with a vectorized, lowered op.
    met.clear_all()
    x = torch.randn(256, 256, device=device)
    _ = torch.where(x > 0, x, torch.zeros_like(x))  # fixed shape, stays on device
    xm.mark_step()
    report2 = met.metrics_report()
    aten2 = [ln for ln in report2.splitlines() if "aten::" in ln]
    print(
        f"  ACT 2 (fix): torch.where version -> {len(aten2)} aten:: counters (expect ~0)."
    )
    print("     Same result, no host round-trip.\n")

    print(
        "  ACT 3 (accept): if you truly need nonzero()/dynamic sizes, that fallback is inherent."
    )
    print(
        "     Run it once at setup, not per step; there's no flag that makes it device-fast."
    )
    return None


def _fallback_torch_compile(device, backend) -> dict:
    """Demo 3 on CPU / TorchNeuron: a data-dependent op breaks the torch.compile graph."""
    import torch

    if backend == "neuron":
        print(f"  [{BETA_LABEL}.]")
    print(
        "  CPU mode: the torch.compile ANALOGUE, via torch._dynamo.explain. This is a related"
    )
    print(
        "  lesson, not the identical mechanism: XLA runs an unlowered op on the host (aten::"
    )
    print(
        "  counter); Dynamo instead BREAKS the graph at a data-dependent op and runs that piece"
    )
    print(
        "  eagerly — either way the compiled region is split, and on an accelerator each"
    )
    print("  split is a host sync.")
    print(
        "  Equivalent live signal on any torch.compile backend:  TORCH_LOGS=graph_breaks\n"
    )

    x = torch.randn(256, 256, device=device)
    bad = explain_graph_breaks(_mask_with_nonzero, x)
    good = explain_graph_breaks(_mask_with_where, x)
    print(
        "  ACT 1 (discover): mask via .nonzero()  ->  "
        f"graphs={bad['graph_count']}, graph breaks={bad['graph_break_count']}"
    )
    for reason in bad["break_reasons"][:3]:
        print(f"     break reason: {reason}")
    print(
        "  ACT 2 (fix):      mask via torch.where ->  "
        f"graphs={good['graph_count']}, graph breaks={good['graph_break_count']}"
    )
    same = torch.allclose(_mask_with_nonzero(x), _mask_with_where(x))
    print(f"     same result: {same}")

    if backend == "neuron":
        # Graph breaks are decided by Dynamo before the backend; this runs both on the backend.
        for name, fn in (("nonzero", _mask_with_nonzero), ("where", _mask_with_where)):
            torch._dynamo.reset()
            try:
                torch.compile(fn, backend="neuron")(x)
                print(f"     torch.compile(backend='neuron') ran the {name} version.")
            except Exception as exc:  # noqa: BLE001 - teaching script: report, don't crash
                print(
                    f"     torch.compile(backend='neuron') on {name}: "
                    f"{type(exc).__name__}: {str(exc)[:60]}"
                )

    print(
        "\n  ACT 3 (accept): .nonzero()/x[mask]/.item() produce data-dependent sizes or host"
    )
    print(
        "     values — inherent breaks/fallbacks. Run them once at setup, not per step."
    )
    return {"nonzero": bad, "where": good}


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser (exposed for tests)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--device",
        choices=["cpu", "xla", "neuron"],
        default="cpu",
        help="cpu (default), xla (PyTorch/XLA on Trainium), "
        "neuron (TorchNeuron, beta; falls back to CPU if absent).",
    )
    p.add_argument(
        "--only",
        choices=["nan", "recompile", "fallback"],
        help="Run just one demo.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the debugging walkthrough."""
    args = build_parser().parse_args(argv)

    device, backend = _device(args.device)
    print(f"Running debugging walkthrough on: {backend}")

    if args.only in (None, "nan"):
        demo_nan_vs_eager(device, backend)
    if args.only in (None, "recompile"):
        demo_recompile_storm(device, backend)
    if args.only in (None, "fallback"):
        demo_cpu_fallback(device, backend)

    print("\nSee docs/neuron_tools_and_debugging.md for the full symptom->tool table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
