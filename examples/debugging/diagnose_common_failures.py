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

It is written to run two ways:
  * **On CPU (default, free):** demonstrates the *diagnosis logic* — it can reproduce the nan→finite
    contrast numerically (SDPA vs eager) on CPU, and explains what you'd see on hardware. No Neuron
    needed.
  * **On a Neuron instance:** add `--device xla` to also count real compilations and show the
    recompile-storm contrast live.

Run:
    python examples/debugging/diagnose_common_failures.py            # CPU walkthrough
    python examples/debugging/diagnose_common_failures.py --device xla   # on Trainium

See docs/neuron_tools_and_debugging.md for the tools referenced here, and
docs/trainium_development_best_practices.md for the underlying lessons.
"""

from __future__ import annotations

import argparse


def _device(requested: str):
    """Resolve a torch device; fall back to CPU off-hardware."""
    import torch

    if requested == "xla":
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device(), "xla"
        except ImportError:
            print("⚠️  torch_xla not present; running the CPU walkthrough instead.")
            return torch.device("cpu"), "cpu"
    return torch.device("cpu"), "cpu"


def demo_nan_vs_eager(device, backend) -> None:
    """Failure 1: a forward pass that nans, and the attention-implementation fix.

    On CPU in fp32 both implementations are finite; the nan is specifically the Neuron bf16 path.
    To make the *lesson* visible everywhere, we run a fully-masked-row attention forward — the
    pattern most likely to produce 0/0 → nan — under both attention implementations and report
    whether the loss is finite. On hardware (bf16) this is where SDPA diverges and eager holds.
    """
    import torch
    from transformers import AutoModelForTokenClassification

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

    vocab, n_labels, seqlen = 4, 3, 16
    ids = torch.randint(0, vocab, (2, seqlen))
    mask = torch.ones(2, seqlen, dtype=torch.long)
    labels = torch.zeros(2, seqlen, dtype=torch.long)

    for attn in ("sdpa", "eager"):
        try:
            model = AutoModelForTokenClassification.from_pretrained(
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
            loss = float(out.loss)
            verdict = "finite ✅" if loss == loss else "nan ❌"  # nan != nan
            print(f"  attn_implementation={attn:6s} -> loss={loss:.4f} ({verdict})")
        except Exception as exc:  # noqa: BLE001 - teaching script: report, don't crash
            print(
                f"  attn_implementation={attn:6s} -> raised {type(exc).__name__}: {str(exc)[:60]}"
            )

    print("\n  On real Trainium (bf16) we measured: sdpa -> nan, eager -> 1.13.")
    print("  The validated example therefore defaults to attn_implementation='eager'.")


def demo_recompile_storm(device, backend) -> None:
    """Failure 2: variable shapes cause a recompile per shape; fixed shapes compile ~once."""
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
        print(
            "  (CPU mode: no real compilation happens. The contrast below is what you'd see"
        )
        print(
            "   on hardware — run with --device xla on a Neuron instance to measure it.)"
        )
        print(
            "   ragged batches (drop_last=False): a NEW graph per distinct batch shape (7+)."
        )
        print("   fixed batches (drop_last=True):   ONE graph, reused (~1-2).")
        return

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


def demo_cpu_fallback(device, backend) -> None:
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
    """
    print("\n" + "=" * 70)
    print("FAILURE 3: silent CPU fallback  (symptom: slow; no error; device underused)")
    print("=" * 70)
    print("Detector (the key tool):  torch_xla.debug.metrics.metrics_report()")
    print(
        "  -> ANY counter named 'aten::<op>' = that op ran on CPU, not the NeuronCore.\n"
    )

    if backend != "xla":
        print(
            "  (CPU mode: there's no device to fall back FROM, so the metrics report won't show"
        )
        print(
            "   the contrast. Run with --device xla on a Neuron instance to see it live.)"
        )
        print("\n  What you'd see on hardware:")
        print(
            "   BAD  (mask via .nonzero()/.item()):  metrics_report() lists  aten::nonzero ,"
        )
        print("        aten::_local_scalar_dense  -> each step round-trips to CPU.")
        print(
            "   GOOD (mask via torch.where / boolean mul): no aten:: counters; stays on device."
        )
        print("\n  Fix recipe:")
        print(
            "   • Replace data-dependent indexing (x[x>0], .nonzero(), .item()) with vectorized"
        )
        print(
            "     ops that keep a FIXED shape: torch.where(cond, a, b), cond.float()*x, masking."
        )
        print(
            "   • Avoid .item() in the loop — it forces a CPU sync (it IS a fallback trigger)."
        )
        print("\n  When you CAN'T fix it (and how to know):")
        print(
            "   • aten::nonzero / aten::_local_scalar_dense on genuinely dynamic-shape problems are"
        )
        print(
            "     inherent — the result SIZE depends on data, which the static-shape device can't do."
        )
        print("     The PyTorch/XLA docs explicitly treat these as expected.")
        print(
            "   • Decision: if the fallback op runs ONCE at setup, accept it. If it's per-step in"
        )
        print(
            "     the hot loop, it must be removed or the model rethought — there's no flag that"
        )
        print("     makes a dynamic-size op fast on a static-shape accelerator.")
        return

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


def main(argv: list[str] | None = None) -> int:
    """Run the debugging walkthrough."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "xla"], default="cpu")
    p.add_argument(
        "--only",
        choices=["nan", "recompile", "fallback"],
        help="Run just one demo.",
    )
    args = p.parse_args(argv)

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
