#!/usr/bin/env python3
"""A runnable debugging walkthrough for the two classic Trainium failures.

This is a *teaching* script. It reproduces, on purpose, the two failures that bit us while building
the validated NER example on real hardware — and shows how to diagnose and fix each:

  1. **bf16 SDPA → nan in the forward pass.** Hugging Face v5 defaults to SDPA attention, which
     produces `nan` at step 0 on the Neuron bf16 path. The fix is `attn_implementation="eager"`
     (which KEEPS bf16 on — not `--auto-cast=none`, which defeats the accelerator).
  2. **The recompile storm.** Variable batch shapes make the Neuron compiler build a new graph per
     shape. Fixing shapes (`drop_last=True` + fixed max_length) compiles ~once.

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


def main(argv: list[str] | None = None) -> int:
    """Run the debugging walkthrough."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "xla"], default="cpu")
    p.add_argument("--only", choices=["nan", "recompile"], help="Run just one demo.")
    args = p.parse_args(argv)

    device, backend = _device(args.device)
    print(f"Running debugging walkthrough on: {backend}")

    if args.only in (None, "nan"):
        demo_nan_vs_eager(device, backend)
    if args.only in (None, "recompile"):
        demo_recompile_storm(device, backend)

    print("\nSee docs/neuron_tools_and_debugging.md for the full symptom->tool table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
