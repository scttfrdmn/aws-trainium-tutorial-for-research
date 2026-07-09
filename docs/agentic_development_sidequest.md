# Side-quest: AWS's agentic Neuron tooling (Claude Code / Kiro)

> **Assumed knowledge:** you've worked through some of this tutorial — enough to know what NKI
> kernels, the compile step, profiling, and "port a GPU model to Neuron" mean. You're using an AI
> coding assistant (**Claude Code** or **Kiro**).
> **What you'll get:** AWS's *official* agents + skills for Neuron development, installed into your
> assistant — so the manual workflows this tutorial teaches get an AI accelerator.

This is an **optional side-quest**, not part of the core 2.9/XLA path. It's here because much of this
tutorial's audience is reading it *inside Claude Code right now*, and AWS ships tooling that plugs
straight into that.

---

## What it is

[`aws-neuron/neuron-agentic-development`](https://github.com/aws-neuron/neuron-agentic-development)
(Apache-2.0) is an official AWS repo that installs **agents** and **skills** into Claude Code or Kiro.
Rather than a chat you copy-paste from, these are structured agent definitions your assistant can
invoke — for writing NKI kernels, debugging compile errors, profiling, and porting models.

> **Honesty note:** this is a pointer to an external, actively-evolving AWS repo — not something this
> tutorial has hardware-validated. Treat the agent/skill names below as current-as-of-writing; check
> the repo's README and the
> [Neuron agentic-development overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/agentic-development-overview.html)
> for the live list and any setup changes. It targets **Claude Code** and **Kiro** (no Cursor path as
> of writing).

## Install

```bash
# 1. Install the package (from the Neuron pip index)
pip install --upgrade neuron-agentic-development \
    --extra-index-url https://pip.repos.neuron.amazonaws.com

# 2. Deploy the agents + skills into your assistant
deploy-neuron-agentic-development-to-claude     # for Claude Code
# or:
deploy-neuron-agentic-development-to-kiro        # for Kiro
```

The deploy step writes the agent/skill definitions into your assistant's config so they're available
in your sessions. (The repo also supports install from a wheel or a local `git clone && pip install .`)

## What it gives you — and where it maps to this tutorial

The agentic toolkit automates, with an AI in the loop, several workflows this tutorial teaches you to
do **by hand**. Do the manual version first (so you understand what the agent is doing), then let the
agent accelerate the repetitive parts.

| Agent / skill | What it does | This tutorial's manual version |
|---|---|---|
| `neuron-nki-writing`, `neuron-nki-writer-agent` | Write/modify NKI kernels from PyTorch / NumPy / natural language | [Novel kernels on Trainium](novel_kernels_on_trainium.md) |
| `neuron-nki-debugging`, `neuron-nki-debugger-agent` | Autonomously debug NKI **compilation errors** | [Neuron tools & debugging](neuron_tools_and_debugging.md) (symptom→tool table) |
| `neuron-nki-profiling`, `neuron-nki-profile-analysis-agent` | Profile NKI kernels on hardware; find bottlenecks | the profiler/Neuron-Explorer section of [tools & debugging](neuron_tools_and_debugging.md) |
| `neuron-nki-profile-querying` | Query Neuron Explorer profile data (parquet) via SQL / Python | same — turns a profile into queryable data |
| `neuron-framework-autoport`, `neuron-framework-autoport-agent` | Port a GPU-compatible model to NeuronX Distributed Inference | the CUDA→Neuron migration chapter in the [full tutorial](main_tutorial_doc.md) |
| `neuron-framework-equivalence` | Verify functional equivalence between two model implementations | our recurring habit: *does the Neuron result match CPU/fp32?* (e.g. the [NER bf16→`nan` fix](../examples/use_cases/biomedical_ner.py)) |
| `neuron-nki-docs` | Research NKI documentation / APIs on demand | replaces manual doc-diving |

## When to reach for it

- **You've hit the NKI chapter and want to actually write a kernel.** The writer/debugger agents turn
  "stare at the `neuronxcc.nki.language` API" into an iterative loop with the compiler in the feedback
  path. You still need to understand tiles / SBUF / PSUM (that's what our
  [novel kernels chapter](novel_kernels_on_trainium.md) is for) — the agent handles the boilerplate
  and the compile-error grind.
- **You're porting a real GPU model.** The autoport + equivalence agents automate the
  "port → compile → run → check it matches" cycle we do manually.
- **You're profiling.** The profile-querying skill (SQL over Neuron Explorer parquet) is a big step up
  from eyeballing a timeline.

## When *not* to

- **Learning the fundamentals.** If you let the agent write your first kernel, you'll skip the mental
  model that tells you *why* a kernel is slow or wrong. Do the manual version once first.
- **This tutorial's core examples.** The 2.9/XLA training examples here don't need it — they're
  plain PyTorch/XLA. The agentic toolkit shines on **NKI** and **porting**, not on `xm.mark_step()`
  training loops.

---

> **Where this fits:** a capstone. You've learned the Trainium model by hand throughout this tutorial;
> this is AWS's official way to put an AI coding assistant to work on the same tasks. Start from the
> [novel kernels chapter](novel_kernels_on_trainium.md) if NKI is your goal, or the
> [tools & debugging chapter](neuron_tools_and_debugging.md) for profiling.
