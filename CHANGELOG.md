# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Project work (milestones, issues, labels) is tracked on
[GitHub](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research) — not in local files.

## [Unreleased]

### Added
- Hardware-validation harness (`validation/`): provenance capture, example registry,
  `spawn`/`awscli` launcher with auto-terminate + cost ceiling, and `VALIDATED.md` status
  rendering. ([#1])
- Research-grade biomedical NER fine-tune example (`examples/use_cases/biomedical_ner.py`) on the
  real NCBI-disease corpus, with entity-level F1 and the `run(config)` harness contract.
- `docs/trainium_development_best_practices.md` — "build in the form the hardware wants",
  grounded in findings reproduced on a real trn1.2xlarge:
  - HF v5 default **SDPA attention produces `nan` on the Neuron bf16 path**; `attn_implementation=
    "eager"` fixes it (loss 1.13, matching CPU fp32) **without disabling bf16**. ([#5])
  - Fixed batch shapes (`drop_last`) compile in ~2 graphs vs 7+ for ragged shapes.
- `CHANGELOG.md` (Keep a Changelog) and SemVer 2.0.0 adoption.
- GitHub project structure: milestones (v0.1.0 → v1.0.0), `type:`/`area:`/`status:` label
  taxonomy, and tracking issues.
- **Learner-expectations framing** throughout — main tutorial and each chapter/example open with
  "assumed knowledge / what you'll be able to do".
- `docs/choose_your_path.md` — decision guide mapping research domains and problem-shapes to
  Trainium fit (with honest "use a GPU instead" verdicts) and a starting example.
- `docs/neuron_tools_and_debugging.md` — Neuron tools/profiling/tracing/simulator chapter
  (neuron-ls/top/monitor, Neuron Explorer, torch_xla profiler, NKI simulation, compile cache),
  with a symptom→tool table. Public sources; thin areas flagged.
- `examples/debugging/diagnose_common_failures.py` (+README) — runnable reproduction of the
  bf16 SDPA→`nan` and recompile-storm failures, with diagnosis + fix.
- `docs/novel_kernels_on_trainium.md` — Trainium architecture (quoted from the public NKI
  architecture guide) + a "does your problem map?" framework + the novel-kernel thesis: a
  **better numerical result** via FP32-resident fusion (PSUM always accumulates FP32), not just speed.
- `examples/distributed/data_parallel_ner.py` (+README) — real multi-NeuronCore data-parallel
  training (torchrun + XLA DDP), demonstrating gradient all-reduce and Neuron-correct checkpointing.

### Changed
- Tutorial refreshed for **June 2026**: Neuron SDK 2.30.0, PyTorch 2.9 (XLA path), a brief
  public note that 2.9 is the last XLA version, and Trainium-for-inference positioning.
- Tooling modernized to **ruff** (lint+format), **uv**, pinned Python (3.12), and CI.
- Fixed on-ramp blockers: README/quick-start/local-setup dead links, wrong script names, and
  stale versions reconciled to the uv + Neuron 2.30 / PyTorch 2.9 path.

### Removed
- Redundant `setup.py` (consolidated into `pyproject.toml`).
- Toy/synthetic examples (easter-eggs, `np.random` "genomics"/"computer vision"); remaining
  mock/stale/illustrative examples quarantined to `examples/_legacy/` with an honest banner.

[Unreleased]: https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/compare/main...HEAD
[#1]: https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/issues/1
