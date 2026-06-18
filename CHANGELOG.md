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
- Satellite land-cover CV example (`examples/use_cases/satellite_landcover.py`, +README) — EuroSAT
  10-class residual CNN with a CPU smoke path, the `run(config)` harness contract, and an optional
  `torchrun` data-parallel mode.
- **`examples/use_cases/cv_utilization_spike.py` (+README) — *measures* Trainium under-utilization.**
  A controlled CNN-vs-ViT experiment (same device/input/batch) reporting achieved TFLOP/s.
  **Measured on a real trn1.2xlarge: the ViT achieves 5.1× the small CNN's TFLOP/s while doing 2.7×
  fewer FLOPs (CNN 1.07 vs ViT 5.51 TFLOP/s; 307 ms vs 22 ms/step)** — proving a small CNN starves the
  128×128 systolic array. On CPU the ordering flips, isolating the cause to the hardware's shape. This
  turns "build in the form the hardware wants" from advice into a number.

### Changed
- **Qwen3-8B LoRA validated through a full epoch on trn1.32xlarge (32 NeuronCores).** Loss 1.93→1.43,
  steady-state ~5 s/step / ~13.5k tok/s / MFU ~29%. Documented the measured **per-step compile decay**
  (step 1 ~119 s → step 4+ ~5 s, a ~20-40× cliff) and that a warm S3 cache removes the cold phase on
  re-run, plus three real gotchas (32-rank HF-hub download race → pre-fetch once; `HF_HUB_OFFLINE`
  breaks the post-train hub-cache sync; `TrainOutput.training_loss` reports `nan` on XLA so the metric
  is now derived from `trainer.state.log_history`). Best-practices §1c gained a "slow first step is
  compilation" subsection.
- **Compile-cache guidance promoted for the cloud workflow** (best-practices §1b): on a fresh
  instance a *local* cache is always cold, so an **S3** `NEURON_COMPILE_CACHE_URL` is what actually
  saves the recompile tax across reprovisions. Added compiler-version pinning + cache-key caveats, and
  wired a `--cache-url` flag into the validation harness (exports the env on the launched instance).
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
