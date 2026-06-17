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

### Changed
- Tutorial refreshed for **June 2026**: Neuron SDK 2.30.0, PyTorch 2.9 (XLA path), a brief
  public note that 2.9 is the last XLA version, and Trainium-for-inference positioning.
- Tooling modernized to **ruff** (lint+format), **uv**, pinned Python (3.12), and CI.

### Removed
- Redundant `setup.py` (consolidated into `pyproject.toml`).

[Unreleased]: https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/compare/main...HEAD
[#1]: https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/issues/1
