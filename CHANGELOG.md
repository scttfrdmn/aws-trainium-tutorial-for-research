# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Project work (milestones, issues, labels) is tracked on
[GitHub](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research) — not in local files.

## [Unreleased]

### Validated (hardware, us-west-2 with S3 compile cache)
- **5/6 examples now hardware-validated.** Re-ran in us-west-2 with `NEURON_COMPILE_CACHE_URL` on S3
  (the cold-compile cost — not bad hosts — was what made earlier runs look stuck; the cache is the
  fix the tutorial itself teaches). Fresh artifacts: **cv_utilization_spike** ViT **5.18× the CNN's
  TFLOP/s** (reproduced 5.1×; CNN 1.069 / ViT 5.535 TFLOP/s), **crystal_cif_slm** perplexity 1.735
  with the new **`validity_rate` measured at 0.0%** — an honest result (a 1-epoch char-GPT learns CIF
  syntax but not yet parseable structures; exactly why validity is reported, not gated). The RODA
  **satellite** example reads real Sentinel-2+WorldCover patches and trains on Trainium, but a clean
  auto-validated run is still pending: on a fresh lock-free cache prefix, its residual-CNN training
  graph took **≥14 min to compile on a single trn1.2xlarge and didn't finish** in the window (a slow
  compile of this multi-ResBlock+BatchNorm graph on 8 vCPUs — not a cache lock, not a confirmed Neuron
  defect). Consistent with the example's own lesson that a small-conv CNN isn't the shape the array
  wants. Re-try options: compile on trn1.32xlarge, `neuron_parallel_compile`, or a ViT-shaped model.
  (Earlier in this work a separate **stale-S3-cache-lock** failure mode — killing a run mid-compile
  leaves a `.lock` with no `.neff` and wedges the next run — was found and is documented; fix is a
  per-run cache prefix. Eval was also restored to the Trainium device, not CPU.)

### Changed (RODA open data)
- **`satellite_landcover.py` now builds its training set from the AWS Registry of Open Data**, not a
  pre-tiled Hugging Face benchmark. It reads **Sentinel-2 L2A** COGs (`s3://sentinel-cogs`) + **ESA
  WorldCover** label rasters (`s3://esa-worldcover`) anonymously via `rasterio`, co-registers them,
  tiles into 64×64 patches, and labels each by majority WorldCover class — real `(image → land-cover)`
  pairs from live open data. Harder than EuroSAT (majority-label noise), so the gate is 0.60 (CPU
  smoke already hits ~0.77). The CNN/training/utilization-lesson code is unchanged.
- Added an honest **"why Hugging Face, not RODA?"** note to the other five examples — RODA is
  geospatial/genomics/climate-centric and has no NLP/protein/materials/LLM corpus, so only the
  geospatial example converts. (The earlier fake `aws_open_data.py` that *pretended* to serve RODA
  was deleted in the audit; this is the real thing.)
- `rasterio` + `pymatgen` added to the `science` extra.

### Added (crystal validity + bookkeeping)
- **Crystal-CIF example now reports a pymatgen-based `validity_rate`** — the fraction of generated
  CIFs that parse into a real structure. Reported, NOT gated (validity on a small/short run is
  noisy; perplexity stays the gate). pymatgen is optional — absent → `validity_rate = -1.0` and the
  run still works. Unit-tested locally (0.5 on a valid+garbage CIF pair); first *hardware* artifact
  pending a clean re-run (the trn1.2xlarge spot hosts drawn this round were environmentally degraded —
  Neuron compiler not progressing — so the re-validation was deferred rather than forced).
- README gained an at-a-glance hardware-validation pointer to `VALIDATED.md`.
- Closed GitHub issues #2 (Qwen3 LoRA — shipped + validated) and #5 (NER bf16 exemplar — shipped).

### Added (SLM / "build a small model" track)
- **All three new SLM examples are now hardware-validated on trn1.2xlarge** (Neuron 2.30, torch 2.9.1):
  distillation **student F1 0.5732** (71.4% of the 0.80 teacher at 3.8× fewer params), antibody
  affinity **Spearman 0.542**, crystal-CIF **perplexity 1.735**. Validation surfaced (and fixed) three
  real hardware bugs, each now a documented teaching point: a **dynamic-shape recompile storm** from
  boolean-mask token selection in the distill loss (→ dense masked KL), a **bf16 `-inf`-mask → nan** in
  the crystal GPT's causal attention (→ `-1e9`), and a from-scratch student that under-learns (→ init
  from pretrained `bert-small`). Added per-step **progress logging** (`_progress.py`) with a
  "first step is compiling, not hung" heads-up so long runs are legible to learners.
- **Three new academic-domain examples** centered on making small/specialized models — all built to
  the gold-standard template (`run(config)` contract, CPU smoke path, companion `.md`, registry
  entry):
  - `distill_ner_slm.py` — knowledge distillation (temperature KL + hard-label CE) of the validated
    NER teacher into a ~4-layer student SLM; reports student-vs-teacher F1 retention + compression.
  - `antibody_affinity_slm.py` — fine-tunes a small **ESM-2** protein LM to predict antibody
    **binding affinity** from sequence on the public AbBibench benchmark; scored by Spearman.
  - `crystal_cif_slm.py` — a **CrystaLLM**-style character GPT trained from scratch to generate
    crystal-structure **CIFs** from a composition; reports perplexity + a sampled structure (LM
    quality only — structural-validity scoring is a documented follow-up).

### Changed (train→serve example)
- Brought `examples/complete_workflow/trainium_to_inferentia_pipeline.py` up to the repo's tone:
  assumed-knowledge/what-you'll-get header, the train-graph-vs-inference-graph (`trace()`) lesson
  front-and-center, a single labeled `ILLUSTRATIVE_HOURLY_USD` cost table (no hand-typed savings
  multipliers), and a DRY-by-default `main()` (`--run` to actually launch). Added companion README.

### Changed (learner-experience audit)
- **Standardized on a single Python version (3.12) + uv everywhere.** Removed per-OS native Python
  installs and plain-`pip`/`venv` paths from the setup guide and README; `requires-python` tightened
  to `>=3.12,<3.13` and CI matrix reduced to 3.12 only. `pip install awscli` (deprecated v1) replaced
  with a pointer to the AWS CLI v2 install.
- **Fixed first-five-minutes blockers:** README's first real command now includes the required
  `--email`; clarified that the CPU smoke run yields near-zero F1 (plumbing check, not the 0.846
  validated number); fixed dead links (`docs/README.md`, `scripts/neuron_migration.py`,
  `examples/quickstart/`, dashboard path) and pruned the README/tutorial TOCs to real sections.
- **Discoverability:** rewrote `examples/use_cases/README.md` with a suggested learning order
  surfacing all five validated examples (NER → satellite → utilization spike → distributed → Qwen3
  LoRA); added a "Start here" reading path and reverse-nav "Where this fits" footers across the
  conceptual docs and the debugging front door.
- **Pedagogy gaps closed:** define **MFU** and the **128×128 systolic array** in plain language in
  the entry doc (`choose_your_path.md`); added the bf16-inputs/**FP32-accumulation** reassurance and
  a lazy-execution/`mark_step` mental-model intro to best-practices; corrected "two failures" → three
  in the debugging walkthrough.

### Removed (learner-experience audit)
- Deleted stale/contradictory legacy material that undercut the validated tier: the CUDA-API-ridden
  `docs/error_handling_debugging.md` and `docs/troubleshooting/` (superseded by
  `neuron_tools_and_debugging.md` + the runnable debugging walkthrough), and two examples that
  contradicted the "real data, no toys" promise — `financial_modeling.py` (CPU-only, synthetic
  fallback, hand-typed cost claims) and `aws_open_data.py` (all loaders returned `np.random` mock
  data). Stripped the false "✅ validated" claim from `security_compliance_patterns.py` and labeled it
  a read-don't-run reference.

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
