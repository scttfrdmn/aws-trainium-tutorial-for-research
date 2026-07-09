# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Project work (milestones, issues, labels) is tracked on
[GitHub](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research) — not in local files.

## [Unreleased]

### Changed (version refresh → Neuron 2.31.0)
- **Bumped the tutorial's target to Neuron SDK 2.31.0** (released 2026-07-07; `torch-neuronx`
  2.9.0.2.15) across the README badge/status box, `VERSION_MATRIX.md`, `docs/quick-start.md`,
  `main_tutorial_doc.md`, `local_setup_guide.md`, and the docker-image tags. 2.31 is the **same
  PyTorch 2.9 / XLA stack** as 2.30 — no code or guidance changes; the native (non-XLA) **TorchNeuron**
  backend is still **Private Preview** (not GA), still slated for PyTorch 2.10+.
- **Hardware-validation records intentionally left at 2.30.** `VALIDATED.md` and the per-example
  "measured on real trn1.2xlarge" notes keep **Neuron 2.30.10 / torch-neuronx 2.9.0.2.14** — that's
  what actually ran. The target-version lines now say "validated on 2.30; 2.31 is the same PyTorch
  2.9/XLA stack" so "latest" and "what was proven" are both honest and distinguishable.

### Added (multi-core parallelism: measured on real hardware)
- **Data-parallel throughput measured (`data_parallel_ner.py`).** Same NER job single-core vs. 2-core
  DDP on one trn1.2xlarge, warm compile cache: **90.1 → 119.8 samples/s (~1.33×, not 2×)**. Added
  `train_wall_s` / `train_throughput_samples_s` metrics + `DDP_*` env knobs so a benchmark can match a
  single-core run's config. The README explains the ~1.33× (per-step overhead + all-reduce + serial
  rank-0 eval dominate at small scale) and why to benchmark **warm** (single-core cold spent 201 s of
  844 s on first-step compile; warm 844 s → 36 s).
- **Tensor-parallel full fine-tune (`examples/distributed/tensor_parallel_full_finetune.py`, +README).**
  The load-bearing counterpart to data parallelism: a **full** (non-LoRA) fine-tune too big for one
  NeuronCore, via optimum-neuron `NeuronModelForCausalLM` + `tensor_parallel_size`. **Validated on a
  real trn1.2xlarge** across two models — the honest finding is that TP is *necessary but, on 2 cores,
  not sufficient* for full FT. Each core has a hard **16.00 GB** HBM ceiling: 1 core always OOMs
  (Qwen3-1.7B 17.87 GB at compile, `NCC_EOOM001`; Llama-3.2-1B at runtime, `NRT_RESOURCE`); TP=2 shards
  the model and Llama-3.2-1B trains steps but tips over at **15.958 GB** (32 MB short), Qwen3-1.7B needs
  19.59 GB/core. This is why the Qwen3 example uses **LoRA** on trn1.2xlarge and reserves **full** FT for
  trn1.32xlarge/Trn2 (more cores → more aggregate HBM + a data-parallel dimension for ZeRO-1). The
  example catches both compile- and runtime-OOMs and explains them; registered in `TORCHRUN_EXAMPLES`.
- `VALIDATED.md` manual table now renders **⏳ pending hardware** for torchrun examples whose
  `validated_note` is empty/placeholder, so it never claims "validated" before a real run.

### Removed / fixed (repo audit)
- **Deleted stale pre-revamp clutter** that contradicted reality and the repo's own CONTRIBUTING
  policy: `PROJECT_STATUS.md`, `NEXT_ACTIONS.md`, `RESUMPTION_GUIDE.md`, `DEVELOPMENT_COMPLETE.md`,
  `docs/REVAMP_PLAN.md` (a "not yet executed" plan that's fully done), `docs/research_papers/`
  (papers for the deleted, partly-fabricated genomics/finance examples), `docs/video_tutorials/`
  (pinned retired versions: torch-neuronx 2.2.0, plain pip), and `docs/aws_engagement/` (re:Invent-2025
  outreach). Project tracking lives in CHANGELOG + GitHub issues.
- **Made the validation count honest.** `VALIDATED.md` now also lists the two `torchrun`-validated
  examples (`qwen3_lora`, `ddp_ner`) in a separate "validated by manual launch" table; the README no
  longer claims "6/6 registered" (the registry has 8 — 6 auto + 2 manual). Added a `validated_note`
  field to the registry to record their observed hardware results.
- **Consistency fixes:** README install now `[dev,science]` (the examples need rasterio/pymatgen/…);
  "real run" command flagged as on-Trainium-only (it silently CPU-degrades off-device); VERSION_MATRIX
  reconciled to "this repo pins 3.12" + TensorFlow row marked archived (was ✅) + unsubstantiated
  Lightning row dropped; crystal `run()` docstring fixed (`inv_val_perplexity`, not the nonexistent
  `neg_val_perplexity`); cv_spike "EuroSAT" comments removed (it's synthetic input); `main_tutorial_doc`
  body headings renumbered 1–9 to match the TOC; macOS setup uses uv (not plain venv); fixed the
  `_legacy/README` link to a deleted file and the `validation/` refs to the deleted REVAMP_PLAN.

### Validated (hardware — all on trn1.2xlarge, with an S3 compile cache)
- **6/6 examples now hardware-validated**, each with a captured provenance artifact. Metrics:
  ner_biomedical f1=0.846, satellite_landcover (RODA) eval_acc=0.75, cv_utilization_spike ViT
  **5.18× the CNN's TFLOP/s** (CNN 1.069 / ViT 5.535), distill_ner_slm student_f1=0.573,
  antibody_affinity_slm spearman=0.542, crystal_cif_slm perplexity 1.735 + `validity_rate`=0.0%
  (honest — a 1-epoch char-GPT learns CIF syntax but not yet parseable structures; why validity is
  reported, not gated).
- **Satellite compile story (a real teaching artifact).** The small-conv residual CNN is *slow to
  compile* on the trn1.2xlarge's 8 vCPUs — a cold compile took **~44 min** (compilation is a
  CPU-bound, ahead-of-time step; `neuronx-cc` lowers the whole graph to a NEFF before step 1). The fix
  is the standard Neuron pattern, **not** a bigger accelerator: `neuron_parallel_compile` + a
  persistent **S3 compile cache** made it one-time, and the validated **warm re-run finished in
  ~1.5 min** (`Using a cached neff`, 0 recompiles). Eval runs on-device (Trainium tutorial).
- Process lessons captured along the way: the **stale-S3-cache-lock** failure mode (a run killed
  mid-compile leaves a `.lock` with no `.neff` and wedges the next run on a *shared* prefix → use a
  per-run prefix), and that the cold-compile cost — not bad hosts — was what made earlier runs look
  "degraded." Both folded into best-practices §1.

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
  run still works. Unit-tested locally (0.5 on a valid+garbage CIF pair); hardware-validated on
  trn1.2xlarge — `validity_rate` measured at 0.0% (a 1-epoch char-GPT learns CIF syntax, not yet
  parseable structures; see the 6/6 summary above).
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
