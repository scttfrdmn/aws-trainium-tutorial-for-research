# Hardware Validation Status

This file is **generated** by `validation/render_status.py` from the provenance artifacts in `validation/results/`. Do not edit by hand. Each row reflects a real run on real Neuron hardware (or marks the example as not-yet-validated).

_Last rendered: 2026-09-06_

**Coverage: 5/6 examples validated on hardware.**

| Example | Status | Instance | Neuron SDK | torch-neuronx | Key metric | Wall clock | Commit | When |
|---------|--------|----------|-----------|---------------|-----------|-----------|--------|------|
| `ner_biomedical` | ✅ passed | trn1.2xlarge | 2.31.15 | 2.9.0.2.15.32035+de43f57c | eval_f1=0.8467 | 1302.75s | 1f72cf3 | 2026-09-05 |
| `satellite_landcover` | ❌ failed | trn1.2xlarge | 2.31.15 | 2.9.0.2.15.32035+de43f57c | — | 2129.97s | 1f72cf3 | 2026-09-05 |
| `cv_utilization_spike` | ✅ passed | trn1.2xlarge | 2.31.15 | 2.9.0.2.15.32035+de43f57c | vit_over_cnn_tflops=4.9870 | 617.96s | 1f72cf3 | 2026-09-05 |
| `distill_ner_slm` | ✅ passed | trn1.2xlarge | 2.31.15 | 2.9.0.2.15.32035+de43f57c | student_f1=0.5859 | 1314.46s | 1f72cf3 | 2026-09-05 |
| `antibody_affinity_slm` | ✅ passed | trn1.2xlarge | 2.31.15 | 2.9.0.2.15.32035+de43f57c | spearman=0.5420 | 928.81s | 1f72cf3 | 2026-09-05 |
| `crystal_cif_slm` | ✅ passed | trn1.2xlarge | 2.31.15 | 2.9.0.2.15.32035+de43f57c | inv_val_perplexity=0.5778 | 682.12s | 1f72cf3 | 2026-09-05 |

## Multi-process examples (torchrun — validated by manual launch)

These need one process per NeuronCore (`torchrun`), which the single-device auto-harness doesn't orchestrate, so they're validated by a manual launch and recorded here rather than in the auto-table above.

| Example | Status | Instance | Observed | Notes |
|---------|--------|----------|----------|-------|
| `qwen3_lora` | ✅ validated (manual) | trn1.32xlarge | Qwen3-8B, full epoch on 32 cores: loss 1.93→1.43, ~5s/step, MFU ~29% | Qwen3 LoRA SFT via optimum-neuron (torchrun; hardware-only). |
| `ddp_ner` | ✅ validated (manual) | trn1.2xlarge | 2-core data-parallel: eval_f1 0.826, gradient all-reduce verified | Data-parallel NER across NeuronCores (torchrun XLA DDP). |
| `tp_full_finetune` | ✅ validated (manual) | trn1.2xlarge | Full FT, 16.00 GB/core HBM ceiling: 1 core OOMs (Qwen3-1.7B 17.87 GB compile; Llama-3.2-1B runtime). TP=2 shards the model + trains steps (Llama), but full FT is marginal — Qwen3-1.7B 19.59 GB/core, Llama-3.2-1B 15.958 GB (32 MB over). TP necessary, not sufficient on 2 cores → LoRA here, full FT on trn1.32xlarge. | Tensor-parallel FULL fine-tune: 1 core OOMs; TP=2 shards but stays tight (optimum-neuron; torchrun; hardware-only). |

### Legend
- ✅ **passed** — ran on the listed instance and met its registry thresholds.
- ❌ **failed** — ran but missed a threshold or errored (see the artifact's `error`).
- ⚠️ **unvalidated** — no provenance artifact yet; not proven on hardware.

Artifacts: `validation/results/*.json` · Logs: `validation/logs/`
