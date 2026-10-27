# Hardware Validation Status

This file is **generated** by `validation/render_status.py` from the provenance artifacts in `validation/results/`. Do not edit by hand. Each row reflects a real run on real Neuron hardware (or marks the example as not-yet-validated).

_Last rendered: 2026-10-27_

**Coverage: 6/6 examples validated on hardware.** (6 validated on more than one instance — one row each below.)

| Example | Status | Instance | Neuron SDK | torch-neuronx | Key metric | Wall clock | Commit | When |
|---------|--------|----------|-----------|---------------|-----------|-----------|--------|------|
| `ner_biomedical` | ✅ passed | trn1.2xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | eval_f1=0.8480 | 1019.17s | 5351772 | 2026-10-21 |
| `ner_biomedical` | ✅ passed | trn2.3xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | eval_f1=0.8465 | 962.6s | 9da6b94 | 2026-10-26 |
| `satellite_landcover` | ✅ passed | trn1.2xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | eval_acc=0.7500 | 2054.31s | 5351772 | 2026-10-21 |
| `satellite_landcover` | ✅ passed | trn2.3xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | eval_acc=0.7500 | 636.46s | 9da6b94 | 2026-10-26 |
| `cv_utilization_spike` | ✅ passed | trn1.2xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | vit_over_cnn_tflops=4.6480 | 450.66s | 5351772 | 2026-10-21 |
| `cv_utilization_spike` | ❌ failed | trn2.3xlarge | — | 2.9.0.2.15.32035+de43f57c | — | 49.28s | 9da6b94 | 2026-10-26 |
| `distill_ner_slm` | ✅ passed | trn1.2xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | student_f1=0.5732 | 1027.26s | f6e9475 | 2026-10-21 |
| `distill_ner_slm` | ✅ passed | trn2.3xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | student_f1=0.5732 | 1272.6s | 9da6b94 | 2026-10-26 |
| `antibody_affinity_slm` | ✅ passed | trn1.2xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | spearman=0.4738 | 643.46s | f6e9475 | 2026-10-21 |
| `antibody_affinity_slm` | ❌ failed | trn2.3xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | spearman=0.3867 | 599.1s | 9da6b94 | 2026-10-26 |
| `crystal_cif_slm` | ✅ passed | trn1.2xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | inv_val_perplexity=0.5764 | 502.98s | f6e9475 | 2026-10-21 |
| `crystal_cif_slm` | ✅ passed | trn2.3xlarge | 2.33.38 | 2.9.0.2.15.32035+de43f57c | inv_val_perplexity=0.5764 | 635.11s | 9da6b94 | 2026-10-26 |

## TorchNeuron (native backend, public beta; Trn2/Trn3 only)

The same examples run on AWS's native PyTorch backend (`torch-neuronx` 2.13+, `device="neuron"`). Not counted in the coverage line above.

| Example | Status | Instance | Neuron SDK | torch-neuronx | Key metric | Wall clock | Commit | When |
|---------|--------|----------|-----------|---------------|-----------|-----------|--------|------|
| `ner_biomedical_torchneuron` | ✅ passed | trn2.3xlarge | — | 2.14.3.0.1+b1b6c66 | eval_f1=0.8491 | 245.01s | 504209b | 2026-10-26 |
| `ner_biomedical_torchneuron_compile` | ✅ passed | trn2.3xlarge | — | 2.14.3.0.1+b1b6c66 | eval_f1=0.8373 | 845.94s | e8f1c90 | 2026-10-26 |
| `satellite_landcover_torchneuron` | ⚠️ unvalidated | — | — | — | — | — | — | — |
| `cv_utilization_spike_torchneuron` | ⚠️ unvalidated | — | — | — | — | — | — | — |
| `distill_ner_slm_torchneuron` | ⚠️ unvalidated | — | — | — | — | — | — | — |
| `antibody_affinity_slm_torchneuron` | ⚠️ unvalidated | — | — | — | — | — | — | — |
| `crystal_cif_slm_torchneuron` | ⚠️ unvalidated | — | — | — | — | — | — | — |

## Multi-process examples (torchrun — validated by manual launch)

These need one process per NeuronCore (`torchrun`), which the single-device auto-harness doesn't orchestrate, so they're validated by a manual launch and recorded here rather than in the auto-table above.

| Example | Status | Instance | Observed | Notes |
|---------|--------|----------|----------|-------|
| `qwen3_lora` | ✅ validated (manual) | trn1.32xlarge | Qwen3-8B, full epoch on 32 cores: loss 1.93→1.43, ~5s/step, MFU ~29% | Qwen3 LoRA SFT via optimum-neuron (torchrun; hardware-only). |
| `ddp_ner` | ✅ validated (manual) | trn1.2xlarge, trn2.3xlarge | 2-core data-parallel: eval_f1 0.826, gradient all-reduce verified (trn1.2xlarge); trn2.3xlarge / Neuron 2.33, 2 logical cores: eval_f1 0.826, 11.6 samples/s | Data-parallel NER across NeuronCores (torchrun XLA DDP). |
| `tp_full_finetune` | ✅ validated (manual) | trn1.2xlarge | Full FT, 16.00 GB/core HBM ceiling: 1 core OOMs (Qwen3-1.7B 17.87 GB compile; Llama-3.2-1B runtime). TP=2 shards the model + trains steps (Llama), but full FT is marginal — Qwen3-1.7B 19.59 GB/core, Llama-3.2-1B 15.958 GB (32 MB over). TP necessary, not sufficient on 2 cores → LoRA here, full FT on trn1.32xlarge. | Tensor-parallel FULL fine-tune: 1 core OOMs; TP=2 shards but stays tight (optimum-neuron; torchrun; hardware-only). |

### Legend
- ✅ **passed** — ran on the listed instance and met its registry thresholds.
- ❌ **failed** — ran but missed a threshold or errored (see the artifact's `error`).
- ⚠️ **unvalidated** — no provenance artifact yet; not proven on hardware.

Artifacts: `validation/results/*.json` · Logs: `validation/logs/`
