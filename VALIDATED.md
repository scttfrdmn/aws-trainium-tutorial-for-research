# Hardware Validation Status

This file is **generated** by `validation/render_status.py` from the provenance artifacts in `validation/results/`. Do not edit by hand. Each row reflects a real run on real Neuron hardware (or marks the example as not-yet-validated).

_Last rendered: 2026-07-07_

**Coverage: 6/6 examples validated on hardware.**

| Example | Status | Instance | Neuron SDK | torch-neuronx | Key metric | Wall clock | Commit | When |
|---------|--------|----------|-----------|---------------|-----------|-----------|--------|------|
| `ner_biomedical` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | eval_f1=0.8462 | 1387.59s | — | 2026-06-17 |
| `satellite_landcover` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | eval_acc=0.7500 | 94.66s | — | 2026-06-25 |
| `cv_utilization_spike` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | vit_over_cnn_tflops=5.1780 | 714.35s | — | 2026-06-24 |
| `distill_ner_slm` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | student_f1=0.5732 | 113.49s | — | 2026-06-24 |
| `antibody_affinity_slm` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | spearman=0.5420 | 1007.43s | — | 2026-06-24 |
| `crystal_cif_slm` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | inv_val_perplexity=0.5764 | 976.55s | — | 2026-06-24 |

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
