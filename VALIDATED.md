# Hardware Validation Status

This file is **generated** by `validation/render_status.py` from the provenance artifacts in `validation/results/`. Do not edit by hand. Each row reflects a real run on real Neuron hardware (or marks the example as not-yet-validated).

_Last rendered: 2026-06-24_

**Coverage: 5/6 examples validated on hardware.**

| Example | Status | Instance | Neuron SDK | torch-neuronx | Key metric | Wall clock | Commit | When |
|---------|--------|----------|-----------|---------------|-----------|-----------|--------|------|
| `ner_biomedical` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | eval_f1=0.8462 | 1387.59s | — | 2026-06-17 |
| `satellite_landcover` | ⚠️ unvalidated | — | — | — | — | — | — | — |
| `cv_utilization_spike` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | vit_over_cnn_tflops=5.1780 | 714.35s | — | 2026-06-24 |
| `distill_ner_slm` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | student_f1=0.5732 | 113.49s | — | 2026-06-24 |
| `antibody_affinity_slm` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | spearman=0.5420 | 1007.43s | — | 2026-06-24 |
| `crystal_cif_slm` | ✅ passed | trn1.2xlarge | 2.30.10 | 2.9.0.2.14.27725+e2ff0410 | inv_val_perplexity=0.5764 | 976.55s | — | 2026-06-24 |

### Legend
- ✅ **passed** — ran on the listed instance and met its registry thresholds.
- ❌ **failed** — ran but missed a threshold or errored (see the artifact's `error`).
- ⚠️ **unvalidated** — no provenance artifact yet; not proven on hardware.

Artifacts: `validation/results/*.json` · Logs: `validation/logs/`
