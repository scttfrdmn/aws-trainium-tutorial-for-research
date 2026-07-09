"""Catalog of validatable tutorial examples.

Each entry declares how to run an example under the harness and what "passing" means. Keeping this
as typed Python (not YAML) avoids a parser dependency and lets the thresholds be reviewed in code.

An example is *validatable* iff it exposes a module-level callable:

    def run(config: dict) -> dict[str, float]: ...

returning a flat dict of metric_name -> value. The harness imports the module, calls run(), times
it, and checks the returned metrics against `thresholds` (metric >= min).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Example:
    """A validatable example and its pass criteria."""

    key: str  # short id, e.g. "ner_biomedical"
    module: str  # importable module path, e.g. "examples.use_cases.biomedical_ner"
    entrypoint: str = "run"  # callable name inside the module
    instances: tuple[str, ...] = ("trn1.2xlarge",)  # validated baselines
    thresholds: dict[str, float] = field(default_factory=dict)  # metric -> minimum
    smoke_config: dict = field(default_factory=dict)  # tiny config for CPU smoke test
    full_config: dict = field(default_factory=dict)  # real config for hardware run
    est_runtime_min: float = 0.0  # rough wall-clock estimate for planning
    description: str = ""
    validated_note: str = (
        ""  # for TORCHRUN_EXAMPLES: the observed hardware result (manual launch)
    )


# The registry. Phase 1 ships exactly one entry (the NER exemplar); more are added as they are
# converted to research-grade + given a run() entrypoint.
EXAMPLES: tuple[Example, ...] = (
    Example(
        key="ner_biomedical",
        module="examples.use_cases.biomedical_ner",
        entrypoint="run",
        instances=("trn1.2xlarge",),
        # seqeval F1 on the validation split. A real fine-tuned token classifier on BC5CDR/NCBI
        # clears ~0.80 F1 comfortably; we gate at 0.75 to catch a broken training loop without
        # being flaky on small epoch counts.
        thresholds={"eval_f1": 0.75},
        smoke_config={
            "epochs": 1,
            "max_train_samples": 64,
            "max_eval_samples": 64,
            "device": "cpu",
        },
        # Tuned for efficient hardware validation: a larger batch (fewer steps/epoch on a single
        # NeuronCore) and 2 epochs reach F1>=0.75 on NCBI-disease without a long run. eager
        # attention (the example default) is required -- sdpa produces nan on the bf16 Neuron path.
        full_config={
            "epochs": 3,
            "device": "xla",
            "train_batch_size": 32,
            "eval_batch_size": 32,
            # NCBI-disease sentences are short; max_length=64 covers the vast majority and ~halves
            # per-step compute vs 128 (single-NeuronCore eager BERT is throughput-bound, not
            # compile-bound, once shapes are fixed). Keeps the validation run to ~15 min.
            "max_length": 64,
        },
        est_runtime_min=18.0,
        description="Token-classification NER fine-tune on a real biomedical corpus (XLA/Trainium).",
    ),
    Example(
        key="satellite_landcover",
        module="examples.use_cases.satellite_landcover",
        entrypoint="run",
        instances=("trn1.2xlarge",),
        # RODA Sentinel-2 + WorldCover land-cover. Patches are labeled by MAJORITY WorldCover class
        # (mixed-class patches → inherent label noise), so this is harder than the pre-tiled EuroSAT
        # benchmark; gate at 0.60 (the 1-epoch CPU smoke already hits ~0.77) to catch a broken
        # pipeline without being flaky. Needs `rasterio` on the instance to read COGs from S3.
        thresholds={"eval_acc": 0.60},
        smoke_config={
            "device": "cpu",
            "epochs": 1,
            "scenes": ("32/U/PU/2021/7/S2A_32UPU_20210705_0_L2A",),
            "patches_per_side": 16,
            "train_batch_size": 16,
            "eval_batch_size": 16,
        },
        # One scene's patches (484) at 16 patches/side, 5 epochs — enough to clear the 0.60 gate on a
        # single NeuronCore in a reasonable wall-clock (the small CNN is slow per-step on 1 core; more
        # scenes/epochs raise accuracy but cost time). Eval runs on CPU (see _evaluate note).
        full_config={
            "device": "xla",
            "epochs": 5,
            "scenes": ("32/U/PU/2021/7/S2A_32UPU_20210705_0_L2A",),
            "patches_per_side": 16,
        },
        est_runtime_min=20.0,
        description="Satellite land-cover CNN: RODA Sentinel-2 + WorldCover (vision; XLA/Trainium).",
    ),
    Example(
        key="cv_utilization_spike",
        module="examples.use_cases.cv_utilization_spike",
        entrypoint="run",
        instances=("trn1.2xlarge",),
        # The spike's thesis: a ViT-shaped model fills the 128x128 systolic array far better than a
        # small CNN. Measured 5.1x on trn1.2xlarge; gate at 2.0x so a regression that inverts the
        # relationship fails, without being flaky on step-time noise. (On CPU the ratio flips < 1,
        # which is why this gates only the hardware run.)
        thresholds={"vit_over_cnn_tflops": 2.0},
        smoke_config={
            "device": "cpu",
            "warmup_steps": 1,
            "timed_steps": 2,
            "batch_size": 8,
            "depth": 2,
        },
        full_config={"device": "xla"},
        est_runtime_min=12.0,
        description="Measures Trainium under-utilization: CNN vs ViT achieved TFLOP/s (XLA/Trainium).",
    ),
    Example(
        key="distill_ner_slm",
        module="examples.use_cases.distill_ner_slm",
        entrypoint="run",
        instances=("trn1.2xlarge",),
        # Distill the NER teacher into a small student. Gate on student_f1: a 4-layer student on
        # NCBI-disease clears ~0.55 comfortably while staying well below the ~0.80 teacher, so the
        # gate catches a broken distillation loop without being flaky. (Also checks the student
        # actually learned something, vs. the smoke run's near-zero F1.)
        thresholds={"student_f1": 0.55},
        smoke_config={
            "device": "cpu",
            "teacher_epochs": 1,
            "distill_epochs": 1,
            "max_train_samples": 64,
            "max_eval_samples": 64,
            # From-scratch tiny student for a fast offline CPU smoke (no pretrained download).
            "student_from_pretrained": None,
            "student_layers": 2,
            "student_hidden": 128,
            "student_heads": 2,
            "student_intermediate": 256,
        },
        # Cap train/eval so the inline teacher fine-tune + distillation finish in a watchable time on
        # a single NeuronCore (the full NCBI set × 5 total epochs is hours on 1 core). 2000 sentences
        # is ample for a distilled 4-layer student to clear the 0.55 F1 gate.
        full_config={
            "device": "xla",
            "teacher_epochs": 2,
            "distill_epochs": 3,
            "max_train_samples": 2000,
            "max_eval_samples": 600,
        },
        est_runtime_min=30.0,
        description="Knowledge distillation: NER teacher → small student SLM (XLA/Trainium).",
    ),
    Example(
        key="antibody_affinity_slm",
        module="examples.use_cases.antibody_affinity_slm",
        entrypoint="run",
        instances=("trn1.2xlarge",),
        # ESM-2 fine-tuned to predict antibody binding affinity. Gate on Spearman rank correlation:
        # a fine-tuned small protein LM on a single AbBibench complex should clear ~0.40 comfortably;
        # gate there so a broken regressor (correlation ~0) fails without being flaky on a hard task.
        thresholds={"spearman": 0.40},
        smoke_config={
            "device": "cpu",
            "epochs": 1,
            "max_train_samples": 48,
            "max_eval_samples": 32,
        },
        full_config={"device": "xla", "epochs": 8},
        est_runtime_min=15.0,
        description="Antibody binding-affinity regression with ESM-2 protein LM (XLA/Trainium).",
    ),
    Example(
        key="crystal_cif_slm",
        module="examples.use_cases.crystal_cif_slm",
        entrypoint="run",
        instances=("trn1.2xlarge",),
        # CrystaLLM-style char GPT generating crystal CIFs. Gated metric is inv_val_perplexity
        # (= 1/perplexity, in (0,1], higher is better). A trained char-LM on structured CIF text
        # reaches low single-digit perplexity (inv >= 0.1, i.e. perplexity <= 10); gate there so a
        # broken/untrained model (smoke perplexity ~79 → inv ~0.013) fails. The example ALSO reports
        # a pymatgen-based `validity_rate` (fraction of generated CIFs that parse to a real
        # structure) — reported, NOT gated, since validity on a short run is noisy.
        thresholds={"inv_val_perplexity": 0.1},
        smoke_config={
            "device": "cpu",
            "epochs": 1,
            "max_train_samples": 64,
            "max_eval_samples": 32,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 64,
            "block_size": 128,
            "sample_tokens": 64,
            "validity_samples": 4,
        },
        # Cap to 8000 CIFs for the validation run so a single-core pass is watchable (the full 36k
        # corpus is a much longer run — scale up for a stronger model). 8k structured CIFs is plenty
        # for the char-LM to clear the perplexity gate.
        full_config={
            "device": "xla",
            "epochs": 1,
            "max_train_samples": 8000,
            "max_eval_samples": 1000,
            "validity_samples": 20,
        },
        est_runtime_min=40.0,
        description="CrystaLLM-style char GPT: composition → crystal-structure CIF (XLA/Trainium).",
    ),
)

# Examples that are NOT in the single-process auto-registry above because they require a multi-process
# torchrun launch (one process per NeuronCore), which `run_on_hardware.py` does not yet orchestrate.
# They are built to the same standards and validated by launching torchrun manually. Listed here so
# they're discoverable and so VALIDATED.md isn't polluted with misleading "failed" rows.
TORCHRUN_EXAMPLES: tuple[Example, ...] = (
    Example(
        key="qwen3_lora",
        module="examples.use_cases.qwen3_lora_finetune",
        entrypoint="run",
        instances=("trn1.32xlarge",),
        thresholds={},  # validated by inspecting train_loss from a manual torchrun launch
        est_runtime_min=30.0,
        description="Qwen3 LoRA SFT via optimum-neuron (torchrun; hardware-only).",
        validated_note="Qwen3-8B, full epoch on 32 cores: loss 1.93→1.43, ~5s/step, MFU ~29%",
    ),
    Example(
        key="ddp_ner",
        module="examples.distributed.data_parallel_ner",
        entrypoint="run",
        instances=("trn1.2xlarge",),
        thresholds={"eval_f1": 0.75},
        est_runtime_min=20.0,
        description="Data-parallel NER across NeuronCores (torchrun XLA DDP).",
        validated_note="2-core data-parallel: eval_f1 0.826, gradient all-reduce verified",
    ),
    Example(
        key="tp_full_finetune",
        module="examples.distributed.tensor_parallel_full_finetune",
        entrypoint="run",
        instances=("trn1.2xlarge",),
        thresholds={},  # torchrun example — validated by manual launch, not auto-gated
        est_runtime_min=25.0,
        description="Tensor-parallel FULL fine-tune: 1 core OOMs; TP=2 shards but stays tight (optimum-neuron; torchrun; hardware-only).",
        validated_note=(
            "Full FT, 16.00 GB/core HBM ceiling: 1 core OOMs (Qwen3-1.7B 17.87 GB compile; "
            "Llama-3.2-1B runtime). TP=2 shards the model + trains steps (Llama), but full FT is "
            "marginal — Qwen3-1.7B 19.59 GB/core, Llama-3.2-1B 15.958 GB (32 MB over). TP necessary, "
            "not sufficient on 2 cores → LoRA here, full FT on trn1.32xlarge."
        ),
    ),
)


def get(key: str) -> Example | None:
    """Return the registry entry for `key`, or None if absent."""
    for ex in EXAMPLES:
        if ex.key == key:
            return ex
    return None
