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
        instances=("trn1.2xlarge", "trn1.32xlarge"),
        thresholds={},  # validated by inspecting train_loss from a manual torchrun launch
        est_runtime_min=30.0,
        description="Qwen3 LoRA SFT via optimum-neuron (torchrun; hardware-only).",
    ),
    Example(
        key="ddp_ner",
        module="examples.distributed.data_parallel_ner",
        entrypoint="run",
        instances=("trn1.2xlarge", "trn1.32xlarge"),
        thresholds={"eval_f1": 0.75},
        est_runtime_min=20.0,
        description="Data-parallel NER across NeuronCores (torchrun XLA DDP).",
    ),
)


def get(key: str) -> Example | None:
    """Return the registry entry for `key`, or None if absent."""
    for ex in EXAMPLES:
        if ex.key == key:
            return ex
    return None
