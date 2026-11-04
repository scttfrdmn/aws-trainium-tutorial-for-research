# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Command construction for the four backends. Pure functions: no side effects, fully testable.

Each builder returns an `Invocation`: argv, the environment overrides it needs, where its
predictions will land, and (optionally) a separate profiled invocation. Timing runs are never
profiled; the profile comes from an extra run so profiler overhead cannot leak into wall times.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from harness import pins

PY = sys.executable


@dataclass(frozen=True)
class RunConfig:
    """One (target, backend, seed, ds) run."""

    backend: str
    target_id: str
    input_path: Path
    out_dir: Path
    seed: int
    diffusion_samples: int = 1
    recycling_steps: int = pins.RECYCLING_STEPS
    sampling_steps: int = pins.SAMPLING_STEPS
    lnc: int | None = None
    visible_core: int | None = None
    atom_graph_break: int | None = None
    max_msa_seqs: int | None = None
    max_parallel_samples_cap: bool = False
    nim: bool = False
    # openfold-trt only
    alignment_dir: Path | None = None
    template_mmcif_dir: Path | None = None
    jax_param_path: Path | None = None
    trt_engine_dir: Path | None = None
    openfold_dir: Path | None = None


@dataclass
class Invocation:
    """A command to run, plus what it needs and produces."""

    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    timing_json: Path | None = None
    predictions_dir: Path | None = None
    prediction_glob: str = "*_model_*.cif"
    profile_argv: list[str] | None = None
    profile_env: dict[str, str] = field(default_factory=dict)
    profile_path: Path | None = None


class ConfigError(ValueError):
    """A run configuration the work plan does not allow."""


def _boltz_args(c: RunConfig) -> list[str]:
    args = [
        "predict",
        str(c.input_path),
        "--out_dir",
        str(c.out_dir),
        "--recycling_steps",
        str(c.recycling_steps),
        "--sampling_steps",
        str(c.sampling_steps),
        "--diffusion_samples",
        str(c.diffusion_samples),
        "--num_workers",
        "0",
        "--seed",
        str(c.seed),
        "--output_format",
        "mmcif",
        "--override",
    ]
    if c.max_msa_seqs is not None:
        args += ["--max_msa_seqs", str(c.max_msa_seqs)]
    return args


def _predictions_dir(c: RunConfig) -> Path:
    # boltz writes to <out_dir>/boltz_results_<input stem>/predictions/<record id>/
    return (
        c.out_dir / f"boltz_results_{c.input_path.stem}" / "predictions" / c.target_id
    )


def _entry(
    target: str, timing: Path, boltz_args: list[str], profile: Path | None = None
) -> list[str]:
    argv = [PY, "-m", "harness.entry", "--target", target, "--timing-out", str(timing)]
    if profile is not None:
        argv += ["--torch-profile", str(profile)]
    return [*argv, "--", *boltz_args]


def neuron(c: RunConfig) -> Invocation:
    """boltz-neuron on trn2: per-logical-core worker, LNC=1/2, optional atom-graph split."""
    if c.lnc not in (1, 2):
        raise ConfigError("neuron backend needs lnc=1 or lnc=2")
    if c.visible_core is None:
        raise ConfigError(
            "neuron backend needs visible_core (one worker per logical core)"
        )
    max_core = 8 if c.lnc == 1 else 4
    if not 0 <= c.visible_core < max_core:
        raise ConfigError(f"visible_core must be in [0, {max_core}) at LNC={c.lnc}")
    if c.sampling_steps not in (pins.SAMPLING_STEPS, pins.REPRO_SAMPLING_STEPS):
        raise ConfigError(
            "sampling_steps must be 200 (comparison) or 50 (task-0 reproduction)"
        )
    env = {
        "NEURON_RT_VISIBLE_CORES": str(c.visible_core),
        "NEURON_LOGICAL_NC_CONFIG": str(c.lnc),
    }
    if c.atom_graph_break is not None:
        env["BOLTZ_NEURON_ATOM_GRAPH_BREAK"] = str(c.atom_graph_break)
    if c.max_parallel_samples_cap:
        env["BOLTZ_FIX_MAX_PARALLEL_SAMPLES"] = "1"
    timing = c.out_dir / "timing.json"
    prof_dir = c.out_dir / "neuron-profile"
    return Invocation(
        argv=_entry("boltz_neuron", timing, _boltz_args(c)),
        env=env,
        timing_json=timing,
        predictions_dir=_predictions_dir(c),
        profile_argv=_entry(
            "boltz_neuron", c.out_dir / "timing.profiled.json", _boltz_args(c)
        ),
        # Neuron runtime system/device profile of every NEFF executed by the profiled run.
        profile_env={
            **env,
            "NEURON_RT_INSPECT_ENABLE": "1",
            "NEURON_RT_INSPECT_OUTPUT_DIR": str(prof_dir),
        },
        profile_path=prof_dir,
    )


def gpu(c: RunConfig) -> Invocation:
    """Stock Boltz-2 on CUDA with cuEquivariance kernels (Boltz's default; never --no_kernels)."""
    if c.nim:
        raise ConfigError(
            "the NIM column is a separate, licence-gated runner (see RUNBOOK-A)"
        )
    timing = c.out_dir / "timing.json"
    prof = c.out_dir / "torch-profile.json"
    args = [*_boltz_args(c), "--accelerator", "gpu", "--devices", "1"]
    return Invocation(
        argv=_entry("boltz", timing, args),
        env={"CUDA_VISIBLE_DEVICES": "0"},
        timing_json=timing,
        predictions_dir=_predictions_dir(c),
        profile_argv=_entry(
            "boltz", c.out_dir / "timing.profiled.json", args, profile=prof
        ),
        profile_env={"CUDA_VISIBLE_DEVICES": "0"},
        profile_path=prof,
    )


def cpu(c: RunConfig) -> Invocation:
    """CPU fp32 oracle: stock Boltz-2 forced to Lightning precision 32-true; ds=1 only."""
    if c.diffusion_samples != 1:
        raise ConfigError(
            "the CPU oracle runs at ds=1 only (ds=5 is gated against g7e)"
        )
    timing = c.out_dir / "timing.json"
    args = [*_boltz_args(c), "--accelerator", "cpu"]
    return Invocation(
        argv=_entry("boltz_fp32", timing, args),
        timing_json=timing,
        predictions_dir=_predictions_dir(c),
    )


def openfold_trt(c: RunConfig) -> Invocation:
    """OpenFold with the TensorRT Evoformer, BF16, model_1 weights, one unrelaxed structure."""
    need = {
        "alignment_dir": c.alignment_dir,
        "template_mmcif_dir": c.template_mmcif_dir,
        "jax_param_path": c.jax_param_path,
        "trt_engine_dir": c.trt_engine_dir,
        "openfold_dir": c.openfold_dir,
    }
    missing = [k for k, v in need.items() if v is None]
    if missing:
        raise ConfigError(f"openfold-trt needs {missing}")
    if c.diffusion_samples != 1:
        raise ConfigError(
            "openfold-trt produces one structure per query (ds does not apply)"
        )
    assert c.openfold_dir and c.template_mmcif_dir and c.alignment_dir
    argv = [
        PY,
        str(c.openfold_dir / "run_pretrained_openfold.py"),
        str(c.input_path.parent),  # fasta_dir: one FASTA per target
        str(c.template_mmcif_dir),
        "--use_precomputed_alignments",
        str(c.alignment_dir),
        "--output_dir",
        str(c.out_dir),
        "--model_device",
        "cuda:0",
        "--config_preset",
        "model_1",
        "--jax_param_path",
        str(c.jax_param_path),
        "--skip_relaxation",
        "--cif_output",
        "--data_random_seed",
        str(c.seed),
        "--precision",
        "bf16",
        "--trt_mode",
        "run",
        "--trt_engine_dir",
        str(c.trt_engine_dir),
        "--trt_max_sequence_len",
        "1024",
    ]
    return Invocation(
        argv=argv,
        env={"CUDA_VISIBLE_DEVICES": "0"},
        predictions_dir=c.out_dir / "predictions",
        prediction_glob=f"{c.target_id}*_unrelaxed.cif",
    )


BUILDERS = {"neuron": neuron, "gpu": gpu, "cpu": cpu, "openfold-trt": openfold_trt}


def build(c: RunConfig) -> Invocation:
    """Dispatch on backend, enforcing the shared rules first."""
    if c.backend not in BUILDERS:
        raise ConfigError(f"unknown backend {c.backend!r}")
    if c.seed not in pins.SEEDS:
        raise ConfigError(f"seed must be one of {pins.SEEDS}")
    if (
        c.backend != "openfold-trt"
        and c.diffusion_samples not in pins.DIFFUSION_SAMPLES
    ):
        raise ConfigError(f"diffusion_samples must be one of {pins.DIFFUSION_SAMPLES}")
    if c.backend != "neuron" and c.sampling_steps != pins.SAMPLING_STEPS:
        raise ConfigError("comparison runs use Boltz-2's default 200 sampling steps")
    if c.recycling_steps != pins.RECYCLING_STEPS:
        raise ConfigError("comparison runs use Boltz-2's default 3 recycling steps")
    return BUILDERS[c.backend](c)
