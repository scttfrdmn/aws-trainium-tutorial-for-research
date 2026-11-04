# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Pinned versions and commits for Phase A. Changing any of these is a new measurement.

Re-validation on a later Neuron SDK is a separate task (work plan, section 2 and risks).
"""

NEURON_SDK = "2.33.0"
TORCH = "2.14.0"
TORCH_NEURONX = "2.14.3"
NEURONX_CC = "2.28.7631"  # as shipped in the SDK 2.33 DLAMI venv (boltz-neuron README)
BOLTZ = "2.2.1"

# jimburtoft/boltz-neuron HEAD on 2026-11-04 ("README: cause of the inf2 1,088-token limit").
BOLTZ_NEURON_REPO = "https://github.com/jimburtoft/boltz-neuron"
BOLTZ_NEURON_COMMIT = "7cf012186a08cff5fff0d6718049d4e73df8b8d9"
BOLTZ_NEURON_VENV = "/opt/aws_neuronx_venv_torch_neuronx_2_14_0"

# aqlaboratory/openfold main on 2026-11-04; carries NVIDIA's upstreamed TensorRT path
# (--trt_mode/--trt_engine_dir/...) and the cuEquivariance flags in run_pretrained_openfold.py.
OPENFOLD_REPO = "https://github.com/aqlaboratory/openfold"
OPENFOLD_COMMIT = "be2ec1841f16c966c65ae0e7599ebbadc725757d"
TENSORRT = "10.13"
CUDA = "13.0"

# DLAMI lookup for trn2 (boltz-neuron README). The label's "2.14.0" is the torch version.
TRN2_DLAMI_SSM = (
    "/aws/service/neuron/dlami/torch-neuronx-2.14.0/ubuntu-24.04/latest/image_id"
)

# Prebuilt NEFF caches published with the port (Hugging Face dataset jburtoft/boltz-neuron-cache).
NEFF_CACHE_DATASET = "jburtoft/boltz-neuron-cache"
NEFF_CACHE_TRN2 = {
    2: "boltz_neuron_cache_v9_sdk233_trn2_lnc2_ds1-8-25.tgz",
    1: "boltz_neuron_cache_v8_sdk233_trn2_ds1-8-25.tgz",
}

# Boltz-2 defaults for every comparison run (work plan section 2).
RECYCLING_STEPS = 3
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = (1, 5)
SEEDS = (0, 1)
# Reproduction target only (Phase A task 0), never a comparison row.
REPRO_SAMPLING_STEPS = 50
