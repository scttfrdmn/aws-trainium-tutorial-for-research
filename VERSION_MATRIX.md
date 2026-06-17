# Version Compatibility Matrix

This document tracks software versions, dependencies, and the current platform direction for the AWS Trainium & Inferentia tutorial components.

**Last Updated**: 2026-06-16
**Reference**: [AWS Neuron release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html)
**Legend**: ✅ Current/recommended | ⚠️ Legacy/maintenance | ❌ Archived/EOL | 🔬 Preview

> **How to read version claims here.** The versions below reflect the AWS Neuron release notes
> as of June 2026. They are documented compatibility targets, not the result of a fresh
> end-to-end hardware run on every cell — re-validate against the live release notes before
> pinning a production environment.

## 🧠 Core Neuron Stack (Neuron SDK 2.30.0 — May 21, 2026)

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| **AWS Neuron SDK** | 2.30.0 | ✅ | Latest release (2026-05-21) |
| **torch-neuronx** (PyTorch/XLA) | 2.9.x (`2.9.0.2.14.*`) | ✅ | PyTorch 2.9 — **last XLA-based version** |
| **TorchNeuron** (native PyTorch backend) | — | 🔬 | Private preview; non-XLA, targets **PyTorch 2.10** (see below) |
| **NxD Training** (NeuronX Distributed) | 1.x | ✅ | Recommended path for distributed training |
| **NxD Inference** (NeuronX Distributed) | 0.x | ✅ | Recommended serving lib; **Trn2+ only since 2.29** (Inf2/Trn1 → pin 2.28) |
| **vLLM Neuron plugin** | current | ✅ | Standard high-throughput inference serving on Neuron |
| **optimum-neuron** | current | ✅ | Hugging Face integration |
| **NKI (Neuron Kernel Interface)** | 0.4.0 | ✅ | FP8 matmul, Trn3 support |
| **jax-neuronx** | 0.10.x | ⚠️ | Beta; research focus |
| **tensorflow-neuronx** | — | ❌ | **Archived** — TensorFlow Neuron is no longer developed |
| **transformers-neuronx** | — | ⚠️ | Being folded into NxD Inference; install via Neuron pip index |

> **Verify before pinning:** exact patch versions for NxD Training/Inference, the vLLM plugin, and
> optimum-neuron move with each SDK release. Check the
> [component release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html)
> for the precise version bundled with your SDK.

## 🔀 The PyTorch/XLA → TorchNeuron transition

Historically, PyTorch on Neuron runs through **PyTorch/XLA**: you select an XLA device
(`xm.xla_device()`), build a lazy graph, and materialize it with `xm.mark_step()`. Most code in
this tutorial uses that model.

As of June 2026, AWS has announced **TorchNeuron**, an open-source **native** PyTorch backend:

- Registers Trainium as a native PyTorch device via the **`PrivateUse1`** mechanism — *"standard
  PyTorch runs unchanged on Trainium without platform-specific modifications."*
- Provides **eager-mode execution**, native distributed APIs (FSDP, DTensor), and
  **`torch.compile`** support for optimization.
- Status: **private preview** (access via AWS account teams), licensed Apache 2.0.
- Targets **PyTorch 2.10** support, planned for a future Neuron release.

**Key consequences for this tutorial:**

| | PyTorch/XLA (today) | TorchNeuron (coming) |
|---|---|---|
| Device | `xm.xla_device()` | native `PrivateUse1` device |
| Execution | lazy graph + `xm.mark_step()` | eager + `torch.compile` |
| Last/first PyTorch | **2.9 is the last XLA version** | **2.10+** |
| Status | ✅ current, GA | 🔬 private preview |

> **Guidance:** Keep using `torch-neuronx`/XLA for production on PyTorch ≤ 2.9. Don't over-invest
> in deep `mark_step`-style restructuring for new code — expect to migrate to the native backend.
> Source: [Neuron "What's New"](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html).

## 🎯 When to use Inferentia2 vs Trainium2 for inference

AWS has **not announced an Inferentia3**, and all recent accelerator announcements (Trn2, Trn3,
UltraServers) are Trainium. The modern serving library **NxD Inference dropped Inf2/Trn1 support in
Neuron 2.29** (NKI 0.3.0 kernels are no longer supported on that hardware; *"NxD Inference models are
now only supported on Trn2 and newer hardware"*) — to use it on Inf2 you must **pin to Neuron 2.28**.
Inferentia2 is therefore best understood as a **GA, cost-optimized inference option in maintenance
mode**, not the forward-looking platform.

**Use Inferentia2 (Inf2) when:**
- Serving small-to-mid models (roughly ≤ ~20B params) that fit comfortably in Inf2 memory.
- Latency-sensitive, low-/moderate-throughput single-stream inference.
- You want the lowest absolute $/hour at modest scale for a short-lived research deployment.
- You don't need the newest NxD Inference / vLLM features (or are content pinning to Neuron 2.28).

**Use Trainium2 (Trn2) for inference when:**
- Serving large models or running high-throughput / batch inference.
- You want to stay on the **actively developed** software path (NxD Inference + vLLM plugin).
- You're standing up a **new production service** intended to last.
- You want training and inference on one chip family.

> **Strategic note:** Inferentia2 appears to be the last of its line. Don't architect *new* work
> around an Inf2-only assumption; prefer Trn2 unless a specific cost/latency reason favors Inf2.
> Verify the current Inf2-vs-Trn2 price/latency crossover against live pricing before committing.

## 🐍 Python Environment

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| **Python** | 3.11 / 3.12 | ✅ | Recommended; current Neuron DLAMIs ship these |
| **Python** | 3.13 | ✅ | Available in recent DLAMIs (e.g. JAX SF) |
| **Python** | 3.10 | ✅ | Supported |
| **Python** | 3.9 | ⚠️ | Minimum; nearing end of upstream support |
| **Python** | 3.8 | ❌ | EOL (Oct 2024) — do not use |

## 🔥 ML Frameworks

### PyTorch Stack (Neuron 2.30.0)
| Component | Version | Status | Neuron Compatible |
|-----------|---------|--------|-------------------|
| **PyTorch** | 2.9 | ✅ | ✅ **Current — last PyTorch/XLA version** |
| **PyTorch** | 2.7 / 2.8 | ⚠️ | EOL in 2.29; pin to Neuron 2.28 if required |
| **PyTorch** | 2.10 | 🔬 | Planned via **TorchNeuron** native backend (preview) |
| **torch-neuronx** | 2.9.x | ✅ | XLA-based path; required for Trainium today |
| **torch-xla** | 2.9 | ✅ | Underpins `torch-neuronx` |
| **transformers** | recent | ✅ | Use version paired with current `optimum-neuron` |

### TensorFlow Stack — ❌ Archived
TensorFlow Neuron (`tensorflow-neuronx` / `tensorflow-neuron`) is **archived** and no longer
developed. New work should use the PyTorch path. Existing TF deployments can continue on older
SDKs but will not receive new features or hardware support.

### JAX Stack (Neuron 2.30.0)
| Component | Version | Status | Neuron Compatible |
|-----------|---------|--------|-------------------|
| **JAX-NeuronX** | 0.10.x | ⚠️ | Beta; research focus, not GA |
| **JAX** | paired w/ 0.10.x plugin | ⚠️ | Follow the JAX-NeuronX release notes for the exact pin |

## 🏗️ Infrastructure Components

### AWS Instance Types

> Hourly costs are approximate on-demand list prices and vary by region; **confirm against the
> [EC2 pricing pages](https://aws.amazon.com/ec2/pricing/on-demand/) before budgeting.** Spot is
> typically much cheaper but interruptible.

| Instance | Chip | Status | Use Case |
|----------|------|--------|----------|
| **trn2.48xlarge** | Trainium2 (16 chips, 1,536 GiB device mem) | ✅ | Current flagship — large-scale training **and inference** |
| **trn1.32xlarge** | Trainium1 (16 chips) | ✅ | Large-scale training |
| **trn1.2xlarge** | Trainium1 (1 chip) | ✅ | Entry-level training / experimentation |
| **Trn3 UltraServer** | Trainium3 | 🔬 | Announced re:Invent 2025; private preview, UltraServer-scale |
| **inf2.xlarge → inf2.48xlarge** | Inferentia2 | ⚠️ | GA but maintenance mode; cost-optimized inference for smaller models (see decision guide above) |
| **inf1.\*** | Inferentia1 | ❌ | Legacy; `torch-neuron` path is archived |

Per-chip Trainium2 (from the trn2.48xlarge spec / 16 chips): ~96 GiB HBM, ~2.9 TB/s memory
bandwidth, ~1.3 PFLOPS FP8 per chip. Confirm exact figures on the
[Trn2 architecture page](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn2-arch.html).

### AWS Services
| Service | Version/API | Status | Last Tested | Notes |
|---------|-------------|--------|-------------|-------|
| **SageMaker** | 2024.12.1 | ✅ | 2024-12-19 | Neuron container support |
| **S3** | v4 API | ✅ | 2024-12-19 | Data storage |
| **CloudWatch** | v2 API | ✅ | 2024-12-19 | Monitoring integration |
| **Lambda** | Python 3.11 | ✅ | 2024-12-18 | Pipeline orchestration |
| **EKS** | 1.28 | ✅ | 2024-12-17 | Kubernetes deployment |

## 📊 Data Sources (AWS Open Data)

| Dataset | Version/Date | Status | Last Tested | Size | Format |
|---------|--------------|--------|-------------|------|--------|
| **NASA Global Climate** | 2024.12 | ✅ | 2024-12-19 | 2.3GB | NetCDF |
| **NOAA GFS** | 2024.12.01 | ✅ | 2024-12-18 | 15.7GB | GRIB2 |
| **1000 Genomes** | Phase 3 | ✅ | 2024-12-17 | 847GB | VCF |
| **TCGA** | 2024.11 | ✅ | 2024-12-16 | 456GB | Various |
| **Landsat 8** | 2024.12 | ✅ | 2024-12-15 | 1.2TB | GeoTIFF |
| **Common Crawl** | 2024.12 | ⚠️ | 2024-12-10 | 3.6TB | WARC |

## 🧪 Testing Status by Component

### Examples Status

> **Validation honesty:** "✅ Runs" below means the example executes as written in a dev
> environment. It does **not** imply a full run on live Trainium/Inferentia hardware unless the
> "Notes" column says so. Several examples use mock data or mocked deployment steps for
> illustration — treat their performance/cost outputs as estimates, not measurements.

| Example | Runs | On real hardware? | Notes |
|---------|------|-------------------|-------|
| **Basic Hello Trainium** | ✅ | not verified here | Smoke test |
| **Climate Prediction** | ✅ | not verified here | Illustrative |
| **Genomics Analysis** | ✅ | not verified here | **Mock data** |
| **Financial Monte Carlo** | ✅ | not verified here | Illustrative |
| **Matrix Operations** | ✅ | not verified here | JAX path experimental |
| **Precision Emulation** | ✅ | not verified here | Illustrative |
| **End-to-End Pipeline** | ⚠️ | no | **Deployment mocked**; training loop pattern corrected (see code) |

### Framework Integration Status
| Framework | Training | Inference | Compilation | Mixed Precision | Issues |
|-----------|----------|-----------|-------------|-----------------|--------|
| **PyTorch + Neuronx** | ✅ | ✅ | ✅ | ✅ | None |
| **TensorFlow + Neuronx** | ✅ | ✅ | ✅ | ✅ | None |
| **JAX + Neuronx** | ⚠️ | ✅ | ✅ | ⚠️ | Experimental features |
| **Transformers + Optimum** | ✅ | ✅ | ✅ | ✅ | None |
| **Lightning + Neuronx** | ✅ | ✅ | ✅ | ✅ | None |
| **XGBoost + Neuronx** | ⚠️ | ⚠️ | ⚠️ | ❌ | Beta quality |

## 🐛 Known Issues and Workarounds

### Critical Issues
| Issue | Severity | Affected Versions | Workaround | ETA Fix |
|-------|----------|------------------|------------|---------|
| None currently | - | - | - | - |

### Minor Issues
| Issue | Affected Components | Workaround | Status |
|-------|-------------------|------------|--------|
| NxD Inference dropped Inf2/Trn1 | NxD Inference ≥ 2.29 | Pin to Neuron 2.28, or move to Trn2+ | Expected (hardware focus shift) |
| JAX path is beta | jax-neuronx 0.10.x | Use PyTorch path for production | Upstream beta |

## 🔄 Testing

The repo ships a `pytest` suite (`tests/`, `test_suite.py`) that runs in CI on commit. These tests
exercise the cost/budget/ephemeral-instance logic and example imports with **mocked AWS calls** —
they do **not** run workloads on live Neuron hardware. Coverage numbers should be read from the CI
run, not from a static table here.

Recommended OS baseline matches the current Neuron DLAMI: **Ubuntu 24.04** (or Amazon Linux 2023)
with Python 3.10–3.13.

## 📦 Installation Requirements

### Minimum System Requirements
```bash
# System Requirements
CPU: x86_64 (for development)
Memory: 8GB RAM minimum, 16GB recommended
Storage: 50GB free space
OS: Linux (Amazon Linux 2, Ubuntu 20.04+, RHEL 8+)

# AWS Requirements
AWS CLI: v2.15.0+
Boto3: v1.34.0+
Valid AWS credentials with Neuron permissions
```

### Installation (Neuron SDK 2.30.0, PyTorch 2.9 / XLA path)

> The fastest, most reliable way to get a known-good stack is the **AWS Neuron DLAMI** (Deep
> Learning AMI) or a Neuron Deep Learning Container — they bundle matched driver + runtime +
> framework versions. The pip recipe below is for custom environments. Always confirm exact
> versions against the
> [PyTorch Neuron setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/setup/pytorch-install.html).

```bash
# Core Neuron installation (PyTorch/XLA path — current as of Neuron 2.30.0)
# torch-neuronx pulls a matched torch / torch-xla; let the Neuron index resolve the pin.
python -m pip install --upgrade pip
pip install torch-neuronx neuronx-cc \
    --extra-index-url https://pip.repos.neuron.amazonaws.com
pip install transformers "optimum[neuron]"

# Distributed training / inference (recommended libraries)
pip install neuronx-distributed neuronx-distributed-inference \
    --extra-index-url https://pip.repos.neuron.amazonaws.com

# JAX (beta, research only)
# pip install jax-neuronx --extra-index-url https://pip.repos.neuron.amazonaws.com

# NOTE: TensorFlow Neuron is archived — no install recipe is provided.
# NOTE: PyTorch 2.10+ will use the native TorchNeuron backend (private preview); the install
#       flow will change at that point. Track the Neuron "What's New" page.

# Development dependencies (latest tested versions)
pip install \
    boto3==1.35.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    matplotlib==3.9.0 \
    scikit-learn==1.5.0 \
    pytest==8.2.2 \
    black==24.4.2 \
    flake8==7.1.0
```

### Docker Images
Use the latest **AWS Deep Learning Containers for Neuron** rather than hardcoding a tag — image
URIs and SDK versions change every release. Look up the current PyTorch Neuron training/inference
image for your region in the
[AWS Deep Learning Containers reference](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-containers).
A current image tag looks like:

```bash
# Pattern (resolve the exact current tag from the DLC reference above):
763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training-neuronx:<torch>-neuronx-py3xx-sdk2.30.x-ubuntu24.04
763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-inference-neuronx:<torch>-neuronx-py3xx-sdk2.30.x-ubuntu24.04
# (TensorFlow Neuron containers are no longer produced — the framework is archived.)
```

## 🔬 Performance & Cost (illustrative)

> ⚠️ **These are illustrative planning figures, not verified measurements from a controlled run
> on the current SDK.** Throughput and compile times depend heavily on SDK version, batch size,
> sequence length, and compiler flags; costs depend on region and spot pricing. Reproduce on your
> own instance with your own workload before relying on any number here. Earlier versions of this
> file labeled these "Validated/Verified" — that overstated the evidence and has been corrected.

### Performance (order-of-magnitude, trn1.2xlarge class)
| Model Type | Rough throughput | Notes |
|------------|------------------|-------|
| **BERT-Base** | hundreds of samples/sec | Compile once, then steady-state |
| **GPT-2** | ~hundreds of tokens/sec | Depends on seq length |
| **ResNet-50** | ~1k+ images/sec | Vision baseline |

### Cost (estimate from published pricing)
Estimate cost as `instance $/hr × hours`. Use current spot/on-demand prices from the
[EC2 pricing pages](https://aws.amazon.com/ec2/pricing/) — do not rely on the static figures that
previously appeared here.

## 📈 Platform Direction (2026)

| Area | Now (June 2026) | Next |
|------|-----------------|------|
| **Neuron SDK** | 2.30.0 | Rolling ~monthly releases |
| **PyTorch backend** | PyTorch/XLA (`torch-neuronx`), PyTorch 2.9 | **TorchNeuron** native backend, PyTorch 2.10 (preview → GA) |
| **Inference serving** | NxD Inference + vLLM plugin (Trn2+) | Continued Trainium focus |
| **Hardware** | Trn2 GA, Trn3 preview | Trn3 broader availability |

### Legacy / archived
| Component | Status | Migration Path |
|-----------|--------|----------------|
| **Python 3.8** | EOL | Use 3.10–3.13 |
| **TensorFlow Neuron** | Archived | Move to PyTorch path |
| **`torch-neuron` (Inf1)** | Archived | Use `torch-neuronx` on Trn/Inf2 |
| **NxD Inference on Inf2/Trn1** | Dropped in 2.29 | Pin Neuron 2.28, or move to Trn2+ |

## 🛠️ Development Environment Setup

### Recommended Development Stack
```bash
# Create virtual environment
python3.11 -m venv neuron-env
source neuron-env/bin/activate

# Install exact tested versions
pip install -r requirements-exact.txt

# Verify installation
python -c "import torch_neuronx; print('Neuron version:', torch_neuronx.__version__)"
```

### VS Code Configuration (Tested)
```json
{
    "python.defaultInterpreterPath": "./neuron-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

## 📞 Support and Troubleshooting

### Getting Help
1. Check the [Neuron release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html) for the exact versions in your SDK.
2. Prefer a Neuron DLAMI/DLC so driver + runtime + framework are matched for you.
3. Report issues with exact version information (`pip list | grep -i neuron`).
4. Include environment details from `neuron-ls` and `python -m torch_neuronx.analyze`.

---

**Sourcing**: Versions and platform direction in this file reflect the AWS Neuron release notes and
"What's New" page as of **2026-06-16**. Treat performance/cost tables as planning estimates, not
measured benchmarks. Re-check the live release notes before pinning a production environment.
