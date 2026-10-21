# Version Compatibility Matrix

This document tracks software versions, dependencies, and the current platform direction for the AWS Trainium & Inferentia tutorial components.

**Last Updated**: 2026-10-20
**Reference**: [AWS Neuron release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html)
**Legend**: ✅ Current/recommended | ⚠️ Legacy/maintenance | ❌ Archived/EOL | 🔬 Preview

> **How to read version claims here.** The versions below reflect the AWS Neuron release notes
> as of October 2026 (SDK 2.33.0). They are documented compatibility targets, not the result of a fresh
> end-to-end hardware run on every cell — re-validate against the live release notes before
> pinning a production environment. (The exact SDK each example's hardware result was captured on is
> recorded per row in [`VALIDATED.md`](VALIDATED.md). The XLA `torch-neuronx` build is identical from
> 2.31.1 through 2.33 — only the compiler changed.)

## 🧠 Core Neuron Stack (Neuron SDK 2.33.0 — October 19, 2026)

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| **AWS Neuron SDK** | 2.33.0 | ✅ | Latest release (2026-10-19) |
| **torch-neuronx 2.9** (PyTorch/XLA) | `2.9.0.2.15` | ✅ ⚠️ | PyTorch 2.9, the **last XLA version** and **this tutorial's path**. Same build since 2.31.1; runs on Trn1, Trn2, Trn3, Inf2. AWS now calls XLA the *legacy* path for new work |
| **TorchNeuron** (`torch-neuronx` 2.13 / 2.14) | `2.13.3.0` / `2.14.3.0` | 🔬 | Native PyTorch backend (`neuron` device, eager + `torch.compile`). **Public beta in 2.33; Trn2/Trn3 only.** See [XLA vs TorchNeuron](#-the-pytorch-path-xla-today) |
| **Neuron Compiler** (`neuronx-cc`) | 2.28.7631 | ✅ | 2.27.5334 in 2.32; 2.26.6360 in 2.31 |
| **NxD Training** (NeuronX Distributed) | 1.x | ⚠️ | XLA-based; AWS lists it under *legacy training frameworks* |
| **NxD Inference** (NeuronX Distributed) | 0.10.x | ⚠️ | Trn2+ only since 2.29 (Inf2/Trn1 → pin 2.28); maintenance mode since 2.32; **APIs end on Trn2/Trn3 in 2.34** |
| **vLLM Neuron plugin** | 0.29.x (`0.29.0.1.2.0`) | ✅ | vLLM 0.29.0; compiles with `torch.compile` by default. Its XLA path ends on Trn2/Trn3 in 2.34 |
| **optimum-neuron** | current | ✅ | Hugging Face integration |
| **NKI (Neuron Kernel Interface)** | 0.7.0 | ✅ | Current in 2.33 (0.6.0 added `nisa.topk`, variable-length collectives, `fori_loop`/`while_loop`); FP8 matmul, Trn3 support |
| **jax-neuronx** | 0.10.x | ⚠️ | Beta; research focus. 2.33 supports JAX 0.7–0.10 |
| **tensorflow-neuronx** | — | ❌ | **Archived** — TensorFlow Neuron is no longer developed |
| **transformers-neuronx** | — | ⚠️ | Being folded into NxD Inference; install via Neuron pip index |

> **Verify before pinning:** exact patch versions for NxD Training/Inference, the vLLM plugin, and
> optimum-neuron move with each SDK release. Check the
> [component release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html)
> for the precise version bundled with your SDK.

## 🔀 The PyTorch path: XLA today

This tutorial targets the **PyTorch/XLA** path (`torch-neuronx` 2.9): select an XLA device
(`xm.xla_device()`), build a lazy graph, and materialize it with `xm.mark_step()`. Every example here
uses it, and it is the **only** PyTorch path on Trn1.

As of **Neuron 2.33 (October 2026)** AWS ships two PyTorch backends under the same `torch-neuronx`
package name:

| | PyTorch/XLA (this tutorial) | TorchNeuron (native) |
|---|---|---|
| Package | `torch-neuronx` 2.9.0.2.15 | `torch-neuronx` 2.13.3.0 / 2.14.3.0 |
| PyTorch | 2.9 (the last XLA version) | 2.13 / 2.14 |
| Programming model | lazy graph + `xm.mark_step()` | eager `.to("neuron")`, or `torch.compile(backend="neuron")` |
| Hardware | Trn1, Trn2, Trn3, Inf2 | **Trn2, Trn3 only** |
| Status | Production; AWS now calls it *legacy* for new work | **Public beta** |
| Inference on Trn2/Trn3 | **Ends in Neuron 2.34** | The stated replacement |

**Which should you use?**
- **Trn1:** XLA is the only option, and that's what this tutorial teaches.
- **Existing XLA code on Trn2/Trn3:** training keeps working (AWS has announced no end date for XLA
  *training*), but serving on the XLA path ends in 2.34, so plan a migration.
- **New work on Trn2/Trn3:** AWS recommends TorchNeuron. It's beta, so validate it on your workload
  before depending on it; see [Trying TorchNeuron](docs/main_tutorial_doc.md#trying-torchneuron).

The concepts this tutorial teaches (compile cost, static shapes, bf16, filling the 128×128 array)
apply to both backends: `torch.compile` on TorchNeuron still compiles a graph on first call and
recompiles when shapes change.

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

> **This repo pins Python 3.12 + uv** (`.python-version`); that's what every tutorial command and the
> CI assume. The table below is the broader range *Neuron itself* supports, for reference.

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| **Python** | 3.12 | ✅ | **This tutorial's pinned version** (uv-managed) |
| **Python** | 3.13 | ✅ | Available in recent DLAMIs (e.g. JAX SF) |
| **Python** | 3.10 / 3.11 | ✅ | Supported by Neuron, but not what this repo targets |
| **Python** | 3.9 | ⚠️ | Neuron minimum; nearing end of upstream support |
| **Python** | 3.8 | ❌ | EOL (Oct 2024) — do not use |

## 🔥 ML Frameworks

### PyTorch Stack (Neuron 2.33.0)
| Component | Version | Status | Neuron Compatible |
|-----------|---------|--------|-------------------|
| **PyTorch** | 2.9 | ✅ | ✅ **Current — last PyTorch/XLA version; this tutorial's target** |
| **PyTorch** | 2.7 / 2.8 | ⚠️ | EOL in 2.29; pin to Neuron 2.28 if required |
| **PyTorch** | 2.13 / 2.14 | 🔬 | TorchNeuron (native backend), public beta, Trn2/Trn3 only; not used by this tutorial's examples |
| **torch-neuronx** | 2.9.0.2.15 | ✅ | XLA path; required on Trn1 |
| **torch-xla** | 2.9 | ✅ | Underpins `torch-neuronx` |
| **transformers** | recent | ✅ | Use version paired with current `optimum-neuron` |

### TensorFlow Stack — ❌ Archived
TensorFlow Neuron (`tensorflow-neuronx` / `tensorflow-neuron`) is **archived** and no longer
developed. New work should use the PyTorch path. Existing TF deployments can continue on older
SDKs but will not receive new features or hardware support.

### JAX Stack (Neuron 2.33.0)
| Component | Version | Status | Neuron Compatible |
|-----------|---------|--------|-------------------|
| **JAX-NeuronX** | 0.10.x | ⚠️ | Beta; research focus, not GA (2.33 supports JAX 0.7–0.10) |
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
| **PyTorch + Neuronx** | ✅ | ✅ | ✅ | ✅ | None — the supported path |
| **TensorFlow + Neuronx** | ❌ | ❌ | ❌ | ❌ | **Archived** — no longer developed (see above) |
| **JAX + Neuronx** | ⚠️ | ✅ | ✅ | ⚠️ | Experimental features |
| **Transformers + Optimum** | ✅ | ✅ | ✅ | ✅ | None |
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
| NxD Inference now in maintenance mode | NxD Inference (2.32) | Feature work is slowing; track the release notes | As of Neuron 2.32 |
| NumPy 2.1 dropped | Neuron 2.32 | Use NumPy 2.2 or 2.3 | As of Neuron 2.32 |
| NxD Inference APIs + PyTorch/XLA inference end on Trn2/Trn3 | Neuron 2.34 | Move serving to vLLM Neuron / TorchNeuron (`torch.compile`) | Announced in 2.33 |
| TorchNeuron is Trn2/Trn3-only | `torch-neuronx` 2.13/2.14 | Stay on XLA `torch-neuronx` 2.9 on Trn1 | Beta scope |
| Amazon Linux 2 dropped | Neuron 2.33 | Use Ubuntu 24.04/26.04 or AL2023 | As of Neuron 2.33 |
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

### Installation (Neuron SDK 2.33.0, PyTorch 2.9 / XLA path)

> The fastest, most reliable way to get a known-good stack is the **AWS Neuron DLAMI** (Deep
> Learning AMI) or a Neuron Deep Learning Container — they bundle matched driver + runtime +
> framework versions. The pip recipe below is for custom environments. Always confirm exact
> versions against the
> [PyTorch Neuron setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/setup/pytorch-install.html).

```bash
# Core Neuron installation (PyTorch/XLA path — current as of Neuron 2.33.0)
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
# NOTE: TorchNeuron (torch-neuronx 2.13/2.14, Trn2/Trn3, public beta) installs differently;
#       see docs/main_tutorial_doc.md "Trying TorchNeuron".

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
The Neuron container lineup changed: AWS discontinued the `pytorch-training-neuronx` image in 2.31.0
and `pytorch-inference-neuronx` in 2.32.0. 2.33 DLCs are published on **Ubuntu 26.04 only**, and the
new torch-neuronx images ship PyTorch 2.14 with TorchTitan (that's the TorchNeuron backend, not the
XLA path). Look up the current image for your region in the
[AWS Deep Learning Containers reference](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-containers)
rather than hardcoding a tag. For this tutorial's XLA examples, use a DLAMI (below).
TensorFlow Neuron containers are no longer produced.

**DLAMI note (verified 2026-10-20):** the *multi-framework* Neuron DLAMI
(`/aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id`) tracks the latest SDK
(2.33.0). The single-framework `pytorch-2.9` DLAMI still lagged at 2.31.15.

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

| Area | Now (October 2026) | Next |
|------|-----------------|------|
| **Neuron SDK** | 2.33.0 | Rolling ~monthly releases |
| **PyTorch backend** | PyTorch/XLA 2.9 (all Trainium; this tutorial) + TorchNeuron 2.13/2.14 public beta (Trn2/Trn3) | XLA inference ends on Trn2/Trn3 in 2.34; no TorchNeuron GA date announced |
| **Inference serving** | vLLM Neuron 0.29 (`torch.compile` by default); NxD Inference in maintenance | NxD Inference APIs end on Trn2/Trn3 in 2.34 |
| **Hardware** | Trn2 GA, Trn3 preview | Trn3 broader availability |

### Legacy / archived
| Component | Status | Migration Path |
|-----------|--------|----------------|
| **Python 3.8** | EOL | Use 3.10–3.13 |
| **TensorFlow Neuron** | Archived | Move to PyTorch path |
| **`torch-neuron` (Inf1)** | Archived | Use `torch-neuronx` on Trn/Inf2 |
| **NxD Inference on Inf2/Trn1** | Dropped in 2.29 | Pin Neuron 2.28, or move to Trn2+ |
| **NxD Inference APIs / PyTorch/XLA inference on Trn2/Trn3** | Ends in 2.34 | vLLM Neuron or TorchNeuron + `torch.compile` |
| **Amazon Linux 2** | Dropped in 2.33 | Ubuntu 24.04/26.04 or AL2023 |

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
"What's New" page as of **2026-10-20**. Treat performance/cost tables as planning estimates, not
measured benchmarks. Re-check the live release notes before pinning a production environment.
