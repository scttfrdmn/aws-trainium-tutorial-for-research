# AWS Trainium & Inferentia Tutorial for Research

[![Open Source](https://img.shields.io/badge/Open%20Source-MIT-blue)](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research)
[![GitHub Stars](https://img.shields.io/github/stars/scttfrdmn/aws-trainium-tutorial-for-research)](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/stargazers)
[![Version](https://img.shields.io/badge/Version-2026.1.0-green)](VERSION_MATRIX.md)
[![Neuron SDK](https://img.shields.io/badge/Neuron%20SDK-2.30.0-orange)](https://awsdocs-neuron.readthedocs-hosted.com/)

## 🎯 Overview

A comprehensive, research-focused tutorial for AWS Trainium and Inferentia. This tutorial provides what researchers and organizations need to leverage AWS Neuron hardware for cost-effective ML research and production deployment.

> ### 📅 Status as of June 2026
>
> This tutorial targets **Neuron SDK 2.30.0** (released May 21, 2026) and **PyTorch 2.9** on the
> **PyTorch/XLA** path (`torch_xla`, `xm.xla_device()`, `xm.mark_step()`) — the production path
> available today. Two things to keep in mind:
>
> - **PyTorch path.** Per AWS's public [Neuron "What's New"](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html), **PyTorch 2.9 is the last version using PyTorch/XLA**, and a future release moves to a native (non-XLA) backend at **PyTorch 2.10+**. That path isn't generally available yet, so this tutorial stays on XLA; the native backend is a separate, forward-looking track.
> - **Trainium is now the path for both training and inference.** AWS has not announced an Inferentia3, and the modern serving library (NxD Inference) **dropped Inf2/Trn1 support in Neuron 2.29** (pin to 2.28 if you need it on Inf2). Inferentia2 remains GA and useful for cost-optimized, latency-sensitive, smaller-model inference, but new work should generally target **Trainium2 (Trn2)**. See the [Inferentia decision guide](VERSION_MATRIX.md#-when-to-use-inferentia2-vs-trainium2-for-inference).

## Key Features

- **FinOps-First Approach**: Built-in cost controls and monitoring
- **Ephemeral Computing**: Auto-terminating resources to prevent runaway costs
- **Complete Workflows**: End-to-end examples from training to inference
- **Advanced Patterns**: NKI development, RAG implementations, distributed training
- **Real-Time Monitoring**: S3-hosted dashboard for cost and resource tracking
- **Domain-Specific Examples**: Climate science, biomedical, social sciences
- **Production Ready**: Container-based workflows with AWS Batch

## Cost Savings

- **Training**: 30-75% savings vs comparable NVIDIA GPU instances
- **Inference**: substantial savings with spot instances on Inf2/Trn2
- **Example**: Llama 2 7B training - ~$144 (Trainium2) vs ~$295 (H100)

> ⚠️ **On the numbers in this tutorial:** cost figures are illustrative estimates based on
> published on-demand/spot pricing and public benchmarks, not measured results from a single
> controlled run unless explicitly stated. Spot prices vary by region and time. Always confirm
> current pricing in the [AWS pricing pages](https://aws.amazon.com/ec2/pricing/) before
> planning a budget, and treat throughput/cost tables as planning aids rather than guarantees.

## Table of Contents

1. [Introduction & Prerequisites](#introduction)
2. [FinOps First: Cost Control](#finops-first)
3. [AWS Fundamentals](#aws-fundamentals)
4. [Chip Comparison](#chip-comparison)
5. [CUDA to Neuron Migration](#cuda-migration)
6. [Ephemeral Environments](#ephemeral-setup)
7. [Container Workflows](#container-workflows)
8. [Complete Trainium → Inferentia Workflow](#complete-workflow)
9. [Cost Monitoring Dashboard](#monitoring-dashboard)
10. [Domain-Specific Examples](#research-examples)
11. [Advanced Optimization](#advanced-optimization)
12. [Performance Benchmarks](#benchmarks)
13. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Set up AWS credentials
aws configure

# 2. Create budget alerts
python scripts/setup_budget.py --limit 500

# 3. Launch an ephemeral, auto-terminating instance
python scripts/ephemeral_instance.py \
    --name "bert-test" \
    --instance-type "trn1.2xlarge" \
    --max-hours 4

# 4. Monitor costs
python scripts/cost_monitor.py
```

## Repository Structure

```
aws-trainium-tutorial-for-research/
├── README.md                          # This file
├── docs/                              # Full tutorial + best-practices + troubleshooting
├── scripts/                           # Utility scripts (budget, ephemeral instance, monitor)
├── validation/                        # Hardware-validation harness + provenance artifacts
├── examples/
│   ├── use_cases/                    # biomedical_ner (validated), financial_modeling
│   ├── complete_workflow/            # Trainium → Inferentia pipeline
│   ├── deployment/ · integration/    # serving + MLflow/Kubeflow/CI templates
│   └── advanced/                     # NKI patterns (illustrative)
├── advanced/                          # Advanced NKI patterns (illustrative)
├── monitoring/                        # Cost monitoring dashboard
└── VALIDATED.md                       # Generated: which examples are hardware-validated
```

## Key Examples

### Biomedical NER fine-tune — ⭐ hardware-validated

The reference example: a real disease-NER fine-tune on the NCBI-disease corpus, **validated on a
real `trn1.2xlarge`** (`eval_f1 = 0.846` — see [`VALIDATED.md`](VALIDATED.md)). It's the model the
other examples aim to match: real data, honest metrics, the Trainium-native lessons baked in.

```bash
# Laptop smoke test (CPU, proves the code path):
NER_SMOKE=1 python examples/use_cases/biomedical_ner.py

# Real run on a Trainium instance:
python examples/use_cases/biomedical_ner.py
```

See [`examples/use_cases/biomedical_ner.md`](examples/use_cases/biomedical_ner.md) and the
[Trainium development best practices](docs/trainium_development_best_practices.md) it demonstrates.

### Complete Training → Inference Pipeline

[`examples/complete_workflow/trainium_to_inferentia_pipeline.py`](examples/complete_workflow/trainium_to_inferentia_pipeline.py)
shows the full lifecycle — train on Trainium (XLA path), then compile the trained model for
Inferentia serving — with real `run_instances` calls, runtime DLAMI resolution, and auto-termination.
It's a teaching template (edit the S3 bucket; it's not a hardened service).

### Advanced NKI Development

> **Note:** The snippet below is **illustrative pseudocode** to convey the shape of an NKI
> kernel and the NeuronCore memory hierarchy — it is not a drop-in, runnable kernel. The real
> `neuronxcc.nki.language` API operates on tiles with explicit partition/free-axis reductions and
> masking. For working kernels, start from the
> [official NKI documentation and samples](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html).

```python
# Custom Flash Attention kernel for Trainium (ILLUSTRATIVE — not runnable as-is)
@nki.jit
def flash_attention_kernel(q_tensor, k_tensor, v_tensor, scale):
    # Optimized for NeuronCore v2 memory hierarchy
    # - SBUF: 24MB on-chip memory
    # - PSUM: 2MB for matrix multiply results
    # - Native 128x128 matrix operations
    
    q_tile = nl.load(q_tensor)
    k_tile = nl.load(k_tensor) 
    v_tile = nl.load(v_tensor)
    
    scores = nl.matmul(q_tile, k_tile.transpose())
    attn_weights = nl.softmax(scores * scale)
    output = nl.matmul(attn_weights, v_tile)
    
    return output
```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/scttfrdmn/aws-trainium-tutorial-for-research
   cd aws-trainium-tutorial-for-research
   ```

2. **Set up the environment with [uv](https://docs.astral.sh/uv/)** (recommended):
   ```bash
   # Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
   uv python install 3.12        # uses the version pinned in .python-version
   uv venv
   uv pip install -e ".[dev]"    # dev tooling (ruff, mypy, pytest)

   # Neuron wheels come from the AWS Neuron index (run on a Neuron instance/DLAMI):
   uv pip install torch-neuronx neuronx-cc \
       --extra-index-url https://pip.repos.neuron.amazonaws.com
   ```
   <details><summary>Prefer plain pip?</summary>

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```
   </details>

3. **Configure AWS**:
   ```bash
   python scripts/aws_environment_checker.py
   ```

4. **Run the tests** to confirm your setup:
   ```bash
   uv run pytest -m "not aws and not neuron"
   ```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **AWS Neuron Community**: [AWS Neuron SDK GitHub](https://github.com/aws-neuron/aws-neuron-sdk)
- **GitHub Issues**: [Report bugs or request features](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/issues)
- **Documentation**: [Full tutorial documentation](docs/README.md)

## Citation

If you use this tutorial in your research, please cite:

```bibtex
@misc{aws_trainium_tutorial2026,
  title={AWS Trainium \& Inferentia: Complete Tutorial for Academic Researchers},
  author={Friedman, Scott},
  year={2026},
  publisher={GitHub},
  url={https://github.com/scttfrdmn/aws-trainium-tutorial-for-research}
}
```