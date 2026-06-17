# AWS Trainium & Inferentia Tutorial for Research

[![Open Source](https://img.shields.io/badge/Open%20Source-MIT-blue)](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research)
[![GitHub Stars](https://img.shields.io/github/stars/scttfrdmn/aws-trainium-tutorial-for-research)](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/stargazers)
[![Version](https://img.shields.io/badge/Version-2026.1.0-green)](VERSION_MATRIX.md)
[![Neuron SDK](https://img.shields.io/badge/Neuron%20SDK-2.30.0-orange)](https://awsdocs-neuron.readthedocs-hosted.com/)

## 🎯 Overview

A comprehensive, research-focused tutorial for AWS Trainium and Inferentia. This tutorial provides what researchers and organizations need to leverage AWS Neuron hardware for cost-effective ML research and production deployment.

> ### 📅 Status as of June 2026
>
> This tutorial targets **Neuron SDK 2.30.0** (released May 21, 2026) and **PyTorch 2.9**. Two platform shifts shape how you should read it:
>
> - **PyTorch backend transition.** AWS has announced **TorchNeuron**, a native (non-XLA) PyTorch backend that registers Trainium as a native device via PyTorch's `PrivateUse1` mechanism, with eager mode and `torch.compile` support. It is in **private preview** and targets **PyTorch 2.10**. **PyTorch 2.9 is the last version using PyTorch/XLA.** Most code here uses the XLA path (`torch_xla`, `xm.xla_device()`, `xm.mark_step()`) — the *outgoing* model. See [VERSION_MATRIX.md](VERSION_MATRIX.md#-the-pytorchxla--torchneuron-transition) for the migration outlook.
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

# 3. Launch ephemeral experiment
python scripts/ephemeral_experiment.py \
    --name "bert-test" \
    --instance-type "trn1.2xlarge" \
    --max-hours 4

# 4. Monitor costs
python scripts/cost_monitor.py
```

## Repository Structure

```
aws-trainium-inferentia-tutorial/
├── README.md                          # This file
├── docs/                              # Full tutorial documentation
├── scripts/                           # Utility scripts
├── examples/                          # Complete examples
│   ├── climate_prediction/           # Climate science example
│   ├── protein_structure/            # Biomedical example
│   ├── social_media_analysis/        # Social sciences example
│   └── rag_pipeline/                 # Modern RAG implementation
├── containers/                        # Docker configurations
├── monitoring/                        # Cost monitoring dashboard
├── advanced/                          # Advanced patterns (NKI, etc.)
└── benchmarks/                        # Performance data and comparisons
```

## Key Examples

### Complete Training → Inference Pipeline

Train a climate prediction model on Trainium, then deploy on Inferentia:

```python
# Train on Trainium2
pipeline = TrainiumToInferentiaPipeline('climate-prediction-v1')
training_result = pipeline.train_on_trainium(
    model_class='ClimateTransformer',
    config={'epochs': 100, 'instance_type': 'trn2.48xlarge'}
)

# Deploy on Inferentia2  
inference_result = pipeline.deploy_on_inferentia(
    model_path=training_result['model_path']
)

# Cost analysis shows 60-70% savings vs GPU approach
```

### Modern RAG Implementation

```python
# RAG pipeline optimized for AWS ML chips
rag = NeuronRAGPipeline(
    embedding_model='BGE-base-en-v1.5',  # On Inferentia2
    llm_model='Llama-2-7B'               # On Trainium2
)

# Index documents
rag.index_documents(research_papers)

# Query with cost tracking
result = rag.generate("What are the latest findings in climate modeling?")
print(f"Answer: {result['answer']}")
print(f"Cost: ${result['inference_cost']:.4f}")
```

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