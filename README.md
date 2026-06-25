# AWS Trainium & Inferentia Tutorial for Research

[![Open Source](https://img.shields.io/badge/Open%20Source-MIT-blue)](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research)
[![GitHub Stars](https://img.shields.io/github/stars/scttfrdmn/aws-trainium-tutorial-for-research)](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/stargazers)
[![Version](https://img.shields.io/badge/Version-2026.1.0-green)](VERSION_MATRIX.md)
[![Neuron SDK](https://img.shields.io/badge/Neuron%20SDK-2.30.0-orange)](https://awsdocs-neuron.readthedocs-hosted.com/)

## 🎯 Overview

A comprehensive, research-focused tutorial for AWS Trainium and Inferentia. This tutorial provides what researchers and organizations need to leverage AWS Neuron hardware for cost-effective ML research and production deployment.

> ✅ **Hardware-validated on real Trainium** (Neuron 2.30 / PyTorch 2.9): **6 single-device examples
> auto-validated on `trn1.2xlarge`** (each with a captured provenance artifact), **plus 2 multi-process
> examples** (`qwen3_lora`, `ddp_ner`) validated by manual `torchrun` launch — see
> [`VALIDATED.md`](VALIDATED.md). Every performance number traces to a real run; nothing is hand-typed.
> (One example, the satellite CNN, is slow to *compile* the first time — Trainium compiles your whole
> model before running it, unlike a GPU. That first-compile cost and how to make it one-time is a core
> lesson, explained in the [CUDA→Neuron chapter](docs/main_tutorial_doc.md#cuda-migration).)

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

## ⏱️ Time & cost expectations

Pick the path that matches your goal — they differ by an order of magnitude in time and cost:

| Path | What you do | Time | AWS cost* |
|---|---|---|---|
| **Read + laptop smoke tests** | Read the core docs; run each example's CPU `*_SMOKE` path on your laptop (proves the code, no Trainium) | **~3–4 hrs** | **$0** |
| **Guided hands-on** | The above **+ launch one `trn1.2xlarge`** and run 2–3 examples end-to-end on real hardware | **~half a day** | **~$1–3** |
| **Full hardware lab** | Run **all** examples on Trainium, including the multi-core (`qwen3_lora`, `ddp_ner`) and train→serve pipeline | **~1–2 days** (mostly unattended) | **~$10–25** |

\* `trn1.2xlarge` is ≈ **$0.40/hr spot / ~$1.34/hr on-demand** (us-east/us-west, illustrative — confirm
current pricing). Auto-terminating scripts + budget alerts (Quick Start steps 1–2) keep this bounded.

**Expect the wall-clock to be *compile*-dominated, not compute-dominated** — that's the tutorial's
central lesson, not a bug. A first run pays an ahead-of-time compile (minutes for small models,
**~40 min for the satellite CNN** on the small box); an **S3 compile cache** makes it a one-time cost,
so warm re-runs finish in **~1–2 min**. Budget generously for your *first* launch, then it's fast.

## Start here (reading order)

New to this tutorial? Follow this path — each step builds on the last:

1. **[Local setup guide](docs/local_setup_guide.md)** — get your machine ready (Python 3.12 + uv, AWS CLI).
2. **[Quick start](docs/quick-start.md)** — your first success: run the NER smoke test, then a real
   Trainium run.
3. **[Choose your path](docs/choose_your_path.md)** — honest verdict on whether Trainium fits *your*
   workload, and which example to start from.
4. **[Full tutorial](docs/main_tutorial_doc.md)** — the in-depth chapters (FinOps, CUDA→Neuron
   migration, the complete workflow) with its own table of contents.
5. **[Trainium development best practices](docs/trainium_development_best_practices.md)** — the
   compile / bf16 / static-shape rules. Best read **after your first real Trainium run**, when its
   "we hit a `nan` at step 0 / 7 recompiles" field notes match what you just saw (read it earlier for
   the rules, and revisit).

Hit an error? Jump to **[Neuron tools & debugging](docs/neuron_tools_and_debugging.md)** (symptom→tool table).

## Quick Start

```bash
# 1. Set up AWS credentials
aws configure

# 2. Create budget alerts (--email is required — alerts are time-sensitive)
python scripts/setup_budget.py --limit 500 --email your-email@university.edu

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
│   ├── use_cases/                    # biomedical_ner, satellite_landcover, cv_utilization_spike, distill/antibody/crystal SLMs, qwen3_lora (all hardware-validated)
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
# Laptop smoke test (CPU, proves the code path — F1 will be near zero on the tiny 1-epoch
# subset; this checks plumbing, NOT accuracy. The 0.846 above is the full Trainium run):
NER_SMOKE=1 python examples/use_cases/biomedical_ner.py

# Real run — execute this ON a Trainium instance / DLAMI (it reaches eval_f1 = 0.846).
# Off-Trainium it silently falls back to a slow full CPU run, so only run it on a Neuron box:
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

> **Note:** This is advanced material — skim it on a first read. The snippet below is **illustrative
> pseudocode** to convey the shape of an NKI kernel and the NeuronCore memory hierarchy — it is not a
> drop-in, runnable kernel. The terms it uses (SBUF/PSUM on-chip memories, the 128×128 systolic
> array) are explained from scratch in the
> [novel kernels chapter](docs/novel_kernels_on_trainium.md); the real `neuronxcc.nki.language` API
> operates on tiles with explicit partition/free-axis reductions and masking. For working kernels,
> start from the
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

2. **Set up the environment with [uv](https://docs.astral.sh/uv/)** (this repo standardizes on
   **uv + Python 3.12** — pinned in `.python-version`):
   ```bash
   # Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
   uv python install 3.12        # the one supported version (pinned in .python-version)
   uv venv
   uv pip install -e ".[dev,science]"  # dev tooling + the science deps the examples need
                                       # (rasterio, pymatgen, biopython, …). `make install-dev` is equivalent.

   # Neuron wheels come from the AWS Neuron index (run on a Neuron instance/DLAMI):
   uv pip install torch-neuronx neuronx-cc \
       --extra-index-url https://pip.repos.neuron.amazonaws.com
   ```

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
- **Documentation**: [Full tutorial](docs/main_tutorial_doc.md) · [Quick start](docs/quick-start.md) · [Choose your path](docs/choose_your_path.md)

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