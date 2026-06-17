# AWS Trainium & Inferentia: A FinOps-First Tutorial for Academic Researchers

## Table of Contents
1. [Introduction & Prerequisites](#introduction)
2. [FinOps First: Cost Control from Day One](#finops-first)
3. [AWS Fundamentals for ML Researchers](#aws-fundamentals)
4. [Understanding AWS ML Chips vs NVIDIA GPUs](#chip-comparison)
5. [Migrating from CUDA to Neuron](#cuda-migration)
6. [Setting Up Ephemeral Environments](#ephemeral-setup)
7. [Container-Based Workflows with AWS Batch](#container-workflows)
8. [Complete Trainium → Inferentia Workflow](#complete-workflow)
9. [Real-Time Cost Monitoring Dashboard](#monitoring-dashboard)
10. [Domain-Specific Research Examples](#research-examples)
11. [Advanced Cost Optimization](#advanced-optimization)
12. [Performance Benchmarks & Sizing](#benchmarks)
13. [Advanced Patterns: NKI & Modern Architectures](#advanced-patterns)
14. [Compilation: Why Your First Step Takes Forever](#compilation)
15. [The Neuron Simulator: What It Can & Can't Do](#simulator)
16. [Troubleshooting & Resources](#troubleshooting)

---

## 1. Introduction & Prerequisites {#introduction}

### Who This Tutorial Is For
This tutorial is designed for academic researchers who:
- Are new to AWS but familiar with deep learning concepts
- Currently use NVIDIA GPUs (local or HPC clusters) 
- Want to minimize cloud costs while maximizing research output
- Need to run experiments without breaking the budget

### Knowledge Prerequisites

| Topic | Required Level | If You Need to Learn More |
|-------|---------------|--------------------------|
| Python | Intermediate | [Python for Data Science - AWS ML University](https://aws.amazon.com/machine-learning/mlu/) |
| Deep Learning | Basic understanding | [Deep Learning Fundamentals - AWS ML University](https://www.youtube.com/playlist?list=PL8P_Z6C4GcuWfAq8Pt6PBYlck4OprHXsw) |
| PyTorch/TensorFlow | Basic usage | [PyTorch Tutorials](https://pytorch.org/tutorials/) |
| Linux/Bash | Basic commands | [Linux Basics for AWS](https://aws.amazon.com/getting-started/hands-on/run-command-linux/) |
| Git | Basic usage | [Git Handbook](https://guides.github.com/introduction/git-handbook/) |
| Docker | Helpful but not required | [Docker 101 Tutorial](https://www.docker.com/101-tutorial/) |

### Why AWS ML Chips for Academic Research?

**The Cost Reality Check:**
- NVIDIA A100 (p4d.24xlarge): $32.77/hour
- NVIDIA H100 (p5.48xlarge): $98.32/hour  
- AWS Trainium2 (trn2.48xlarge): ~$40/hour (est.)
- AWS Inferentia2 (inf2.48xlarge): $12.98/hour

**For a typical 2-week experiment:**
- H100: $98.32 × 24 × 14 = **$33,035**
- Trainium2: $40 × 24 × 14 = **$13,440** (59% savings)
- With spot instances: ~$4,032 (88% savings)

### What you'll be able to do by the end

Concrete, checkable outcomes — by the end of this tutorial you can:

1. **Set up cost guardrails** so a forgotten instance can't run up a bill (budgets, ephemeral
   auto-terminating instances, emergency shutdown).
2. **Launch a Trainium instance and run a real fine-tune** on the PyTorch/XLA path — and read the
   first-step compile cost and throughput honestly.
3. **Apply the Trainium-native rules** that separate "it works" from "it crawls": static shapes to
   avoid recompilation, bf16-stable model choices (e.g. eager attention), and `xm.mark_step()`
   placement. (See the [best-practices chapter](trainium_development_best_practices.md).)
4. **Decide whether Trainium fits your problem** at all, using the
   [domain decision guide](choose_your_path.md).
5. **Use the Neuron tools** to diagnose `nan`s, slow training, and compile problems
   ([tools & debugging chapter](neuron_tools_and_debugging.md)).
6. **Reason about custom kernels** — what Trainium's architecture does that's genuinely different,
   and how to tell if your problem maps ([novel kernels chapter](novel_kernels_on_trainium.md)).
7. **Validate your own work on real hardware** with the
   [validation harness](../validation/README.md) — provenance, not hand-typed numbers.

> **How this tutorial sets expectations:** every chapter and example opens with a short
> *"assumed knowledge / what you'll be able to do"* block. We don't assume you "just know" Neuron —
> if a concept matters, it's introduced before it's used.

> ⚠️ The cost figures above are **illustrative** list/spot estimates for planning, not quotes —
> confirm current [EC2 pricing](https://aws.amazon.com/ec2/pricing/) before budgeting.

---

## 2. FinOps First: Cost Control from Day One {#finops-first}

### The #1 Rule: Nothing Runs Permanently

Before we write any ML code, let's set up safeguards to prevent runaway costs.

#### Step 1: Set Up Budget Alerts (5 minutes)

```bash
# First, install AWS CLI if you haven't
pip install awscli

# Configure with your academic credentials
aws configure
# Enter your Access Key ID, Secret Access Key, Region (us-east-1), and output format (json)
```

#### Step 2: Auto-Termination Scripts

**Never manually manage instances.** Always use scripts that auto-terminate. See `scripts/ephemeral_instance.py` for the complete implementation.

#### Step 3: Cost Dashboard

Create a simple cost tracking dashboard - see `monitoring/` directory for the complete S3-hosted dashboard.

---

## 3. AWS Fundamentals for ML Researchers {#aws-fundamentals}

### Essential AWS Concepts (15-minute crash course)

| AWS Concept | What It Means for ML Research | Academic Analogy |
|-------------|------------------------------|------------------|
| **EC2 Instance** | Virtual computer for training/inference | Like reserving a GPU node on HPC cluster |
| **S3 Bucket** | Cloud storage for datasets/models | Shared network drive, but accessible anywhere |
| **IAM Role** | Permissions for resources | Like your university login credentials |
| **VPC** | Private network for your resources | Your lab's private network |
| **AMI** | Pre-configured OS image | Like a Docker image for entire OS |
| **Spot Instance** | Discounted compute (can be interrupted) | Like standby queue on HPC |

---

## 4. Understanding AWS ML Chips vs NVIDIA GPUs {#chip-comparison}

### Comprehensive Performance Comparison

> Figures below are approximate, drawn from vendor specs and public benchmarks for planning only;
> verify against the [Trn2 architecture page](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn2-arch.html)
> and current [EC2 pricing](https://aws.amazon.com/ec2/pricing/) before relying on them. **Trainium3
> (Trn3)** was announced at re:Invent 2025 and is in private preview at UltraServer scale.

| Metric | V100 (16GB) | A100 (40GB) | H100 (80GB) | Trainium1 | Trainium2 | Inferentia2 |
|--------|-------------|-------------|-------------|-----------|-----------|-------------|
| **Performance** |  |  |  |  |  |  |
| FP32 TFLOPS | 15.7 | 19.5 | 60 | 190 | 275 | 190 |
| FP16 TFLOPS | 125 | 312 | 1,000 | 210 | 550 | 190 |
| BF16 TFLOPS | - | 312 | 1,000 | 210 | 550 | 190 |
| FP8 TFLOPS | - | - | 2,000 | - | 1,300 | - |
| **Memory** |  |  |  |  |  |  |
| Capacity | 16/32 GB | 40/80 GB | 80 GB | 32 GB | 96 GB | 32 GB |
| Bandwidth | 900 GB/s | 1.6 TB/s | 3.35 TB/s | 820 GB/s | 2.9 TB/s | 820 GB/s |
| **AWS Instance** |  |  |  |  |  |  |
| Type | p3.2xlarge | p4d.24xlarge | p5.48xlarge | trn1.2xlarge | trn2.48xlarge | inf2.xlarge |
| GPUs/Chips | 1 | 8 | 8 | 1 | 16 | 1 |
| On-Demand $/hr | $3.06 | $32.77 | $98.32 | $1.34 | ~$40 | $0.758 |
| Spot $/hr (avg) | $0.918 | $9.83 | ~$29.50 | $0.40 | ~$12 | $0.227 |

---

## 5. Migrating from CUDA to Neuron {#cuda-migration}

> **Note (June 2026):** this chapter teaches the **PyTorch/XLA** path (`torch_xla`,
> `xm.xla_device()`, lazy graphs + `xm.mark_step()`) — the production path on **PyTorch 2.9**, which
> AWS's public docs note is the **last XLA-based version**. A future, non-XLA PyTorch path is
> mentioned for **PyTorch 2.10+** but is not generally available yet, so everything below targets
> the XLA path you can actually run today. See
> [VERSION_MATRIX.md](../VERSION_MATRIX.md#-the-pytorch-path-xla-today).

### Understanding the Differences (PyTorch/XLA path)

| CUDA Concept | Neuron Equivalent | Key Differences |
|--------------|-------------------|-----------------|
| `torch.cuda.is_available()` | `torch_xla.core.xla_model.xla_device()` | XLA device abstraction |
| `.cuda()` | `.to(xla_device)` | Explicit device placement |
| CUDA kernels | Neuron Kernel Interface (NKI) | Different syntax |
| cuDNN | Neuron optimized ops | Automatic in most cases |
| Mixed precision (AMP) | Native BF16 support | Built into hardware |
| NCCL | Neuron Collective Comm | Different but similar API |

See `scripts/neuron_migration.py` for the complete migration helper.

---

## 8. Complete Trainium → Inferentia Workflow {#complete-workflow}

> **📌 2026 framing: "train on Trainium, serve on Inferentia" is now one option, not the default.**
> AWS has not announced an Inferentia3, and the modern serving stack (NxD Inference) **dropped
> Inf2/Trn1 support in Neuron 2.29** (pin to 2.28 to keep using it on Inf2). AWS now positions
> **Trainium2 for both training and inference**. Inferentia2 is still GA and is a fine,
> cost-optimized target for **smaller, latency-sensitive models** — but for new or large-scale
> serving, consider keeping inference on **Trn2** with NxD Inference + the vLLM plugin. See the
> [Inferentia vs Trainium decision guide](../VERSION_MATRIX.md#-when-to-use-inferentia2-vs-trainium2-for-inference).

### End-to-End Example: Climate Model Training and Deployment

This example shows a complete workflow of training a model on Trainium and deploying it on
Inferentia for cost-effective inference. The same pattern works with Trn2 as the inference target.

**Key Features:**
- Training on Trainium2 with automatic cost tracking
- Model compilation for both training and inference
- Deployment on Inferentia2 with Flask API
- Real-time cost monitoring throughout
- 60-70% cost savings vs traditional GPU approaches

See `examples/complete_workflow/` for the full implementation.

**Cost Breakdown:**
- Training Phase: $144 for Llama 2 7B (vs $295 on H100)
- Inference Phase: $0.227/hour (vs $3.06/hour on GPU)
- Total Monthly Savings: $2,000+ for typical research workload

---

## 13. Advanced Patterns: NKI & Modern Architectures {#advanced-patterns}

### Neuron Kernel Interface (NKI) Development

For advanced users who want hardware-level optimization.

> **Note:** The kernel below is **illustrative pseudocode** showing the *shape* of an NKI kernel
> and the NeuronCore memory hierarchy — it will not run verbatim. The real `neuronxcc.nki.language`
> API works on tiles with explicit partition/free-axis reductions, masking, and accumulation. Use
> the [official NKI guide and sample kernels](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
> as your starting point for production kernels.

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def flash_attention_kernel(q_tensor, k_tensor, v_tensor, scale):
    """Illustrative Flash Attention sketch for Trainium (NOT runnable as-is)"""
    # NeuronCore has 24MB SBUF and 2MB PSUM
    # Tile computations to fit in on-chip memory
    
    batch, seq_len, d_head = q_tensor.shape
    assert seq_len <= nl.tile_size.pmax  # 128 max
    
    # Load Q, K, V tiles to SBUF (on-chip memory)
    q_tile = nl.load(q_tensor)
    k_tile = nl.load(k_tensor)
    v_tile = nl.load(v_tensor)
    
    # Compute attention scores in PSUM buffer
    scores = nl.matmul(q_tile, k_tile.transpose(), dtype=nl.float32)
    scores = nl.multiply(scores, scale)
    
    # Softmax (custom implementation for NeuronCore)
    scores_max = nl.max(scores, axis=-1, keepdim=True)
    scores_exp = nl.exp(nl.subtract(scores, scores_max))
    scores_sum = nl.sum(scores_exp, axis=-1, keepdim=True)
    attn_weights = nl.divide(scores_exp, scores_sum)
    
    # Apply attention to values
    output = nl.matmul(attn_weights, v_tile)
    
    # Store result back to HBM
    output_tensor = nl.ndarray(output.shape, dtype=q_tensor.dtype)
    nl.store(output_tensor, output)
    
    return output_tensor
```

### Modern RAG Implementation on AWS ML Chips

Complete RAG pipeline optimized for Trainium/Inferentia:

```python
class NeuronRAGPipeline:
    """Complete RAG pipeline optimized for AWS ML chips"""
    
    def __init__(self, embedding_model_path, llm_model_path):
        # Load embedding model on Inferentia2
        self.embedder = self._load_embedding_model(embedding_model_path)
        
        # Load LLM on Trainium/Inferentia2
        self.llm = self._load_llm(llm_model_path)
        
        # Initialize vector store (FAISS on CPU)
        self.index = None
        self.documents = []
        
    def generate(self, query, max_length=512):
        """Complete RAG pipeline with cost tracking"""
        
        # Retrieve relevant documents
        docs, scores = self.retrieve(query, k=3)
        
        # Build context
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response on Trainium/Inferentia
        response = self.llm.sample(
            prompt,
            max_length=max_length,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        return {
            'answer': response,
            'sources': docs,
            'scores': scores.tolist(),
            'inference_cost': self._calculate_inference_cost()
        }
```

See `examples/rag_pipeline/` for the complete implementation.

### Advanced Distributed Training Patterns

```python
class TrainiumDistributedTrainer:
    """Advanced distributed training utilizing Trainium2 features"""
    
    @staticmethod
    def train_with_neuron_features(model_fn, dataset, config):
        """Utilize Trainium2-specific optimizations"""
        
        def _mp_fn(rank, flags):
            # Trainium2 specific features:
            # - 4x sparsity support (16:4)
            # - Stochastic rounding for better accuracy
            # - Dedicated collective engines
            
            # Enable Trainium2 optimizations
            import os
            os.environ['NEURON_CC_FLAGS'] = ' '.join([
                '--model-type=transformer',
                '--enable-saturate-infinity',
                '--enable-stochastic-rounding',  # Trainium2 feature
                '--enable-sparse-compute',  # 4x sparsity
                '--collective-engine-mode=dedicated'  # Use dedicated engines
            ])
            
            device = xm.xla_device()
            
            # Model with FSDP
            model = model_fn()
            model = FSDP(
                model,
                compute_dtype=torch.bfloat16,
                # Trainium2 optimizations
                forward_prefetch=True,
                backward_prefetch=True,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                use_orig_params=True
            )
```

---

## Compilation: why your first step takes forever (and how to tame it) {#compilation}

The single biggest surprise for people coming from CUDA is **compilation time**. Neuron is an
ahead-of-time (AOT) compiled platform: the Neuron compiler (`neuronx-cc`) turns your model graph
into a NEFF (Neuron Executable File Format) binary before it can run. This is very different from
eager CUDA execution.

### Why compiles can take *hours*

- **Every distinct graph shape triggers a compile.** Under PyTorch/XLA, a new input shape, batch
  size, or sequence length produces a new graph. Dynamic shapes (variable sequence lengths,
  `if`-on-tensor control flow, padding that changes per batch) can cause **recompilation on almost
  every step** — this is the classic "my training loop never makes progress" symptom.
- **Large models = large graphs.** Multi-billion-parameter models, big fused attention graphs, and
  aggressive optimization levels expand compiler search/scheduling time substantially. Hours-long
  compiles for large LLMs are not unusual on a first run.
- **First-run cost is paid once — if you let it cache.** Without caching, you re-pay on every job.

### How to tame it

1. **Use the persistent compile cache.** Point `NEURON_COMPILE_CACHE_URL` at a local dir or an S3
   bucket so compiled NEFFs are reused across runs and across nodes:
   ```bash
   export NEURON_COMPILE_CACHE_URL="s3://my-bucket/neuron-cache"   # or a local path
   ```
   A warm cache turns an hours-long startup into seconds. Share the cache across a cluster.

2. **Pre-compile ahead of the run with `neuron_parallel_compile`.** Instead of compiling lazily
   during the first (very slow) training epoch, do a trial run that *collects* all graphs and
   compiles them in parallel, populating the cache:
   ```bash
   neuron_parallel_compile python train.py     # collects graphs, compiles in parallel → cache
   python train.py                             # real run, now uses the warm cache
   ```
   The first command runs your script with execution stubbed so it only captures graphs.

3. **Keep shapes static.** Pad to fixed sequence lengths and use fixed batch sizes so the same
   graph is reused. Bucketing (a small set of fixed shapes) beats fully dynamic shapes. Watch for
   accidental recompiles with `NEURON_RT_LOG_LEVEL=INFO` or the profiler.

4. **Right-size the optimization level.** The compiler accepts flags via `NEURON_CC_FLAGS`
   (e.g. `--optlevel`/`-O`); lower levels compile faster for debugging, higher levels run faster.
   Iterate at a low level, then compile once at the high level for the production run.

> **Rule of thumb:** if "training" seems stuck with high host CPU and no step progress, you are
> almost certainly recompiling every step. Fix the shapes, then warm the cache.

## The Neuron simulator / NKI simulation: what it can and can't do {#simulator}

You do **not** need a Trainium/Inferentia instance to start developing — especially for custom
**NKI** kernels. NKI provides a **simulation mode** that runs a kernel on CPU using NumPy semantics
so you can develop and validate numerically before paying for hardware.

```python
import neuronxcc.nki as nki
import numpy as np

# Run an NKI kernel on CPU via simulation (no Trainium required):
out = nki.simulate_kernel(my_kernel, a_tensor, b_tensor)   # returns NumPy results
np.testing.assert_allclose(out, reference, rtol=1e-2)
```

### What the simulator **can** do
- **Validate numerical correctness** of an NKI kernel against a NumPy/PyTorch reference.
- **Develop without hardware** — iterate on tiling, indexing, and math on a laptop or CI runner.
- **Catch logic bugs early** (wrong reductions, indexing, accumulation) before a real compile.
- **Run in CI** — kernel correctness tests don't need a Neuron instance.

### What the simulator **cannot** do
- **It does not give you real performance numbers.** It models semantics, not the hardware's
  timing, memory bandwidth, or engine scheduling — no throughput/latency you can trust.
- **It is not cycle-accurate** and won't surface real-hardware bottlenecks (SBUF/PSUM pressure,
  DMA stalls, engine occupancy). For those, profile on a real instance with the Neuron profiler.
- **It doesn't replace end-to-end model compilation.** It's for NKI kernels, not a substitute for
  compiling and running your full model on Trainium/Inferentia.
- **Coverage can lag the hardware.** Newer instructions/intrinsics may not be fully modeled.

> **Workflow:** develop and unit-test NKI kernels in simulation (cheap, fast, CI-friendly) → then
> compile and **profile on real hardware** for performance. Simulation answers "is it correct?";
> only hardware answers "is it fast?".

---

## Performance Benchmarks (illustrative)

> ⚠️ The table below contains **illustrative estimates** for planning, not measurements from a
> single controlled run on the current SDK. Times, costs, and quality vary with SDK version, batch
> size, hyperparameters, and region/spot pricing. Reproduce with your own workload before relying
> on any figure.

### Estimated Benchmark Results

| Model | Platform | Time (hours) | Cost ($) | Quality Score |
|-------|----------|--------------|----------|---------------|
| **BERT-Large Fine-tuning** |
| | H100 | 0.6 | 17.70 | F1: 91.3 |
| | Trainium2 | 0.9 | 10.80 | F1: 91.2 |
| | Trainium1 | 2.4 | 0.96 | F1: 91.1 |
| **Llama 2 7B Training** |
| | H100 | 10 | 295.00 | PPL: 5.2 |
| | Trainium2 | 12 | 144.00 | PPL: 5.3 |
| **Stable Diffusion XL** |
| | H100 | 36 | 1062.00 | FID: 12.2 |
| | Trainium2 | 42 | 504.00 | FID: 12.3 |

**Key Insights:**
- Trainium2 offers 30-60% cost savings with comparable quality
- Training times are competitive, especially considering cost
- Quality metrics (F1, perplexity, FID) are within acceptable ranges

---

## Getting Started Checklist

Before starting your research on AWS:

- [ ] Set up AWS account with billing alerts (`scripts/setup_budget.py`)
- [ ] Complete IAM setup with minimal permissions
- [ ] Install and configure AWS CLI
- [ ] Create S3 bucket for data/models
- [ ] Set up auto-termination scripts (`scripts/ephemeral_instance.py`)
- [ ] Test with small experiment first (`examples/quickstart/`)
- [ ] Deploy cost monitoring dashboard (`monitoring/dashboard.html`)
- [ ] Join Neuron community channels
- [ ] Have emergency shutdown script ready (`scripts/emergency_shutdown.py`)

---

## Conclusion

This tutorial provides everything needed to start using AWS Trainium and Inferentia for academic research while maintaining strict cost control. The combination of advanced optimization techniques, real-world examples, and comprehensive monitoring makes it possible to achieve significant cost savings (30-88%) while maintaining research quality.

**Key Takeaways:**
1. Always use ephemeral resources with auto-termination
2. Start small and scale gradually
3. Monitor costs in real-time
4. Leverage advanced patterns for maximum efficiency
5. Join the community for support

The future of academic ML research is cost-efficient, and AWS ML chips provide the path forward.

---

*Last Updated: June 16, 2026*
*Tutorial Version: 2026.1.0 — targets Neuron SDK 2.30.0 / PyTorch 2.9 (PyTorch/XLA)*