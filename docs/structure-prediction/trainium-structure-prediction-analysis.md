<!-- Copyright 2026 Scott Friedman. SPDX-License-Identifier: Apache-2.0 -->

# AF3-class protein structure prediction on AWS Trainium

State of play as of 31 October 2026, architecture analysis, and the comparison against the NVIDIA RTX PRO 6000 (EC2 G7e).

Scott Friedman. Prepared from a review of the Neuron 2.33 release, the one existing Trainium port of an AF3-class model, and NVIDIA's published benchmarks on the same model class.

---

## 1. Summary

- AlphaFold2/OpenFold and AlphaFold3/OpenFold3/Boltz-2 are attention-based deep networks (transformer-family with geometric inductive biases). They do not run on Amazon Bedrock. On AWS they run on self-managed accelerators (EC2, Batch, ParallelCluster, SageMaker, HealthOmics workflows).
- The Neuron 2.33 release (19–20 October 2026) ships TorchNeuron, a native PyTorch backend for Trainium2/3 via the `PrivateUse1` device mechanism, with eager mode, `torch.compile(backend="neuron")`, FSDP/DTensor, and NKI for custom kernels. It replaces the XLA-based `torch-neuronx` lineage. Inferentia is not a target.
- One port of an AF3-class model to this stack exists: `jimburtoft/boltz-neuron`, running stock Boltz-2 v2.2.1 on trn2 (and inf2) through TorchNeuron, with measured latency, throughput, memory limits, and accuracy. No port of OpenFold/AlphaFold2 or OpenFold3 to Trainium exists.
- Per Trn2 chip, dense BF16 throughput is 667 TFLOPS, against 989 for an H100 SXM, 2,250 for B200/B300, and roughly 3,600–4,500 projected for Rubin. Trn3 holds BF16 flat at 671 TFLOPS; its gains are in MXFP8/MXFP4, HBM, and interconnect. For a BF16-bound geometric workload, the per-chip gap to NVIDIA widens each generation.
- The RTX PRO 6000 Blackwell Server Edition (EC2 G7e) has 503 dense BF16 TFLOPS, below a Trn2 chip. NVIDIA has published Boltz-2 and OpenFold-TensorRT timings on this exact part for a fixed public target set. It is the comparator that isolates the Trainium software gap from the raw-FLOPS gap.
- The workload-level question is whether the O(L³) pair-representation operations (triangle attention, triangle multiplicative update) can be made tensor-engine-bound on NeuronCore-v3/v4, where the tensor-to-vector throughput ratio is roughly 80:1 versus roughly 15:1 on an H100. That is a kernel-engineering question, not an architecture question; AlphaFold2 and AlphaFold3 were developed and run on TPUs, which share the same shape.

---

## 2. The model class and its compute profile

### 2.1 AlphaFold2 / OpenFold

AlphaFold2 (2021) is a supervised end-to-end network. The trunk ("Evoformer", 48 blocks) maintains two representations: an MSA representation (N_seq × L × 256) and a pair representation (L × L × 128). Each block applies MSA row attention (with pair bias), MSA column attention, a transition, outer-product-mean to update the pair rep, two triangle multiplicative updates, two triangle attention passes, and a pair transition. A structure module (8 blocks, shared weights) uses invariant point attention over per-residue rigid frames to produce coordinates. Training uses frame-aligned point error plus auxiliary losses; inference recycles the output three times.

OpenFold is the AlQuraishi lab's trainable PyTorch reimplementation. It reproduces the AlphaFold2 architecture in full with weight interoperability, and adds its own training code, memory-efficient attention, low-memory attention for long chains, FlashAttention for MSA attention, custom CUDA kernels (modified from FastFold) for triangle attention and in-place attention, and DeepSpeed's Evoformer attention kernel. Its performance depends on those kernels.

### 2.2 AlphaFold3 / OpenFold3 / Boltz-2

AlphaFold3 (2024) keeps an attention trunk but replaces the Evoformer with a Pairformer (no MSA column attention; a lighter MSA module feeds the pair rep) and replaces the structure module with a diffusion decoder that denoises atomic coordinates conditioned on the trunk outputs, handling proteins, nucleic acids, ligands, ions and modified residues as atoms. Boltz-1/2 (MIT license, released Nov 2024 / Jun 2025) and OpenFold3 (Apache 2.0; preview released Oct 2025, later release status not checked for this document) are open reimplementations of this architecture; Chai-1, Protenix and RoseTTAFold3 are others.

### 2.3 Where the time goes

Three operations dominate the trunk, all O(L³) in sequence length:

| Operation | Shape | Arithmetic character |
|---|---|---|
| Triangle multiplicative update | per channel (c=128), an L×L by L×L matmul | matmul-dominated; tensor-engine friendly |
| Triangle attention | L² query rows attending over L keys; 4 heads × d_head 32; additive bias from the pair rep | small head dimension; softmax (exp, max, sum, normalise) per score; low arithmetic intensity |
| MSA row/column attention (AF2 only) | N_seq × L rows; 8 heads × d_head 32 | same small-head-dimension pattern |

The structure module (IPA, frame gathers, torsions) is small and irregular. The AF3-class diffusion decoder is 200 sampling steps × N samples of a token-level transformer plus windowed (local) atom attention, at standard head dimensions and static shapes.

The MSA search stage (jackhmmer/HHblits against ~2.5 TB of databases, or MMseqs2) runs on CPU or GPU before the model and often dominates end-to-end wall time. MSA-free models (ESMFold; Boltz-2 in single-sequence mode) remove this stage at an accuracy cost that is target-dependent.

---

## 3. The Neuron stack as of October 2026

TorchNeuron (`torch-neuronx` ≥ 2.14) is an open-source (Apache 2.0) native PyTorch backend for Trainium, in public beta with Neuron SDK 2.33. It registers Trainium as a standard PyTorch device via `PrivateUse1`; existing PyTorch code runs by changing the device placement from `cuda` to `neuron`. It provides eager execution, `torch.compile(model, backend="neuron")` through a TorchDynamo backend, and standard distributed APIs (FSDP, DTensor, DDP, tensor parallelism), with context parallelism for scaled dot-product attention via PyTorch's experimental API. It supports Trn2 and Trn3 only; Trn1 and Inferentia stay on the XLA-based line before 2.14. PyTorch 2.9 is the last XLA-based version; 2.10 and later are native.

Properties relevant to this workload:

- `torch.compile` with the Neuron backend compiles one NEFF per distinct input shape. A captured operator the backend cannot lower fails compilation by default. Compilation of a Boltz-2 diffusion-model graph needs roughly 40 GB of host RAM and takes minutes per shape.
- NeuronCore-v3/v4 hardware supports control flow and dynamic shapes; Trainium3's NeuronCore-v4 adds ISA extensions for them. Whether and how TorchNeuron exposes dynamic shapes without recompilation is not documented for this model class.
- NKI is the kernel language for the four engines and the software-managed SBUF/PSUM SRAM. Fused attention kernels for LLM-shaped attention exist in the TorchNeuron examples; none exist for triangle attention, triangle multiplication, or windowed atom attention.

---

## 4. Existing ports

### 4.1 Boltz-2 on trn2 (jimburtoft/boltz-neuron)

The author, Jim Burtoft, is an AWS Solutions Architect; the repository is a personal GitHub project, not an aws-neuron or aws-samples release. The package wraps stock Boltz-2 v2.2.1 unchanged and applies a configuration automatically: bf16 inference, the diffusion model compiled with `torch.compile(backend="neuron")`, token-count padding to multiples of 32, a NEFF cache and an AOTAutograd cache that survive restarts. It runs on the public SDK 2.33 torch-neuronx DLAMI (torch 2.14.0, torch-neuronx 2.14.3, neuronx-cc 2.28). Validated on trn2.3xlarge (one Trainium2 device) for single chains, protein–ligand complexes including affinity, and protein–protein complexes.

Measured on trn2.3xlarge, LNC=2 (four 24 GB logical cores per device), bf16, `--recycling_steps 3 --sampling_steps 50`, fk_steering off, single-sequence inputs, warm:

| Tokens | Latency, 1 logical core (s), ds=1 / 8 / 25 | Throughput per device, 4 workers (structures/s), ds=1 / 8 / 25 |
|---|---|---|
| 64 | 4.0 / 7.2 / 8.0 | 0.984 / 4.14 / 11.61 |
| 128 | 4.5 / 8.6 / 10.0 | 0.889 / 3.53 / 9.14 |
| 192 | 7.5 / 10.6 / 16.7 | 0.535 / 2.89 / 5.81 |
| 256 | 9.4 / 13.8 / 25.8 | 0.427 / 2.26 / 3.85 |
| 384 | 16.5 / 23.1 / 49.5 | 0.242 / 1.37 / 2.00 |
| 512 | 31.1 / 41.5 / 66.5* | 0.129 / 0.764 / 1.49* |
| 640 | 54.6 / 73.8 / 116.1* | 0.073 / 0.432 / 0.857* |
| 960 | 143.8 / 186.1 / 282.1 | 0.028 / 0.172 / 0.348 |
| 1,024 | 201.8 / 258.1 / — | 0.020 / 0.124 / — |
| 1,056 | 205.4 / 269.7 / — | 0.019 / 0.118 / — |

\* with samples run in chunks (`BOLTZ_FIX_MAX_PARALLEL_SAMPLES=1`).

With an MSA (`--max_msa_seqs 2048`), one logical core: PDB 3HTB (192 tokens) 7.8 s at ds=1; 1M17 (352) 17.6 s; 3W7T (832) 106.3 s; 6EA2 (928) 141.9 s.

Input ceilings per logical core at LNC=2: 1,408 tokens at ds=1, 1,280 at ds=8, 960 at ds=25; deep MSAs lower these. LNC=1 (eight 12 GB cores) gives 6–58% more throughput per device below ~640 tokens and loses above ~700.

Accuracy: bf16, padding and caches validated against fp32, unpadded and uncached runs; caches give bit-identical output. On ~765-residue proteins with MSAs (3W7T, 5A7M; two seeds) 0.61–0.92 Å CA-RMSD, lDDT-CA 0.979–0.983 on trn2; on ~890-residue proteins (6EA2, 3B34) 0.47–0.50 Å, lDDT-CA 0.991–0.996. Ligand and complex targets (3HTB, 1M17, 1BRS) match CPU fp32 stock Boltz-2 within seed-to-seed variation.

A compiler-correctness defect was found and worked around: on SDK 2.33 at LNC=2, one compiled graph in the diffusion model (the atom/token one-hot matmul fused with the following windowed atom attention) returns wrong values for some atom counts (e.g. 5,952–6,400 atoms), moving predictions to 1.6–2.2 Å from the crystal. Splitting the graph (`BOLTZ_NEURON_ATOM_GRAPH_BREAK=1`, set automatically on trn2 at LNC=2) restores accuracy at no measurable speed cost.

Caveats for comparison against published GPU numbers: these runs use `--sampling_steps 50`; the Boltz-2 default is 200, and the diffusion loop is most of the runtime. Multi-device scaling on trn2.48xlarge was not measured; the repository extrapolates linearly from measured inf2.24xlarge scaling. Each prediction runs on one logical core (24 GB at LNC=2); the port does not shard a single prediction across cores, so inputs above the per-core ceiling (1,408 tokens at ds=1, lower with deep MSAs or more samples) do not run at all, whereas a 96 GB GPU runs them. Large antibody–antigen, multi-chain and nucleic-acid complexes fall in that range.

### 4.2 OpenFold / AlphaFold2

No port exists. OpenFold's speed rests on CUDA kernels (triangle attention and multiplicative update modified from FastFold, DeepSpeed Evoformer attention, FlashAttention on the MSA stack) that have no Neuron equivalent. Run through `.to('neuron')` it would take the reference PyTorch path for the O(L³) operations.

### 4.3 OpenFold3

No port exists. Architecturally it is closer to Boltz-2 (Pairformer + diffusion) than to OpenFold, so the Boltz-2 port is the closer precedent.

### 4.4 Other accelerators, for calibration

Tenstorrent's TT-Bio package runs Boltz-2, ESMFold2, Protenix, OpenFold3, OpenBind-0, OpenDDE, RoseTTAFold3, BoltzGen, RFdiffusion3, PXDesign, ESMC and SaProt on Blackhole and Wormhole. AlphaFold2 and AlphaFold3 were developed, trained and run on Google TPUs (JAX/XLA). Protein language models (ESM-2 class, BERT-style encoders) have run on AWS Inferentia in production (Metagenomi reported a 56% cost reduction).

---

## 5. Trainium architecture against the workload

### 5.1 NeuronCore engines

NeuronCore-v3 (Trainium2) is an independent heterogeneous unit with four engines and software-managed on-chip SRAM (28 MiB per core):

| Engine | Function | NeuronCore-v3 throughput |
|---|---|---|
| Tensor | power-optimised systolic array: GEMM, CONV, transpose; mixed precision cFP8/FP16/BF16/TF32/FP32 | 79 BF16/FP16/TF32 TFLOPS; 158 cFP8; 316 with 2:4-class structured sparsity |
| Vector | element-wise ops where each output depends on multiple inputs (axpy, LayerNorm, reductions) | 1 TFLOPS FP32 |
| Scalar | ops where each output depends on one input (activations, exp) | 1.2 TFLOPS FP32 |
| GPSIMD | eight programmable 512-bit vector processors; general C code | — |

A Trainium2 chip has eight NeuronCore-v3 (667 dense BF16 TFLOPS, 1,299 FP8; the per-core figures above are rounded and sum to 632 and 1,264, so the chip figures are used for chip-level comparisons throughout), 96 GiB HBM at 2.9 TB/s, 224 MiB SBUF, 128 DMA engines, 20 collective-communication cores, and NeuronLink-v3 at 1.28 TB/s per chip. Logical NeuronCore configuration (LNC) combines physical cores; trn2 defaults to LNC=2.

NeuronCore-v4 (Trainium3) keeps the tensor engine at 79 BF16/FP16/TF32 TFLOPS and 20 FP32, raises it to 315 MXFP8/MXFP4 TFLOPS, raises SRAM to 32 MiB per core with near-memory accumulation, adds vector-engine quantisation into MXFP8, and adds a fast exponential on the vector engine at 4× the scalar engine's throughput, intended for self-attention. A Trainium3 chip: 671 dense BF16 TFLOPS, 2,517 MXFP8/MXFP4, 144 GiB HBM3e at 4.9 TB/s, 256 MiB SBUF, NeuronLink-v4 at 2.56 TB/s per chip.

### 5.2 Tensor-to-vector ratio

Per NeuronCore-v3, tensor:vector+scalar throughput is roughly 79 : 2.2, or about 36:1 (about 80:1 against the vector engine alone). An H100 SXM is roughly 989 dense BF16 tensor TFLOPS against 67 FP32 CUDA-core TFLOPS, about 15:1. TPUs (MXU vs VPU) share Trainium's shape.

### 5.3 Consequence per operation

Triangle multiplicative update: c independent L×L×L matmuls per block. Tensor-engine work; the systolic array is well suited. Fusing the preceding LayerNorm and the gating sigmoid into the matmul kernel avoids an L²×c round trip to HBM.

Triangle attention: per attention score, about 128 matmul FLOPs (QKᵀ and PV, each 2×32 at d_head=32) and six to eight vector/scalar operations (bias add, mask, running max, subtract, exp, sum, normalise, cast). On one NeuronCore-v3, tensor time ∝ 128 / 79e12 ≈ 1.6e-12 and vector/scalar time ∝ 7 / 2.2e12 ≈ 3.2e-12; the softmax side is roughly 2× the matmul side, leaving the tensor engine idle about half the time in an unfused implementation. The same arithmetic on an H100 (≈15:1) is roughly balanced to tensor-bound. These are order-of-magnitude estimates; the Phase A profile replaces them with measured occupancy. The small head dimension is the cause; LLM attention at d_head=128 has four times the matmul work per score. Flash-style fusion (online softmax, bias tile resident in SBUF) raises the tensor-engine fraction; Trn3's 4× exponential raises the scalar/vector ceiling for exactly this operation.

MSA attention (AF2 only): same pattern.

Diffusion decoder (AF3 class): token transformer at standard head dimensions, windowed atom attention over 32-atom windows, 200 steps × N samples. Static shapes; tensor-engine friendly. The Boltz-2 port's throughput at ds=25 being 8–12× its ds=1 throughput across the measured sizes (11.8 at 64 tokens, 8.3 at 384, 12.4 at 960) is consistent with the trunk being amortised over the samples.

Structure module (AF2): irregular gathers and small matmuls; unfavourable on every accelerator and a small fraction of runtime.

### 5.4 What follows

- An AF3-class model has a higher matmul share than an AF2-class model (no MSA column attention, fewer trunk blocks per unit accuracy, a large static-shape diffusion loop). This is why the one port that exists is Boltz-2 and why it works without model changes.
- With fused NKI kernels for triangle attention, triangle multiplication and windowed atom attention, the trunk can approach tensor-bound; without them, naive `.to('neuron')` underperforms a GPU by more than the FLOPS ratio.
- Static-shape training (AF3-class models train on fixed 384- or 640-token crops) removes the compile-per-shape cost entirely.
- Variable-length inference requires shape bucketing and a prebuilt NEFF cache; this amortises over batch screening and penalises interactive single-structure use.
- The per-core software-managed SRAM (24–32 MiB) is what makes a fused triangle kernel feasible: the L×L bias tile and a strip of the pair representation can stay resident.
- The AlphaFold family's development on TPUs shows the architecture class handles the workload when kernels are co-designed. The difference between Trainium and TPU for this workload is the absence of those kernels, not the hardware.

---

## 6. Generational comparison

Dense BF16 per accelerator, with memory:

| Part | Dense BF16 TFLOPS | HBM | Bandwidth | Notes |
|---|---|---|---|---|
| Trainium2 (chip) | 667 | 96 GiB | 2.9 TB/s | 8 × NeuronCore-v3 |
| Trainium3 (chip) | 671 | 144 GiB | 4.9 TB/s | 8 × NeuronCore-v4; 2,517 MXFP8/MXFP4 |
| H100 SXM | 989 | 80 GB | 3.35 TB/s | 1,979 FP8 |
| H100 PCIe | ~756 | 80 GB | 2.0 TB/s | the part in NVIDIA's folding benchmark |
| RTX PRO 6000 Blackwell SE | 503 | 96 GB GDDR7 | ~1.6 TB/s | EC2 G7e |
| B200 / B300 | 2,250 | 192 / 288 GB | 8 TB/s | 4,500 FP8 |
| Rubin (R100/R200), projected | ~3,600–4,500 | 288 GB HBM4 | up to 22 TB/s | 17,500 dense FP8; BF16 growth 1.6–2× over Blackwell per reports |

NVIDIA raised BF16 from Hopper to Blackwell (2.3×) and is reported to raise it again for Rubin while putting most new transistors into FP4/FP8. AWS held BF16 flat from Trn2 to Trn3 and put the gains into MXFP8/MXFP4, HBM, and interconnect. The per-chip BF16 gap is 1.5× (H100), 3.4× (B200), and 5–7× (Rubin).

For this workload the accuracy-bearing path is geometric and currently BF16-floored. Whether FP8 is accuracy-neutral for the diffusion decoder or the Pairformer linears is unmeasured. If it is, Trn3's 2,517 MXFP8 TFLOPS competes with B200's 4,500 FP8 at 1.8× rather than 3.4×. If it is not, Trainium's position on this workload is a Trn2 price-per-hour and availability position that does not improve with Trn3.

---

## 7. The G7e comparison

### 7.1 The instance

EC2 G7e instances carry up to 8 NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (96 GB per GPU, 768 GB total), 5th-generation Intel Xeon (Emerald Rapids), up to 192 vCPUs, up to 2,048 GiB system memory, up to 15.2 TB local NVMe, up to 1,600 Gbps EFA with GPUDirect P2P and RDMA. Relative to G6e (L40S): 2× GPU memory, 1.85× memory bandwidth, 4× network bandwidth, up to 2.3× inference performance. Sizes start at g7e.2xlarge (one GPU). Purchasing: On-Demand, Spot, Savings Plans, Dedicated Instances/Hosts; no Capacity Block requirement. Regions as of May 2026: US East (N. Virginia, Ohio), US West (Oregon), Europe (Spain, London), Asia Pacific (Tokyo, Seoul), plus the Los Angeles Local Zone (us-west-2-lax-1b).

### 7.2 Published numbers on this GPU

NVIDIA ("Highly efficient protein structure prediction from compact computers to datacenters", Didi et al., 2025) benchmarked 20 hard CASP14 targets (95–949 residues), MSAs from ColabFold-search with MMseqs2-GPU, single sequence at a time, first AlphaFold2 weight set, one unrelaxed structure per query. Average seconds per target:

| Configuration | Hardware | MSA (s) | DL (s) | Total (s) | TM-score |
|---|---|---|---|---|---|
| AlphaFold2 (JackHMMER+HHblits, JAX) | x86 + L40S | 1,977.36 | 447.50 | 2,424.86 | 0.69±0.05 |
| ColabFold (MMseqs2-GPU + ColabFold-batch) | x86 + RTX PRO 6000 | 10.33 | 34.33 | 44.66 | 0.71±0.05 |
| OpenFold-PyTorch (TF32) | x86 + RTX PRO 6000 | 10.33 | 14.25 | 24.57 | 0.68±0.05 |
| OpenFold-TRT (BF16 Evoformer) | x86 + RTX PRO 6000 | 10.33 | 5.60 | 15.93 | 0.67±0.05 |
| Boltz-2 NIM v1.1.0 | x86 + RTX PRO 6000 | 10.33 | 6.83 | 17.16 | 0.68±0.05 |
| Boltz-2 NIM v1.1.0 | x86 + H100 PCIe | 12.12 | 4.81 | 16.92 | 0.67±0.05 |
| OpenFold-TRT | DGX GH200 | 11.22 | 5.40 | 16.62 | 0.67±0.05 |

The Boltz-2 NIM uses cuEquivariance triangle attention and multiplication kernels and TensorRT compilation. The H100 PCIe is 1.42× faster than the RTX PRO 6000 on Boltz-2 DL time, tracking the ~1.5× BF16 ratio, which indicates the NIM is close to tensor-bound. OpenFold-TRT on an L40S (48 GB) ran out of memory at 496 residues and on an H100 (80 GB) at 949; the RTX PRO 6000 (96 GB) inferred all 20 targets. Accelerations were upstreamed to MMseqs2, OpenFold and TensorRT (CUDA 13.0, TRT 10.13).

### 7.3 Why this is the comparator

- Trn2 has 1.33× the dense BF16 of an RTX PRO 6000 and ~1.8× the memory bandwidth. A Trn2 loss against it is attributable to kernels and compiler, not to FLOPS; a Trn2 loss against H100/B200 is confounded.
- The GPU side is already measured on a public, fixed target set with fixed MSAs.
- G7e is available on-demand and spot in six regions plus a Local Zone, which is the instance a research group is most likely to be offered. The Trainium availability argument has to be made against G7e, not against p5.
- On G7e the MSA stage runs on the same GPU (MMseqs2-GPU with Blackwell DPX optimisations, 10.3 s average on the CASP14 set). On trn2 the MSA stage runs on the host CPU, so end-to-end carries an extra cost on the Trainium side.

### 7.4 What is not yet comparable

The trn2 Boltz-2 numbers are at 50 sampling steps, not the default 200; single-device only; and at 192–1,056 tokens on single sequences, not the CASP14 set with MSAs. A like-for-like table requires rerunning the trn2 side at defaults on the CASP14 targets with the same MSAs.

---

## 8. Conclusions

1. Structure prediction on AWS is an accelerator workload (EC2/Batch/ParallelCluster/SageMaker/HealthOmics), with Bedrock relevant only for adjacent reasoning, annotation and orchestration.
2. The October 2026 Neuron release makes porting AF3-class models a device-swap plus `torch.compile`, as the Boltz-2 port demonstrates, with a per-shape compile model and a demonstrated compiler-correctness risk. It does not supply the O(L³) kernels; those remain to be written in NKI.
3. AF2-class models (OpenFold) have no Trainium path without those kernels; AF3-class models (Boltz-2, OpenFold3) have a working but kernel-less path.
4. Trainium's dense BF16 per chip is below current and projected NVIDIA datacenter parts and is flat across Trn2→Trn3. Its arguments on this workload are price per hour, availability, static-shape training, and a measured FP8 result that does not yet exist.
5. The RTX PRO 6000 on G7e has lower dense BF16 than a Trn2 chip and has published Boltz-2 and OpenFold-TRT timings on a public target set. A like-for-like run against it isolates the kernel gap and produces the first defensible $/structure comparison for Trainium.

---

## Sources

- AWS Neuron: TorchNeuron overview — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/torch/index.rst
- AWS Neuron: TorchNeuron release notes, Neuron 2.33.0 — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/release-notes/components/torchneuron.rst
- AWS Neuron: native PyTorch inference — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/inference-native-pytorch.rst
- aws-neuron/torch-neuronx (GitHub) — https://github.com/aws-neuron/torch-neuronx
- AWS Neuron: NeuronCore-v3 architecture — https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/about-neuron/arch/neuron-hardware/neuron-core-v3.rst
- AWS Neuron: NeuronCore-v4 architecture — https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/about-neuron/arch/neuron-hardware/neuron-core-v4.html
- AWS Neuron: Trainium2 architecture — https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.0/_sources/about-neuron/arch/neuron-hardware/trainium2.rst
- AWS Neuron: Trainium3 architecture — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium3.html
- jimburtoft/boltz-neuron (GitHub) — https://github.com/jimburtoft/boltz-neuron
- jburtoft/boltz-neuron-cache (Hugging Face) — https://huggingface.co/datasets/jburtoft/boltz-neuron-cache
- Didi et al., "Highly efficient protein structure prediction from compact computers to datacenters" (NVIDIA, 2025) — https://research.nvidia.com/labs/dbr/assets/data/manuscripts/fast_folding.pdf
- Amazon EC2 G7e instances — https://aws.amazon.com/ec2/instance-types/g7e/
- AWS News Blog: Announcing G7e — https://aws.amazon.com/blogs/aws/announcing-amazon-ec2-g7e-instances-accelerated-by-nvidia-rtx-pro-6000-blackwell-server-edition-gpus
- G7e in Europe (London) — https://aws.amazon.com/about-aws/whats-new/2026/05/amazon-ec2-g7e-london-region/
- G7e in Los Angeles Local Zone — https://aws.amazon.com/about-aws/whats-new/2026/04/ec2-g7e-instances-local-zones/
- Ahdritz et al., "OpenFold: retraining AlphaFold2…", Nature Methods 2024 — https://www.nature.com/articles/s41592-024-02272-z
- OpenFold3 training data, Registry of Open Data on AWS — https://registry.opendata.aws/openfold3/
- aqlaboratory/openfold triangular_attention.py — https://github.com/aqlaboratory/openfold/blob/main/openfold/model/triangular_attention.py
- tt-bio (PyPI) — https://pypi.org/project/tt-bio/0.8.0/
- Rubin vs Blackwell vs Hopper spec comparison — https://www.spheron.network/blog/nvidia-rubin-vs-blackwell-vs-hopper
- SemiAnalysis on Rubin BF16 scaling — https://newsletter.semianalysis.com/i/188150420
- The Next Platform on Trainium3/4 — https://www.nextplatform.com/2025/12/03/with-trainium4-aws-will-crank-up-everything-but-the-clocks/
- AWS Trainium & Inferentia ecosystem guide (Metagenomi) — https://introl.com/blog/aws-trainium-inferentia-silicon-ecosystem-guide-2025
