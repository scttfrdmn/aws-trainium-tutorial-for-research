<!-- Copyright 2026 Scott Friedman. SPDX-License-Identifier: Apache-2.0 -->
<!-- Pre-registration artifact: section 5 of the work plan, copied verbatim on 2026-11-04,
     before any measurement. Do not edit; Phase A results are scored against it on the
     Phase A tracking issue (#55). -->

## 5. Expectations

Stated before measurement so the results can be scored against them.

### Phase A

- Expected: trn2 (one chip, LNC=2, 4 workers, kernels as shipped) is 2–4× slower than one RTX PRO 6000 on Boltz-2 DL time per structure at defaults, despite 1.33× the BF16 FLOPS. Attribution: unfused triangle attention and multiplication on the vector/scalar engines, per-shape padding overhead, and host-side glue.
- Expected occupancy: tensor engine < 30% busy during the trunk; vector+scalar the binding engines in triangle attention; the diffusion loop closer to 50–60% tensor-busy.
- Expected end-to-end: trn2 further behind because the MSA stage runs on host CPU; MMseqs2-CPU on the trn2.3xlarge host's small vCPU count will be the largest single term for short targets. (Host vCPU/RAM for trn2.3xlarge to be recorded in the Phase A manifest rather than assumed here.)
- Expected $/structure at on-demand pricing: g7e ahead on DL-only; the end-to-end gap wider. A trn2 win in Phase A is not expected.
- Confidence: high for the direction; the 2–4× range is the uncertain part.

### Phase B

- Expected: fused triangle attention is the single largest gain, 2–4× at the op level, taking the trunk from vector-bound to near tensor-bound. Triangle multiplication fusion a further 1.2–1.5× on that op. Atom attention kernel mainly a correctness and robustness gain with modest speed.
- Expected end-to-end: 1.5–2.5× over the Phase A trn2 baseline on the CASP14 set, bringing trn2 to roughly 1.0–1.6× the RTX PRO 6000's DL time per structure (chip vs GPU).
- Expected $/structure after Phase B: decided by the trn2.3xlarge vs g7e.2xlarge hourly ratio. If trn2.3xlarge is priced at or below g7e.2xlarge, trn2 reaches parity-to-advantage on DL-only $/structure; end-to-end still carries the MSA cost unless Phase D co-locates MMseqs2 on a larger host.
- Confidence: moderate. The main uncertainty is NKI maturity on NeuronCore-v3 for a d_head=32 attention with an additive bias and whether the compiler will accept the fused graph without further defects. Expect at least one more compiler-correctness issue to be found and worked around.

### Phase C

- Expected: FP8 on the diffusion decoder's linears is accuracy-neutral within the seed spread (the decoder is a standard transformer and those matmuls are the least geometry-sensitive). FP8 on the Pairformer linears is uncertain; FP8 on the triangle ops is not attempted.
- Consequence if neutral: Trn3's MXFP8 path applies to roughly the diffusion share of runtime (the majority at ds ≥ 5), and the Trn3 argument becomes real for inference.
- Consequence if not neutral: Trainium's position on this workload is a Trn2 BF16 price position with no Trn3 improvement; this is stated plainly in the write-up.
- Confidence: moderate for the decoder result; low for the Pairformer.

### Phase D

- Expected: linear scaling across 16 chips (the port's inf2.24xlarge measurement scaled linearly; no cross-chip communication in inference). Batched same-bucket inference gives ≥ 1.5× throughput at ≤ 256 tokens by filling the tensor engine; little gain above 512 tokens where single-structure work already fills it. LNC=1 wins below ~640 tokens, LNC=2 above, as the port measured.
- Expected end-to-end on trn2.48xlarge: co-locating MMseqs2-CPU on 192 vCPUs makes the MSA stage comparable to the GPU path for throughput workloads, though not for latency.
- Confidence: high.

### Phase E

- Expected: this is the phase most likely to show a Trainium advantage. Static crops remove the compile tax; the pair-representation activations fit 96 GB HBM; NeuronLink carries FSDP all-reduce. Expected tensor-engine MFU 35–50% with the Phase B kernels and their backward passes, against ~40–55% MFU for the GPU run with cuEquivariance.
- Expected $/step: trn2.48xlarge between 0.6× and 1.2× p6-b200.48xlarge at on-demand pricing. The outcome within that range is set by the hourly price ratio more than by MFU.
- Confidence: low-to-moderate. Backward passes for the fused kernels are the largest engineering item in the plan, and OpenFold3's trainer on TorchNeuron has not been exercised by anyone.

### Overall

- Probable outcomes: (a) the first published engine-level profile of an AF3-class model on Trainium; (b) three reusable NKI kernels that apply to Boltz, OpenFold3, Chai, Protenix and RF3; (c) a like-for-like $/structure table against G7e in which trn2 is at or near parity on DL-only after Phase B and behind end-to-end unless MMseqs2 is co-located.
- Uncertain outcomes: a measured FP8 result that either extends or closes the Trn3 story; a training $/step result that favours Trainium.
- Not expected: trn2 beating the RTX PRO 6000 on per-structure latency, or any Trainium part beating B200/B300 on BF16 throughput.
