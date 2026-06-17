# Choose Your Path: does Trainium fit *your* problem, and where do you start?

> **Assumed knowledge:** you know your own workload (what you train/run and roughly how big). No
> Neuron experience needed.
> **What you'll get:** an honest verdict on whether Trainium suits your problem, and a concrete
> first example to follow.

Most "should I use Trainium?" guides only say "yes, it's cheaper." That's not useful when your
workload is a bad fit — you'll fight the hardware and lose. This guide routes you by **problem
shape**, tells you when the answer is *"use a GPU instead,"* and points you at the right starting
example.

---

## Step 1 — What shape is your workload?

Trainium is an **ahead-of-time-compiled, bf16-native, static-shape, matmul-optimized** accelerator
(see [best practices](trainium_development_best_practices.md) and
[novel kernels](novel_kernels_on_trainium.md) for *why*). That shape determines fit.

| Your workload is mostly… | Fit | Why |
|---|---|---|
| **Transformer training / fine-tuning** (LLM, BERT-family, ViT) | ✅ Strong | Dense matmul + attention; static shapes; the mainstream Neuron path. |
| **Large-batch / high-throughput inference** | ✅ Strong | Throughput-oriented; NxD Inference + vLLM target this. |
| **CNN / dense vision** training | ✅ Good | Convolutions map to the tensor engine; keep input sizes fixed. |
| **Classic dense ML** (MLPs, matrix factorization, big linear algebra) | ✅ Good | Matmul-heavy, regular. |
| **Scientific compute that's matmul/FFT-heavy with regular tiling** | 🟡 Promising | Possible via NKI; see the novel-kernels chapter — but expect real kernel work. |
| **Heavy dynamic shapes** (wildly varying seq lengths, ragged batches) | 🟡 Caution | Each new shape recompiles. Workable with bucketing/padding, but it's effort. |
| **Data-dependent control flow / pointer-chasing / graph traversal** | 🔴 Poor | The compiled, tiled model dislikes irregular branching and random access. |
| **Tiny models / tiny data, latency-trivial** | 🔴 Overkill | Compile overhead and instance cost aren't worth it; use CPU/GPU. |
| **Single-GPU-fits, you need it *today*, one-off** | ⚪ Maybe not | If a GPU already works and the run is short, the porting effort may not pay back. |

If you're in a 🔴 row, **stop here** — use a GPU. Being honest about this is the whole point.

---

## Step 2 — The five questions that decide it

Score your workload. The more "yes", the better the fit (details + the *why* in
[novel kernels: does your problem map?](novel_kernels_on_trainium.md#does-your-problem-map)):

1. **Matmul-dominated?** Is most of the FLOPs in dense matrix multiply / attention / convolution?
2. **Static shapes?** Are tensor shapes fixed (or a small bucketed set) across steps?
3. **Regular data access?** Bulk, predictable reads — not random scatter/gather or pointer chasing?
4. **bf16-tolerant?** Can the math live in bf16 (with fp32 accumulation) without falling apart?
5. **Reused enough to amortize compilation?** You'll run it many times (training loop, served model)
   — not a single 30-second job?

- **4–5 yes →** strong fit. Proceed.
- **2–3 yes →** workable with effort (bucketing, NKI, precision care). Read best-practices first.
- **0–1 yes →** likely the wrong tool. Use a GPU and move on.

---

## Step 3 — Pick your starting point

| If you're doing… | Start with | Then |
|---|---|---|
| **Any first run / NLP fine-tune** | [Biomedical NER](../examples/use_cases/biomedical_ner.py) (validated) | best-practices → tools & debugging |
| **An LLM fine-tune (LoRA)** | [Qwen3 LoRA fine-tune](../examples/use_cases/qwen3_lora_finetune.py) (optimum-neuron) | best-practices + sizing for your instance |
| **Train then serve** | [Trainium → Inferentia pipeline](../examples/complete_workflow/trainium_to_inferentia_pipeline.py) | [Inferentia vs Trn2 decision guide](../VERSION_MATRIX.md#-when-to-use-inferentia2-vs-trainium2-for-inference) |
| **Satellite / image classification (CNN)** | [Satellite land-cover](../examples/use_cases/satellite_landcover.py) | keep tiles fixed-size |
| **Quant finance / time series** | [Financial modeling](../examples/use_cases/financial_modeling.py) | static-shape + bf16 review |
| **A custom kernel / new operator** | [Novel kernels on Trainium](novel_kernels_on_trainium.md) | NKI simulation → hardware |
| **Multi-NeuronCore / bigger models** | [Distributed training](../examples/distributed/) | best-practices (sharding) |
| **"It's slow / it `nan`s / it won't compile"** | [Neuron tools & debugging](neuron_tools_and_debugging.md) | the symptom→tool table |

---

## Step 4 — Domain notes (honest fit by field)

- **NLP / LLMs** — the strongest, best-trodden fit. Start at the NER example; scale to LoRA.
- **Genomics / bioinformatics** — *sequence-model* tasks (NER over literature, protein/DNA
  transformers) fit well; *alignment/assembly/variant-calling* algorithms (pointer-heavy, irregular)
  generally do **not** — keep those on CPU and use Trainium only for the ML stages.
- **Climate / geospatial** — gridded tensor models (CNN/transformer over fixed grids) fit; irregular
  mesh / adaptive-resolution solvers map poorly without serious kernel work.
- **Quant finance** — Monte Carlo and dense linear algebra fit; tick-by-tick event simulation does not.
- **Physics / chemistry** — dense methods (DFT-style matmul, MP2-type contractions) are promising via
  NKI; irregular n-body / sparse solvers are hard.
- **Computer vision** — fixed-size image pipelines fit; variable-resolution / heavy dynamic
  preprocessing needs bucketing.

> When a domain says "poor fit," that's not a failure of the tutorial — it's the honest answer.
> Trainium is a specialized tool, not a universal GPU replacement.

---

## Still unsure?

Run the cheapest possible probe: take the [NER example](../examples/use_cases/biomedical_ner.py),
swap in a small slice of *your* data and *your* model, and run the CPU smoke path
(`NER_SMOKE=1 …`). If it adapts cleanly with static shapes, you're likely a good fit — validate on
real hardware next. If you find yourself fighting dynamic shapes or control flow at every turn,
that's your answer too.
