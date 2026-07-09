# Train on Trainium → serve on Inferentia (end-to-end pipeline)

> **Assumed knowledge:** you've run the [biomedical NER example](../use_cases/biomedical_ner.py) so
> the PyTorch/XLA training path is familiar, and read the
> [best-practices chapter](../../docs/trainium_development_best_practices.md).
> **What you'll get:** the *shape* of a real research deployment — train on Trainium, compile once
> for inference, serve on Inferentia — and the one boundary people get wrong (training vs inference
> graphs). This is a **template to adapt**, not a validated single-file example.

This is the tutorial's **"train then serve"** example, the last row of the
[choose-your-path guide](../../docs/choose_your_path.md). It orchestrates two EC2 phases:

1. **Train** on a Trainium instance using the XLA lazy-tensor path (`xm.xla_device()`,
   `xm.optimizer_step()`, `xm.mark_step()`), with a cost monitor logging spend every 5 min.
2. **Compile + serve** on an Inferentia instance: the trained checkpoint is
   `torch_neuronx.trace()`-d **once** into a frozen inference graph and served behind a small Flask
   `/predict` endpoint, again with cost tracking.

## ⚠️ Illustrative orchestration — read it, adapt it, don't run it blind

Unlike the [validated examples](../use_cases/), this one **launches real, billable EC2 instances**
and serves an **unauthenticated** endpoint, and it trains on a **placeholder dataset** (a `train.csv`
you supply in an S3 tarball). It is intentionally **not** in the validation harness — the harness
validates single-device `run(config)` examples; this is a multi-instance workflow. Every cost and
latency number it prints is an **estimate**.

So running it does nothing dangerous by default:

```bash
# DRY RUN (default) — explains what it would do + the cost/auth caveats, launches nothing:
python examples/complete_workflow/trainium_to_inferentia_pipeline.py

# Actually run it (launches instances — costs money). Supply your own bucket + dataset:
python examples/complete_workflow/trainium_to_inferentia_pipeline.py --run \
    --bucket my-ml-bucket --dataset-path s3://my-ml-bucket/data.tar.gz
```

## The one lesson: training graph ≠ inference graph

| Phase | Path | Key call |
|---|---|---|
| **Train** (Trainium) | XLA lazy tensors — gradients flow | `xm.optimizer_step()` + `xm.mark_step()` |
| **Serve** (Inferentia) | a **frozen** compiled graph — no gradients | `torch_neuronx.trace(model, example_inputs)` once |

You **cannot backprop through a traced graph** — `trace()` produces an inference-only artifact.
Tracing during training is the classic mistake; this example keeps the boundary explicit (train on
device, then reload best weights on CPU and trace once for serving).

## Cost numbers are illustrative

All `$/hr` rates live in one labeled table — `ILLUSTRATIVE_HOURLY_USD` at the top of the script —
and the comparison report derives every figure from it (no hand-typed savings percentages). They're
rough, region- and time-dependent estimates, **not** quotes or measured results. Confirm current
pricing at the [AWS pricing pages](https://aws.amazon.com/ec2/pricing/) before budgeting.

## Platform note (July 2026)

AWS positions **Trainium2 for both training and inference**, and NxD Inference dropped Inf2/Trn1
support in Neuron 2.29. This example serves on **Inf2** to show the classic train-Trn/serve-Inf
split; for new or large-scale serving, prefer **Trn2 + NxD Inference + the vLLM plugin**. See the
[Inferentia vs Trn2 decision guide](../../VERSION_MATRIX.md#-when-to-use-inferentia2-vs-trainium2-for-inference).
