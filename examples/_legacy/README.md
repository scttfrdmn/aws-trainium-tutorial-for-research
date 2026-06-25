# Legacy / not-validated examples

⚠️ **These examples are kept for reference only. They are NOT hardware-validated, and several use
mock or synthetic data, stale APIs, or illustrative pseudocode.** Do not treat their outputs
(metrics, costs, timings) as real.

They were moved here during the move to a *hardware-validated, real-data* tutorial. Each has at
least one of: fabricated metrics presented as results, the `trace()`-then-`backward()` antipattern
(see [best practices](../../docs/trainium_development_best_practices.md)), stale version pins, broken
imports, or all-mock AWS infrastructure.

**Use the validated examples instead** (see [`/VALIDATED.md`](../../VALIDATED.md)):
- [`examples/use_cases/`](../use_cases/) — six hardware-validated examples: biomedical NER, RODA
  satellite land-cover, the CV utilization spike, and the distillation / antibody / crystal SLMs.
- [`examples/complete_workflow/`](../complete_workflow/) — train → Inferentia pipeline (real AWS calls).
- [`examples/debugging/`](../debugging/) — runnable failure-diagnosis walkthrough.

If you want to revive one of these, the bar is the one in
[CONTRIBUTING.md](../../CONTRIBUTING.md): real (or clearly-labeled) data, a Trainium-friendly model,
a `run(config) -> dict[str, float]` entrypoint, and honest metrics — then it can be validated by the
[harness](../../validation/README.md) and graduate out of `_legacy/`.

Contents: `domain_specific/`, `end_to_end/`, `rag_pipeline/`, `frameworks/`, `deployment/`,
`integration/`, `benchmarking/`, `nki_optimization.py`.
