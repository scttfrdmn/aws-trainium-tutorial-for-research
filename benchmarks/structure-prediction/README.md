<!-- Copyright 2026 Scott Friedman. SPDX-License-Identifier: Apache-2.0 -->

# Structure prediction on Trainium2 vs G7e (Phase A harness)

A measurement harness for AF3-class protein structure prediction (Boltz-2 now, OpenFold3 later)
on AWS Trainium2 (trn2.3xlarge, via `jimburtoft/boltz-neuron`) against the NVIDIA RTX PRO 6000
(EC2 g7e.2xlarge), reported as $/structure with accuracy held within seed-to-seed noise of a CPU
fp32 oracle. Phase A profiles and baselines both platforms before any kernel work.

**No numbers exist yet.** Nothing here has run on Trainium or GPU hardware. The harness is
implemented and unit-tested on a workstation; every hardware step is an open issue labelled
`status:needs-hardware`.

Status, decisions and progress live on GitHub, not in this directory:

* Charter (objective, decisions, risks, reporting rules): [#54](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/issues/54)
* Phase A milestone and tracking issue: [Phase A — Profile and baseline](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/milestone/5), tracking issue [#55](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/issues/55)
* Background and numbers: [`docs/structure-prediction/`](../../docs/structure-prediction/)

## Layout

| path | what |
|---|---|
| `targets/` | `build_targets.py` (CASP14 set from Didi et al. 2025 + ground truth), `targets.json`, `targets.fasta` (generated, committed), `ceiling.json` |
| `msa/` | ColabFold-search with MMseqs2-GPU (g7e) and MMseqs2-CPU timing (r7a); `RUNBOOK.md` |
| `harness/manifest.py` | run manifest: instance, AMI, versions, commits, LNC, env, live prices |
| `harness/run.py` | CLI: `neuron`, `gpu`, `openfold-trt`, `cpu` backends; `--dry-run` |
| `harness/entry.py` | in-process wrapper: per-prediction timing, peak memory, fp32 for the oracle |
| `harness/score.py` | CA-RMSD, lDDT-CA, ligand RMSD, TM-score; the work plan's accuracy gate |
| `harness/profile_classify.py` | profiled time by operation class and engine |
| `harness/cost_table.py` | `results/phaseA.csv` and Markdown $/structure tables |
| `harness/oracle.py` | CPU fp32 oracle: parallel pinned workers, calibration, budget estimate |
| `harness/directory.py` | end-to-end batch: the whole target directory, one process per worker |
| `harness/stages.py` | per-host stage timer (boot, cache restore, MSA, ...) for the batch column |
| `infra/` | host definitions and launch/teardown (spawn where supported, `aws ec2` otherwise) |
| `RUNBOOK-A0.md`, `RUNBOOK-A.md` | the reproduction check and the full Phase A sequence |
| `results/EXPECTATIONS.md` | the work plan's expectations, copied before measurement |

## Use

```bash
uv pip install -e ".[dev,science]"          # from the repository root
cd benchmarks/structure-prediction
PYTHONPATH=. python -m harness.run predict --backend neuron --lnc 2 --core 0 \
  --target-id T1031 --input inputs/T1031.yaml --out-dir /tmp/sp --seed 0 --ds 1 --dry-run
```

Tests: `uv run pytest benchmarks/structure-prediction/tests` from the repository root.

## Pins

Neuron SDK 2.33.0, torch 2.14.0, torch-neuronx 2.14.3, boltz 2.2.1, boltz-neuron
`7cf012186a08cff5fff0d6718049d4e73df8b8d9`, OpenFold `be2ec1841f16c966c65ae0e7599ebbadc725757d`,
ColabFold v1.6.3, MMseqs2 18-8cc5c (`harness/pins.py`).

## License

Apache License 2.0 (this directory; see `LICENSE`). The rest of the repository is MIT.
Copyright 2026 Scott Friedman.
