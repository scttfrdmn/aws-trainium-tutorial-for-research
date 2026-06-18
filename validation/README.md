# Hardware Validation Harness

This directory holds the machinery that keeps one promise: **every tutorial example is proven to
run on real AWS Neuron hardware**, with the proof captured as a provenance artifact rather than a
hand-typed claim. See [`docs/REVAMP_PLAN.md`](../docs/REVAMP_PLAN.md) for the full design.

## How it works

```
registry.py         catalog of validatable examples + pass thresholds
provenance.py       result schema + environment capture (SDK/instance/commit/metrics)
launcher.py         spawn|awscli launcher (auto-terminate + cost ceiling)
run_on_hardware.py  orchestrator (local launch  OR  --in-instance execution)
render_status.py    results/*.json  ->  /VALIDATED.md
results/            captured provenance artifacts (committed — the evidence)
logs/               raw run logs (gitignored)
```

An example is **validatable** when it exposes `def run(config: dict) -> dict[str, float]`. The
harness imports it, runs it, times it, and checks the returned metrics against the registry
thresholds (`metric >= min`).

## Usage

```bash
# 1) From your workstation — see exactly what would launch (no cost):
python -m validation.run_on_hardware --instance trn1.2xlarge --region us-east-2 --example ner_biomedical

# 2) Launch for real (spot + auto-terminate + --cost-limit). Needs AWS creds:
python -m validation.run_on_hardware --instance trn1.2xlarge --region us-east-2 --example ner_biomedical --yes

# 2b) Reuse compiled graphs across throwaway instances with an S3 compile cache (no cold recompile
#     on every fresh box). Highly recommended in the cloud — see best-practices §1b:
python -m validation.run_on_hardware --instance trn1.2xlarge --region us-east-2 \
    --example ner_biomedical --cache-url s3://my-bucket/neuron-cache --yes

# 3) On a Neuron instance — run examples and write artifacts:
python -m validation.run_on_hardware --in-instance --all

# 4) Regenerate the status table:
python -m validation.render_status            # writes /VALIDATED.md
python -m validation.render_status --check    # exit 1 if anything is unvalidated (CI gate)
```

## Launcher: spawn vs awscli

Per [REVAMP_PLAN §0b](../docs/REVAMP_PLAN.md), the harness prefers [`spawn`](https://github.com/spore-host/spawn)
when it's installed (launch + lifecycle + guaranteed auto-terminate + `--cost-limit`), and falls
back to raw `aws ec2 run-instances` otherwise — so it stays runnable without spore.host tools.
Choose explicitly with `--launcher spawn|awscli|auto`.

> Trn2 single-device (`trn2.3xlarge`) via `spawn` is pending [spore-host/spawn#205](https://github.com/spore-host/spawn/issues/205);
> until then it uses the awscli launcher.

## Running long jobs on the instance

The launcher runs validation inside a **detached tmux session** (`tmux new-session -d -s validate`)
with **unbuffered output** (`python3 -u … | tee ~/validate.log`). This matters:

- **tmux** keeps the run alive across SSH disconnects; re-attach with `tmux attach -t validate`.
- **`-u`** makes epoch/progress prints appear live. Without it, `tee` block-buffers stdout and you
  see *nothing* until the run ends — which looks like a hang on a slow single-NeuronCore run.

To watch progress without holding a connection: `spawn connect <id> -- 'tail -f ~/validate.log'`,
or attach the tmux session. Results land in `validation/results/*.json` regardless.

## Safety

Every launch path enforces **auto-termination** (spawn `--on-complete terminate --ttl`, or awscli
`instance-initiated-shutdown-behavior=terminate` + a `timeout` in user-data) plus a dollar ceiling,
so a failed or hung run cannot leak cost. Nothing is launched without an explicit `--yes`.
