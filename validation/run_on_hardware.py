#!/usr/bin/env python3
"""Validation orchestrator.

Two modes:

  * `--in-instance` : run ON a Neuron box. Imports each registered example, calls its run()
    entrypoint, times it, captures provenance, checks thresholds, writes results/<example>.json.
    This is what actually proves an example works on hardware.

  * (default/local)  : run on your workstation. Builds a LaunchPlan for the target instance and,
    only with `--yes`, provisions it (spawn or awscli) to execute the in-instance mode remotely.
    Without `--yes` it prints the plan and exits (dry run) -- no paid resources are touched.

Examples:
    # Local dry-run: show exactly what would be launched (no cost):
    python -m validation.run_on_hardware --instance trn1.2xlarge --region us-east-2

    # On a Trainium instance: validate everything and write artifacts:
    python -m validation.run_on_hardware --in-instance --all

    # On a Trainium instance: just one example, real config:
    python -m validation.run_on_hardware --in-instance --example ner_biomedical
"""

from __future__ import annotations

import argparse
import importlib
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Allow `python validation/run_on_hardware.py` as well as `-m validation.run_on_hardware`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation import launcher as launcher_mod  # noqa: E402
from validation import registry  # noqa: E402
from validation.provenance import (  # noqa: E402
    ValidationResult,
    capture_environment,
    meets_thresholds,
)

RESULTS_DIR = _REPO_ROOT / "validation" / "results"
LOGS_DIR = _REPO_ROOT / "validation" / "logs"


def _run_one(
    example: registry.Example, *, smoke: bool, clock: str | None
) -> ValidationResult:
    """Import and run a single example's entrypoint, returning a populated ValidationResult."""
    result = ValidationResult(
        example=example.module, status="failed", thresholds=dict(example.thresholds)
    )
    result = capture_environment(result, clock=clock)
    result.thresholds = dict(example.thresholds)
    config = dict(example.smoke_config if smoke else example.full_config)

    try:
        module = importlib.import_module(example.module)
    except Exception as exc:  # import failure is a real, reportable validation failure
        result.error = f"import failed: {exc}"
        return result

    entry = getattr(module, example.entrypoint, None)
    if not callable(entry):
        result.error = (
            f"module '{example.module}' has no callable '{example.entrypoint}'"
        )
        return result

    start = time.time()
    try:
        metrics = entry(config) or {}
    except Exception:
        result.wall_clock_s = round(time.time() - start, 2)
        result.error = "run() raised:\n" + traceback.format_exc(limit=6)
        return result
    result.wall_clock_s = round(time.time() - start, 2)

    if not isinstance(metrics, dict):
        result.error = (
            f"run() returned {type(metrics).__name__}, expected dict[str, float]"
        )
        return result

    result.metrics = {
        k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
    }

    # Smoke runs don't gate on the real thresholds (tiny data, 1 epoch); they only prove the code
    # path executes. Hardware (full) runs are gated.
    if smoke:
        result.status = "passed"
        result.error = None
        return result

    ok, failures = meets_thresholds(result.metrics, example.thresholds)
    result.status = "passed" if ok else "failed"
    if not ok:
        result.error = "threshold(s) not met: " + "; ".join(failures)
    return result


def run_in_instance(keys: list[str], *, smoke: bool, clock: str | None) -> int:
    """Run the selected examples on the current host and write artifacts. Returns exit code."""
    examples = (
        registry.EXAMPLES if not keys else [e for k in keys if (e := registry.get(k))]
    )
    if keys and len(examples) != len(keys):
        missing = [k for k in keys if registry.get(k) is None]
        print(f"ERROR: unknown example key(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    failed = 0
    for ex in examples:
        mode = "smoke" if smoke else "full"
        print(f"▶ validating {ex.key} ({ex.module}) [{mode}] ...", flush=True)
        result = _run_one(ex, smoke=smoke, clock=clock)
        path = result.write(RESULTS_DIR)
        status_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭"}.get(
            result.status, "?"
        )
        print(
            f"  {status_icon} {result.status}  ({result.wall_clock_s}s) -> {path.name}"
        )
        if result.error:
            print(f"     {result.error.splitlines()[0]}")
        if result.status == "failed":
            failed += 1

    print(f"\n{len(examples) - failed}/{len(examples)} passed.")
    return 1 if failed else 0


# Extra Python deps the six single-device examples import beyond the Neuron DLAMI's preinstalled
# stack. Deliberately does NOT include torch/torch-neuronx (preinstalled; reinstalling risks
# clobbering the Neuron build) — none of these pull torch as a hard dependency.
_EXAMPLE_DEPS = "transformers datasets rasterio pymatgen biopython scikit-learn scipy"


def _bootstrap_prefix(repo_url: str, git_ref: str) -> str:
    """Bash that prepares a bare Neuron DLAMI: activate its venv, clone the repo, install deps.

    Leaves the shell cd'd into the repo so the harness invocation that follows runs in-tree.
    """
    repo_dir = repo_url.rstrip("/").rsplit("/", 1)[-1]
    return (
        "set -e; "
        # Activate the DLAMI's preinstalled PyTorch-Neuron venv (name is SDK-versioned; glob it).
        "for v in /opt/aws_neuronx_venv_pytorch_2_9 /opt/aws_neuronx_venv_pytorch*; do "
        '[ -f "$v/bin/activate" ] && . "$v/bin/activate" && break; done; '
        "cd ~; "
        f"[ -d {repo_dir} ] || git clone --depth 1 --branch {git_ref} {repo_url}; "
        f"cd {repo_dir}; "
        f"pip install --no-input {_EXAMPLE_DEPS}; "
    )


def run_local(args: argparse.Namespace) -> int:
    """Build a launch plan and (only with --yes) provision the instance. Returns exit code."""
    keys = args.example or [e.key for e in registry.EXAMPLES]
    example_flag = (
        "--all" if not args.example else f"--example {' '.join(args.example)}"
    )
    # Run inside a detached tmux session so the validation survives SSH disconnects and can be
    # re-attached for live progress (`tmux attach -t validate`) without racing connection timeouts.
    # The harness still writes results/<example>.json + logs regardless of the session.
    # python3 -u keeps stdout unbuffered so epoch/progress prints appear live in the log/tmux pane
    # (without it, tee block-buffers and you see nothing until the run ends -- a real gotcha).
    # A persistent compile cache makes a fresh cloud instance reuse compiled graphs instead of
    # recompiling from cold every launch. An s3:// URL is what survives reprovisioning (a local dir
    # would be empty on each new box). Exported before the run so every example's compiles hit it.
    cache_env = (
        f"export NEURON_COMPILE_CACHE_URL={shlex.quote(args.cache_url)}; "
        if args.cache_url
        else ""
    )
    # After the run, optionally hand artifacts + log back via S3 so a self-terminating instance
    # doesn't take them to the grave (no inbound SSH once it's gone). Uses `;` (not `&&`) so it runs
    # even if the harness exits non-zero -- a failed run's log is exactly what you need. Requires an
    # instance profile with s3:PutObject on the bucket (see --iam-instance-profile).
    if args.results_s3:
        base = args.results_s3.rstrip("/")
        results_sync = (
            f"; aws s3 sync validation/results {base}/results/ "
            f"; aws s3 cp ~/validate.log {base}/logs/{keys[0] if len(keys) == 1 else 'all'}.log"
        )
    else:
        results_sync = ""

    # Where the harness runs from: a bootstrap run clones + cds into the repo itself; otherwise assume
    # the box already has the checkout.
    if args.bootstrap:
        setup = _bootstrap_prefix(args.repo_url, args.git_ref)
    else:
        setup = "cd /opt/tutorial 2>/dev/null || cd ~/tutorial 2>/dev/null || cd ~/aws-trainium-tutorial-for-research; "

    inner = (
        f"{setup}"
        f"{cache_env}"
        f"python3 -u -m validation.run_on_hardware --in-instance {example_flag} 2>&1 | tee ~/validate.log"
        f"{results_sync}"
    )
    # tmux keeps an interactive (spawn/SSH) run alive across disconnects; a headless awscli/user-data
    # launch has no tty, so run inline there. Either way the harness writes artifacts + the S3 sync.
    launcher = launcher_mod.choose_launcher(args.instance, args.launcher)
    if launcher == "awscli":
        remote = inner
    else:
        remote = (
            "set -e; "
            f"tmux new-session -d -s validate {shlex.quote(inner)} || {inner}; "
            "echo 'validation running in tmux session \"validate\" (tmux attach -t validate)'"
        )
    plan = launcher_mod.build_plan(
        args.instance,
        args.region,
        name=f"neuron-validate-{keys[0] if len(keys) == 1 else 'all'}",
        remote_command=remote,
        prefer=args.launcher,
        max_hours=args.max_hours,
        use_spot=not args.on_demand,
        cost_limit_usd=args.cost_limit,
        iam_instance_profile=args.iam_instance_profile,
        image_id=args.image_id,
    )

    print(plan.describe())
    print()
    if not args.yes:
        print(
            "DRY RUN. No instance launched. Re-run with --yes to actually provision (this costs money)."
        )
        return 0

    print(f"Launching via {plan.launcher} ...")
    try:
        completed = subprocess.run(plan.command, check=False)
    except (OSError, ValueError) as exc:
        print(f"ERROR launching: {exc}", file=sys.stderr)
        return 1
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    p = argparse.ArgumentParser(
        description="Run tutorial examples on real Neuron hardware and capture provenance."
    )
    p.add_argument(
        "--in-instance",
        action="store_true",
        help="Run examples on THIS host (use on a Neuron instance).",
    )
    p.add_argument(
        "--all", action="store_true", help="Validate all registered examples."
    )
    p.add_argument(
        "--example",
        nargs="*",
        metavar="KEY",
        help="Specific example key(s) from the registry.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny CPU smoke run (proves code path; skips thresholds).",
    )
    p.add_argument(
        "--instance",
        default="trn1.2xlarge",
        help="Instance type for local launch (default: trn1.2xlarge).",
    )
    p.add_argument(
        "--region",
        default="us-east-2",
        help="AWS region for local launch (default: us-east-2).",
    )
    p.add_argument(
        "--launcher",
        choices=["auto", "spawn", "awscli"],
        default="auto",
        help="Launcher preference.",
    )
    p.add_argument(
        "--max-hours", type=float, default=2.0, help="Auto-terminate cap (hours)."
    )
    p.add_argument(
        "--cost-limit",
        type=float,
        default=5.0,
        help="Hard $ ceiling via spawn --cost-limit (0=disable).",
    )
    p.add_argument(
        "--on-demand", action="store_true", help="Use on-demand instead of spot."
    )
    p.add_argument(
        "--cache-url",
        default=None,
        metavar="URL",
        help="NEURON_COMPILE_CACHE_URL for the run (e.g. s3://bucket/neuron-cache). On a fresh "
        "cloud instance a local cache is always cold; an s3:// URL persists compiled graphs across "
        "reprovisions so you don't recompile every launch. See best-practices §1b.",
    )
    p.add_argument(
        "--iam-instance-profile",
        default=None,
        metavar="NAME",
        help="Attach this IAM instance profile (awscli launcher) so the in-instance run can push "
        "artifacts to S3 without SSH. Pair with --results-s3.",
    )
    p.add_argument(
        "--image-id",
        default=None,
        metavar="AMI",
        help="Pin a specific AMI (e.g. a private/pre-release Neuron image) instead of the public "
        "DLAMI resolved from SSM. Use to validate an SDK version before AWS publishes its DLAMI.",
    )
    p.add_argument(
        "--bootstrap",
        action="store_true",
        help="Prepend repo clone + example-dependency install to the remote command, for a bare "
        "Neuron DLAMI that doesn't already have the tutorial checked out. Activates the DLAMI's "
        "preinstalled PyTorch-Neuron venv; does NOT reinstall torch/torch-neuronx.",
    )
    p.add_argument(
        "--repo-url",
        default="https://github.com/scttfrdmn/aws-trainium-tutorial-for-research",
        help="Repo to clone when --bootstrap is set.",
    )
    p.add_argument(
        "--git-ref",
        default="main",
        help="Branch/tag/SHA to clone when --bootstrap is set.",
    )
    p.add_argument(
        "--results-s3",
        default=None,
        metavar="URL",
        help="After the run, `aws s3 sync validation/results` to this s3:// URL (artifact hand-back "
        "from a self-terminating instance). Requires --iam-instance-profile with s3:PutObject.",
    )
    p.add_argument(
        "--yes", action="store_true", help="Actually launch (omit for a dry run)."
    )
    p.add_argument(
        "--clock",
        default=None,
        help="Fixed ISO timestamp for deterministic artifacts (testing).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    if args.in_instance:
        keys = args.example or ([] if args.all else [])
        if not keys and not args.all:
            print("Specify --all or --example KEY for --in-instance.", file=sys.stderr)
            return 2
        return run_in_instance(keys, smoke=args.smoke, clock=args.clock)
    return run_local(args)


if __name__ == "__main__":
    raise SystemExit(main())
