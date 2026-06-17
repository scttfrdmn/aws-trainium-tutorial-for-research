"""Provenance capture for hardware validation runs.

The whole point of this module: a validation result is only trustworthy if it carries *where* and
*with what* it was produced. Every number the tutorial publishes must be traceable to a captured
environment, never typed by hand. This module defines the result schema and the (best-effort)
environment probes that fill it in.

Nothing here imports torch or Neuron at module load time -- it must be importable on a laptop (for
rendering/inspection) as well as on a Trainium instance (for capture).
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], timeout: int = 10) -> str | None:
    """Run a command and return stripped stdout, or None on any failure.

    Used for best-effort environment probes (neuron-ls, git, pip). A missing tool or non-zero
    exit is expected off-hardware and must never raise.
    """
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def utc_now_iso(clock: str | None = None) -> str:
    """Return an ISO-8601 UTC timestamp.

    `clock` lets callers inject a fixed time (tests, deterministic replay) since the workflow
    sandbox forbids argless clock reads. When None, uses the real wall clock.
    """
    if clock:
        return clock
    return datetime.now(timezone.utc).isoformat()


def detect_instance_type() -> str | None:
    """Return the EC2 instance type via IMDSv2, or None if not on EC2.

    Uses the token-based metadata flow (IMDSv2) because IMDSv1 is disabled on modern AMIs.
    """
    try:
        import urllib.request

        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_req, timeout=2).read().decode()  # noqa: S310
        meta_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(meta_req, timeout=2).read().decode()  # noqa: S310
    except Exception:
        return None


def detect_region() -> str | None:
    """Return the AWS region from env or IMDSv2 placement, or None."""
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        import urllib.request

        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_req, timeout=2).read().decode()  # noqa: S310
        meta_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/placement/region",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(meta_req, timeout=2).read().decode()  # noqa: S310
    except Exception:
        return None


def detect_neuron_sdk() -> dict[str, str | None]:
    """Best-effort capture of the installed Neuron / framework versions.

    Reads pip metadata (importlib.metadata) for the relevant packages plus the Neuron runtime
    version from `neuron-ls` if available. All fields are optional -- off-hardware they are None.
    """
    from importlib.metadata import PackageNotFoundError, version

    def pkg(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    info: dict[str, str | None] = {
        "torch": pkg("torch"),
        "torch_neuronx": pkg("torch-neuronx"),
        "neuronx_cc": pkg("neuronx-cc"),
        "neuronx_distributed": pkg("neuronx-distributed"),
        "transformers": pkg("transformers"),
        "optimum_neuron": pkg("optimum-neuron"),
        "neuron_sdk": None,
    }

    # The umbrella "Neuron SDK" version isn't a pip package; parse it from `neuron-ls` if present.
    ls = _run(["neuron-ls", "--version"]) or _run(["neuron-ls"])
    if ls:
        m = re.search(r"(\d+\.\d+\.\d+)", ls)
        if m:
            info["neuron_sdk"] = m.group(1)
    return info


def git_commit() -> str | None:
    """Return the short git SHA of the repo, or None outside a checkout."""
    return _run(["git", "rev-parse", "--short", "HEAD"])


@dataclass
class ValidationResult:
    """One example's validation outcome -- the unit of provenance.

    Serializes to validation/results/<example>.json. `status` is "passed" only if the example ran
    and met its registry thresholds; otherwise "failed" (with `error`) or "skipped".
    """

    example: str
    status: str  # "passed" | "failed" | "skipped"
    instance_type: str | None = None
    region: str | None = None
    commit: str | None = None
    timestamp: str = ""
    python: str = field(default_factory=platform.python_version)
    launcher: str | None = None
    wall_clock_s: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    cost_estimate_usd: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str | None] = field(default_factory=dict)
    log_path: str | None = None
    error: str | None = None

    def to_json(self) -> str:
        """Return a stable, pretty-printed JSON representation."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, results_dir: Path) -> Path:
        """Write this result to results_dir/<example-slug>.json and return the path."""
        results_dir.mkdir(parents=True, exist_ok=True)
        slug = self.example.replace("/", "__").removesuffix(".py")
        out = results_dir / f"{slug}.json"
        out.write_text(self.to_json())
        return out


def capture_environment(
    result: ValidationResult, clock: str | None = None
) -> ValidationResult:
    """Fill in the environment fields of `result` from the current host (in place).

    Safe to call anywhere: off-hardware the instance/region/Neuron fields simply come back None.
    """
    result.instance_type = result.instance_type or detect_instance_type()
    result.region = result.region or detect_region()
    result.commit = result.commit or git_commit()
    result.timestamp = result.timestamp or utc_now_iso(clock)
    result.versions = result.versions or detect_neuron_sdk()
    return result


def meets_thresholds(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> tuple[bool, list[str]]:
    """Check captured metrics against registry thresholds (`metric >= min` semantics).

    Returns (passed, failures) where failures lists human-readable reasons. A threshold whose
    metric is missing counts as a failure -- we don't pass on absent evidence.
    """
    failures: list[str] = []
    for key, minimum in thresholds.items():
        if key not in metrics:
            failures.append(f"missing metric '{key}' (needed >= {minimum})")
        elif metrics[key] < minimum:
            failures.append(f"{key}={metrics[key]:.4f} < required {minimum}")
    return (not failures), failures
