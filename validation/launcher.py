"""Hardware launcher abstraction: spawn when available, boto3/aws-cli otherwise.

This runs on your workstation. It provisions a Neuron instance, runs the in-instance half of the
harness there, and brings results back. The design decision:

  * Prefer `spawn` (spore.host) -- it gives launch + guaranteed auto-terminate + lifecycle for free.
  * Fall back to raw `aws ec2` so the harness stays runnable for someone without spore.host tools.

Both launchers enforce the same non-negotiable safety property: **the instance auto-terminates**,
so a failed or hung validation run cannot leak cost.

Phase 1 implements launch *planning* and the spawn/awscli command construction with a real
dry-run, plus the AMI/region resolution. Actually spawning a paid instance is gated behind an
explicit --yes from the operator (never launched implicitly).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# Single-Trainium-device baselines we validate on. Trn2 single-device (trn2.3xlarge) is added once
# spawn supports it (see spore-host/spawn#205); until then it uses the awscli launcher.
SPAWN_SUPPORTED_TODAY = {"trn1.2xlarge", "inf2.xlarge", "inf2.8xlarge", "trn1.32xlarge"}


@dataclass
class LaunchPlan:
    """A fully-resolved, reviewable description of what *would* be launched.

    Printing this (and getting operator sign-off) is mandatory before any paid launch.
    """

    instance_type: str
    region: str
    launcher: str  # "spawn" | "awscli"
    ami_id: str | None
    max_hours: float
    use_spot: bool
    name: str
    command: list[str]
    cost_limit_usd: float | None = None

    def describe(self) -> str:
        """Return a human-readable summary for operator confirmation."""
        spot = "spot" if self.use_spot else "on-demand"
        ami = self.ami_id or "(resolved on launch)"
        cost = f"${self.cost_limit_usd}" if self.cost_limit_usd else "(none)"
        return (
            f"Launch plan:\n"
            f"  launcher      : {self.launcher}\n"
            f"  instance type : {self.instance_type} ({spot})\n"
            f"  region        : {self.region}\n"
            f"  AMI           : {ami}\n"
            f"  auto-terminate: {self.max_hours} h (ttl) + on-complete\n"
            f"  cost ceiling  : {cost}\n"
            f"  name/tag      : {self.name}\n"
            f"  command       : {' '.join(self.command)}"
        )


def spawn_available() -> bool:
    """True if the `spawn` CLI is on PATH."""
    return shutil.which("spawn") is not None


def choose_launcher(instance_type: str, prefer: str = "auto") -> str:
    """Pick a launcher for an instance type.

    `prefer` is "auto" | "spawn" | "awscli". "auto" uses spawn when it's installed AND known to
    support the instance type today, else falls back to awscli.
    """
    if prefer == "spawn":
        return "spawn"
    if prefer == "awscli":
        return "awscli"
    if spawn_available() and instance_type in SPAWN_SUPPORTED_TODAY:
        return "spawn"
    return "awscli"


def resolve_neuron_dlami(
    region: str, pytorch_version: str = "2.9", os_version: str = "ubuntu-24.04"
) -> str | None:
    """Resolve the current Neuron DLAMI image id from SSM (None if unavailable).

    The parameter path is versioned by PyTorch release, e.g. (verified us-west-2, June 2026):
        /aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id
    """
    param = f"/aws/service/neuron/dlami/pytorch-{pytorch_version}/{os_version}/latest/image_id"
    try:
        out = subprocess.run(
            [
                "aws",
                "ssm",
                "get-parameter",
                "--region",
                region,
                "--name",
                param,
                "--query",
                "Parameter.Value",
                "--output",
                "text",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    val = out.stdout.strip()
    return val if (out.returncode == 0 and val.startswith("ami-")) else None


def build_plan(
    instance_type: str,
    region: str,
    *,
    name: str,
    remote_command: str,
    prefer: str = "auto",
    max_hours: float = 2.0,
    use_spot: bool = True,
    cost_limit_usd: float | None = 5.0,
) -> LaunchPlan:
    """Construct a LaunchPlan (no side effects beyond read-only AMI resolution).

    `remote_command` is the shell to run on the instance (the in-instance harness invocation).
    `cost_limit_usd` maps to spawn's --cost-limit (hard $ ceiling); None disables it.
    """
    launcher = choose_launcher(instance_type, prefer)
    ami = resolve_neuron_dlami(region)

    if launcher == "spawn":
        # spawn owns AMI selection, spot, and auto-terminate via its own flags. Flags verified
        # against `spawn launch --help` (June 2026): name is positional; --command runs the
        # workload; --on-complete terminate + --ttl give belt-and-suspenders auto-terminate;
        # --cost-limit is a hard dollar ceiling.
        cmd = [
            "spawn",
            "launch",
            name,
            "--instance-type",
            instance_type,
            "--region",
            region,
            "--on-complete",
            "terminate",  # terminate when the workload signals completion
            "--ttl",
            f"{int(max_hours)}h",  # hard cap even if the command hangs
        ]
        if use_spot:
            cmd.append("--spot")
        if cost_limit_usd:
            cmd += ["--cost-limit", str(cost_limit_usd)]
        cmd += ["--command", remote_command]
    else:
        # awscli fallback: a minimal, explicit run-instances with auto-terminate baked into
        # user-data (shutdown after the command) plus instance-initiated-shutdown-behavior=terminate.
        cmd = [
            "aws",
            "ec2",
            "run-instances",
            "--region",
            region,
            "--instance-type",
            instance_type,
            "--image-id",
            ami or "AMI_UNRESOLVED",
            "--instance-initiated-shutdown-behavior",
            "terminate",
            "--instance-market-options",
            "MarketType=spot" if use_spot else "",
            "--tag-specifications",
            f"ResourceType=instance,Tags=[{{Key=Name,Value={name}}},{{Key=spore,Value=validation}}]",
            "--count",
            "1",
            # user-data wraps the remote command with a hard-timeout shutdown for cost safety.
            "--user-data",
            _awscli_user_data(remote_command, max_hours),
        ]
        cmd = [c for c in cmd if c != ""]  # drop empty market-options when on-demand

    return LaunchPlan(
        instance_type=instance_type,
        region=region,
        launcher=launcher,
        ami_id=ami,
        max_hours=max_hours,
        use_spot=use_spot,
        name=name,
        command=cmd,
        cost_limit_usd=cost_limit_usd if launcher == "spawn" else None,
    )


def _awscli_user_data(remote_command: str, max_hours: float) -> str:
    """Build user-data that runs the command then terminates, with a hard timeout backstop."""
    timeout_s = int(max_hours * 3600)
    return (
        "#!/bin/bash\n"
        "set -x\n"
        f"timeout {timeout_s} bash -lc {remote_command!r}\n"
        "shutdown -h now\n"
    )
