# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Launch, list and tear down the Phase A hosts from infra/hosts.yaml.

    python infra/launch.py plan     trn2 [--spot] [--az us-east-1a] [--key-name K]
    python infra/launch.py launch   trn2 --yes [...]
    python infra/launch.py list
    python infra/launch.py teardown trn2 --yes

`plan` (the default without --yes) prints the exact command and runs nothing except the
read-only SSM lookup of the AMI. Every launch auto-terminates: spawn `--ttl` and `--on-complete
terminate`, or for the awscli path `--instance-initiated-shutdown-behavior terminate` plus a
user-data `shutdown -h +TTL`. Data volumes are created with DeleteOnTermination=true.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

HOSTS = Path(__file__).with_name("hosts.yaml")

Runner = Callable[[list[str]], str]


def run(argv: list[str]) -> str:
    """Run a command and return stdout."""
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout


def load_hosts(path: Path = HOSTS) -> dict[str, Any]:
    """Parse hosts.yaml."""
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    return data


def resolve_ami(ssm: str, region: str, runner: Runner = run) -> str:
    """AMI ID from an SSM public parameter (read-only)."""
    return runner(
        ["aws", "ssm", "get-parameter", "--region", region, "--name", ssm,
         "--query", "Parameter.Value", "--output", "text"]
    ).strip()  # fmt: skip


def user_data(ttl_hours: float, data_mount: str | None) -> str:
    """Boot script: hard TTL shutdown (terminates), and format/mount the data volume if any."""
    lines = ["#!/bin/bash", f"shutdown -h +{int(ttl_hours * 60)}"]
    if data_mount:
        # The data volume is the unmounted EBS disk larger than 1.5 TB (instance-store NVMe,
        # present on some g7e sizes, reports a different model string and is left alone).
        lines += [
            "dev=$(lsblk -bdnpo NAME,SIZE,MODEL | awk '/Elastic Block Store/ && $2 > 1500000000000 {print $1}' | head -1)",
            'if [ -n "$dev" ] && ! findmnt -S "$dev" >/dev/null; then',
            '  mkfs.xfs -f "$dev"',
            f'  mkdir -p {data_mount} && mount "$dev" {data_mount} && chown 1000:1000 {data_mount}',
            "fi",
        ]
    return "\n".join(lines) + "\n"


def tags(defaults: dict[str, Any], role: str, name: str) -> dict[str, str]:
    """Instance and volume tags (never an account ID)."""
    return {**defaults.get("tags", {}), "Name": name, "Role": role}


def plan(
    role: str,
    cfg: dict[str, Any],
    *,
    ami: str,
    spot: bool = False,
    az: str | None = None,
    key_name: str | None = None,
    iam_instance_profile: str | None = None,
    reservation_id: str | None = None,
    region: str | None = None,
) -> list[str]:
    """The launch argv for one host role.

    `reservation_id` targets a purchased EC2 Capacity Block for ML (its region and AZ must
    match the launch); spot does not apply to a block.
    """
    d, h = cfg["defaults"], cfg["hosts"][role]
    region = region or host_region(cfg, role)
    if reservation_id and spot:
        raise ValueError("a Capacity Block launch cannot also be spot")
    name = f"sp-phaseA-{role}"
    tg = tags(d, role, name)
    root = h.get("root_gb", d["root_gb"])
    if h["launcher"] == "spawn":
        if h.get("data_gb"):
            raise ValueError(
                f"{role}: spawn cannot create an empty data volume; use awscli"
            )
        argv = [
            "spawn", "launch", name,
            "--instance-type", h["instance_type"],
            "--region", region,
            "--ami", ami,
            "--volume-size", str(root),
            "--ttl", f"{h['ttl_hours']}h",
            "--on-complete", "terminate",
        ]  # fmt: skip
        for k, v in tg.items():
            if k != "Name":
                argv += ["--tag", f"{k}={v}"]
        if az:
            argv += ["--az", az]
        if spot:
            argv.append("--spot")
        if reservation_id:
            argv += ["--capacity-block", "--reservation-id", reservation_id]
        if key_name:
            argv += ["--key-name", key_name]
        if iam_instance_profile:
            argv += ["--instance-profile", iam_instance_profile]
        return argv
    bdm = [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": root,
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }
    ]
    if h.get("data_gb"):
        bdm.append(
            {
                "DeviceName": "/dev/sdf",
                "Ebs": {
                    "VolumeSize": h["data_gb"],
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        )
    tag_list = ",".join(f"{{Key={k},Value={v}}}" for k, v in tg.items())
    argv = [
        "aws", "ec2", "run-instances",
        "--region", region,
        "--instance-type", h["instance_type"],
        "--image-id", ami,
        "--count", "1",
        "--instance-initiated-shutdown-behavior", "terminate",
        "--block-device-mappings", json.dumps(bdm),
        "--tag-specifications",
        f"ResourceType=instance,Tags=[{tag_list}]",
        f"ResourceType=volume,Tags=[{tag_list}]",
        "--user-data", user_data(h["ttl_hours"], h.get("data_mount")),
    ]  # fmt: skip
    if spot:
        argv += ["--instance-market-options", "MarketType=spot"]
    if reservation_id:
        argv += [
            "--instance-market-options",
            "MarketType=capacity-block",
            "--capacity-reservation-specification",
            f"CapacityReservationTarget={{CapacityReservationId={reservation_id}}}",
        ]
    if az:
        argv += ["--placement", f"AvailabilityZone={az}"]
    if key_name:
        argv += ["--key-name", key_name]
    if iam_instance_profile:
        argv += ["--iam-instance-profile", f"Name={iam_instance_profile}"]
    return argv


def host_region(cfg: dict[str, Any], role: str) -> str:
    """A host's region: its own `region` in hosts.yaml, else the default."""
    region: str = cfg["hosts"][role].get("region", cfg["defaults"]["region"])
    return region


def list_argv(region: str) -> list[str]:
    """Describe this project's running instances."""
    return [
        "aws", "ec2", "describe-instances", "--region", region,
        "--filters", "Name=tag:Project,Values=structure-prediction",
        "Name=instance-state-name,Values=pending,running,stopping,stopped",
        "--query", "Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,Tags[?Key=='Role']|[0].Value,LaunchTime]",
        "--output", "text",
    ]  # fmt: skip


def teardown_argvs(role: str, region: str, runner: Runner = run) -> list[list[str]]:
    """Terminate every instance of this project with the given role."""
    out = runner(
        ["aws", "ec2", "describe-instances", "--region", region,
         "--filters", "Name=tag:Project,Values=structure-prediction", f"Name=tag:Role,Values={role}",
         "Name=instance-state-name,Values=pending,running,stopping,stopped",
         "--query", "Reservations[].Instances[].InstanceId", "--output", "text"]
    )  # fmt: skip
    ids = out.split()
    return (
        [
            [
                "aws",
                "ec2",
                "terminate-instances",
                "--region",
                region,
                "--instance-ids",
                *ids,
            ]
        ]
        if ids
        else []
    )


def main(argv: list[str] | None = None, runner: Runner = run) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="infra/launch.py")
    ap.add_argument("action", choices=("plan", "launch", "list", "teardown"))
    ap.add_argument("role", nargs="?")
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--az")
    ap.add_argument("--key-name")
    ap.add_argument("--iam-instance-profile")
    ap.add_argument("--reservation-id", help="purchased Capacity Block (cr-...)")
    ap.add_argument("--region", help="override the host's region for this command")
    ap.add_argument("--yes", action="store_true", help="actually launch / terminate")
    ns = ap.parse_args(argv)
    cfg = load_hosts()
    if ns.action == "list":
        regions = (
            [ns.region]
            if ns.region
            else sorted({host_region(cfg, r) for r in cfg["hosts"]})
        )
        for reg in regions:
            print(f"# {reg}")
            print(runner(list_argv(reg)))
        return 0
    if ns.role not in cfg["hosts"]:
        ap.error(f"role must be one of {sorted(cfg['hosts'])}")
    region = ns.region or host_region(cfg, ns.role)
    if ns.action == "teardown":
        cmds = teardown_argvs(ns.role, region, runner)
        for c in cmds:
            print(" ".join(c))
            if ns.yes:
                print(runner(c))
        if not cmds:
            print(f"no {ns.role} instances")
        return 0
    h = cfg["hosts"][ns.role]
    ami = resolve_ami(h["ami_ssm"], region, runner)
    cmd = plan(ns.role, cfg, ami=ami, spot=ns.spot, az=ns.az, key_name=ns.key_name,
               iam_instance_profile=ns.iam_instance_profile,
               reservation_id=ns.reservation_id, region=region)  # fmt: skip
    print(f"# {ns.role}: {h['instance_type']} via {h['launcher']} — {h['purpose']}")
    print(" ".join(repr(a) if " " in a or "\n" in a else a for a in cmd))
    if ns.action == "launch":
        if not ns.yes:
            print("# plan only; pass --yes to launch")
            return 0
        print(runner(cmd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
