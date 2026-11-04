# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Run manifest: everything needed to reproduce or price a run, captured at run time.

`capture_manifest()` is the one entry point. Every external lookup goes through injectable
callables (a command runner, an HTTP getter for instance metadata, and boto3-style pricing/EC2
clients), so the whole function is unit-tested with mocks. Lookups that fail are recorded as
null with the error, never guessed: a manifest says what it could not find.

Prices are fetched, never hard-coded: on-demand from the AWS Price List API (`pricing`
GetProducts), spot from EC2 `describe-spot-price-history` for the run's AZ.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "sp-manifest/1.0"
IMDS = "http://169.254.169.254/latest"
ENV_PREFIXES = (
    "BOLTZ_NEURON_",
    "NEURON_",
    "BOLTZ_FIX_",
    "TORCH_NEURONX_",
    "CUDA_VISIBLE",
)

# Region code -> Price List API "location" name (the API filters on the display name).
REGION_LOCATION = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-2": "US West (Oregon)",
    "eu-west-2": "EU (London)",
    "eu-south-2": "EU (Spain)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
}

Runner = Callable[[list[str]], str]
HttpGet = Callable[[str, Mapping[str, str]], str]


def run_command(argv: list[str]) -> str:
    """Run a command and return stdout; raise on failure or a missing binary."""
    out = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=True)
    return out.stdout


def http_get(url: str, headers: Mapping[str, str]) -> str:
    """GET a URL (used for IMDSv2); short timeout because off-EC2 it never answers."""
    import urllib.request

    req = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=2) as resp:
        body: bytes = resp.read()
    return body.decode()


def _imds_put_token() -> str:
    import urllib.request

    req = urllib.request.Request(
        f"{IMDS}/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"},
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        body: bytes = resp.read()
    return body.decode()


def _try(errors: dict[str, str], key: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except (
        Exception
    ) as exc:  # recorded, not raised: the manifest reports what it could not find
        errors[key] = f"{type(exc).__name__}: {exc}"
        return None


def instance_identity(
    http: HttpGet = http_get, token: Callable[[], str] = _imds_put_token
) -> dict[str, Any]:
    """Instance type, AZ, region and AMI from IMDSv2 (identity document)."""
    tok = token()
    doc = json.loads(
        http(
            f"{IMDS}/dynamic/instance-identity/document",
            {"X-aws-ec2-metadata-token": tok},
        )
    )
    return {
        "instance_type": doc["instanceType"],
        "availability_zone": doc["availabilityZone"],
        "region": doc["region"],
        "ami_id": doc["imageId"],
        "instance_id": doc["instanceId"],
    }


def host_resources(meminfo: str | None = None) -> dict[str, Any]:
    """Host vCPU count and RAM (GiB) from the OS."""
    if meminfo is None:
        meminfo = Path("/proc/meminfo").read_text()
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo, re.M)
    return {
        "vcpus": os.cpu_count(),
        "ram_gib": round(int(m.group(1)) / 1024**2, 1) if m else None,
    }


def neuron_inventory(run: Runner) -> dict[str, Any]:
    """`neuron-ls --json-output` plus the Neuron package versions."""
    devices = json.loads(run(["neuron-ls", "--json-output"]))
    pkgs = _pip_versions(
        run,
        (
            "neuronx-cc",
            "torch-neuronx",
            "torch",
            "libneuronxla",
            "nki",
            "neuronx-runtime-lib",
        ),
    )
    rt = _try({}, "rt", lambda: run(["dpkg-query", "-W", "aws-neuronx-runtime-lib"]))
    drv = _try({}, "drv", lambda: run(["dpkg-query", "-W", "aws-neuronx-dkms"]))
    return {
        "kind": "neuron",
        "devices": devices,
        "device_count": len(devices) if isinstance(devices, list) else None,
        "packages": pkgs,
        "runtime_deb": rt.split()[-1] if rt else None,
        "driver_deb": drv.split()[-1] if drv else None,
    }


def gpu_inventory(run: Runner) -> dict[str, Any]:
    """`nvidia-smi` GPU list, driver and CUDA versions, plus cuEquivariance/TensorRT."""
    q = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader",
        ]
    )
    gpus = [
        dict(
            zip(
                ("name", "memory", "driver", "bus"),
                (c.strip() for c in line.split(",")),
                strict=True,
            )
        )
        for line in q.strip().splitlines()
    ]
    head = run(["nvidia-smi"])
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", head)
    pkgs = _pip_versions(
        run,
        (
            "torch",
            "cuequivariance-torch",
            "cuequivariance-ops-torch-cu12",
            "cuequivariance-ops-torch-cu13",
            "tensorrt",
        ),
    )
    return {
        "kind": "gpu",
        "gpus": gpus,
        "device_count": len(gpus),
        "driver_version": gpus[0]["driver"] if gpus else None,
        "cuda_version": cuda.group(1) if cuda else None,
        "packages": pkgs,
    }


def _pip_versions(run: Runner, names: tuple[str, ...]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            # importlib.metadata, not pip: uv-created venvs ship without pip.
            code = f"import importlib.metadata as m; print(m.version({name!r}))"
            text = run([sys.executable, "-c", code]).strip()
            out[name] = text or None
        except Exception:
            out[name] = None
    return out


def git_commit(run: Runner, path: str | os.PathLike[str] | None) -> str | None:
    """HEAD commit of the checkout at `path` (None if not a checkout)."""
    if not path:
        return None
    try:
        return run(["git", "-C", str(path), "rev-parse", "HEAD"]).strip()
    except Exception:
        return None


def relevant_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Every BOLTZ_NEURON_* / NEURON_* (and related) variable, verbatim."""
    return {k: v for k, v in sorted(environ.items()) if k.startswith(ENV_PREFIXES)}


def on_demand_price(pricing: Any, instance_type: str, region: str) -> dict[str, Any]:
    """Linux, shared-tenancy, no-preinstalled-software on-demand $/h from the Price List API."""
    location = REGION_LOCATION.get(region)
    if location is None:
        raise ValueError(f"no Price List location name for region {region}")
    filters = [
        {"Type": "TERM_MATCH", "Field": k, "Value": v}
        for k, v in (
            ("instanceType", instance_type),
            ("location", location),
            ("operatingSystem", "Linux"),
            ("tenancy", "Shared"),
            ("preInstalledSw", "NA"),
            ("capacitystatus", "Used"),
        )
    ]
    resp = pricing.get_products(ServiceCode="AmazonEC2", Filters=filters, MaxResults=10)
    prices = []
    for raw in resp.get("PriceList", []):
        item = json.loads(raw) if isinstance(raw, str) else raw
        for term in item.get("terms", {}).get("OnDemand", {}).values():
            for dim in term.get("priceDimensions", {}).values():
                usd = float(dim["pricePerUnit"]["USD"])
                if usd > 0:
                    prices.append(usd)
    if len(set(prices)) != 1:
        raise ValueError(f"expected one on-demand price, got {sorted(set(prices))}")
    return {
        "usd_per_hour": prices[0],
        "source": "pricing:GetProducts",
        "location": location,
    }


def spot_price(ec2: Any, instance_type: str, availability_zone: str) -> dict[str, Any]:
    """Most recent Linux/UNIX spot price for the run's AZ."""
    resp = ec2.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        AvailabilityZone=availability_zone,
        MaxResults=1,
    )
    hist = resp.get("SpotPriceHistory", [])
    if not hist:
        raise ValueError(
            f"no spot price history for {instance_type} in {availability_zone}"
        )
    rec = hist[0]
    ts = rec.get("Timestamp")
    return {
        "usd_per_hour": float(rec["SpotPrice"]),
        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else ts,
        "source": "ec2:DescribeSpotPriceHistory",
    }


def capture_manifest(
    *,
    accelerator: str,
    lnc: int | None = None,
    repo_paths: Mapping[str, str | None] | None = None,
    extra: Mapping[str, Any] | None = None,
    run: Runner = run_command,
    http: HttpGet = http_get,
    imds_token: Callable[[], str] = _imds_put_token,
    pricing_client: Any = None,
    ec2_client: Any = None,
    environ: Mapping[str, str] | None = None,
    meminfo: str | None = None,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> dict[str, Any]:
    """Capture the manifest for one run.

    Args:
        accelerator: "neuron", "gpu" or "cpu" (selects the inventory probe).
        lnc: logical NeuronCore config in force (Neuron only).
        repo_paths: name -> checkout path for git commits (boltz, boltz-neuron, this repo, ...).
        extra: caller fields copied in verbatim (e.g. the backend's pinned versions).
        run: command runner (neuron-ls, nvidia-smi, pip, git).
        http: HTTP getter for instance metadata.
        imds_token: IMDSv2 token provider.
        pricing_client: boto3 `pricing` client (us-east-1); created if None.
        ec2_client: boto3 `ec2` client for the run's region; created if None.
        environ: environment to read variables from (default os.environ).
        meminfo: /proc/meminfo contents (read from the host if None).
        now: clock for the capture timestamp.

    Returns:
        A JSON-serialisable dict. Fields that could not be determined are null and the reason is
        under "errors".
    """
    environ = os.environ if environ is None else environ
    errors: dict[str, str] = {}
    ident = _try(
        errors, "instance_identity", lambda: instance_identity(http, imds_token)
    )
    hostres = _try(errors, "host", lambda: host_resources(meminfo))

    inventory: Any = None
    if accelerator == "neuron":
        inventory = _try(errors, "accelerator", lambda: neuron_inventory(run))
    elif accelerator == "gpu":
        inventory = _try(errors, "accelerator", lambda: gpu_inventory(run))
    elif accelerator == "cpu":
        inventory = {"kind": "cpu", "packages": _pip_versions(run, ("torch",))}
    else:
        raise ValueError(f"unknown accelerator {accelerator!r}")

    commits = {name: git_commit(run, path) for name, path in (repo_paths or {}).items()}

    prices: dict[str, Any] = {"on_demand": None, "spot": None}
    if ident:
        if pricing_client is None or ec2_client is None:
            import boto3

            pricing_client = pricing_client or boto3.client(
                "pricing", region_name="us-east-1"
            )
            ec2_client = ec2_client or boto3.client("ec2", region_name=ident["region"])
        prices["on_demand"] = _try(
            errors,
            "price_on_demand",
            lambda: on_demand_price(
                pricing_client, ident["instance_type"], ident["region"]
            ),
        )
        prices["spot"] = _try(
            errors,
            "price_spot",
            lambda: spot_price(
                ec2_client, ident["instance_type"], ident["availability_zone"]
            ),
        )
    else:
        errors.setdefault(
            "prices", "not on EC2 (no instance identity), prices not fetched"
        )

    if lnc is None and accelerator == "neuron":
        raw = environ.get("NEURON_LOGICAL_NC_CONFIG")
        lnc = int(raw) if raw and raw.isdigit() else None

    return {
        "schema": MANIFEST_SCHEMA,
        "captured_at": now().isoformat(),
        "instance": ident,
        "host": hostres,
        "python": platform.python_version(),
        "accelerator": inventory,
        "lnc": lnc,
        "git_commits": commits,
        "env": relevant_env(environ),
        "prices": prices,
        "extra": dict(extra or {}),
        "errors": errors,
    }


def write_manifest(manifest: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write a manifest as JSON (2-space indent) and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI: capture a manifest on this host and write it as JSON."""
    import argparse

    ap = argparse.ArgumentParser(prog="harness.manifest")
    ap.add_argument("--accelerator", choices=("neuron", "gpu", "cpu"), required=True)
    ap.add_argument("--lnc", type=int, choices=(1, 2))
    ap.add_argument("--repo", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)
    repos = {k: (v or None) for k, _, v in (r.partition("=") for r in ns.repo)}
    doc = capture_manifest(accelerator=ns.accelerator, lnc=ns.lnc, repo_paths=repos)
    write_manifest(doc, ns.out)
    print(json.dumps({"out": str(ns.out), "errors": doc["errors"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
