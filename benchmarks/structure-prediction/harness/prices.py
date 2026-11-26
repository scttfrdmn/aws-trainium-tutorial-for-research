# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Instance prices fetched now, for runs whose manifests could not record them (#75).

    python -m harness.prices fetch --instance g7e.2xlarge@us-east-2 \
        --instance trn2.3xlarge@sa-east-1 --out results/prices.json

Per instance type and region, each fetched live and never hard-coded:
* on_demand:      Price List API (Linux, shared tenancy), $/h;
* spot:           the lowest of the latest Linux/UNIX spot price in each AZ, with that AZ;
* capacity_block: the cheapest 24 h Capacity Block offering, as an effective $/h (upfront fee /
                  hours; assumes the whole block is used).
A price that is not offered (trn2.3xlarge has no on-demand or spot) is null with the reason
under "errors". `harness.cost_table --prices` uses this file where a run manifest has no price,
and records which source each row used.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from harness.manifest import _try, capacity_block_price, on_demand_price

PRICES_SCHEMA = "sp-prices/1.0"


def lowest_spot(ec2: Any, instance_type: str, now: dt.datetime) -> dict[str, Any]:
    """Lowest of the most recent Linux/UNIX spot price per AZ (last 6 hours)."""
    resp = ec2.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        StartTime=now - dt.timedelta(hours=6),
        MaxResults=1000,
    )
    latest: dict[str, Any] = {}
    for rec in resp.get("SpotPriceHistory", []):
        az = rec["AvailabilityZone"]
        if az not in latest or rec["Timestamp"] > latest[az]["Timestamp"]:
            latest[az] = rec
    if not latest:
        raise ValueError(f"no spot price history for {instance_type}")
    best = min(latest.values(), key=lambda r: float(r["SpotPrice"]))
    ts = best["Timestamp"]
    return {
        "usd_per_hour": float(best["SpotPrice"]),
        "availability_zone": best["AvailabilityZone"],
        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else ts,
        "source": "ec2:DescribeSpotPriceHistory (lowest AZ)",
    }


def _one(
    pricing: Any, ec2: Any, itype: str, region: str, t: dt.datetime
) -> dict[str, Any]:
    errors: dict[str, str] = {}
    return {
        "instance_type": itype,
        "region": region,
        "on_demand": _try(
            errors, "on_demand", lambda: on_demand_price(pricing, itype, region)
        ),
        "spot": _try(errors, "spot", lambda: lowest_spot(ec2, itype, t)),
        "capacity_block": _try(
            errors, "capacity_block", lambda: capacity_block_price(ec2, itype)
        ),
        "errors": errors,
    }


def fetch(
    instances: Sequence[str],
    *,
    pricing: Any = None,
    ec2_for: Callable[[str], Any] | None = None,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> dict[str, Any]:
    """Prices for each 'type@region'."""
    if pricing is None or ec2_for is None:
        import boto3

        pricing = pricing or boto3.client("pricing", region_name="us-east-1")
        ec2_for = ec2_for or (lambda r: boto3.client("ec2", region_name=r))
    t = now()
    out: dict[str, Any] = {
        "schema": PRICES_SCHEMA,
        "fetched_at": t.isoformat(),
        "prices": {},
    }
    for spec in instances:
        itype, _, region = spec.partition("@")
        if not region:
            raise ValueError(f"expected TYPE@REGION, got {spec!r}")
        out["prices"][spec] = _one(pricing, ec2_for(region), itype, region, t)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="harness.prices")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fetch")
    pf.add_argument("--instance", action="append", required=True, metavar="TYPE@REGION")
    pf.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)
    doc = fetch(ns.instance)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(doc, indent=2) + "\n")
    for spec, p in doc["prices"].items():
        cols = [
            f"{k} {p[k]['usd_per_hour']:.4f}" if p[k] else f"{k} —"
            for k in ("on_demand", "spot", "capacity_block")
        ]
        print(spec, "|", ", ".join(cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
