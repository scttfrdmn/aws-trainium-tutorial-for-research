# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import datetime as dt
import json

import pytest
from sp_testutil import FakeEC2, FakePricing

from harness import manifest, prices


def test_capacity_block_effective_rate():
    cb = manifest.capacity_block_price(FakeEC2(), "trn2.3xlarge")
    assert cb["usd_per_hour"] == pytest.approx(2.0) and cb["upfront_usd"] == 48.0
    assert cb["hours"] == 24 and cb["availability_zone"] == "sa-east-1a"

    class Empty:
        def describe_capacity_block_offerings(self, **k):
            return {"CapacityBlockOfferings": []}

    with pytest.raises(ValueError, match="no 24 h Capacity Block"):
        manifest.capacity_block_price(Empty(), "g7e.2xlarge")


def test_new_regions_have_price_list_locations():
    for r in ("sa-east-1", "ap-southeast-4"):
        assert r in manifest.REGION_LOCATION


class _SpotEC2(FakeEC2):
    def describe_spot_price_history(self, **kw):
        t0 = dt.datetime(2026, 11, 25, 12, tzinfo=dt.UTC)
        return {"SpotPriceHistory": [
            {"AvailabilityZone": "us-east-2a", "SpotPrice": "2.30", "Timestamp": t0},
            {"AvailabilityZone": "us-east-2a", "SpotPrice": "2.10", "Timestamp": t0 + dt.timedelta(hours=1)},
            {"AvailabilityZone": "us-east-2c", "SpotPrice": "2.17", "Timestamp": t0},
        ]}  # fmt: skip


def test_fetch_records_each_basis_and_why_one_is_missing(tmp_path):
    now = dt.datetime(2026, 11, 25, 14, tzinfo=dt.UTC)
    doc = prices.fetch(["g7e.2xlarge@us-east-2", "trn2.3xlarge@mars-1"], pricing=FakePricing(),
                       ec2_for=lambda r: _SpotEC2(), now=lambda: now)  # fmt: skip
    g = doc["prices"]["g7e.2xlarge@us-east-2"]
    assert (
        g["spot"]["usd_per_hour"] == 2.10
        and g["spot"]["availability_zone"] == "us-east-2a"
    )
    assert (
        g["on_demand"]["usd_per_hour"] > 0
        and g["capacity_block"]["usd_per_hour"] == 2.0
    )
    t = doc["prices"]["trn2.3xlarge@mars-1"]
    assert (
        t["on_demand"] is None and "no Price List location" in t["errors"]["on_demand"]
    )
    assert doc["fetched_at"] == now.isoformat()
    json.dumps(doc)  # serialisable
    with pytest.raises(ValueError):
        prices.fetch(
            ["g7e.2xlarge"], pricing=FakePricing(), ec2_for=lambda r: _SpotEC2()
        )
