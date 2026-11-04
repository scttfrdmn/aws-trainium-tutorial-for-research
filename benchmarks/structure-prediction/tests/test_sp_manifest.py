# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from sp_testutil import FakeEC2, FakeHost, FakePricing

from harness import manifest as m


def test_neuron_manifest_captures_everything(fake_manifest_kwargs, tmp_path):
    doc = m.capture_manifest(
        accelerator="neuron",
        lnc=2,
        repo_paths={
            "boltz-neuron": "/opt/boltz-neuron",
            "this-repo": "/opt/repo",
            "missing": None,
        },
        **fake_manifest_kwargs,
    )
    assert doc["schema"] == m.MANIFEST_SCHEMA
    assert doc["instance"]["instance_type"] == "trn2.3xlarge"
    assert doc["instance"]["availability_zone"] == "us-east-1a"
    assert doc["instance"]["ami_id"].startswith("ami-")
    assert doc["host"]["ram_gib"] == 124.0
    assert doc["accelerator"]["device_count"] == 1
    assert doc["accelerator"]["packages"]["torch-neuronx"] == "2.14.3"
    assert doc["accelerator"]["packages"]["libneuronxla"] is None
    assert doc["accelerator"]["runtime_deb"] == "2.28.10.0"
    assert doc["lnc"] == 2
    assert (
        doc["git_commits"]["boltz-neuron"] == "0123456789abcdef0123456789abcdef01234567"
    )
    assert doc["git_commits"]["missing"] is None
    assert doc["env"] == {
        "BOLTZ_NEURON_ATOM_GRAPH_BREAK": "1",
        "NEURON_RT_VISIBLE_CORES": "0",
    }
    assert doc["prices"]["on_demand"]["usd_per_hour"] == pytest.approx(2.235)
    assert doc["prices"]["spot"]["usd_per_hour"] == pytest.approx(0.9)
    assert doc["errors"] == {}
    path = m.write_manifest(doc, tmp_path / "manifest.json")
    assert json.loads(path.read_text())["lnc"] == 2


def test_pricing_filters_are_specific():
    pricing = FakePricing()
    m.on_demand_price(pricing, "g7e.2xlarge", "us-east-1")
    fields = {f["Field"]: f["Value"] for f in pricing.kwargs["Filters"]}
    assert fields == {
        "instanceType": "g7e.2xlarge",
        "location": "US East (N. Virginia)",
        "operatingSystem": "Linux",
        "tenancy": "Shared",
        "preInstalledSw": "NA",
        "capacitystatus": "Used",
    }


def test_ambiguous_on_demand_price_is_an_error_not_a_guess():
    class Two(FakePricing):
        def get_products(self, **kwargs):
            a = json.loads(super().get_products(**kwargs)["PriceList"][0])
            b = json.loads(json.dumps(a).replace(self.usd, "9.99"))
            return {"PriceList": [json.dumps(a), json.dumps(b)]}

    with pytest.raises(ValueError, match="expected one on-demand price"):
        m.on_demand_price(Two(), "g7e.2xlarge", "us-east-1")


def test_gpu_inventory_parses_nvidia_smi():
    host = FakeHost("gpu")
    inv = m.gpu_inventory(host.run)
    assert inv["gpus"][0]["name"].startswith("NVIDIA RTX PRO 6000")
    assert inv["cuda_version"] == "13.0"
    assert inv["driver_version"] == "580.82.07"


def test_off_ec2_records_errors_and_no_prices(fake_manifest_kwargs):
    def no_imds():
        raise OSError("no route to IMDS")

    kw = {**fake_manifest_kwargs, "imds_token": no_imds}
    doc = m.capture_manifest(accelerator="cpu", **kw)
    assert doc["instance"] is None
    assert doc["prices"] == {"on_demand": None, "spot": None}
    assert "instance_identity" in doc["errors"]
    assert "prices" in doc["errors"]


def test_failed_price_lookup_is_null_with_reason(fake_manifest_kwargs):
    class Broken(FakeEC2):
        def describe_spot_price_history(self, **kwargs):
            return {"SpotPriceHistory": []}

    doc = m.capture_manifest(
        accelerator="neuron", **{**fake_manifest_kwargs, "ec2_client": Broken()}
    )
    assert doc["prices"]["spot"] is None
    assert "no spot price history" in doc["errors"]["price_spot"]
    assert doc["prices"]["on_demand"]["usd_per_hour"] == pytest.approx(2.235)


def test_lnc_falls_back_to_environment(fake_manifest_kwargs):
    kw = {**fake_manifest_kwargs, "environ": {"NEURON_LOGICAL_NC_CONFIG": "1"}}
    assert m.capture_manifest(accelerator="neuron", **kw)["lnc"] == 1


def test_unknown_accelerator_rejected(fake_manifest_kwargs):
    with pytest.raises(ValueError):
        m.capture_manifest(accelerator="tpu", **fake_manifest_kwargs)


def test_unknown_region_has_no_price_location():
    with pytest.raises(ValueError, match="no Price List location"):
        m.on_demand_price(FakePricing(), "g7e.2xlarge", "mars-1")
