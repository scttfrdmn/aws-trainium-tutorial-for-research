# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from harness import cost_table as ct
from harness import profile_classify as pc
from harness.schema import COST_COLUMNS

FIX = Path(__file__).parent / "fixtures"


def by_class(res):
    return {r["class"]: r for r in res["classes"]}


def test_gpu_trace_classification():
    res = pc.classify_gpu_trace(
        json.loads((FIX / "synthetic_torch_trace.json").read_text())
    )
    c = by_class(res)
    assert res["unit"] == "us"
    expect = {
        "triangle_attention": 80,
        "triangle_multiplication": 60,
        "pairformer_transition": 20,
        "pairformer_other": 10,
        "msa_module": 10,
        "windowed_atom_attention": 25,
        "diffusion_token_transformer": 100,
        "dma_transfer": 15,
        "other": 5,
        "host_glue": 635,
    }
    assert {k: v["time"] for k, v in c.items()} == expect
    assert all(v["attribution"] == "certain" for v in c.values())
    assert sum(v["fraction"] for v in c.values()) == pytest.approx(1.0)


def test_neuron_profile_classification_and_engines():
    res = pc.classify_neuron_profile(
        json.loads((FIX / "synthetic_neuron_profile.json").read_text())
    )
    c = by_class(res)
    ta = c["triangle_attention"]
    assert ta["time"] == 400 and ta["attribution"] == "certain"
    eng = ta["engine_busy_fraction"]
    assert (eng["tensor"], eng["scalar"], eng["vector"]) == (0.25, 0.75, 0.625)
    assert c["triangle_multiplication"]["engine_busy_fraction"]["tensor"] == 1.0
    assert c["pairformer_transition"]["time"] == 50
    assert c["diffusion_token_transformer"]["time"] == 100
    assert c["windowed_atom_attention"]["attribution"] == "heuristic"
    assert c["dma_transfer"]["time"] == 40
    assert c["host_glue"]["time"] == 160
    assert c["other"]["time"] == 10


def test_neuron_field_adapter():
    doc = [{"q": "Tensor", "t0": 0, "dt": 10, "name": "TriangleMultiplicationIncoming"}]
    res = pc.classify_neuron_profile(
        doc,
        {"list": "x", "engine": "q", "start": "t0", "duration": "dt", "label": "name"},
    )
    assert (
        by_class(res)["triangle_multiplication"]["engine_busy_fraction"]["tensor"]
        == 1.0
    )


@pytest.mark.parametrize(
    "stack, cls",
    [
        (
            [
                "Boltz2",
                "PairformerModule",
                "PairformerLayer",
                "TriangleAttentionEndingNode",
            ],
            "triangle_attention",
        ),
        (
            [
                "Boltz2",
                "TemplateV2Module",
                "PairformerNoSeqModule",
                "TriangleMultiplicationIncoming",
            ],
            "triangle_multiplication",
        ),
        (["Boltz2", "MSAModule", "MSALayer", "Transition"], "msa_module"),
        (
            [
                "Boltz2",
                "DiffusionModule",
                "AtomAttentionDecoder",
                "AtomTransformer",
                "DiffusionTransformer",
            ],
            "windowed_atom_attention",
        ),
        (
            [
                "Boltz2",
                "DiffusionModule",
                "DiffusionTransformer",
                "DiffusionTransformerLayer",
            ],
            "diffusion_token_transformer",
        ),
        (["Boltz2", "InputEmbedder"], "other"),
    ],
)
def test_classify_stack_rules(stack, cls):
    assert pc.classify_stack(stack) == cls


def test_profile_cli(tmp_path):
    out = tmp_path / "c.json"
    assert (
        pc.main(
            [
                "neuron",
                "--profile",
                str(FIX / "synthetic_neuron_profile.json"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["unit"] == "ns"


# ---------------------------------------------------------------- cost table
def write_result(
    root,
    tid,
    seed,
    *,
    backend="neuron",
    lnc=2,
    workers=4,
    ds=1,
    status="measured",
    model_s=10.0,
    tokens=100,
    inst="trn2.3xlarge",
    od=4.0,
    spot=1.0,
):
    d = root / f"{backend}-{lnc}-{ds}" / tid / f"s{seed}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "instance": {"instance_type": inst},
                "prices": {
                    "on_demand": {"usd_per_hour": od},
                    "spot": {"usd_per_hour": spot},
                },
            }
        )
    )
    measured = status == "measured"
    (d / "result.json").write_text(json.dumps({
        "target_id": tid, "seed": seed, "backend": backend, "lnc": lnc, "workers": workers,
        "diffusion_samples": ds, "status": status, "model_s": model_s if measured else None,
        "tokens": tokens, "manifest_path": "manifest.json",
    }))  # fmt: skip


TARGETS = {"targets": [
    {"id": "T1", "tokens": 100, "gated": True},
    {"id": "T2", "tokens": 200, "gated": True},
    {"id": "CEIL", "tokens": 1600, "gated": False},
]}  # fmt: skip


def test_cost_rows_trn2_with_separate_msa_host(tmp_path):
    for s in (0, 1):
        write_result(tmp_path, "T1", s, model_s=10.0 + s * 2)
        write_result(tmp_path, "T2", s, model_s=40.0, tokens=200)
        write_result(tmp_path, "CEIL", s, status="did-not-run", tokens=1600)
    msa_cpu = {
        "host_instance_type": "r7a.48xlarge",
        "per_target_s": {"T1": 60.0, "T2": 120.0},
        "price_od_usd_per_h": 12.0,
        "price_spot_usd_per_h": 3.0,
    }
    rows = ct.build_rows(tmp_path, TARGETS, {}, msa_cpu)
    r1 = next(r for r in rows if r["target_id"] == "T1")
    assert r1["dl_s_per_structure"] == pytest.approx(
        11.0 / 4
    )  # mean of seeds / 4 workers
    assert r1["dl_usd_per_structure_od"] == pytest.approx(11.0 / 4 * 4.0 / 3600)
    assert r1["e2e_usd_per_structure_od"] == pytest.approx(
        11.0 / 4 * 4.0 / 3600 + 60 * 12.0 / 3600
    )
    assert r1["msa_host"] == "r7a.48xlarge"
    assert r1["ceiling_tokens"] == 200
    ceil = next(r for r in rows if r["target_id"] == "CEIL")
    assert ceil["ran"] is False and ceil["dl_s_per_structure"] is None
    mean = next(r for r in rows if r["target_id"] == "__mean__")
    assert mean["dl_s_per_structure"] == pytest.approx((11.0 / 4 + 10.0) / 2)
    assert set(rows[0]) == set(COST_COLUMNS)


def test_cost_row_needs_both_seeds(tmp_path):
    write_result(tmp_path, "T1", 0)
    rows = ct.build_rows(tmp_path, TARGETS, {}, {})
    r = rows[0]
    assert (
        r["ran"] is False
        and r["dl_usd_per_structure_od"] is None
        and "both seeds" in r["note"]
    )


def test_gpu_msa_billed_on_same_host(tmp_path):
    for s in (0, 1):
        write_result(
            tmp_path,
            "T1",
            s,
            backend="gpu",
            lnc=None,
            workers=None,
            inst="g7e.2xlarge",
            od=3.0,
            spot=1.5,
        )
    rows = ct.build_rows(
        tmp_path,
        TARGETS,
        {"host_instance_type": "g7e.2xlarge", "per_target_s": {"T1": 10.0}},
        {},
    )
    r = rows[0]
    assert r["platform"] == "g7e.2xlarge gpu"
    assert r["e2e_usd_per_structure_spot"] == pytest.approx(
        10.0 * 1.5 / 3600 + 10.0 * 1.5 / 3600
    )


def test_cost_cli_writes_csv_and_markdown(tmp_path):
    for s in (0, 1):
        write_result(tmp_path / "r", "T1", s)
    tj = tmp_path / "t.json"
    tj.write_text(json.dumps(TARGETS))
    assert (
        ct.main(
            [
                "--results",
                str(tmp_path / "r"),
                "--targets",
                str(tj),
                "--out",
                str(tmp_path / "a.csv"),
                "--markdown",
                str(tmp_path / "a.md"),
            ]
        )
        == 0
    )
    head = (tmp_path / "a.csv").read_text().splitlines()[0].split(",")
    assert tuple(head) == COST_COLUMNS
    assert "| **mean** |" in (tmp_path / "a.md").read_text()
