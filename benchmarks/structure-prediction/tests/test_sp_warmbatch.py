# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from harness import backends, run, schema, warmbatch

TARGETS = {
    "targets": [
        {"id": t, "tokens": n}
        for t, n in (("A", 949), ("B", 496), ("C", 300), ("D", 100), ("E", 95))
    ]
}


def inputs(tmp_path):
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    for t in TARGETS["targets"]:
        (d / f"{t['id']}.yaml").write_text("version: 1\n")
    return d


def test_assign_longest_first_balances():
    shards = warmbatch.assign({t["id"]: t["tokens"] for t in TARGETS["targets"]}, 2)
    assert shards[0] == ["A"] and sorted(shards[1]) == ["B", "C", "D", "E"]


def test_cpu_sets_disjoint():
    assert warmbatch.cpu_sets(4, 12) == ["0-2", "3-5", "6-8", "9-11"]
    assert warmbatch.cpu_sets(8, 12) == [f"{i}-{i}" for i in range(8)]


def test_order_from_timing_uses_execution_order():
    pr = {"B_w2": 9.0, "C": 30.0, "B": 7.0, "C_w2": 8.0}
    assert warmbatch.order_from_timing(pr, "B") == ("B_w2", "B")
    assert warmbatch.order_from_timing(pr, "C") == ("C", "C_w2")
    assert warmbatch.order_from_timing({"B": 1.0}, "B") is None


def fake_exec(order_dup_first=("D",), fail_shard=None):
    def ex(argv, env):
        sdir = Path(argv[argv.index("predict") + 1])
        timing = Path(argv[argv.index("--timing-out") + 1])
        out = Path(argv[argv.index("--out_dir") + 1])
        ids = sorted({p.stem.removesuffix(warmbatch.DUP) for p in sdir.glob("*.yaml")})
        if fail_shard and fail_shard in ids:
            return 1, "NRT_RESOURCE: out of memory", 5.0
        per = {}
        for t in ids:
            a, b = (
                (t + warmbatch.DUP, t)
                if t in order_dup_first
                else (t, t + warmbatch.DUP)
            )
            per[a], per[b] = 30.0, 7.5  # first call, then warm
            pd = out / f"boltz_results_{sdir.name}" / "predictions" / a
            pd.mkdir(parents=True, exist_ok=True)
            for r in (1, 0):
                (pd / f"{a}_model_{r}.cif").write_text("x")
        timing.parent.mkdir(parents=True, exist_ok=True)
        timing.write_text(
            json.dumps(
                {"per_record_s": per, "precision": "32-true", "peak_accel_mem_bytes": 1}
            )
        )
        return 0, "", 60.0

    return ex


def go(tmp_path, ex, **kw):
    base = {"backend": "neuron", "inputs": inputs(tmp_path), "targets": TARGETS, "out_root": tmp_path / "out",
            "seed": 0, "ds": 1, "lnc": 2, "workers": 2, "executor": ex, "manifest_fn": lambda **k: {}, "vcpus": 12}  # fmt: skip
    base.update(kw)
    return warmbatch.run_warmbatch(**base)


def test_warmbatch_times_second_occurrence_scores_first(tmp_path):
    res = go(tmp_path, fake_exec())
    assert res["measured"] == res["of"] == 5 and res["threads_per_worker"] == 6
    g = tmp_path / "out" / res["run_group"]
    d = json.loads((g / "D" / "result.json").read_text())
    schema.validate_result(d)
    assert (d["model_s"], d["wall_warm_s"], d["wall_cold_s"]) == (7.5, 7.5, 30.0)
    assert d["timing_method"] == "in-process-warm"
    assert (
        Path(d["predictions"][0]).name == "D_w2_model_0.cif"
    )  # first-run copy is the scored one
    a = json.loads((g / "A" / "result.json").read_text())
    assert Path(a["predictions"][0]).parent.name == "A"
    assert a["command"][:3] == ["taskset", "-c", "0-5"] or a["command"][0] != "taskset"
    assert a["env"]["OMP_NUM_THREADS"] == "6"
    for p in (g / "shards").rglob("*.yaml"):
        assert p.stem.removesuffix(warmbatch.DUP) in "ABCDE"
    assert len(list((g / "shards").rglob("*.yaml"))) == 10


def test_oom_in_a_shard_marks_its_targets_did_not_run(tmp_path):
    res = go(tmp_path, fake_exec(fail_shard="A"))
    g = tmp_path / "out" / res["run_group"]
    a = json.loads((g / "A" / "result.json").read_text())
    assert a["status"] == "did-not-run" and a["model_s"] is None
    assert json.loads((g / "B" / "result.json").read_text())["status"] == "measured"


def test_only_variant_and_validation(tmp_path):
    res = go(
        tmp_path,
        fake_exec(),
        only=["A"],
        workers=1,
        variant="msa2048",
        max_msa_seqs=2048,
    )
    g = tmp_path / "out" / res["run_group"]
    a = json.loads((g / "A" / "result.json").read_text())
    assert a["variant"] == "msa2048" and "--max_msa_seqs" in a["command"]
    with pytest.raises(backends.ConfigError, match="one GPU"):
        go(tmp_path, fake_exec(), backend="gpu", lnc=None, workers=2)
    with pytest.raises(backends.ConfigError, match="1..4"):
        go(tmp_path, fake_exec(), workers=5)


def test_cli_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "capture_manifest", lambda **k: {})
    tj = tmp_path / "t.json"
    tj.write_text(json.dumps(TARGETS))
    rc = run.main(["warmbatch", "--backend", "gpu", "--inputs", str(inputs(tmp_path)), "--targets-json", str(tj),
                   "--out-root", str(tmp_path / "o"), "--seed", "1", "--ds", "5", "--dry-run"])  # fmt: skip
    assert rc == 0
