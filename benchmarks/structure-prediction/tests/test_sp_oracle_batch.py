# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from harness import backends, directory, oracle, run, stages

TARGETS = {"targets": [
    {"id": "T_small", "tokens": 100, "gated": True},
    {"id": "T_big", "tokens": 900, "gated": True},
    {"id": "T1047s1", "tokens": 232, "gated": False, "timing_only": True},
    {"id": "CEIL", "tokens": 1600, "gated": False},
]}  # fmt: skip


def test_oracle_jobs_cover_gated_targets_largest_first():
    jobs = oracle.oracle_jobs(TARGETS)
    assert jobs == [
        ("T_big", 0, 900),
        ("T_big", 1, 900),
        ("T_small", 0, 100),
        ("T_small", 1, 100),
    ]
    assert oracle.oracle_jobs(TARGETS, only=["CEIL"]) == [
        ("CEIL", 0, 1600),
        ("CEIL", 1, 1600),
    ]


def test_cores_for_workers():
    assert [oracle.cores_for(i, 24) for i in (0, 1, 7)] == ["0-23", "24-47", "168-191"]


def test_run_oracle_pins_each_worker_and_writes_summary(tmp_path):
    seen = []

    def fake_predict(cfg, **kw):
        seen.append(
            (
                cfg.target_id,
                cfg.seed,
                cfg.cpu_cores,
                cfg.threads,
                cfg.out_dir,
                kw["worker_index"],
            )
        )
        return {"status": "dry-run", "model_s": None}

    res = oracle.run_oracle(
        oracle.oracle_jobs(TARGETS),
        inputs=tmp_path,
        out_root=tmp_path / "o",
        workers=2,
        threads=24,
        predict=fake_predict,
        dry_run=True,
    )
    assert len(seen) == 4 and len(res["jobs"]) == 4
    for tid, seed, cores, threads, out, w in seen:
        assert cores == oracle.cores_for(w, 24) and threads == 24
        assert out == tmp_path / "o" / tid / f"seed{seed}"
    assert (tmp_path / "o" / "oracle_run.json").exists()


def test_run_oracle_variant_directory(tmp_path):
    out = []
    oracle.run_oracle(
        [("T_big", 0, 900)],
        inputs=tmp_path,
        out_root=tmp_path / "o",
        workers=1,
        threads=8,
        predict=lambda c, **k: out.append(c) or {"status": "dry-run"},
        variant="msa2048",
        max_msa_seqs=2048,
        dry_run=True,
    )
    assert out[0].out_dir.name == "seed0-msa2048" and out[0].max_msa_seqs == 2048


def test_calibration_and_budget(tmp_path):
    def fake_predict(cfg, **kw):
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (cfg.out_dir / "timing.json").write_text(
            json.dumps({"peak_host_rss_bytes": 4 * 2**30})
        )
        model = 100.0 if kw["workers"] == 1 else 125.0
        return {
            "status": "measured",
            "model_s": model,
            "wall_cold_s": model + 30.0,
            "manifest_path": str(cfg.out_dir / "manifest.json"),
        }

    cal = oracle.calibrate(
        target_id="T1042",
        input_path=tmp_path / "x.yaml",
        tokens=300,
        out_root=tmp_path / "cal",
        workers=4,
        threads=24,
        predict=fake_predict,
    )
    assert (
        cal["ok"]
        and cal["slowdown"] == pytest.approx(1.25)
        and cal["startup_s"] == pytest.approx(30.0)
    )
    b = oracle.budget(
        cal, TARGETS, workers=4, extra=["T_big:msa2048"], host_ram_gib=384
    )
    assert b["jobs"] == 6
    k2, k3 = b["estimates"]["k=2"], b["estimates"]["k=3"]
    big2 = (30 + 100 * 1.25 * (900 / 300) ** 2) / 3600
    assert (
        k2["longest_job_h"] == pytest.approx(big2)
        and k3["makespan_h"] > k2["makespan_h"]
    )
    assert b["memory"]["projected_largest_rss_gib_l2"] == pytest.approx(4 * 9)
    assert b["memory"]["fits"] is False or b["memory"]["fits"] is True


def test_budget_refuses_failed_calibration():
    with pytest.raises(ValueError, match="calibration"):
        oracle.budget({"ok": False}, TARGETS, 8)


def test_lpt_makespan():
    # LPT: 5|4 -> 5|7 -> 8|7 -> 8|10 (LPT is a heuristic; the optimum here is 9)
    assert oracle.lpt_makespan([5, 4, 3, 3, 3], 2) == 10


def test_directory_dry_run_shards_inputs(tmp_path):
    inputs = tmp_path / "in"
    inputs.mkdir()
    for t in ("A", "B", "C"):
        (inputs / f"{t}.yaml").write_text("version: 1\n")
    doc = directory.run_directory(
        backend="neuron",
        inputs=inputs,
        out_dir=tmp_path / "d",
        seed=0,
        ds=1,
        lnc=2,
        workers=2,
        dry_run=True,
        executor=None,
        manifest_fn=lambda **k: {},
    )
    assert (
        doc["status"] == "dry-run"
        and doc["n_targets"] == 3
        and doc["group_wall_s"] is None
    )
    assert sorted(p.name for p in (tmp_path / "d" / "shards" / "w0").iterdir()) == [
        "A.yaml",
        "C.yaml",
    ]
    argv = doc["workers_detail"][1]["argv"]
    assert str(tmp_path / "d" / "shards" / "w1") in argv


def test_directory_measured_and_validation(tmp_path):
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "A.yaml").write_text("version: 1\n")

    def ex(argv, env):
        t = Path(argv[argv.index("--timing-out") + 1])
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(json.dumps({"per_record_s": {"A": 12.0}}))
        return 0, "", 40.0

    doc = directory.run_directory(
        backend="gpu",
        inputs=inputs,
        out_dir=tmp_path / "d",
        seed=1,
        ds=5,
        lnc=None,
        workers=1,
        dry_run=False,
        executor=ex,
        manifest_fn=lambda **k: {},
    )
    assert doc["status"] == "measured" and doc["workers_detail"][0][
        "per_record_model_s"
    ] == {"A": 12.0}
    with pytest.raises(backends.ConfigError, match="one GPU"):
        directory.run_directory(
            backend="gpu",
            inputs=inputs,
            out_dir=tmp_path / "e",
            seed=0,
            ds=1,
            lnc=None,
            workers=2,
            dry_run=True,
            executor=ex,
            manifest_fn=lambda **k: {},
        )
    with pytest.raises(backends.ConfigError, match="neuron and gpu"):
        directory.run_directory(
            backend="cpu",
            inputs=inputs,
            out_dir=tmp_path / "f",
            seed=0,
            ds=1,
            lnc=None,
            workers=1,
            dry_run=True,
            executor=ex,
            manifest_fn=lambda **k: {},
        )


def test_stages_boot_time_and_billing(tmp_path):
    f = tmp_path / "stages.json"
    stages.record_boot(
        f, "trn2", uptime=lambda: 75.0, instance_type=lambda: "trn2.3xlarge"
    )
    stages.record_boot(
        f, "trn2", uptime=lambda: 999.0, instance_type=lambda: "x"
    )  # first call wins
    stages.time_stage(f, "cache_restore", ["true"], runner=lambda a: 0)
    stages.time_stage(f, "setup", ["true"], runner=lambda a: 0)
    stages.time_stage(f, "db_preload", ["false"], runner=lambda a: 1)
    doc = json.loads(f.read_text())
    assert (
        doc["instance_type"] == "trn2.3xlarge"
        and doc["stages"]["boot"]["wall_s"] == 75.0
    )
    total, used = stages.billed_seconds(doc)
    assert used == [
        "boot",
        "cache_restore",
    ]  # failed db_preload and unbilled setup excluded
    assert stages.billed_seconds(doc, bill_setup=True)[1] == [
        "boot",
        "cache_restore",
        "setup",
    ]


def test_cli_oracle_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "capture_manifest", lambda **k: {})
    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/taskset")
    tj = tmp_path / "t.json"
    tj.write_text(json.dumps(TARGETS))
    rc = run.main(
        [
            "oracle",
            "--targets-json",
            str(tj),
            "--inputs",
            str(tmp_path),
            "--out-root",
            str(tmp_path / "o"),
            "--workers",
            "2",
            "--threads",
            "4",
            "--dry-run",
        ]
    )
    assert rc == 0
    doc = json.loads((tmp_path / "o" / "T_big" / "seed0" / "result.json").read_text())
    assert (
        doc["command"][:2] == ["taskset", "-c"]
        and doc["env"]["SP_TORCH_THREADS"] == "4"
    )
    assert (
        run.main(
            [
                "oracle",
                "--targets-json",
                str(tj),
                "--inputs",
                str(tmp_path),
                "--out-root",
                str(tmp_path / "o"),
                "--workers",
                "1",
                "--threads",
                "4",
                "--variant",
                "msa2048",
                "--dry-run",
            ]
        )
        == 2
    )


def test_per_lnc_stage_names_replace_the_plain_stage():
    doc = {
        "stages": {
            "boot": {"wall_s": 10, "rc": 0},
            "cache_restore_lnc2": {"wall_s": 30, "rc": 0},
            "cache_restore_lnc1": {"wall_s": 50, "rc": 0},
        }
    }
    assert stages.billed_seconds(doc, lnc=2) == (40.0, ["boot", "cache_restore_lnc2"])
    assert stages.billed_seconds(doc, lnc=1)[0] == 60.0


def test_dtype_audit_catches_autocast_inside_a_submodule():
    torch = pytest.importorskip("torch")
    from harness.entry import DtypeAudit

    class Diffusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            with torch.autocast(
                "cpu", dtype=torch.bfloat16
            ):  # what a trainer override misses
                return self.lin(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pairformer_module = torch.nn.Linear(4, 4)
            self.msa_module = torch.nn.Linear(4, 4)
            self.structure_module = Diffusion()

        def forward(self, x):
            return self.structure_module(self.msa_module(self.pairformer_module(x)))

    m = Model()
    audit = DtypeAudit()
    audit.attach(m)
    m(torch.zeros(2, 4))
    rep = audit.report()
    assert not rep["fp32_ok"]
    assert rep["by_top_module"]["structure_module"]["nonfp32"] >= 1
    assert "structure_module.lin" in rep["autocast_active_in"]
    assert rep["by_top_module"]["pairformer_module"]["nonfp32"] == 0
