# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from harness import backends, pins, run, schema


def cfg(tmp_path, **kw):
    base = {
        "backend": "neuron",
        "target_id": "T1031",
        "input_path": tmp_path / "T1031.yaml",
        "out_dir": tmp_path / "out",
        "seed": 0,
        "diffusion_samples": 1,
        "lnc": 2,
        "visible_core": 0,
    }
    base.update(kw)
    return backends.RunConfig(**base)


def fake_manifest(**kwargs):
    return {"schema": "sp-manifest/1.0", "prices": {}, "kwargs": sorted(kwargs)}


# ---------------------------------------------------------------- backends
def test_neuron_command_uses_boltz_defaults_and_wrapper(tmp_path):
    inv = backends.build(cfg(tmp_path, atom_graph_break=0, diffusion_samples=5))
    a = inv.argv
    assert a[1:5] == ["-m", "harness.entry", "--target", "boltz_neuron"]
    rest = a[a.index("--") + 1 :]
    assert rest[0] == "predict"
    for flag, val in (
        ("--recycling_steps", "3"),
        ("--sampling_steps", "200"),
        ("--diffusion_samples", "5"),
        ("--num_workers", "0"),
        ("--seed", "0"),
    ):
        assert rest[rest.index(flag) + 1] == val
    assert "--use_potentials" not in rest  # fk_steering off
    assert inv.env == {
        "NEURON_RT_VISIBLE_CORES": "0",
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "BOLTZ_NEURON_ATOM_GRAPH_BREAK": "0",
    }
    assert inv.profile_env["NEURON_RT_INSPECT_ENABLE"] == "1"
    assert (
        inv.predictions_dir
        == tmp_path / "out" / "boltz_results_T1031" / "predictions" / "T1031"
    )


def test_atom_graph_break_is_left_to_the_port_when_unset(tmp_path):
    assert "BOLTZ_NEURON_ATOM_GRAPH_BREAK" not in backends.build(cfg(tmp_path)).env


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"lnc": 3}, "lnc=1 or lnc=2"),
        ({"visible_core": None}, "visible_core"),
        ({"lnc": 2, "visible_core": 4}, r"\[0, 4\)"),
        ({"seed": 2}, "seed"),
        ({"diffusion_samples": 8}, "diffusion_samples"),
        ({"sampling_steps": 100}, "sampling_steps"),
        ({"backend": "gpu", "sampling_steps": 50}, "200 sampling steps"),
        ({"backend": "cpu", "diffusion_samples": 5}, "ds=1 only"),
        ({"backend": "gpu", "nim": True}, "licence-gated"),
        ({"backend": "openfold-trt"}, "openfold-trt needs"),
        ({"backend": "tpu"}, "unknown backend"),
    ],
)
def test_disallowed_configs(tmp_path, kw, msg):
    with pytest.raises(backends.ConfigError, match=msg):
        backends.build(cfg(tmp_path, **kw))


def test_lnc1_allows_eight_cores(tmp_path):
    assert (
        backends.build(cfg(tmp_path, lnc=1, visible_core=7)).env[
            "NEURON_RT_VISIBLE_CORES"
        ]
        == "7"
    )


def test_task0_reproduction_allows_50_steps_on_neuron_only(tmp_path):
    inv = backends.build(cfg(tmp_path, sampling_steps=pins.REPRO_SAMPLING_STEPS))
    assert inv.argv[inv.argv.index("--sampling_steps") + 1] == "50"


def test_gpu_keeps_cuequivariance_kernels_on(tmp_path):
    inv = backends.build(cfg(tmp_path, backend="gpu", lnc=None, visible_core=None))
    assert "--no_kernels" not in inv.argv
    assert inv.argv[inv.argv.index("--accelerator") + 1] == "gpu"
    assert "--torch-profile" in inv.profile_argv
    assert "--torch-profile" not in inv.argv  # timing runs are never profiled


def test_cpu_oracle_is_fp32_wrapper(tmp_path):
    inv = backends.build(cfg(tmp_path, backend="cpu", lnc=None, visible_core=None))
    assert inv.argv[inv.argv.index("--target") + 1] == "boltz_fp32"
    assert inv.argv[inv.argv.index("--accelerator") + 1] == "cpu"
    assert inv.profile_argv is None


def test_openfold_trt_command(tmp_path):
    inv = backends.build(
        cfg(
            tmp_path,
            backend="openfold-trt",
            lnc=None,
            visible_core=None,
            alignment_dir=tmp_path / "aln",
            template_mmcif_dir=tmp_path / "tmpl",
            jax_param_path=tmp_path / "params_model_1.npz",
            trt_engine_dir=tmp_path / "trt",
            openfold_dir=tmp_path / "openfold",
        )
    )
    a = inv.argv
    assert a[1].endswith("run_pretrained_openfold.py")
    assert a[a.index("--config_preset") + 1] == "model_1"
    assert a[a.index("--trt_mode") + 1] == "run"
    assert a[a.index("--precision") + 1] == "bf16"
    assert "--skip_relaxation" in a


# ---------------------------------------------------------------- schema
def test_schema_rejects_measured_numbers_on_a_dry_run():
    doc = schema.empty_result(
        target_id="T",
        backend="gpu",
        seed=0,
        diffusion_samples=1,
        recycling_steps=3,
        sampling_steps=200,
    )
    schema.validate_result(doc)
    doc["wall_warm_s"] = 1.0
    with pytest.raises(schema.SchemaError, match="must not carry measured values"):
        schema.validate_result(doc)


def test_schema_requires_manifest_and_walls_when_measured():
    doc = schema.empty_result(
        target_id="T",
        backend="gpu",
        seed=0,
        diffusion_samples=1,
        recycling_steps=3,
        sampling_steps=200,
        status="measured",
        wall_cold_s=2.0,
    )
    with pytest.raises(schema.SchemaError, match="wall_warm_s"):
        schema.validate_result(doc)
    doc["wall_warm_s"] = 1.0
    with pytest.raises(schema.SchemaError, match="manifest"):
        schema.validate_result(doc)
    doc["manifest_path"] = "m.json"
    schema.validate_result(doc)


def test_schema_rejects_bool_as_number_and_unknown_backend():
    doc = schema.empty_result(
        target_id="T",
        backend="gpu",
        seed=0,
        diffusion_samples=1,
        recycling_steps=3,
        sampling_steps=200,
    )
    doc["tokens"] = True
    with pytest.raises(schema.SchemaError):
        schema.validate_result(doc)
    doc["tokens"] = None
    doc["backend"] = "tpu"
    with pytest.raises(schema.SchemaError):
        schema.validate_result(doc)


# ---------------------------------------------------------------- predict / CLI
class Exec:
    def __init__(self, rcs=(0, 0, 0), out="", timing=None):
        self.rcs = list(rcs)
        self.out = out
        self.timing = timing
        self.calls = []

    def __call__(self, argv, env):
        self.calls.append((argv, env))
        if self.timing is not None and "--timing-out" in argv:
            p = Path(argv[argv.index("--timing-out") + 1])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.timing))
        return self.rcs.pop(0), self.out, 10.0 + len(self.calls)


def test_dry_run_runs_nothing_and_writes_schema_valid_result(tmp_path):
    ex = Exec()
    doc = run.predict(
        cfg(tmp_path),
        dry_run=True,
        profile=True,
        tokens=96,
        run_group="g",
        executor=ex,
        manifest_fn=fake_manifest,
    )
    assert ex.calls == []
    assert doc["status"] == "dry-run"
    assert doc["wall_warm_s"] is None and doc["model_s"] is None
    saved = json.loads((tmp_path / "out" / "result.json").read_text())
    schema.validate_result(saved)
    assert saved["command"][0].endswith("python") or "python" in saved["command"][0]
    assert (tmp_path / "out" / "manifest.json").exists()


def test_measured_run_cold_warm_and_profile(tmp_path):
    pdir = tmp_path / "out" / "boltz_results_T1031" / "predictions" / "T1031"
    pdir.mkdir(parents=True)
    for r in (1, 0):
        (pdir / f"T1031_model_{r}.cif").write_text("x")
    ex = Exec(
        timing={
            "model_s": 7.5,
            "peak_accel_mem_bytes": 123,
            "peak_accel_mem_source": "torch.neuron.max_memory_allocated",
        }
    )
    doc = run.predict(
        cfg(tmp_path),
        dry_run=False,
        profile=True,
        tokens=96,
        run_group="g",
        executor=ex,
        manifest_fn=fake_manifest,
    )
    assert doc["status"] == "measured"
    assert (doc["wall_cold_s"], doc["wall_warm_s"]) == (11.0, 12.0)
    assert doc["model_s"] == 7.5
    assert [Path(p).name for p in doc["predictions"]] == [
        "T1031_model_0.cif",
        "T1031_model_1.cif",
    ]
    assert len(ex.calls) == 3
    assert ex.calls[2][1]["NEURON_RT_INSPECT_ENABLE"] == "1"
    assert doc["profile_path"].endswith("neuron-profile")


def test_oom_is_did_not_run_with_no_numbers(tmp_path):
    ex = Exec(rcs=(1,), out="... NRT_RESOURCE: out of device memory")
    doc = run.predict(
        cfg(tmp_path),
        dry_run=False,
        profile=False,
        tokens=1568,
        run_group="g",
        executor=ex,
        manifest_fn=fake_manifest,
    )
    assert doc["status"] == "did-not-run"
    assert doc["wall_cold_s"] is None and doc["wall_warm_s"] is None
    assert "NRT_RESOURCE" in doc["error"]


def test_other_failure_is_failed(tmp_path):
    doc = run.predict(
        cfg(tmp_path),
        dry_run=False,
        profile=False,
        tokens=96,
        run_group="g",
        executor=Exec(rcs=(0, 2), out="Traceback"),
        manifest_fn=fake_manifest,
    )
    assert doc["status"] == "failed"
    assert doc["wall_cold_s"] is None


def test_cpu_oracle_runs_once_and_is_cached(tmp_path):
    targets = tmp_path / "targets.json"
    targets.write_text('{"targets": []}')
    c = cfg(tmp_path, backend="cpu", lnc=None, visible_core=None)
    ex = Exec(timing=GOOD_ORACLE_TIMING)
    first = run.predict(
        c,
        dry_run=False,
        profile=False,
        tokens=96,
        run_group="g",
        targets_json=targets,
        executor=ex,
        manifest_fn=fake_manifest,
    )
    assert (
        first["status"] == "measured"
        and first["wall_warm_s"] is None
        and len(ex.calls) == 1
    )
    again = run.predict(
        c,
        dry_run=False,
        profile=False,
        tokens=96,
        run_group="g",
        targets_json=targets,
        executor=ex,
        manifest_fn=fake_manifest,
    )
    assert len(ex.calls) == 1 and again == first
    targets.write_text('{"targets": [{"id": "new"}]}')  # target set changed -> rerun
    ex.rcs = [0]
    run.predict(
        c,
        dry_run=False,
        profile=False,
        tokens=96,
        run_group="g",
        targets_json=targets,
        executor=ex,
        manifest_fn=fake_manifest,
    )
    assert len(ex.calls) == 2


def test_cli_predict_dry_run_and_config_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, "capture_manifest", fake_manifest)
    inp = tmp_path / "T1031.yaml"
    inp.write_text("version: 1\n")
    base = [
        "predict",
        "--backend",
        "neuron",
        "--target-id",
        "T1031",
        "--input",
        str(inp),
        "--out-dir",
        str(tmp_path / "o"),
        "--seed",
        "0",
        "--lnc",
        "2",
        "--core",
        "1",
        "--dry-run",
    ]
    assert run.main(base, executor=Exec()) == 0
    assert '"status": "dry-run"' in capsys.readouterr().out
    assert run.main([*base[:-3], "--core", "9", "--dry-run"], executor=Exec()) == 2


def test_cli_rejects_bad_seed():
    with pytest.raises(SystemExit):
        run.main(
            [
                "predict",
                "--backend",
                "gpu",
                "--target-id",
                "x",
                "--input",
                "x",
                "--out-dir",
                "x",
                "--seed",
                "3",
            ]
        )


def test_batch_dry_run_shards_over_cores(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "capture_manifest", fake_manifest)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for t in ("A", "B", "C", "D", "E"):
        (inputs / f"{t}.yaml").write_text("version: 1\n")
    rc = run.main(
        [
            "batch",
            "--backend",
            "neuron",
            "--lnc",
            "2",
            "--workers",
            "4",
            "--seed",
            "0",
            "--inputs",
            str(inputs),
            "--out-root",
            str(tmp_path / "r"),
            "--dry-run",
        ],
        executor=Exec(),
    )
    assert rc == 0
    group = next((tmp_path / "r").iterdir())
    summary = json.loads((group / "group.json").read_text())
    assert summary["inputs"] == 5 and summary["group_wall_s"] is None
    cores = sorted(
        json.loads(p.read_text())["env"]["NEURON_RT_VISIBLE_CORES"]
        for p in group.rglob("result.json")
    )
    assert cores == ["0", "0", "1", "2", "3"]


def test_batch_rejects_too_many_workers(tmp_path):
    rc = run.main(
        [
            "batch",
            "--backend",
            "neuron",
            "--lnc",
            "2",
            "--workers",
            "8",
            "--seed",
            "0",
            "--inputs",
            str(tmp_path),
            "--out-root",
            str(tmp_path / "r"),
            "--dry-run",
        ]
    )
    assert rc == 2


def test_inputs_writes_boltz_yaml_with_shared_msa(tmp_path):
    import yaml

    targets = {"targets": [{"id": "C1", "chains": [
        {"chain_id": "A", "entity": "protein", "sequence": "MKV", "msa_id": "C1_P"},
        {"chain_id": "B", "entity": "rna", "sequence": "GGA", "msa_id": None},
    ]}]}  # fmt: skip
    (path,) = run.write_inputs(targets, tmp_path / "msa", tmp_path / "in")
    doc = yaml.safe_load(path.read_text())
    assert doc["sequences"][0]["protein"]["msa"].endswith("msa/C1_P.a3m")
    assert "msa" not in doc["sequences"][1]["rna"]


GOOD_AUDIT = {
    "modules_seen": 1200,
    "nonfp32_modules": [],
    "nonfp32_count": 0,
    "autocast_active_in": [],
    "required_seen": {
        "structure_module": True,
        "pairformer_module": True,
        "msa_module": True,
    },
    "fp32_ok": True,
}
GOOD_ORACLE_TIMING = {
    "model_s": 3600.0,
    "precision": "32-true",
    "dtype_audit": GOOD_AUDIT,
}


@pytest.mark.parametrize(
    "audit, msg",
    [
        (None, "dtype audit failed"),
        (
            {
                **GOOD_AUDIT,
                "fp32_ok": False,
                "nonfp32_modules": ["structure_module.score_model.x"],
            },
            "structure_module",
        ),
        (
            {
                **GOOD_AUDIT,
                "fp32_ok": False,
                "autocast_active_in": ["pairformer_module"],
            },
            "autocast",
        ),
    ],
)
def test_cpu_oracle_rejects_failed_dtype_audit(tmp_path, audit, msg):
    c = cfg(tmp_path, backend="cpu", lnc=None, visible_core=None)
    ex = Exec(timing={"model_s": 1.0, "precision": "32-true", "dtype_audit": audit})
    doc = run.predict(
        c,
        dry_run=False,
        profile=False,
        tokens=96,
        run_group="g",
        executor=ex,
        manifest_fn=fake_manifest,
    )
    assert doc["status"] == "failed" and msg in doc["error"]


def test_primary_rows_keep_default_msa_depth(tmp_path):
    with pytest.raises(backends.ConfigError, match="secondary run"):
        backends.build(cfg(tmp_path, max_msa_seqs=2048))
    inv = backends.build(cfg(tmp_path, max_msa_seqs=2048, variant="msa2048"))
    assert inv.argv[inv.argv.index("--max_msa_seqs") + 1] == "2048"
    with pytest.raises(backends.ConfigError, match="short lowercase"):
        backends.build(cfg(tmp_path, max_msa_seqs=2048, variant="MSA 2048"))


def test_variant_recorded_in_result(tmp_path):
    doc = run.predict(
        cfg(tmp_path, max_msa_seqs=2048, variant="msa2048"),
        dry_run=True,
        profile=False,
        tokens=949,
        run_group="g",
        executor=Exec(),
        manifest_fn=fake_manifest,
    )
    assert (doc["variant"], doc["max_msa_seqs"], doc["schema"]) == (
        "msa2048",
        2048,
        schema.RESULT_SCHEMA,
    )


def test_cpu_threads_and_pinning(tmp_path, monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/taskset")
    inv = backends.build(
        cfg(
            tmp_path,
            backend="cpu",
            lnc=None,
            visible_core=None,
            threads=24,
            cpu_cores="24-47",
        )
    )
    assert inv.argv[:3] == ["taskset", "-c", "24-47"]
    assert inv.env == {
        "OMP_NUM_THREADS": "24",
        "MKL_NUM_THREADS": "24",
        "SP_TORCH_THREADS": "24",
    }
    monkeypatch.setattr(backends.shutil, "which", lambda name: None)
    with pytest.raises(backends.ConfigError, match="taskset"):
        backends.build(
            cfg(
                tmp_path,
                backend="cpu",
                lnc=None,
                visible_core=None,
                threads=24,
                cpu_cores="0-23",
            )
        )


def test_schema_1_0_documents_still_validate():
    doc = schema.empty_result(
        target_id="T",
        backend="gpu",
        seed=0,
        diffusion_samples=1,
        recycling_steps=3,
        sampling_steps=200,
    )
    doc["schema"] = "sp-result/1.0"
    schema.validate_result(doc)


def test_cpu_oracle_rejects_non_fp32(tmp_path):
    c = cfg(tmp_path, backend="cpu", lnc=None, visible_core=None)
    ex = Exec(timing={"model_s": 1.0, "precision": "bf16-mixed"})
    doc = run.predict(
        c,
        dry_run=False,
        profile=False,
        tokens=96,
        run_group="g",
        executor=ex,
        manifest_fn=fake_manifest,
    )
    assert (
        doc["status"] == "failed"
        and "bf16-mixed" in doc["error"]
        and doc["model_s"] is None
    )
