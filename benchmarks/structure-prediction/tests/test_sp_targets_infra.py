# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from harness import entry
from targets import build_targets as bt

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- committed manifest
def test_committed_target_manifest_is_consistent():
    m = json.loads((ROOT / "targets" / "targets.json").read_text())
    casp = [t for t in m["targets"] if t["set"] == "casp14"]
    ceil = [t for t in m["targets"] if t["set"] == "ceiling"]
    assert len(casp) == 20
    gated = [t["id"] for t in casp if t["gated"]]
    timing_only = [t["id"] for t in casp if t.get("timing_only")]
    assert len(gated) == 19 and timing_only == ["T1047s1"]  # owner decision 2026-11-04
    assert all(t["truth"]["source"] for t in casp if t["gated"])
    assert len(ceil) == 2 and not any(t["gated"] for t in ceil)
    assert all(t["tokens"] > 1500 for t in ceil)
    assert {t["length"] for t in casp} >= {95, 949}
    for t in casp:
        assert t["truth"]["source"] in ("pdb", "casp", None)
        if t["truth"]["source"]:
            assert t["truth"]["scoring_unit"] == "whole-target"
    fasta = (ROOT / "targets" / "targets.fasta").read_text()
    assert fasta.count(">") == sum(len(t["chains"]) for t in m["targets"])


# ---------------------------------------------------------------- parsers
def test_supplement_parser():
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MSA_timings"
    ws.append(["Metadata", "Unit is ms"])
    ws.append([None])
    ws.append(["T1031", 95.0, 1971.7])
    ws.append(["T1047s1", 232.0, 1955.0])
    buf = io.BytesIO()
    wb.save(buf)
    assert bt.supplement_targets(buf.getvalue()) == [("T1031", 95), ("T1047s1", 232)]


def test_casp_parsers():
    seqs = bt.parse_casp_fasta(
        ">T1031 S0A2C3d1, , 95 residues|\nMKV\nLL\n>T1033 x|\nAA\n"
    )
    assert seqs == {"T1031": "MKVLL", "T1033": "AA"}
    lst = bt.parse_casp_list(
        "Target;Type;Res;Oligo;a;b;c;d;e;Description\nT1031;All;95;A1;x;x;x;x;-;S0A2C3d1 6vr4  \nT1090;All;193;A1;x;x;x;x;-;NP_595817.1   \n"
    )
    assert lst["T1031"] == {"oligo": "A1", "pdb_hint": "6VR4"}
    assert lst["T1090"]["pdb_hint"] is None
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"ATOM\n"
        info = tarfile.TarInfo("T/T1031.pdb")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    assert bt.casp_files(buf.getvalue()) == {"T1031": b"ATOM\n"}


def test_truth_rule_prefers_pdb_and_falls_back_to_casp(tmp_path, seq30):
    from sp_testutil import helix, write_ca_cif

    xyz = helix(30)
    casp_pdb = tmp_path / "casp.cif"
    write_ca_cif(casp_pdb, seq30, xyz)
    partial = tmp_path / "partial.cif"
    write_ca_cif(partial, seq30[:20], xyz[:20])
    import gzip

    def fake_get(url, data=None):
        if url == bt.RCSB_SEARCH:
            return json.dumps({"result_set": [{"identifier": "1ABC_1"}]}).encode()
        if url == bt.RCSB_GRAPHQL:
            return json.dumps(
                {
                    "data": {
                        "polymer_entities": [
                            {
                                "rcsb_id": "1ABC_1",
                                "rcsb_polymer_entity_container_identifiers": {
                                    "auth_asym_ids": ["A"]
                                },
                                "entry": {
                                    "rcsb_accession_info": {
                                        "initial_release_date": "2021-01-01T00:00:00Z"
                                    }
                                },
                            }
                        ]
                    }
                }
            ).encode()
        return gzip.compress(partial.read_bytes())

    truth = tmp_path / "truth"
    truth.mkdir()
    # PDB covers 20/30 < CASP file's 30 -> CASP file wins, with the reason recorded.
    import gemmi

    pdb_text = gemmi.read_structure(str(casp_pdb))
    pdb_bytes = pdb_text.make_pdb_string().encode()
    res = bt.choose_truth("T9", seq30, None, pdb_bytes, truth, fake_get)
    assert res["source"] == "casp" and "20 matched < 30 needed" in res["reason"]
    # No CASP file: PDB must cover >= 90 %; 20/30 is not enough -> no truth.
    assert bt.choose_truth("T9", seq30, None, None, truth, fake_get) is None


# ---------------------------------------------------------------- infra
def test_launch_plans(tmp_path):
    launch = load(ROOT / "infra" / "launch.py")
    cfg = launch.load_hosts()
    trn2 = launch.plan("trn2", cfg, ami="ami-1", az="us-east-1d")
    assert trn2[:3] == ["aws", "ec2", "run-instances"]
    assert trn2[trn2.index("--instance-initiated-shutdown-behavior") + 1] == "terminate"
    bdm = json.loads(trn2[trn2.index("--block-device-mappings") + 1])
    assert [b["Ebs"]["VolumeSize"] for b in bdm] == [300]
    assert "shutdown -h +1440" in trn2[trn2.index("--user-data") + 1]
    g7e = launch.plan("g7e", cfg, ami="ami-2", spot=True)
    bdm = json.loads(g7e[g7e.index("--block-device-mappings") + 1])
    assert [b["Ebs"]["VolumeSize"] for b in bdm] == [300, 2000]
    assert all(b["Ebs"]["DeleteOnTermination"] for b in bdm)
    assert "MarketType=spot" in g7e
    c7a = launch.plan("c7a", cfg, ami="ami-3")
    assert c7a[:2] == ["spawn", "launch"] and c7a[c7a.index("--ttl") + 1] == "216h"
    assert c7a[c7a.index("--volume-size") + 1] == "300"
    assert "Project=structure-prediction" in c7a
    for argv in (trn2, g7e, c7a):
        assert not any(
            a.isdigit() and len(a) == 12 for a in argv
        )  # never an account ID


def test_spawn_refuses_data_volume():
    launch = load(ROOT / "infra" / "launch.py")
    cfg = launch.load_hosts()
    cfg["hosts"]["r7a"]["launcher"] = "spawn"
    with pytest.raises(ValueError, match="empty data volume"):
        launch.plan("r7a", cfg, ami="ami-4")


def test_teardown_targets_only_this_project():
    launch = load(ROOT / "infra" / "launch.py")
    seen = []

    def fake(argv):
        seen.append(argv)
        return "i-1 i-2"

    (cmd,) = launch.teardown_argvs("g7e", "us-east-1", fake)
    assert cmd[-2:] == ["i-1", "i-2"]
    assert "Name=tag:Project,Values=structure-prediction" in seen[0]
    assert "Name=tag:Role,Values=g7e" in seen[0]


# ---------------------------------------------------------------- entry / msa
def test_timing_recorder_sums_batches(tmp_path):
    ticks = iter([0.0, 2.0, 5.0, 8.5])
    rec = entry.TimingRecorder(tmp_path / "t.json", clock=lambda: next(ticks))
    rec.start(), rec.end(), rec.start(), rec.end()
    rec.write((None, None))
    doc = json.loads((tmp_path / "t.json").read_text())
    assert doc["model_s"] == pytest.approx(5.5) and doc["batch_s"] == [2.0, 3.5]


def test_entry_requires_separator():
    with pytest.raises(SystemExit, match="usage"):
        entry.main(["--target", "boltz", "--timing-out", "x"])


def test_msa_search_one_query_at_a_time(tmp_path, monkeypatch):
    rs = load(ROOT / "msa" / "run_search.py")
    monkeypatch.setattr(
        rs,
        "capture_manifest",
        lambda **k: {
            "instance": {"instance_type": "g7e.2xlarge"},
            "prices": {"on_demand": {"usd_per_hour": 3.0}},
        },
    )
    targets = {"targets": [
        {"chains": [{"entity": "protein", "msa_id": "T1", "sequence": "MKV"}]},
        {"chains": [{"entity": "protein", "msa_id": "C_1", "sequence": "AAA"}, {"entity": "protein", "msa_id": "C_1", "sequence": "AAA"}, {"entity": "rna", "msa_id": None, "sequence": "GG"}]},
    ]}  # fmt: skip
    tj = tmp_path / "t.json"
    tj.write_text(json.dumps(targets))
    calls = []

    def runner(argv, env):
        calls.append(argv)
        out = Path(
            argv[argv.index(str(tmp_path / "db")) + 1]
        )  # argv: ... FASTA DB OUT [--gpu ...]
        out.mkdir(parents=True, exist_ok=True)
        (out / "0.a3m").write_text(">q\nX\n")
        return 0, 1.25

    rc = rs.main(
        [
            "--mode",
            "gpu",
            "--targets",
            str(tj),
            "--db",
            str(tmp_path / "db"),
            "--out",
            str(tmp_path / "msas"),
            "--timing",
            str(tmp_path / "timing.json"),
        ],
        runner=runner,
    )
    assert rc == 0 and len(calls) == 2
    assert calls[0][-4:] == ["--gpu", "1", "--gpu-server", "1"]
    assert sorted(p.name for p in (tmp_path / "msas").iterdir()) == [
        "C_1.a3m",
        "T1.a3m",
    ]
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["per_target_s"] == {"T1": 1.25, "C_1": 1.25}
    assert (
        timing["host_instance_type"] == "g7e.2xlarge"
        and timing["price_od_usd_per_h"] == 3.0
    )


def test_repro_check_uses_median_of_warm_batches():
    from harness import repro_check as rc

    ok = rc.check([60.0, 7.4, 7.9, 7.6], 7.5, 0.10)
    assert ok["median_s"] == 7.6 and ok["within"]
    assert not rc.check([60.0, 8.4, 8.5, 8.6], 7.5, 0.10)["within"]
    with pytest.raises(ValueError):
        rc.check([7.5, 7.5], 7.5, 0.1)


def test_capacity_block_launch_and_per_host_region():
    launch = load(ROOT / "infra" / "launch.py")
    cfg = launch.load_hosts()
    assert launch.host_region(cfg, "trn2") == "sa-east-1"
    assert launch.host_region(cfg, "g7e") == cfg["defaults"]["region"]
    a = launch.plan("trn2", cfg, ami="ami-1", reservation_id="cr-0abc")
    assert a[a.index("--region") + 1] == "sa-east-1"
    assert "MarketType=capacity-block" in a
    assert "CapacityReservationTarget={CapacityReservationId=cr-0abc}" in a
    assert (
        launch.plan("trn2", cfg, ami="ami-1", region="ap-southeast-4")[
            a.index("--region") + 1
        ]
        == "ap-southeast-4"
    )
    s = launch.plan("c7a", cfg, ami="ami-2", reservation_id="cr-0def")
    assert s[s.index("--reservation-id") + 1] == "cr-0def" and "--capacity-block" in s
    with pytest.raises(ValueError, match="cannot also be spot"):
        launch.plan("trn2", cfg, ami="ami-1", reservation_id="cr-0abc", spot=True)


def test_ami_resolved_in_the_hosts_region(capsys):
    launch = load(ROOT / "infra" / "launch.py")
    seen = []

    def fake(argv):
        seen.append(argv)
        return "ami-0123\n"

    assert launch.main(["plan", "trn2", "--reservation-id", "cr-1"], runner=fake) == 0
    assert seen[0][seen[0].index("--region") + 1] == "sa-east-1"
    assert "capacity-block" in capsys.readouterr().out


def test_step_dump_records_every_denoising_call(tmp_path):
    np = pytest.importorskip("numpy")

    class _T:  # minimal tensor stand-in
        def __init__(self, a):
            self.a = a

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class _Diff:
        def preconditioned_network_forward(
            self, noised, sigma, network_condition_kwargs
        ):
            return _T(noised.a * 0.5)

    class _Mod:
        structure_module = _Diff()

    dump = entry.StepDump(tmp_path / "d")
    mod = _Mod()
    dump.attach(mod)
    dump.start("T1037")
    for step, sigma in enumerate((160.0, 80.0)):
        for chunk in (2, 3):  # two chunks per step
            out = mod.structure_module.preconditioned_network_forward(
                _T(np.full((chunk, 4, 3), step + 1.0)),
                sigma,
                network_condition_kwargs={},
            )
            assert out.a.shape == (chunk, 4, 3)
    path = dump.end()
    z = np.load(path)
    assert path.name == "T1037.npz" and z["denoised"].shape == (10, 4, 3)
    assert z["chunk"].tolist() == [2, 3, 2, 3] and z["sigma"].tolist() == [
        160.0,
        160.0,
        80.0,
        80.0,
    ]
    assert z["denoised"][0, 0, 0] == 0.5 and z["denoised"][-1, 0, 0] == 1.0
    dump.start("empty")
    assert dump.end() is None
    with pytest.raises(RuntimeError):
        entry.StepDump(tmp_path).attach(object())


def test_private_neuron_venv_isolates_editable_torch_neuronx(tmp_path):
    # #107: a copied DLAMI venv still imports the shared editable torch_neuronx tree.
    import subprocess

    script = Path(__file__).resolve().parents[1] / "infra" / "private_neuron_venv.sh"
    shared = tmp_path / "source-repos" / "torch-neuronx"
    (shared / "torch_neuronx").mkdir(parents=True)
    (shared / "torch_neuronx" / "__init__.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q", str(shared)], check=True)
    subprocess.run(["git", "-C", str(shared), "add", "."], check=True)
    subprocess.run(["git", "-C", str(shared), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], check=True)  # fmt: skip
    venv = tmp_path / "opt-venv"
    site = venv / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (venv / "bin").mkdir()
    (venv / "bin" / "activate").write_text(f"VIRTUAL_ENV={venv}\n")
    finder = site / "__editable___torch_neuronx_2_14_3_finder.py"
    finder.write_text(
        f"MAPPING: dict[str, str] = {{'torch_mlir': '/x', 'torch_neuronx': "
        f"'{shared / 'torch_neuronx'}'}}\nNAMESPACES = {{}}\n"
    )
    dest = tmp_path / "agent"
    env = {**os.environ, "NEURON_VENV": str(venv), "SKIP_IMPORT_CHECK": "1"}
    subprocess.run(
        ["bash", str(script), str(dest)], check=True, env=env, capture_output=True
    )
    own = dest / "venv" / "lib" / "python3.12" / "site-packages" / finder.name
    assert f"'torch_neuronx': '{dest / 'tn-src' / 'torch_neuronx'}'" in own.read_text()
    assert "'torch_mlir': '/x'" in own.read_text()  # other mappings untouched
    assert (
        str(shared / "torch_neuronx") in finder.read_text()
    )  # the shared venv is untouched
    assert str(dest / "venv") in (dest / "venv" / "bin" / "activate").read_text()
    ok = subprocess.run(
        ["bash", str(script), "--check", str(dest)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    (shared / "torch_neuronx" / "__init__.py").write_text(
        "x = 2  # SHIM\n"
    )  # setup.sh-style edit
    bad = subprocess.run(
        ["bash", str(script), "--check", str(dest)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1 and "shared torch_neuronx source modified" in bad.stdout
