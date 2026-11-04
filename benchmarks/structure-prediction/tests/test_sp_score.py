# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
from sp_testutil import helix, write_ca_cif

from harness import score as s


def rot(theta):
    c, si = np.cos(theta), np.sin(theta)
    return np.array([[c, -si, 0], [si, c, 0], [0, 0, 1.0]])


def test_rmsd_is_invariant_to_rigid_motion():
    x = helix(40)
    y = x @ rot(0.7).T + np.array([5.0, -3.0, 2.0])
    assert s.rmsd_superposed(y, x) == pytest.approx(0.0, abs=1e-6)


def test_rmsd_known_value():
    x = helix(40)
    y = x.copy()
    y[:, 0] += np.where(
        np.arange(40) % 2 == 0, 1.0, -1.0
    )  # +-1 Å, mean-zero, not removable
    assert 0.5 < s.rmsd_superposed(y, x) <= 1.0


def test_lddt_perfect_and_degraded():
    x = helix(40)
    assert s.lddt_ca(x @ rot(1.0).T, x) == pytest.approx(1.0)
    noisy = x + np.random.default_rng(1).normal(0, 1.5, x.shape)
    assert s.lddt_ca(noisy, x) < 0.8


def test_alignment_handles_gaps_and_offsets():
    a = "MKVLAAGIVGLLAQ"
    b = "XXMKVLAGIVGLLAQYY"  # missing one A, extra flanks
    pairs = s.align_sequences(a, b)
    same = [(i, j) for i, j in pairs if a[i] == b[j]]
    assert len(same) == 13


def test_score_prediction_end_to_end(tmp_path, seq30):
    truth_xyz = helix(30)
    # Ground truth is numbered from 101 and is missing residues 1-2 of the model.
    truth = write_ca_cif(
        tmp_path / "truth.cif", seq30[2:], truth_xyz[2:], chain="B", start=103
    )
    pred = write_ca_cif(tmp_path / "pred.cif", seq30, truth_xyz @ rot(0.3).T + 4.0)
    out = s.score_prediction(
        pred, s.load_structure(truth), {"A": "B"}, tm=lambda c: None
    )
    assert out["n_residues"] == 28
    assert out["ca_rmsd"] == pytest.approx(0.0, abs=1e-3)
    assert out["lddt_ca"] == pytest.approx(1.0)
    assert out["ligand_rmsd"] is None


def test_scoring_unit_restricts_residues(tmp_path, seq30):
    xyz = helix(30)
    truth = s.load_structure(write_ca_cif(tmp_path / "t.cif", seq30, xyz))
    pred = write_ca_cif(tmp_path / "p.cif", seq30, xyz)
    out = s.score_prediction(
        pred, truth, {"A": "A"}, unit={"A": [[1, 10], [21, 25]]}, tm=lambda c: None
    )
    assert out["n_residues"] == 15


def test_pocket_aligned_ligand_rmsd(tmp_path, seq30):
    xyz = helix(30)
    lig = {
        "C1": xyz[10] + [6, 0, 0],
        "C2": xyz[11] + [6, 0, 0],
        "C3": xyz[12] + [6, 0, 0],
    }
    truth = s.load_structure(write_ca_cif(tmp_path / "t.cif", seq30, xyz, ligand=lig))
    moved = {
        k: v + [1.0, 0, 0] for k, v in lig.items()
    }  # ligand shifted 1 Å, protein exact
    pred = write_ca_cif(tmp_path / "p.cif", seq30, xyz, ligand=moved)
    out = s.score_prediction(pred, truth, {"A": "A"}, tm=lambda c: None)
    assert out["ligand_rmsd"] == pytest.approx(1.0, abs=1e-3)


def test_usalign_output_parser_reads_tm2():
    text = "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\n/a/model.pdb:A\t/b/ref.pdb:A\t0.8812\t0.9034\t1.20\n"
    assert s.parse_usalign_outfmt2(text) == pytest.approx(0.9034)


def test_tm_score_uses_runner(tmp_path):
    corr = s.Correspondence(helix(10), helix(10), [("A", i) for i in range(10)])
    seen = []

    def fake(argv):
        seen.append(argv)
        return "#PDBchain1\tPDBchain2\tTM1\tTM2\nm\tr\t0.5\t0.6\n"

    assert s.tm_score(corr, run=fake) == pytest.approx(0.6)
    assert seen[0][3:] == ["-TMscore", "1", "-outfmt", "2"]


# ---------------------------------------------------------------- gate
def rows(spec):
    """spec: {(backend, ds): {target: (s0, s1)}} for ca_rmsd; lddt fixed unless given."""
    out = []
    for (backend, ds), per in spec.items():
        for tid, vals in per.items():
            for seed, v in enumerate(vals):
                ca, ld = v if isinstance(v, tuple) else (v, 0.95)
                out.append(
                    {
                        "target_id": tid,
                        "backend": backend,
                        "diffusion_samples": str(ds),
                        "lnc": "",
                        "seed": str(seed),
                        "ca_rmsd": ca,
                        "lddt_ca": ld,
                        "ligand_rmsd": "",
                    }
                )
    return out


IDS = [f"T{i:02d}" for i in range(20)]


def test_gate_pass_with_floor():
    r = rows(
        {
            ("cpu", 1): dict.fromkeys(IDS, (1.0, 1.0)),
            ("neuron", 1): dict.fromkeys(IDS, (1.4, 1.3)),
        }
    )
    res = s.gate(
        r,
        {"backend": "neuron", "diffusion_samples": "1"},
        {"backend": "cpu", "diffusion_samples": "1"},
        IDS,
    )
    assert res["phase_pass"] and res["n_pass"] == 20
    m = res["targets"][0]["metrics"]["ca_rmsd"]
    assert m["tolerance"] == 0.5 and m["deviation"] == pytest.approx(0.4)
    assert "ligand_rmsd" not in res["targets"][0]["metrics"]  # no ligand -> not gated


def test_gate_spread_widens_tolerance():
    r = rows(
        {
            ("cpu", 1): dict.fromkeys(IDS, (1.0, 3.0)),
            ("neuron", 1): dict.fromkeys(IDS, (2.9, 1.2)),
        }
    )
    res = s.gate(r, {"backend": "neuron"}, {"backend": "cpu"}, IDS)
    assert res["n_pass"] == 20
    assert res["targets"][0]["metrics"]["ca_rmsd"]["tolerance"] == 2.0


def test_gate_two_failures_still_pass_three_do_not():
    ref = dict.fromkeys(IDS, (1.0, 1.0))
    acc = dict.fromkeys(IDS, (1.1, 1.1))
    for t in IDS[:2]:
        acc[t] = (1.8, 1.1)  # deviation 0.8 > 0.5 floor, < 2x cap (1.0)
    res = s.gate(
        rows({("cpu", 1): ref, ("neuron", 1): acc}),
        {"backend": "neuron"},
        {"backend": "cpu"},
        IDS,
    )
    assert res["n_pass"] == 18 and res["phase_pass"]
    acc[IDS[2]] = (1.8, 1.1)
    res = s.gate(
        rows({("cpu", 1): ref, ("neuron", 1): acc}),
        {"backend": "neuron"},
        {"backend": "cpu"},
        IDS,
    )
    assert res["n_pass"] == 17 and not res["phase_pass"]


def test_gate_cap_fails_phase_even_with_18_passes():
    ref = dict.fromkeys(IDS, (1.0, 1.0))
    acc = dict.fromkeys(IDS, (1.0, 1.0))
    acc[IDS[0]] = (2.2, 1.0)  # deviation 1.2 > 2 x 0.5
    res = s.gate(
        rows({("cpu", 1): ref, ("neuron", 1): acc}),
        {"backend": "neuron"},
        {"backend": "cpu"},
        IDS,
    )
    assert res["n_pass"] == 19 and res["cap_exceeded"] and not res["phase_pass"]


def test_gate_lddt_floor():
    ref = dict.fromkeys(IDS, ((1.0, 0.9), (1.0, 0.9)))
    acc = dict.fromkeys(IDS, ((1.0, 0.885), (1.0, 0.905)))
    res = s.gate(
        rows({("cpu", 1): ref, ("neuron", 1): acc}),
        {"backend": "neuron"},
        {"backend": "cpu"},
        IDS,
    )
    m = res["targets"][0]["metrics"]["lddt_ca"]
    assert m["tolerance"] == 0.01 and not m["pass_"]


def test_gate_missing_seed_fails_target_and_19_targets_never_pass():
    ref = dict.fromkeys(IDS, (1.0, 1.0))
    acc = dict.fromkeys(IDS, (1.0, 1.0))
    r = [
        x
        for x in rows({("cpu", 1): ref, ("neuron", 1): acc})
        if not (
            x["backend"] == "neuron" and x["target_id"] == "T00" and x["seed"] == "1"
        )
    ]
    res = s.gate(r, {"backend": "neuron"}, {"backend": "cpu"}, IDS)
    assert not res["targets"][0]["pass"] and res["n_pass"] == 19
    assert not s.gate(r, {"backend": "neuron"}, {"backend": "cpu"}, IDS[1:])[
        "phase_pass"
    ]  # 19 gated ids


def test_gate_tm_score_is_not_gated():
    assert "tm_score" not in s.GATE_METRICS


def test_ds5_needs_g7e_reference_and_passed_prereq(tmp_path):
    acc, ref = s.parse_filter("backend=neuron,ds=5"), s.parse_filter("backend=gpu,ds=5")
    with pytest.raises(SystemExit, match="needs --prereq"):
        s.check_prereq(acc, ref, None)
    with pytest.raises(SystemExit, match="g7e"):
        s.check_prereq(acc, s.parse_filter("backend=cpu,ds=1"), None)
    pre = tmp_path / "pre.json"
    pre.write_text(
        json.dumps(
            {
                "acc": {"backend": "gpu", "diffusion_samples": "1"},
                "ref": {"backend": "cpu"},
                "phase_pass": False,
            }
        )
    )
    with pytest.raises(SystemExit, match="has not passed"):
        s.check_prereq(acc, ref, pre)
    pre.write_text(
        json.dumps(
            {
                "acc": {"backend": "gpu", "diffusion_samples": "1"},
                "ref": {"backend": "cpu"},
                "phase_pass": True,
            }
        )
    )
    s.check_prereq(acc, ref, pre)
    s.check_prereq(
        s.parse_filter("backend=neuron,ds=1"), s.parse_filter("backend=cpu,ds=1"), None
    )


def test_gate_markdown_mentions_verdict():
    r = rows(
        {
            ("cpu", 1): dict.fromkeys(IDS, (1.0, 1.0)),
            ("neuron", 1): dict.fromkeys(IDS, (1.0, 1.0)),
        }
    )
    md = s.gate_markdown(s.gate(r, {"backend": "neuron"}, {"backend": "cpu"}, IDS))
    assert "**PASS**" in md and "| T00 | ca_rmsd |" in md


def test_score_results_top_ranked_and_best_of_n(tmp_path, seq30):
    xyz = helix(30)
    tdir = tmp_path / "truth"
    tdir.mkdir()
    write_ca_cif(tdir / "T1.cif", seq30, xyz)
    rdir = tmp_path / "res" / "x"
    rdir.mkdir(parents=True)
    good = write_ca_cif(rdir / "T1_model_1.cif", seq30, xyz)
    bad = write_ca_cif(
        rdir / "T1_model_0.cif",
        seq30,
        xyz + np.random.default_rng(0).normal(0, 1, xyz.shape),
    )
    (rdir / "result.json").write_text(
        json.dumps(
            {
                "status": "measured",
                "target_id": "T1",
                "backend": "neuron",
                "lnc": 2,
                "diffusion_samples": 5,
                "atom_graph_break": None,
                "seed": 0,
                "predictions": [str(bad), str(good)],
            }
        )
    )
    targets = {
        "targets": [
            {
                "id": "T1",
                "gated": True,
                "chains": [{"chain_id": "A", "entity": "protein", "truth_chain": "A"}],
                "truth": {"file": "T1.cif", "unit": None},
            }
        ]
    }
    (row,) = s.score_results(tmp_path / "res", targets, tdir, tm=lambda c: None)
    assert row["ca_rmsd"] > 0.5  # top-ranked (model_0) is what is gated
    assert row["best_ca_rmsd"] == pytest.approx(0.0, abs=1e-3)
    assert row["n_samples"] == 2
