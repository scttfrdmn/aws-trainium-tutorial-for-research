# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from sp_testutil import helix

from harness import gate_selftest as gs
from harness import score as s
from harness.score import Chain, Structure


def test_target_deviation_leave_one_out():
    d = s.target_deviation([1.0, 2.0, 3.0], [2.0])
    # centres: median(2,3)=2.5, median(1,3)=2, median(1,2)=1.5
    assert d["d_ref"] == pytest.approx(1.0)
    assert d["d_acc"] == pytest.approx(1 / 3)
    assert d["excess"] == pytest.approx(-2 / 3)
    assert d["shift"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        s.target_deviation([1.0], [1.0])


def _rows(
    backend, seeds, n_targets=19, shift=0.0, sd=0.3, lddt_shift=0.0, bad=(), rng=None
):
    rng = rng or np.random.default_rng(1)
    rows = []
    for t in range(n_targets):
        base_rmsd, base_lddt = 5.0 + t, 0.6 + 0.01 * t
        for seed in seeds:
            off = 5.0 if f"T{t}" in bad else 0.0
            rows.append({
                "target_id": f"T{t}", "backend": backend, "diffusion_samples": "1",
                "variant": "", "seed": seed,
                "ca_rmsd": base_rmsd + shift + off + rng.normal(0, sd),
                "lddt_ca": base_lddt + lddt_shift + rng.normal(0, 0.005),
                "ligand_rmsd": "",
            })  # fmt: skip
    return rows


GATED = [f"T{t}" for t in range(19)]


def test_calibrated_gate_accepts_same_distribution_and_rejects_defects():
    rng = np.random.default_rng(7)
    ref = _rows("cpu", range(5), rng=rng)
    same = s.calibrated_gate(ref + _rows("x", (0, 1), rng=rng), {"backend": "x"},
                             {"backend": "cpu"}, GATED, n_boot=2000)  # fmt: skip
    assert same["phase_pass"], same["reasons"]
    assert set(same["metrics"]) == {"ca_rmsd", "lddt_ca"}  # no ligands scored
    biased = s.calibrated_gate(ref + _rows("x", (0, 1), lddt_shift=-0.03, rng=rng),
                               {"backend": "x"}, {"backend": "cpu"}, GATED, n_boot=2000)  # fmt: skip
    assert not biased["phase_pass"]
    assert any(r.startswith("lddt_ca") for r in biased["reasons"])
    one_bad = s.calibrated_gate(ref + _rows("x", (0, 1), bad=("T3",), rng=rng),
                                {"backend": "x"}, {"backend": "cpu"}, GATED, n_boot=2000,
                                method="calibrated/1")  # fmt: skip
    assert one_bad["cap_exceeded"] == ["T3"] and not one_bad["phase_pass"]


def test_calibrated_gate_missing_target_fails_and_seed_filters():
    ref = _rows("cpu", range(5))
    acc = [r for r in _rows("x", (0, 1)) if r["target_id"] != "T5"]
    res = s.calibrated_gate(
        ref + acc, {"backend": "x"}, {"backend": "cpu"}, GATED, n_boot=500
    )
    assert res["incomplete"] == ["T5"] and not res["phase_pass"]
    res = s.calibrated_gate(ref, {"backend": "cpu"}, {"backend": "cpu"}, GATED,
                            acc_seeds=[3, 4], ref_seeds=[0, 1, 2], n_boot=500)  # fmt: skip
    assert all(len(m["ref"]) == 3 and len(m["acc"]) == 2
               for t in res["targets"] for m in t["metrics"].values())  # fmt: skip
    assert "Calibrated gate" in s.calibrated_gate_markdown(res)


def _struct(ca):
    return Structure({"A": Chain("A", "A" * len(ca), list(range(1, len(ca) + 1)), ca)})


def test_perturbations():
    x = helix(60)
    st = _struct(x.copy())
    m = gs.mirror(st)
    assert np.allclose(m.chains["A"].ca[:, 0], -x[:, 0]) and np.allclose(
        st.chains["A"].ca, x
    )
    r = gs.register_shift(st)
    moved = np.where(np.any(r.chains["A"].ca != x, axis=1))[0]
    assert moved.min() == 20 and len(moved) == 12
    assert np.allclose(r.chains["A"].ca[20], x[23])
    n = gs.add_noise(st, 1.0, np.random.default_rng(0))
    assert 0.8 < np.std(n.chains["A"].ca - x) < 1.2


def test_selftest_end_to_end_on_synthetic_targets():
    rng = np.random.default_rng(3)
    truths = {f"T{t}": _struct(helix(40 + 5 * t)) for t in range(8)}
    structs = {}
    for tid, tr in truths.items():
        for seed in range(5):
            ca = tr.chains["A"].ca + rng.normal(0, 0.6, tr.chains["A"].ca.shape)
            structs[(tid, seed)] = _struct(ca)

    def scorer(tid, model):
        return s.score_model(model, truths[tid], {"A": "A"}, tm=None)

    res = gs.run_selftest(structs, scorer, list(truths), range(5), replicates=1,
                          local_replicates=1, n_boot=300,
                          only_defects=["mirror", "noise:2.0", "local:1"])  # fmt: skip
    assert len(res["calibrated_aa"]) == 10 and len(res["section2_aa"]) == 30
    assert res["section2_aa_preregistered"]["test"] == [2, 3]
    assert res["planted"]["mirror"]["rejection_rate"] == 1.0
    assert res["planted"]["noise:2.0"]["rejection_rate"] == 1.0
    assert set(res["planted"]) == {"mirror", "noise:2.0", "local:1"}
    assert "planted defects" in gs.selftest_markdown(res)


def test_calibrated2_ca_rmsd_is_a_coarse_check():
    rng = np.random.default_rng(11)
    ref = _rows("cpu", range(5), rng=rng)
    kw = {"n_boot": 2000, "method": "calibrated/2"}
    # a 5 Å CA-RMSD swing on one target: calibrated/1 rejects it, calibrated/2 does not
    swing = ref + _rows("x", (0, 1), bad=("T3",), rng=rng)
    assert s.calibrated_gate(swing, {"backend": "x"}, {"backend": "cpu"}, GATED, **kw)[
        "phase_pass"
    ]
    # an exploded target (+80 Å) is still caught, per target
    rows = ref + _rows("x", (0, 1), rng=rng)
    for r in rows:
        if r["backend"] == "x" and r["target_id"] == "T4":
            r["ca_rmsd"] += 80.0
    res = s.calibrated_gate(rows, {"backend": "x"}, {"backend": "cpu"}, GATED, **kw)
    assert (
        res["cap_exceeded"] == ["T4"] and res["metrics"]["ca_rmsd"]["kind"] == "coarse"
    )
    with pytest.raises(ValueError):
        s.calibrated_gate(
            rows, {"backend": "x"}, {"backend": "cpu"}, GATED, method="nope"
        )


def test_explode_and_held_out_reference_pool():
    x = helix(30)
    e = gs.explode(_struct(x.copy()), 5.0)
    c = x.mean(0)
    assert np.allclose(e.chains["A"].ca - c, 5.0 * (x - c))
    rng = np.random.default_rng(5)
    truths = {f"T{t}": _struct(helix(40 + 5 * t)) for t in range(6)}
    structs = {
        (tid, seed): _struct(tr.chains["A"].ca + rng.normal(0, 0.6, tr.chains["A"].ca.shape))
        for tid, tr in truths.items() for seed in range(7)
    }  # fmt: skip

    def scorer(tid, model):
        return s.score_model(model, truths[tid], {"A": "A"}, tm=None)

    res = gs.run_selftest(structs, scorer, list(truths), [4, 5, 6], ref_pool=[0, 1, 2, 3],
                          replicates=1, local_replicates=2, n_boot=300,
                          only_defects=["explode:1"])  # fmt: skip
    assert [r["ref"] for r in res["calibrated_aa"]] == [[0, 1, 2, 3]] * 3
    assert len(res["section2_aa"]) == 3 * 6
    assert res["planted"]["explode:1"]["trials"] == 6
    assert res["planted"]["explode:1"]["rejection_rate"] == 1.0
    with pytest.raises(ValueError):
        gs.run_selftest(structs, scorer, list(truths), [3, 4], ref_pool=[0, 3])
