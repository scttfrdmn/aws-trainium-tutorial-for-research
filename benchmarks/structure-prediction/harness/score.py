# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Accuracy scoring and the work plan's section-2 accuracy gate.

    python -m harness.score score --results RESULTS_ROOT --targets targets/targets.json \
        --truth-dir truth/ --out scores.csv
    python -m harness.score gate --scores scores.csv --acc backend=neuron,ds=1,lnc=2 \
        --ref backend=cpu,ds=1 --out gate_trn2_lnc2_ds1 [--prereq gate_g7e_ds1.json]

Metrics (all against ground truth, over residues matched by global sequence alignment and, for
CASP evaluation units, restricted to the unit's residue ranges):

* CA-RMSD: Kabsch superposition on all matched CA atoms (implemented here, numpy).
* lDDT-CA: global lDDT on CA atoms, inclusion radius 15 Å in the reference, thresholds
  0.5/1/2/4 Å, pairs within the same residue excluded; score = preserved pair-threshold
  count / (4 × considered pairs). Implemented here (numpy) so the gate does not depend on an
  external tool's version; OpenStructure `lddt` can be used as a cross-check.
* Ligand RMSD (pocket-aligned): superpose on reference CA atoms within 10 Å of any reference
  ligand heavy atom, then RMSD over ligand heavy atoms matched by atom name. No symmetry
  correction (an upper bound for symmetric ligands).
* TM-score: US-align, sequence-dependent (`-TMscore 1`), normalised by the reference length.
  Reported, not gated.

The scored sample is Boltz-2's top-ranked one (`*_model_0.cif`, ranked by confidence_score);
best-of-N over all samples is a secondary column and is not gated.

Gate (work plan section 2), per target and gate metric:
    deviation = max over seeds s in {0,1} of |m(acc, s) - m(ref, s)|
    spread    = |m(ref, 0) - m(ref, 1)|
    pass      if deviation <= max(spread, floor)
Phase pass: at most 2 gated targets fail (>= 17 of the 19 CASP14 targets with ground truth; T1047s1
has none and is timing-only, and if CASP supplies it the same rule gives >= 18 of 20), and no
deviation exceeds 2 x max(spread, floor). ref is CPU fp32 at ds=1; at ds=5 the ref is the g7e ds=5 run, which is
only allowed once the g7e ds=1 run has passed against CPU (`--prereq`). The two ceiling
complexes are never gated. Secondary (variant) rows, e.g. T1061 at --max_msa_seqs 2048, are gated
per target against a reference run with the same variant; they carry no phase verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

GATE_METRICS = ("ca_rmsd", "lddt_ca", "ligand_rmsd")
FLOORS = {"ca_rmsd": 0.5, "lddt_ca": 0.01, "ligand_rmsd": 0.5}
# Phase pass allows this many gated targets to fail: 18/20 as written, 17/19 while T1047s1 has no
# ground truth (owner decision, 2026-11-04).
PHASE_ALLOWED_FAILURES = 2
CAP_FACTOR = 2.0
LDDT_RADIUS = 15.0
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
POCKET_RADIUS = 10.0
WATER = {"HOH", "WAT", "DOD"}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
    "SEC": "U", "PYL": "O",
}  # fmt: skip


# ----------------------------------------------------------------------------- structures
@dataclass
class Chain:
    """Polymer residues of one chain: one-letter code, residue number, CA coordinate."""

    chain_id: str
    seq: str
    resnums: list[int]
    ca: np.ndarray  # (n, 3)


@dataclass
class Structure:
    """Protein chains and ligand heavy atoms of one model."""

    chains: dict[str, Chain]
    ligands: dict[str, dict[str, np.ndarray]] = field(
        default_factory=dict
    )  # key -> name -> xyz


def load_structure(path: str | Path) -> Structure:
    """Read the first model of an mmCIF/PDB file with gemmi."""
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    model = st[0]
    chains: dict[str, Chain] = {}
    ligands: dict[str, dict[str, np.ndarray]] = {}
    for ch in model:
        seq, nums, cas = [], [], []
        for res in ch:
            if res.name in WATER:
                continue
            is_poly = res.entity_type == gemmi.EntityType.Polymer
            if is_poly:
                ca = res.find_atom("CA", "*")
                if not ca:  # gemmi returns a null atom handle when absent
                    continue
                seq.append(THREE_TO_ONE.get(res.name, "X"))
                nums.append(int(res.seqid.num or 0))
                cas.append([ca.pos.x, ca.pos.y, ca.pos.z])
            else:
                key = f"{ch.name}:{res.name}:{res.seqid.num}"
                ligands[key] = {
                    a.name: np.array([a.pos.x, a.pos.y, a.pos.z])
                    for a in res
                    if a.element.name != "H"
                }
        if cas:
            chains[ch.name] = Chain(
                ch.name, "".join(seq), nums, np.array(cas, dtype=float)
            )
    return Structure(chains, ligands)


def align_sequences(a: str, b: str) -> list[tuple[int, int]]:
    """Global alignment of two sequences (cached: it depends on the sequences only)."""
    return list(_align_cached(a, b))


@lru_cache(maxsize=4096)
def _align_cached(a: str, b: str) -> tuple[tuple[int, int], ...]:
    """Global (Needleman-Wunsch) alignment; returns matched index pairs (i in a, j in b).

    Scoring: +2 identity, -1 mismatch, -2 gap. Only aligned identical-or-mismatched columns are
    returned; with a model of the same sequence this is the residue correspondence.
    """
    n, m = len(a), len(b)
    gap, match, mis = -2, 2, -1
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    bv = np.frombuffer(b.encode(), dtype=np.uint8)
    for i in range(1, n + 1):
        sub = np.where(bv == ord(a[i - 1]), match, mis)
        diag = score[i - 1, :-1] + sub
        up = score[i - 1, 1:] + gap
        best = np.maximum(diag, up)
        row = score[i]
        row[1:] = best
        for j in range(1, m + 1):  # left moves depend on the row being built
            if row[j - 1] + gap > row[j]:
                row[j] = row[j - 1] + gap
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        s = int(score[i, j])
        diag_score = match if a[i - 1] == b[j - 1] else mis
        if s == score[i - 1, j - 1] + diag_score:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif s == score[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    return tuple(pairs[::-1])


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |(p @ R.T + t) - q|."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, qc - pc @ r.T


def rmsd_superposed(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD after optimal superposition of p onto q."""
    r, t = kabsch(p, q)
    return float(np.sqrt(((p @ r.T + t - q) ** 2).sum(1).mean()))


def lddt_ca(model: np.ndarray, ref: np.ndarray) -> float:
    """Global lDDT over CA atoms (see module docstring for the exact definition)."""
    dref = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    dmod = np.linalg.norm(model[:, None] - model[None], axis=-1)
    mask = (dref < LDDT_RADIUS) & ~np.eye(len(ref), dtype=bool)
    if not mask.any():
        return float("nan")
    diff = np.abs(dref - dmod)[mask]
    preserved = sum(int((diff < t).sum()) for t in LDDT_THRESHOLDS)
    return preserved / (len(LDDT_THRESHOLDS) * int(mask.sum()))


@dataclass
class Correspondence:
    """Matched CA coordinates (model, reference) and the reference residue keys."""

    model_ca: np.ndarray
    ref_ca: np.ndarray
    ref_keys: list[tuple[str, int]]


def correspond(
    model: Structure,
    ref: Structure,
    chain_map: dict[str, str],
    unit: dict[str, list[list[int]]] | None = None,
) -> Correspondence:
    """Match residues chain by chain (model chain -> ref chain), optionally inside a CASP unit.

    `unit` maps reference chain -> inclusive residue-number ranges (evaluation domain).
    """
    mc, rc, keys = [], [], []
    for mchain, rchain in chain_map.items():
        a, b = model.chains[mchain], ref.chains[rchain]
        for i, j in align_sequences(a.seq, b.seq):
            num = b.resnums[j]
            if (
                unit
                and rchain in unit
                and not any(lo <= num <= hi for lo, hi in unit[rchain])
            ):
                continue
            mc.append(a.ca[i])
            rc.append(b.ca[j])
            keys.append((rchain, num))
    if len(mc) < 3:
        raise ValueError("fewer than 3 matched residues")
    return Correspondence(np.array(mc), np.array(rc), keys)


def ligand_rmsd(model: Structure, ref: Structure, corr: Correspondence) -> float | None:
    """Pocket-aligned ligand RMSD over the ligands the model contains (None when it has none).

    Ground-truth PDB entries routinely carry ions, buffer molecules and ligands bound to other
    chains; only reference ligands whose residue name the model also predicts are scored, so a
    protein-only prediction gets no ligand metric instead of an error.
    """
    model_names = {k.split(":")[1] for k in model.ligands}
    ref = Structure(
        ref.chains,
        {k: v for k, v in ref.ligands.items() if k.split(":")[1] in model_names},
    )
    if not ref.ligands:
        return None
    ref_atoms = np.concatenate(
        [np.stack(list(v.values())) for v in ref.ligands.values()]
    )
    near = (
        np.linalg.norm(corr.ref_ca[:, None] - ref_atoms[None], axis=-1).min(1)
        < POCKET_RADIUS
    )
    if near.sum() < 3:
        raise ValueError("fewer than 3 pocket residues for ligand alignment")
    r, t = kabsch(corr.model_ca[near], corr.ref_ca[near])
    sq = []
    # Pair ligands by residue name, in order of appearance.
    model_by_name: dict[str, list[dict[str, np.ndarray]]] = {}
    for key, atoms in model.ligands.items():
        model_by_name.setdefault(key.split(":")[1], []).append(atoms)
    for key, ratoms in ref.ligands.items():
        name = key.split(":")[1]
        if not model_by_name.get(name):
            raise ValueError(f"model has no ligand {name}")
        matoms = model_by_name[name].pop(0)
        for atom, rxyz in ratoms.items():
            if atom in matoms:
                sq.append(float(((matoms[atom] @ r.T + t - rxyz) ** 2).sum()))
    if not sq:
        raise ValueError("no ligand atoms matched by name")
    return math.sqrt(sum(sq) / len(sq))


def _write_ca_pdb(path: Path, coords: np.ndarray) -> None:
    lines = [
        f"ATOM  {i + 1:5d}  CA  GLY A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        for i, (x, y, z) in enumerate(coords)
    ]
    path.write_text("\n".join([*lines, "END"]) + "\n")


def tm_score(
    corr: Correspondence,
    usalign: str = "USalign",
    run: Callable[[list[str]], str] | None = None,
) -> float | None:
    """US-align TM-score over the matched residues, normalised by the reference."""
    if run is None:
        if shutil.which(usalign) is None:
            return None

        def run(argv: list[str]) -> str:
            return subprocess.run(
                argv, capture_output=True, text=True, check=True
            ).stdout

    with tempfile.TemporaryDirectory() as td:
        mp, rp = Path(td, "model.pdb"), Path(td, "ref.pdb")
        _write_ca_pdb(mp, corr.model_ca)
        _write_ca_pdb(rp, corr.ref_ca)
        out = run([usalign, str(mp), str(rp), "-TMscore", "1", "-outfmt", "2"])
    return parse_usalign_outfmt2(out)


def parse_usalign_outfmt2(text: str) -> float:
    """TM2 (normalised by the second structure, the reference) from `-outfmt 2` output."""
    rows = [
        ln.split("\t")
        for ln in text.strip().splitlines()
        if ln and not ln.startswith("#")
    ]
    head = [ln for ln in text.splitlines() if ln.startswith("#")]
    cols = head[-1].lstrip("#").split("\t") if head else []
    idx = cols.index("TM2") if "TM2" in cols else 3
    return float(rows[-1][idx])


def score_prediction(
    pred: str | Path,
    truth: Structure,
    chain_map: dict[str, str],
    unit: dict[str, list[list[int]]] | None = None,
    tm: Callable[[Correspondence], float | None] = tm_score,
) -> dict[str, float | None]:
    """All metrics for one predicted model."""
    return score_model(load_structure(pred), truth, chain_map, unit, tm)


def score_model(
    model: Structure,
    truth: Structure,
    chain_map: dict[str, str],
    unit: dict[str, list[list[int]]] | None = None,
    tm: Callable[[Correspondence], float | None] | None = tm_score,
) -> dict[str, float | None]:
    """All metrics for one in-memory model (tm=None skips TM-score)."""
    corr = correspond(model, truth, chain_map, unit)
    return {
        "ca_rmsd": rmsd_superposed(corr.model_ca, corr.ref_ca),
        "lddt_ca": lddt_ca(corr.model_ca, corr.ref_ca),
        "ligand_rmsd": ligand_rmsd(model, truth, corr),
        "tm_score": tm(corr) if tm is not None else None,
        "n_residues": float(len(corr.ref_keys)),
    }


# ----------------------------------------------------------------------------- score CLI
SCORE_COLUMNS = (
    "target_id", "backend", "lnc", "diffusion_samples", "atom_graph_break", "variant", "seed",
    "ca_rmsd", "lddt_ca", "ligand_rmsd", "tm_score", "n_residues",
    "best_ca_rmsd", "best_lddt_ca", "best_ligand_rmsd", "best_tm_score", "n_samples",
    "prediction", "result_path",
)  # fmt: skip


def score_results(
    results_root: Path,
    targets: dict[str, Any],
    truth_dir: Path,
    tm: Callable[[Correspondence], float | None] = tm_score,
) -> list[dict[str, Any]]:
    """Score every measured result under results_root (top-ranked and best-of-N)."""
    by_id = {t["id"]: t for t in targets["targets"]}
    rows = []
    truths: dict[str, Structure] = {}
    for res in sorted(results_root.rglob("result.json")):
        doc = json.loads(res.read_text())
        t = by_id.get(doc["target_id"])
        if (
            doc["status"] != "measured"
            or not doc["predictions"]
            or t is None
            or not t.get("gated")
        ):
            continue
        if t["id"] not in truths:
            truths[t["id"]] = load_structure(truth_dir / t["truth"]["file"])
        cmap = {
            ch["chain_id"]: ch["truth_chain"]
            for ch in t["chains"]
            if ch["entity"] == "protein"
        }
        unit = t["truth"].get("unit")
        scored = [
            score_prediction(p, truths[t["id"]], cmap, unit, tm)
            for p in doc["predictions"]
        ]
        top = scored[0]

        def best(
            key: str, lower: bool, scored: list[dict[str, float | None]] = scored
        ) -> float | None:
            vals = [v for s in scored if (v := s[key]) is not None]
            if not vals:
                return None
            return min(vals) if lower else max(vals)

        rows.append(
            {
                "target_id": doc["target_id"],
                "backend": doc["backend"],
                "lnc": doc["lnc"],
                "diffusion_samples": doc["diffusion_samples"],
                "atom_graph_break": doc["atom_graph_break"],
                "variant": doc.get("variant"),
                "seed": doc["seed"],
                **{
                    k: top[k]
                    for k in (
                        "ca_rmsd",
                        "lddt_ca",
                        "ligand_rmsd",
                        "tm_score",
                        "n_residues",
                    )
                },
                "best_ca_rmsd": best("ca_rmsd", True),
                "best_lddt_ca": best("lddt_ca", False),
                "best_ligand_rmsd": best("ligand_rmsd", True),
                "best_tm_score": best("tm_score", False),
                "n_samples": len(scored),
                "prediction": doc["predictions"][0],
                "result_path": str(res),
            }
        )
    return rows


def write_csv(
    rows: Iterable[dict[str, Any]], path: Path, columns: Sequence[str]
) -> None:
    """Write rows with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r.get(k) for k in columns})


# ----------------------------------------------------------------------------- gate
def parse_filter(spec: str) -> dict[str, str]:
    """'backend=neuron,ds=1,lnc=2' -> dict (ds is an alias for diffusion_samples)."""
    out = {}
    for part in filter(None, spec.split(",")):
        k, _, v = part.partition("=")
        out["diffusion_samples" if k == "ds" else k] = v
    return out


def _select(
    rows: list[dict[str, Any]], flt: dict[str, str]
) -> dict[tuple[str, int], dict[str, Any]]:
    sel: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        if all(str(r.get(k, "")) == v for k, v in flt.items()):
            key = (r["target_id"], int(r["seed"]))
            if key in sel:
                raise ValueError(f"filter {flt} matches more than one row for {key}")
            sel[key] = r
    return sel


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    f = float(v)
    return None if math.isnan(f) else f


def gate(
    rows: list[dict[str, Any]],
    acc: dict[str, str],
    ref: dict[str, str],
    gated_ids: Sequence[str],
    timing_only: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply the section-2 gate. Missing data for a gated target counts as a failure.

    Primary rows have an empty variant; unless a filter names one, `variant=""` is implied on
    both sides. A secondary (variant) gate covers only the targets the reference has rows for
    and gives no phase verdict.
    """
    acc = {"variant": "", **acc}
    ref = {"variant": "", **ref}
    a, r = _select(rows, acc), _select(rows, ref)
    secondary = acc["variant"] != ""
    if secondary:
        gated_ids = [t for t in gated_ids if any(k[0] == t for k in r)]
    targets = []
    for tid in gated_ids:
        entry: dict[str, Any] = {
            "target_id": tid,
            "metrics": {},
            "pass": True,
            "cap_exceeded": False,
        }
        for metric in GATE_METRICS:
            ref_vals = [
                _num(r[(tid, s)][metric]) if (tid, s) in r else None for s in (0, 1)
            ]
            acc_vals = [
                _num(a[(tid, s)][metric]) if (tid, s) in a else None for s in (0, 1)
            ]
            if metric == "ligand_rmsd" and all(v is None for v in ref_vals):
                continue  # no ligand in this target
            m: dict[str, Any] = {"ref": ref_vals, "acc": acc_vals}
            if None in ref_vals or None in acc_vals:
                m.update(pass_=False, reason="missing seed result")
                entry["pass"] = False
            else:
                assert all(v is not None for v in (*ref_vals, *acc_vals))
                dev = max(abs(acc_vals[s] - ref_vals[s]) for s in (0, 1))  # type: ignore[operator]
                spread = abs(ref_vals[0] - ref_vals[1])  # type: ignore[operator]
                tol = max(spread, FLOORS[metric])
                ok = dev <= tol
                cap = dev > CAP_FACTOR * tol
                m.update(
                    deviation=dev,
                    spread=spread,
                    tolerance=tol,
                    pass_=ok,
                    cap_exceeded=cap,
                )
                entry["pass"] &= ok
                entry["cap_exceeded"] |= cap
            entry["metrics"][metric] = m
        targets.append(entry)
    n_pass = sum(t["pass"] for t in targets)
    any_cap = any(t["cap_exceeded"] for t in targets)
    required = max(len(targets) - PHASE_ALLOWED_FAILURES, 1)
    return {
        "acc": acc,
        "ref": ref,
        "floors": FLOORS,
        "targets": targets,
        "timing_only": list(timing_only),
        "n_targets": len(targets),
        "n_required": None if secondary else required,
        "n_pass": n_pass,
        "cap_exceeded": any_cap,
        "secondary": secondary,
        "phase_pass": None
        if secondary
        else (bool(targets) and n_pass >= required and not any_cap),
    }


# ----------------------------------------------------------------------------- calibrated gate
# Spread-calibrated gate (#103). It sits beside the section-2 gate above and does not replace it
# until the charter is changed. The tolerance comes from the reference's own seed-to-seed spread
# (>= 2 reference seeds, ideally the 5 CPU oracle seeds), not from two seeds' difference.
#
# Per target t and metric m, with reference seed values R_1..R_k and accuracy-run values A_s:
#   c_j      = median of R without R_j            (leave-one-out centre)
#   d_ref(t) = mean_j |R_j - c_j|                  (how far a fresh reference seed lands)
#   d_acc(t) = mean_{s,j} |A_s - c_j|              (same quantity for the accuracy run)
#   e(t)     = d_acc - d_ref                       (excess deviation; ~0 if A behaves like R)
#   b(t)     = sign_m * (mean_s A_s - median R)    (bias, positive = accuracy run is worse)
# A_s and R_j are both independent of c_j, so under "A is distributed like R" E[e(t)] = 0.
# Phase pass needs, for every metric with data:
#   * excess:  the one-sided (1 - alpha) bootstrap upper bound of mean_t e(t) <= delta_m;
#   * bias:    the two-sided (1 - 2 alpha) bootstrap interval of mean_t b(t) inside +-delta_m;
#   * cap:     no target with d_acc(t) > max(CAL_CAP_FLOOR x delta_m, CAL_CAP_FACTOR x d_ref(t));
# and complete data for every gated target. delta_m = FLOORS. Bootstrap resamples targets.
#
# calibrated/1 applies all three tests to every metric. On the 5-seed CPU oracle it rejected the
# reference against itself in 7 of 10 A/A splits, all through CA-RMSD vs truth: at ~15 Å from
# truth that metric is dominated by domain placement and is multimodal across seeds (#103).
# calibrated/2 keeps lDDT-CA exactly as above and makes CA-RMSD and ligand RMSD *coarse* checks,
# which only look for gross defects (mirror images, exploded structures):
#   * population: the one-sided (1 - alpha) bootstrap upper bound of mean_t e(t) <= COARSE_POP;
#   * per target: e(t) <= COARSE_TARGET;  no bias test.
# Thresholds were set from seeds 0-4 (A/A per-target CA excess max 4.3 Å, mean max 0.23 Å;
# mirror mean ~5 Å; explosions > 70 Å) and are validated on held-out seeds 5-9.
#
# calibrated/2 failed held-out A/A 10/20, every failure the lDDT per-target cap, and failed
# g7e ds=1, every component in the "g7e is better" direction (#103). calibrated/3 asks the
# question the project asks, non-inferiority ("not worse than the reference"), for lDDT-CA:
#   * bias: the one-sided (1 - alpha) bootstrap upper bound of mean_t b(t) <= delta (b > 0 = worse);
#   * cap:  only targets where the run is worse (b(t) > 0) and d_acc(t) > the calibrated/1 cap;
#   * no two-sided excess test (better than the reference is reported, not failed).
# CA-RMSD and ligand RMSD stay the calibrated/2 coarse checks. Use every reference seed available
# (the CPU oracle has seeds 0-9); validated on held-out seeds 10-14.
CAL_ALPHA = 0.05
CAL_BOOTSTRAP = 10_000
CAL_CAP_FACTOR = 3.0
CAL_CAP_FLOOR = 2.0
WORSE_SIGN = {"ca_rmsd": 1.0, "lddt_ca": -1.0, "ligand_rmsd": 1.0}
COARSE_POP = 2.0
COARSE_TARGET = 10.0
CAL_METHODS: dict[str, dict[str, str]] = {
    "calibrated/1": dict.fromkeys(GATE_METRICS, "fine"),
    "calibrated/2": {"ca_rmsd": "coarse", "lddt_ca": "fine", "ligand_rmsd": "coarse"},
    "calibrated/3": {
        "ca_rmsd": "coarse",
        "lddt_ca": "noninferior",
        "ligand_rmsd": "coarse",
    },
}
CAL_DEFAULT = "calibrated/3"


def _seed_values(
    sel: dict[tuple[str, int], dict[str, Any]],
    tid: str,
    metric: str,
    seeds: Sequence[int] | None,
) -> list[float]:
    keys = sorted(k for k in sel if k[0] == tid and (seeds is None or k[1] in seeds))
    return [v for k in keys if (v := _num(sel[k][metric])) is not None]


def target_deviation(
    ref_vals: Sequence[float], acc_vals: Sequence[float]
) -> dict[str, float]:
    """d_ref, d_acc, excess and raw bias for one target and metric (needs >= 2 refs, >= 1 acc)."""
    ref = np.asarray(ref_vals, dtype=float)
    acc = np.asarray(acc_vals, dtype=float)
    k = len(ref)
    if k < 2 or len(acc) < 1:
        raise ValueError("need >= 2 reference values and >= 1 accuracy value")
    centres = np.array([np.median(np.delete(ref, j)) for j in range(k)])
    d_ref = float(np.mean(np.abs(ref - centres)))
    d_acc = float(np.mean(np.abs(acc[:, None] - centres[None, :])))
    return {
        "d_ref": d_ref,
        "d_acc": d_acc,
        "excess": d_acc - d_ref,
        "shift": float(acc.mean() - np.median(ref)),
    }


def _bootstrap_means(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return x[idx].mean(axis=1)


def calibrated_gate(
    rows: list[dict[str, Any]],
    acc: dict[str, str],
    ref: dict[str, str],
    gated_ids: Sequence[str],
    *,
    acc_seeds: Sequence[int] | None = None,
    ref_seeds: Sequence[int] | None = None,
    timing_only: Sequence[str] = (),
    deltas: dict[str, float] | None = None,
    n_boot: int = CAL_BOOTSTRAP,
    alpha: float = CAL_ALPHA,
    rng_seed: int = 0,
    method: str = CAL_DEFAULT,
) -> dict[str, Any]:
    """Spread-calibrated gate (see the block comment above). Seeds default to all present."""
    if method not in CAL_METHODS:
        raise ValueError(f"unknown gate method {method!r}")
    kinds = CAL_METHODS[method]
    deltas = deltas or FLOORS
    acc = {"variant": "", **acc}
    ref = {"variant": "", **ref}
    a, r = _select(rows, acc), _select(rows, ref)
    rng = np.random.default_rng(rng_seed)
    targets: list[dict[str, Any]] = []
    per_metric: dict[str, dict[str, list[float]]] = {
        m: {"excess": [], "bias": []} for m in GATE_METRICS
    }
    for tid in gated_ids:
        entry: dict[str, Any] = {
            "target_id": tid,
            "metrics": {},
            "complete": True,
            "cap_exceeded": False,
        }
        for metric in GATE_METRICS:
            rv = _seed_values(r, tid, metric, ref_seeds)
            av = _seed_values(a, tid, metric, acc_seeds)
            if metric == "ligand_rmsd" and not rv:
                continue  # no ligand scored for this target
            m: dict[str, Any] = {"ref": rv, "acc": av}
            if len(rv) < 2 or not av:
                m["reason"] = f"missing data ({len(rv)} ref, {len(av)} acc values)"
                entry["complete"] = False
            else:
                dev = target_deviation(rv, av)
                bias = WORSE_SIGN[metric] * dev["shift"]
                if kinds[metric] == "coarse":  # cap on the excess itself
                    cap, over = COARSE_TARGET, dev["excess"] > COARSE_TARGET
                else:
                    cap = max(
                        CAL_CAP_FLOOR * deltas[metric], CAL_CAP_FACTOR * dev["d_ref"]
                    )
                    over = dev["d_acc"] > cap
                    if kinds[metric] == "noninferior":
                        over = over and bias > 0
                m.update(dev, bias=bias, cap=cap, cap_exceeded=over, kind=kinds[metric])
                entry["cap_exceeded"] |= m["cap_exceeded"]
                per_metric[metric]["excess"].append(dev["excess"])
                per_metric[metric]["bias"].append(bias)
            entry["metrics"][metric] = m
        targets.append(entry)
    metrics: dict[str, Any] = {}
    for metric, d in per_metric.items():
        ex, bi = np.array(d["excess"]), np.array(d["bias"])
        if len(ex) == 0:
            continue
        bex, bbi = _bootstrap_means(ex, n_boot, rng), _bootstrap_means(bi, n_boot, rng)
        ex_ub = float(np.quantile(bex, 1 - alpha))
        bi_lo, bi_hi = (float(q) for q in np.quantile(bbi, [alpha, 1 - alpha]))
        coarse = kinds[metric] == "coarse"
        delta = COARSE_POP if coarse else deltas[metric]
        metrics[metric] = {
            "kind": kinds[metric],
            "n_targets": len(ex),
            "delta": delta,
            "mean_excess": float(ex.mean()),
            "excess_upper": ex_ub,
            "excess_pass": True if kinds[metric] == "noninferior" else ex_ub <= delta,
            "mean_bias": float(bi.mean()),
            "bias_interval": [bi_lo, bi_hi],
            "bias_pass": (
                True
                if coarse
                else bi_hi <= delta
                if kinds[metric] == "noninferior"
                else (-delta <= bi_lo and bi_hi <= delta)
            ),
        }
    incomplete = [t["target_id"] for t in targets if not t["complete"]]
    capped = [t["target_id"] for t in targets if t["cap_exceeded"]]
    reasons = []
    if not targets:
        reasons.append("no gated targets")
    if incomplete:
        reasons.append(f"incomplete: {', '.join(incomplete)}")
    if capped:
        reasons.append(f"per-target cap exceeded: {', '.join(capped)}")
    for metric, s in metrics.items():
        if not s["excess_pass"]:
            reasons.append(
                f"{metric} excess upper bound {s['excess_upper']:.3f} > {s['delta']}"
            )
        if not s["bias_pass"]:
            lo, hi = s["bias_interval"]
            reasons.append(
                f"{metric} bias interval [{lo:.3f}, {hi:.3f}] not within +-{s['delta']}"
            )
    return {
        "method": method,
        "acc": acc,
        "ref": ref,
        "acc_seeds": list(acc_seeds) if acc_seeds is not None else None,
        "ref_seeds": list(ref_seeds) if ref_seeds is not None else None,
        "params": {
            "alpha": alpha,
            "n_boot": n_boot,
            "cap_factor": CAL_CAP_FACTOR,
            "cap_floor": CAL_CAP_FLOOR,
            "deltas": deltas,
            "kinds": kinds,
            "coarse_pop": COARSE_POP,
            "coarse_target": COARSE_TARGET,
            "rng_seed": rng_seed,
        },
        "targets": targets,
        "timing_only": list(timing_only),
        "metrics": metrics,
        "incomplete": incomplete,
        "cap_exceeded": capped,
        "reasons": reasons,
        "phase_pass": not reasons,
    }


def calibrated_gate_markdown(res: dict[str, Any]) -> str:
    """Summary and per-target table for a calibrated gate result."""

    def f(v: Any, nd: int = 3) -> str:
        return "—" if v is None else f"{v:.{nd}f}"

    verdict = (
        "**PASS**" if res["phase_pass"] else "**FAIL** — " + "; ".join(res["reasons"])
    )
    lines = [
        f"Calibrated gate ({res['method']}): acc `{res['acc']}` vs ref `{res['ref']}` — {verdict}",
        "",
        "| metric | targets | mean excess | excess upper bound | delta | mean bias | bias interval | pass |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for metric, s in res["metrics"].items():
        lo, hi = s["bias_interval"]
        lines.append(
            f"| {metric} | {s['n_targets']} | {f(s['mean_excess'])} | {f(s['excess_upper'])} | "
            f"{s['delta']} | {f(s['mean_bias'])} | [{f(lo)}, {f(hi)}] | "
            f"{'yes' if s['excess_pass'] and s['bias_pass'] else 'no'} |"
        )
    lines += [
        "",
        "| target | metric | ref values | acc values | d_ref | d_acc | excess | cap | over cap |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for t in res["targets"]:
        for metric, m in t["metrics"].items():
            rv = ", ".join(f(v) for v in m["ref"])
            av = ", ".join(f(v) for v in m["acc"])
            lines.append(
                f"| {t['target_id']} | {metric} | {rv} | {av} | {f(m.get('d_ref'))} | "
                f"{f(m.get('d_acc'))} | {f(m.get('excess'))} | {f(m.get('cap'))} | "
                f"{'YES' if m.get('cap_exceeded') else ''}{m.get('reason', '')} |"
            )
    for tid in res.get("timing_only", []):
        lines.append(f"| {tid} | — | | | | | | | no ground truth, timing only |")
    return "\n".join(lines) + "\n"


def check_prereq(acc: dict[str, str], ref: dict[str, str], prereq: Path | None) -> None:
    """ds=5 gating is only valid against a g7e ds=5 ref whose ds=1 run passed against CPU."""
    if acc.get("diffusion_samples") != "5":
        return
    if ref.get("backend") != "gpu" or ref.get("diffusion_samples") != "5":
        raise SystemExit("ds=5 must be gated against the g7e (backend=gpu) ds=5 run")
    if prereq is None or not prereq.exists():
        raise SystemExit(
            "refusing: ds=5 gate needs --prereq, the g7e ds=1 vs CPU gate result"
        )
    pre = json.loads(prereq.read_text())
    ok = (
        pre.get("acc", {}).get("backend") == "gpu"
        and pre.get("acc", {}).get("diffusion_samples") == "1"
        and pre.get("ref", {}).get("backend") == "cpu"
        and pre.get("phase_pass") is True
    )
    if not ok:
        raise SystemExit(
            "refusing: the g7e ds=1 run has not passed the gate against CPU fp32"
        )


def gate_markdown(res: dict[str, Any]) -> str:
    """Pass/fail table as Markdown."""

    def f(v: Any, nd: int = 3) -> str:
        return "—" if v is None else f"{v:.{nd}f}"

    if res.get("secondary"):
        verdict = "secondary row: per-target results only, no phase verdict"
    else:
        verdict = (
            f"**{'PASS' if res['phase_pass'] else 'FAIL'}** "
            f"({res['n_pass']}/{res['n_targets']} targets pass, {res['n_required']} required; "
            f"2× cap {'exceeded' if res['cap_exceeded'] else 'not exceeded'})"
        )
    lines = [
        f"Gate: acc `{res['acc']}` vs ref `{res['ref']}` — {verdict}",
        "",
        "| target | metric | ref s0 | ref s1 | acc s0 | acc s1 | deviation | tolerance | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in res["targets"]:
        for metric, m in t["metrics"].items():
            lines.append(
                f"| {t['target_id']} | {metric} | {f(m['ref'][0])} | {f(m['ref'][1])} | "
                f"{f(m['acc'][0])} | {f(m['acc'][1])} | {f(m.get('deviation'))} | "
                f"{f(m.get('tolerance'))} | {'yes' if m.get('pass_') else 'no'}"
                f"{' (cap)' if m.get('cap_exceeded') else ''}{' — ' + m['reason'] if m.get('reason') else ''} |"
            )
    for tid in res.get("timing_only", []):
        lines.append(f"| {tid} | — | | | | | | | no ground truth, timing only |")
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="harness.score")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("score")
    ps.add_argument("--results", type=Path, required=True)
    ps.add_argument("--targets", type=Path, required=True)
    ps.add_argument("--truth-dir", type=Path, required=True)
    ps.add_argument("--out", type=Path, required=True)
    pg = sub.add_parser("gate")
    pg.add_argument("--scores", type=Path, required=True)
    pg.add_argument("--targets", type=Path, required=True)
    pg.add_argument("--acc", required=True)
    pg.add_argument("--ref", required=True)
    pg.add_argument("--prereq", type=Path)
    pg.add_argument(
        "--out", type=Path, required=True, help="output stem (.json and .md)"
    )
    pc = sub.add_parser("gate-cal", help="spread-calibrated gate (#103)")
    pc.add_argument("--scores", type=Path, required=True)
    pc.add_argument("--targets", type=Path, required=True)
    pc.add_argument("--acc", required=True)
    pc.add_argument("--ref", required=True)
    pc.add_argument("--acc-seeds", type=int, nargs="+")
    pc.add_argument("--ref-seeds", type=int, nargs="+")
    pc.add_argument("--method", choices=sorted(CAL_METHODS), default=CAL_DEFAULT)
    pc.add_argument(
        "--prereq", type=Path, help="ds=5: the g7e ds=1 vs CPU gate-cal result"
    )
    pc.add_argument(
        "--out", type=Path, required=True, help="output stem (.json and .md)"
    )
    ns = ap.parse_args(argv)
    targets = json.loads(ns.targets.read_text())
    if ns.cmd == "score":
        rows = score_results(ns.results, targets, ns.truth_dir)
        write_csv(rows, ns.out, SCORE_COLUMNS)
        print(f"{len(rows)} rows -> {ns.out}")
        return 0
    acc, ref = parse_filter(ns.acc), parse_filter(ns.ref)
    gated = [t["id"] for t in targets["targets"] if t.get("gated")]
    timing_only = [t["id"] for t in targets["targets"] if t.get("timing_only")]
    if ns.cmd == "gate-cal":
        check_prereq(acc, ref, ns.prereq)
        cres = calibrated_gate(
            _read_csv(ns.scores), acc, ref, gated,
            acc_seeds=ns.acc_seeds, ref_seeds=ns.ref_seeds, timing_only=timing_only,
            method=ns.method,
        )  # fmt: skip
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{ns.out}.json").write_text(json.dumps(cres, indent=2) + "\n")
        Path(f"{ns.out}.md").write_text(calibrated_gate_markdown(cres))
        print(("PASS" if cres["phase_pass"] else "FAIL ") + "; ".join(cres["reasons"]))
        return 0 if cres["phase_pass"] else 1
    check_prereq(acc, ref, ns.prereq)
    res = gate(_read_csv(ns.scores), acc, ref, gated, timing_only)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{ns.out}.json").write_text(json.dumps(res, indent=2) + "\n")
    Path(f"{ns.out}.md").write_text(gate_markdown(res))
    if res["secondary"]:
        print(f"SECONDARY {res['n_pass']}/{res['n_targets']} targets pass")
        return 0
    print(
        f"{'PASS' if res['phase_pass'] else 'FAIL'} {res['n_pass']}/{res['n_targets']} "
        f"({res['n_required']} required)"
    )
    return 0 if res["phase_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
