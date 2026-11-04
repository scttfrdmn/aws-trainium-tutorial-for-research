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
Phase pass: >= 18 of the 20 CASP14 targets pass on every gate metric, and no deviation exceeds
2 x max(spread, floor). ref is CPU fp32 at ds=1; at ds=5 the ref is the g7e ds=5 run, which is
only allowed once the g7e ds=1 run has passed against CPU (`--prereq`). The two ceiling
complexes are never gated.
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
from pathlib import Path
from typing import Any

import numpy as np

GATE_METRICS = ("ca_rmsd", "lddt_ca", "ligand_rmsd")
FLOORS = {"ca_rmsd": 0.5, "lddt_ca": 0.01, "ligand_rmsd": 0.5}
PHASE_MIN_PASS = 18
PHASE_TARGETS = 20
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
    return pairs[::-1]


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
    """Pocket-aligned ligand RMSD (None when the reference has no ligand)."""
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
    model = load_structure(pred)
    corr = correspond(model, truth, chain_map, unit)
    return {
        "ca_rmsd": rmsd_superposed(corr.model_ca, corr.ref_ca),
        "lddt_ca": lddt_ca(corr.model_ca, corr.ref_ca),
        "ligand_rmsd": ligand_rmsd(model, truth, corr),
        "tm_score": tm(corr),
        "n_residues": float(len(corr.ref_keys)),
    }


# ----------------------------------------------------------------------------- score CLI
SCORE_COLUMNS = (
    "target_id", "backend", "lnc", "diffusion_samples", "atom_graph_break", "seed",
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
) -> dict[str, Any]:
    """Apply the section-2 gate. Missing data for a gated target counts as a failure."""
    a, r = _select(rows, acc), _select(rows, ref)
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
    return {
        "acc": acc,
        "ref": ref,
        "floors": FLOORS,
        "targets": targets,
        "n_targets": len(targets),
        "n_pass": n_pass,
        "cap_exceeded": any_cap,
        "phase_pass": len(targets) == PHASE_TARGETS
        and n_pass >= PHASE_MIN_PASS
        and not any_cap,
    }


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

    lines = [
        f"Gate: acc `{res['acc']}` vs ref `{res['ref']}` — "
        f"**{'PASS' if res['phase_pass'] else 'FAIL'}** "
        f"({res['n_pass']}/{res['n_targets']} targets pass; 2× cap "
        f"{'exceeded' if res['cap_exceeded'] else 'not exceeded'})",
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
    ns = ap.parse_args(argv)
    targets = json.loads(ns.targets.read_text())
    if ns.cmd == "score":
        rows = score_results(ns.results, targets, ns.truth_dir)
        write_csv(rows, ns.out, SCORE_COLUMNS)
        print(f"{len(rows)} rows -> {ns.out}")
        return 0
    acc, ref = parse_filter(ns.acc), parse_filter(ns.ref)
    check_prereq(acc, ref, ns.prereq)
    gated = [t["id"] for t in targets["targets"] if t.get("gated")]
    res = gate(_read_csv(ns.scores), acc, ref, gated)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{ns.out}.json").write_text(json.dumps(res, indent=2) + "\n")
    Path(f"{ns.out}.md").write_text(gate_markdown(res))
    print(
        f"{'PASS' if res['phase_pass'] else 'FAIL'} {res['n_pass']}/{res['n_targets']}"
    )
    return 0 if res["phase_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
