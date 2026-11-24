# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Self-test of the accuracy gates on the CPU fp32 oracle (#103).

    python -m harness.gate_selftest --oracle-root results/oracle --targets targets/targets.json \
        --truth-dir truth/ --out results/gate_selftest

Two questions, answered from the oracle's own seeds, before either gate judges a platform:

* A/A (false-reject rate). Every split of the oracle seeds into 2 "test" seeds and the rest as
  reference is gated as if the test seeds were a new platform. A test platform that *is* the
  reference should pass. The section-2 gate is run on the pre-registered split (test 2,3 against
  reference 0,1) and on every disjoint (test pair, reference pair); the calibrated gate on every
  test pair against the remaining seeds.
* Planted defects (false-accept rate). The test seeds' predicted coordinates are corrupted, then
  rescored against ground truth and gated against the untouched reference:
    - noise:<sigma>    Gaussian noise on every scored atom of every target (CA and ligand);
    - mirror           x -> -x (chirality flip) on every target;
    - register         one segment per chain (a fifth of it, from a third of the way in) takes
                       the coordinates of the residue 3 positions later;
    - local:<n>        noise of 2 Å on n randomly chosen targets only.
  A gate should reject all of these except, possibly, noise far below the seed-to-seed spread.

Randomness is seeded (`--rng-seed`); TM-score is not computed (it is not gated).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from harness import pins, score
from harness.score import Chain, Structure

Perturb = Callable[[Structure, np.random.Generator], Structure]
LOCAL_SIGMA = 2.0
REGISTER_SHIFT = 3


def copy_structure(st: Structure) -> Structure:
    """Deep copy of the coordinates (sequences and numbering are shared, read-only)."""
    return Structure(
        {k: Chain(c.chain_id, c.seq, c.resnums, c.ca.copy()) for k, c in st.chains.items()},
        {k: {a: x.copy() for a, x in v.items()} for k, v in st.ligands.items()},
    )


def add_noise(st: Structure, sigma: float, rng: np.random.Generator) -> Structure:
    """Gaussian noise (per coordinate, standard deviation sigma) on every CA and ligand atom."""
    out = copy_structure(st)
    for c in out.chains.values():
        c.ca += rng.normal(0.0, sigma, c.ca.shape)
    for atoms in out.ligands.values():
        for a in atoms:
            atoms[a] = atoms[a] + rng.normal(0.0, sigma, 3)
    return out


def mirror(st: Structure) -> Structure:
    """Mirror image through the yz plane."""
    out = copy_structure(st)
    for c in out.chains.values():
        c.ca[:, 0] *= -1.0
    for atoms in out.ligands.values():
        for a in atoms:
            atoms[a] = atoms[a] * np.array([-1.0, 1.0, 1.0])
    return out


def register_shift(st: Structure, shift: int = REGISTER_SHIFT) -> Structure:
    """One segment per chain takes the coordinates of the residue `shift` positions later."""
    out = copy_structure(st)
    for c in out.chains.values():
        n = len(c.ca)
        if n < 3 * shift + 10:
            continue
        start = n // 3
        stop = min(start + max(10, n // 5), n - shift)
        c.ca[start:stop] = st.chains[c.chain_id].ca[start + shift : stop + shift]
    return out


def defects() -> dict[str, tuple[Perturb | None, str]]:
    """name -> (whole-structure perturbation, or None for the local defect; kind)."""
    out: dict[str, tuple[Perturb | None, str]] = {}
    for sigma in (0.25, 0.5, 1.0, 2.0):
        out[f"noise:{sigma}"] = (
            lambda st, rng, s=sigma: add_noise(st, s, rng),
            "random",
        )
    out["mirror"] = (lambda st, rng: mirror(st), "fixed")
    out["register"] = (lambda st, rng: register_shift(st), "fixed")
    out["local:1"] = (None, "local")
    out["local:2"] = (None, "local")
    return out


# ----------------------------------------------------------------------------- loading
def _resolve(pred: str, res_path: Path, root: Path) -> Path:
    p = Path(pred)
    if p.exists():
        return p
    marker = "results/oracle/"
    if marker in pred:
        q = root / pred.split(marker, 1)[1]
        if q.exists():
            return q
    hits = sorted(res_path.parent.rglob(p.name))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"prediction {pred} (from {res_path}) not found under {root}")


def load_oracle(
    root: Path, targets: dict[str, Any], seeds: Sequence[int]
) -> dict[tuple[str, int], Structure]:
    """Top-ranked CPU oracle model per (gated target, seed), primary rows only."""
    gated = {t["id"] for t in targets["targets"] if t.get("gated")}
    out: dict[tuple[str, int], Structure] = {}
    for res in sorted(root.rglob("result.json")):
        doc = json.loads(res.read_text())
        key = (doc.get("target_id"), int(doc.get("seed", -1)))
        if (
            doc.get("backend") != "cpu"
            or doc.get("status") != "measured"
            or doc.get("variant")
            or key[0] not in gated
            or key[1] not in seeds
            or not doc.get("predictions")
        ):
            continue
        if key in out:
            raise ValueError(f"two oracle results for {key}")
        out[key] = score.load_structure(_resolve(doc["predictions"][0], res, root))
    return out


class Scorer:
    """Scores in-memory models against ground truth for the gated targets."""

    def __init__(self, targets: dict[str, Any], truth_dir: Path) -> None:
        self.by_id = {t["id"]: t for t in targets["targets"] if t.get("gated")}
        self.truth_dir = truth_dir
        self.truths: dict[str, Structure] = {}

    def __call__(self, tid: str, model: Structure) -> dict[str, float | None]:
        t = self.by_id[tid]
        if tid not in self.truths:
            self.truths[tid] = score.load_structure(self.truth_dir / t["truth"]["file"])
        cmap = {
            ch["chain_id"]: ch["truth_chain"] for ch in t["chains"] if ch["entity"] == "protein"
        }
        return score.score_model(model, self.truths[tid], cmap, t["truth"].get("unit"), tm=None)


def _row(tid: str, seed: int, backend: str, m: dict[str, float | None]) -> dict[str, Any]:
    return {
        "target_id": tid,
        "backend": backend,
        "diffusion_samples": "1",
        "variant": "",
        "seed": seed,
        **{k: m[k] for k in score.GATE_METRICS},
    }


# ----------------------------------------------------------------------------- self-test
def _gate_summary(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass": res["phase_pass"],
        "reasons": res["reasons"],
        "metrics": res["metrics"],
        "cap_exceeded": res["cap_exceeded"],
    }


def run_selftest(
    structs: dict[tuple[str, int], Structure],
    scorer: Callable[[str, Structure], dict[str, float | None]],
    gated: Sequence[str],
    seeds: Sequence[int],
    *,
    replicates: int = 3,
    local_replicates: int = 5,
    rng_seed: int = 0,
    n_boot: int = score.CAL_BOOTSTRAP,
    only_defects: Sequence[str] | None = None,
) -> dict[str, Any]:
    """A/A and planted-defect results for both gates."""
    rng = np.random.default_rng(rng_seed)
    base = {k: scorer(k[0], st) for k, st in structs.items()}
    cpu_rows = [_row(t, s, "cpu", m) for (t, s), m in base.items()]

    def gate_cal(test_rows: list[dict[str, Any]], ref_seeds: Sequence[int]) -> dict[str, Any]:
        return score.calibrated_gate(
            cpu_rows + test_rows, {"backend": "test"}, {"backend": "cpu"}, gated,
            ref_seeds=ref_seeds, n_boot=n_boot,
        )  # fmt: skip

    def section2(test: Sequence[int], ref: Sequence[int]) -> dict[str, Any]:
        rows = [
            {**_row(t, ref.index(s), "ref", m)} for (t, s), m in base.items() if s in ref
        ] + [{**_row(t, test.index(s), "test", m)} for (t, s), m in base.items() if s in test]
        res = score.gate(rows, {"backend": "test"}, {"backend": "ref"}, gated)
        return {"test": list(test), "ref": list(ref), "n_pass": res["n_pass"],
                "n_targets": res["n_targets"], "cap_exceeded": res["cap_exceeded"],
                "pass": res["phase_pass"]}  # fmt: skip

    splits = [tuple(p) for p in itertools.combinations(seeds, 2)]
    out: dict[str, Any] = {"seeds": list(seeds), "splits": [list(s) for s in splits]}

    # Section-2 gate, A/A. Pre-registered (E1): test 2,3 vs ref 0,1.
    s2 = []
    for test in splits:
        rest = [s for s in seeds if s not in test]
        s2 += [section2(test, list(ref)) for ref in itertools.combinations(rest, 2)]
    out["section2_aa"] = s2
    pre = [r for r in s2 if r["test"] == [2, 3] and r["ref"] == [0, 1]]
    out["section2_aa_preregistered"] = pre[0] if pre else None

    # Calibrated gate, A/A (E2).
    aa = []
    for test in splits:
        ref = [s for s in seeds if s not in test]
        rows = [_row(t, s, "test", m) for (t, s), m in base.items() if s in test]
        aa.append({"test": list(test), "ref": ref, **_gate_summary(gate_cal(rows, ref))})
    out["calibrated_aa"] = aa

    # Planted defects (E3).
    planted: dict[str, Any] = {}
    for name, (fn, kind) in defects().items():
        if only_defects and name not in only_defects:
            continue
        trials = []
        reps = 1 if kind == "fixed" else (local_replicates if kind == "local" else replicates)
        for test in splits:
            ref = [s for s in seeds if s not in test]
            for _ in range(reps):
                hit: list[str] = []
                if kind == "local":
                    n = int(name.split(":")[1])
                    hit = sorted(rng.choice(list(gated), size=n, replace=False).tolist())
                rows = []
                for (t, s), st in structs.items():
                    if s not in test:
                        continue
                    if kind == "local":
                        m = scorer(t, add_noise(st, LOCAL_SIGMA, rng)) if t in hit else base[(t, s)]
                    else:
                        assert fn is not None
                        m = scorer(t, fn(st, rng))
                    rows.append(_row(t, s, "test", m))
                g = gate_cal(rows, ref)
                trials.append({"test": list(test), "targets": hit, **_gate_summary(g)})
        rejected = sum(not t["pass"] for t in trials)
        why: Counter[str] = Counter()
        for t in trials:
            for r in t["reasons"]:
                why[r.split(" ")[0].rstrip(":")] += 1
        planted[name] = {
            "trials": len(trials),
            "rejected": rejected,
            "rejection_rate": rejected / len(trials),
            "reasons": dict(why),
            "detail": trials,
        }
    out["planted"] = planted
    return out


def selftest_markdown(res: dict[str, Any]) -> str:
    """Summary tables for the issue comment."""
    lines = ["**Section-2 gate, A/A**", ""]
    pre = res.get("section2_aa_preregistered")
    if pre:
        lines.append(
            f"- pre-registered split (test {pre['test']} vs ref {pre['ref']}): "
            f"{'PASS' if pre['pass'] else 'FAIL'}, {pre['n_pass']}/{pre['n_targets']} targets pass"
            f"{', 2× cap exceeded' if pre['cap_exceeded'] else ''}"
        )
    s2 = res["section2_aa"]
    n_pass = [r["n_pass"] for r in s2]
    lines.append(
        f"- all {len(s2)} disjoint (test pair, ref pair) combinations: "
        f"{sum(r['pass'] for r in s2)}/{len(s2)} pass; targets passing min {min(n_pass)}, "
        f"median {float(np.median(n_pass)):g}, max {max(n_pass)} of {s2[0]['n_targets']}"
    )
    lines += ["", "**Calibrated gate, A/A**", "",
              "| test seeds | ref seeds | verdict | reasons |", "|---|---|---|---|"]  # fmt: skip
    for r in res["calibrated_aa"]:
        lines.append(
            f"| {r['test']} | {r['ref']} | {'PASS' if r['pass'] else 'FAIL'} | "
            f"{'; '.join(r['reasons'])} |"
        )
    aa = res["calibrated_aa"]
    lines.append(f"\nA/A pass rate: {sum(r['pass'] for r in aa)}/{len(aa)}\n")
    lines += ["**Calibrated gate, planted defects**", "",
              "| defect | trials | rejected | rate | rejected by |", "|---|---:|---:|---:|---|"]  # fmt: skip
    for name, p in res["planted"].items():
        why = ", ".join(f"{k} {v}" for k, v in sorted(p["reasons"].items()))
        lines.append(
            f"| {name} | {p['trials']} | {p['rejected']} | {p['rejection_rate']:.0%} | {why} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="harness.gate_selftest")
    ap.add_argument("--oracle-root", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--truth-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(pins.ORACLE_SEEDS))
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--local-replicates", type=int, default=5)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--defects", nargs="*", help="restrict to these planted defects")
    ap.add_argument("--out", type=Path, required=True, help="output stem (.json and .md)")
    ns = ap.parse_args(argv)
    targets = json.loads(ns.targets.read_text())
    gated = [t["id"] for t in targets["targets"] if t.get("gated")]
    structs = load_oracle(ns.oracle_root, targets, ns.seeds)
    missing = [(t, s) for t in gated for s in ns.seeds if (t, s) not in structs]
    if missing:
        print(f"missing oracle results: {missing}", file=sys.stderr)
        return 2
    res = run_selftest(
        structs, Scorer(targets, ns.truth_dir), gated, ns.seeds,
        replicates=ns.replicates, local_replicates=ns.local_replicates,
        rng_seed=ns.rng_seed, only_defects=ns.defects,
    )  # fmt: skip
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{ns.out}.json").write_text(json.dumps(res, indent=2) + "\n")
    Path(f"{ns.out}.md").write_text(selftest_markdown(res))
    print(selftest_markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
