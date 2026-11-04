# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Build the Phase A target set: 20 CASP14 targets (Didi et al. 2025) + 2 ceiling complexes.

    python -m targets.build_targets --out targets/ --truth-dir ~/sp-data/truth

Writes `targets.json` and `targets.fasta` (committed) and ground-truth mmCIF files into
--truth-dir (not committed).

Sources and rules
-----------------
* Target IDs and the supplement's lengths: the Didi et al. supplement
  (`fast_folding_supplement.xlsx`, sheet `MSA_timings`). The supplement lists IDs and lengths
  only; it has no sequences, PDB IDs or scoring units.
* Sequences: the CASP14 sequence file (predictioncenter.org). Where the supplement's length
  differs from the CASP sequence length, both are recorded (`supplement_length`, `length`).
* Ground truth, per the work plan: the PDB entry where the target was deposited, otherwise the
  CASP Prediction Center's released target file. A PDB entity is accepted when an RCSB sequence
  search finds it at >= 95 % identity and its best chain covers at least as many target residues
  as the CASP file (or >= 90 % of the target when there is no CASP file). PDB codes named in the
  CASP target list are preferred, then the earliest release. Every choice and its reason is
  recorded per target. A target with neither source has `truth: null`; the gate then counts it
  as missing (it is not dropped).
* Scoring unit: the supplement does not say whether it scored domains; its TM-scores are
  whole-target scores, so every CASP14 target is scored as the whole target (all target residues
  present in the ground truth). Recorded per target as `scoring_unit`.
* Ceiling complexes (`ceiling.json`): sequences and chains from RCSB; not gated, no oracle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.score import align_sequences, load_structure

SUPPLEMENT = "https://research.nvidia.com/labs/dbr/assets/data/manuscripts/fast_folding_supplement.xlsx"
CASP_SEQ = "https://predictioncenter.org/download_area/CASP14/sequences/casp14.seq.txt"
CASP_LIST = "https://predictioncenter.org/casp14/targetlist.cgi?type=csv"
CASP_T = "https://predictioncenter.org/download_area/CASP14/targets/casp14.targets.T.public_11.29.2020.tar.gz"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_FILE = "https://files.rcsb.org/download/{}.cif.gz"
SCHEMA = "sp-targets/1.0"
MIN_IDENTITY = 0.95
MIN_COVERAGE_NO_CASP = 0.90

Fetch = Callable[[str, bytes | None], bytes]


def fetch(url: str, data: bytes | None = None) -> bytes:
    """HTTP GET (or POST JSON when data is given)."""
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        body: bytes = resp.read()
    return body


def supplement_targets(xlsx: bytes) -> list[tuple[str, int]]:
    """(target id, supplement length) in the supplement's order (sheet MSA_timings)."""
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(xlsx), read_only=True, data_only=True)
    out = []
    for row in wb["MSA_timings"].iter_rows(values_only=True):
        if row and isinstance(row[0], str) and re.fullmatch(r"T\d{4}(s\d)?", row[0]):
            out.append((row[0], int(float(row[1]))))
    return out


def parse_casp_fasta(text: str) -> dict[str, str]:
    """CASP14 sequence file -> {target: sequence}."""
    seqs: dict[str, str] = {}
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:].split()[0]
            seqs[cur] = ""
        elif cur and line:
            seqs[cur] += line
    return seqs


def parse_casp_list(text: str) -> dict[str, dict[str, Any]]:
    """CASP14 target list CSV -> {target: {oligo, pdb_hint}} (PDB code from the description)."""
    out = {}
    for line in text.splitlines()[1:]:
        cols = line.split(";")
        if len(cols) < 10:
            continue
        m = re.search(r"\b(\d[a-z0-9]{3})\b", cols[9])
        out[cols[0]] = {"oligo": cols[3], "pdb_hint": m.group(1).upper() if m else None}
    return out


def casp_files(tgz: bytes) -> dict[str, bytes]:
    """Public CASP14 target PDB files -> {target: pdb bytes}."""
    out = {}
    with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            if m.isfile() and name.endswith(".pdb"):
                f = tf.extractfile(m)
                if f is not None:
                    out[name[:-4]] = f.read()
    return out


def rcsb_entities(seq: str, get: Fetch) -> list[str]:
    """Polymer entities matching `seq` at >= MIN_IDENTITY."""
    q = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1e-10,
                "identity_cutoff": MIN_IDENTITY,
                "sequence_type": "protein",
                "value": seq,
            },
        },
        "request_options": {"paginate": {"start": 0, "rows": 25}},
        "return_type": "polymer_entity",
    }
    body = get(RCSB_SEARCH, json.dumps(q).encode())
    return [h["identifier"] for h in json.loads(body)["result_set"]] if body else []


def rcsb_entity_info(entity_ids: list[str], get: Fetch) -> dict[str, dict[str, Any]]:
    """Chains and release dates for entities."""
    if not entity_ids:
        return {}
    q = (
        "{ polymer_entities(entity_ids: " + json.dumps(entity_ids) + ") { rcsb_id "
        "rcsb_polymer_entity_container_identifiers { auth_asym_ids } "
        "entry { rcsb_accession_info { initial_release_date } } } }"
    )
    data = json.loads(get(RCSB_GRAPHQL, json.dumps({"query": q}).encode()))["data"]
    return {
        e["rcsb_id"]: {
            "chains": e["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"],
            "released": e["entry"]["rcsb_accession_info"]["initial_release_date"][:10],
        }
        for e in data["polymer_entities"]
    }


def best_chain(
    path: Path, seq: str, chains: list[str] | None = None
) -> dict[str, Any] | None:
    """Chain of `path` covering most target residues with identity; coverage stats."""
    st = load_structure(path)
    best: dict[str, Any] | None = None
    for cid, ch in st.chains.items():
        if chains and cid not in chains:
            continue
        pairs = align_sequences(seq, ch.seq)
        same = [(i, j) for i, j in pairs if seq[i] == ch.seq[j]]
        if not pairs:
            continue
        cand: dict[str, Any] = {
            "chain": cid,
            "matched": len(same),
            "identity": len(same) / len(pairs),
            "coverage": len(same) / len(seq),
            "observed_range": [ch.resnums[same[0][1]], ch.resnums[same[-1][1]]]
            if same
            else None,
        }
        if best is None or cand["matched"] > best["matched"]:
            best = cand
    return best


def choose_truth(
    tid: str,
    seq: str,
    hint: str | None,
    casp_pdb: bytes | None,
    truth_dir: Path,
    get: Fetch,
) -> dict[str, Any] | None:
    """Apply the ground-truth rule; write the chosen file as <truth_dir>/<tid>.cif."""
    import gemmi

    casp_stats = None
    if casp_pdb is not None:
        cp = truth_dir / f"{tid}.casp.pdb"
        cp.write_bytes(casp_pdb)
        casp_stats = best_chain(cp, seq)

    ents = rcsb_entities(seq, get)
    info = rcsb_entity_info(ents, get)
    order = sorted(
        info,
        key=lambda e: (e.split("_")[0] != (hint or ""), info[e]["released"], e),
    )
    need = casp_stats["matched"] if casp_stats else int(MIN_COVERAGE_NO_CASP * len(seq))
    rejected = []
    for ent in order:
        pdb_id = ent.split("_")[0]
        raw = gzip.decompress(get(RCSB_FILE.format(pdb_id), None))
        tmp = truth_dir / f"{pdb_id}.cif"
        tmp.write_bytes(raw)
        stats = best_chain(tmp, seq, info[ent]["chains"])
        if stats and stats["identity"] >= MIN_IDENTITY and stats["matched"] >= need:
            out = truth_dir / f"{tid}.cif"
            out.write_bytes(raw)
            return {
                "source": "pdb",
                "pdb_id": pdb_id,
                "entity": ent,
                "file": out.name,
                **stats,
                "reason": (
                    f"PDB {ent} (released {info[ent]['released']})"
                    + (" named in the CASP target list" if pdb_id == hint else "")
                    + f"; chain {stats['chain']} covers {stats['matched']}/{len(seq)} residues"
                ),
            }
        rejected.append(
            f"{ent}: {stats['matched'] if stats else 0} matched < {need} needed"
        )
    if casp_stats is not None:
        st = gemmi.read_structure(str(truth_dir / f"{tid}.casp.pdb"))
        st.setup_entities()
        out = truth_dir / f"{tid}.cif"
        st.make_mmcif_document().write_file(str(out))
        return {
            "source": "casp",
            "pdb_id": None,
            "entity": None,
            "file": out.name,
            **casp_stats,
            "reason": "no PDB entity met the rule ("
            + ("; ".join(rejected) or "no entity at >= 95 % identity")
            + "); CASP Prediction Center public target file",
        }
    return None


def ceiling_target(c: dict[str, Any], get: Fetch) -> dict[str, Any]:
    """Chains (Boltz chain ids A, B, ...) and token count for a ceiling complex."""
    q = (
        '{ entry(entry_id: "' + c["pdb_id"] + '") { polymer_entities { rcsb_id '
        "entity_poly { rcsb_entity_polymer_type pdbx_seq_one_letter_code_can } "
        "rcsb_polymer_entity_container_identifiers { auth_asym_ids } } } }"
    )
    data = json.loads(get(RCSB_GRAPHQL, json.dumps({"query": q}).encode()))["data"][
        "entry"
    ]
    kinds = {"Protein": "protein", "RNA": "rna", "DNA": "dna"}
    chains = []
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for ent in data["polymer_entities"]:
        kind = kinds[ent["entity_poly"]["rcsb_entity_polymer_type"]]
        seq = ent["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "")
        for auth in ent["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]:
            chains.append(
                {
                    "chain_id": next(letters),
                    "entity": kind,
                    "sequence": seq,
                    "truth_chain": auth,
                    "msa_id": f"{c['id']}_{ent['rcsb_id']}"
                    if kind == "protein"
                    else None,
                }
            )
    tokens = sum(len(ch["sequence"]) for ch in chains)
    return {
        "id": c["id"],
        "set": "ceiling",
        "kind": c["kind"],
        "length": tokens,
        "supplement_length": None,
        "tokens": tokens,
        "gated": False,
        "chains": chains,
        "truth": {
            "source": "pdb",
            "pdb_id": c["pdb_id"],
            "file": None,
            "reason": c["reason"],
        },
    }


def build(out: Path, truth_dir: Path, get: Fetch = fetch) -> dict[str, Any]:
    """Build and write targets.json / targets.fasta; returns the manifest."""
    truth_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        name: get(url, None)
        for name, url in (
            ("supplement", SUPPLEMENT),
            ("casp_seq", CASP_SEQ),
            ("casp_list", CASP_LIST),
            ("casp_targets", CASP_T),
        )
    }
    sup = supplement_targets(raw["supplement"])
    seqs = parse_casp_fasta(raw["casp_seq"].decode())
    clist = parse_casp_list(raw["casp_list"].decode())
    cfiles = casp_files(raw["casp_targets"])

    targets: list[dict[str, Any]] = []
    for tid, sup_len in sup:
        seq = seqs[tid]
        truth = choose_truth(
            tid,
            seq,
            clist.get(tid, {}).get("pdb_hint"),
            cfiles.get(tid),
            truth_dir,
            get,
        )
        if truth is not None:
            truth["scoring_unit"] = "whole-target"
            truth["unit"] = None
        targets.append(
            {
                "id": tid,
                "set": "casp14",
                "length": len(seq),
                "supplement_length": sup_len,
                "tokens": len(seq),
                "gated": True,
                "oligo_state": clist.get(tid, {}).get("oligo"),
                "chains": [
                    {
                        "chain_id": "A",
                        "entity": "protein",
                        "sequence": seq,
                        "truth_chain": truth["chain"] if truth else None,
                        "msa_id": tid,
                    }
                ],
                "truth": truth
                or {
                    "source": None,
                    "file": None,
                    "reason": "no PDB entity at >= 95 % identity and no public CASP target file; "
                    "gated as missing until the owner decides (see the Phase A tracking issue)",
                },
            }
        )
    ceiling = json.loads((Path(__file__).with_name("ceiling.json")).read_text())
    targets += [ceiling_target(c, get) for c in ceiling["complexes"]]

    manifest = {
        "_copyright": "Copyright 2026 Scott Friedman. SPDX-License-Identifier: Apache-2.0",
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sources": {
            name: {"url": url, "sha256": hashlib.sha256(raw[name]).hexdigest()}
            for name, url in (
                ("supplement", SUPPLEMENT),
                ("casp_seq", CASP_SEQ),
                ("casp_list", CASP_LIST),
                ("casp_targets", CASP_T),
            )
        },
        "rules": {
            "min_identity": MIN_IDENTITY,
            "min_coverage_without_casp_file": MIN_COVERAGE_NO_CASP,
            "scoring_unit": "whole-target for every CASP14 target (the supplement does not state domains)",
        },
        "targets": targets,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "targets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fasta = ["; Copyright 2026 Scott Friedman. SPDX-License-Identifier: Apache-2.0"]
    for t in targets:
        for ch in t["chains"]:
            fasta += [f">{t['id']}|{ch['chain_id']}|{ch['entity']}", ch["sequence"]]
    (out / "targets.fasta").write_text("\n".join(fasta) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="targets.build_targets")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    ap.add_argument("--truth-dir", type=Path, required=True)
    ns = ap.parse_args(argv)
    m = build(ns.out, ns.truth_dir)
    for t in m["targets"]:
        tr = t["truth"]
        print(
            f"{t['id']:10s} {t['length']:5d} {str(tr.get('source')):5s} {tr.get('pdb_id') or '-':5s} {tr.get('reason', '')[:90]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
