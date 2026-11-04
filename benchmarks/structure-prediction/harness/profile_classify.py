# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Attribute profiled time to operation classes (Phase A task 4).

    python -m harness.profile_classify gpu    --trace torch-profile.json --out classes.json
    python -m harness.profile_classify neuron --profile neuron-profile.json --out classes.json

Operation classes and how each is identified
--------------------------------------------
Classes are defined by Boltz-2 2.2.1 module boundaries (class names in boltz/model). A time
slice is assigned to the innermost enclosing module that matches a rule below; the rule records
whether the attribution is *certain* (a module boundary seen directly) or *heuristic* (inferred
from op or instruction names).

| class | rule (module boundary) |
|---|---|
| triangle_attention | TriangleAttention, TriangleAttentionStartingNode/EndingNode |
| triangle_multiplication | TriangleMultiplicationOutgoing / TriangleMultiplicationIncoming |
| msa_module | MSAModule (incl. its MSALayer, PairWeightedAveraging, OuterProductMean) |
| pairformer_transition | Transition inside PairformerModule / PairformerNoSeqModule |
| pairformer_other | anything else inside PairformerModule (e.g. AttentionPairBias) |
| windowed_atom_attention | AtomTransformer, AtomAttentionEncoder, AtomAttentionDecoder |
| diffusion_token_transformer | DiffusionTransformer inside DiffusionModule (not inside an atom module) |
| dma_transfer | GPU memcpy/memset events; Neuron DMA-queue instructions |
| host_glue | wall time with no device activity (gaps between kernels/instructions) |
| other | device time inside none of the above |

GPU (torch.profiler Chrome trace, recorded with `with_modules=True` by harness.entry): each GPU
kernel is linked to its CUDA launch through `args.correlation`; the launch's CPU timestamp is
placed inside the `nn.Module: <Class>_<n>` python_function intervals of the launching thread,
which gives the module stack. Module-based attributions are certain.

Neuron (neuron-profile JSON, see `NEURON_FIELDS`): instructions carry an engine and a label
(framework op name / HLO metadata). Where the label contains a Boltz module class name the
attribution is certain; otherwise the instruction is matched by op-name patterns and marked
heuristic. The JSON field names are an adapter (`NEURON_FIELDS`) because the exported format has
not yet been checked against real `neuron-profile view --output-format json` output; that check
is a needs-hardware item (Phase A task 4).

Per class, the output gives device time, fraction of the total, and for Neuron the
tensor/vector/scalar/gpsimd/DMA engine busy fractions over the class's device time.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLASSES = (
    "triangle_attention",
    "triangle_multiplication",
    "msa_module",
    "pairformer_transition",
    "pairformer_other",
    "windowed_atom_attention",
    "diffusion_token_transformer",
    "dma_transfer",
    "host_glue",
    "other",
)
ENGINES = ("tensor", "vector", "scalar", "gpsimd", "dma")

_MOD = re.compile(r"nn\.Module: ([A-Za-z0-9_]+?)(?:_\d+)?$")
ATOM_MODULES = {"AtomTransformer", "AtomAttentionEncoder", "AtomAttentionDecoder"}
PAIRFORMER = {"PairformerModule", "PairformerNoSeqModule"}


def classify_stack(stack: Sequence[str]) -> str:
    """Class for a module stack (outermost first) of Boltz-2 module class names."""
    s = set(stack)
    for name in reversed(stack):  # innermost first
        if name.startswith("TriangleAttention"):
            return "triangle_attention"
        if name.startswith("TriangleMultiplication"):
            return "triangle_multiplication"
    if "MSAModule" in s:
        return "msa_module"
    if s & ATOM_MODULES:
        return "windowed_atom_attention"
    if "DiffusionTransformer" in s and "DiffusionModule" in s:
        return "diffusion_token_transformer"
    if s & PAIRFORMER:
        return (
            "pairformer_transition"
            if stack and "Transition" in stack
            else "pairformer_other"
        )
    return "other"


def _union_length(intervals: Iterable[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None and cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None and cur_s is not None:
        total += cur_e - cur_s
    return total


@dataclass
class _Interval:
    start: float
    end: float
    name: str


def classify_gpu_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Classify a torch.profiler Chrome trace (times in µs)."""
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    modules: dict[Any, list[_Interval]] = defaultdict(list)
    launches: dict[Any, tuple[Any, float]] = {}
    kernels, copies = [], []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, args = e.get("cat", ""), e.get("args", {}) or {}
        if cat == "python_function":
            m = _MOD.match(e.get("name", ""))
            if m:
                modules[e.get("tid")].append(
                    _Interval(e["ts"], e["ts"] + e["dur"], m.group(1))
                )
        elif cat == "cuda_runtime" and "correlation" in args:
            launches[args["correlation"]] = (e.get("tid"), e["ts"])
        elif cat == "kernel":
            kernels.append(e)
        elif cat in ("gpu_memcpy", "gpu_memset"):
            copies.append(e)
    for ivs in modules.values():
        ivs.sort(key=lambda iv: iv.start)
    starts = {tid: [iv.start for iv in ivs] for tid, ivs in modules.items()}

    def stack_at(tid: Any, ts: float) -> list[str]:
        ivs = modules.get(tid, [])
        k = bisect.bisect_right(starts.get(tid, []), ts)
        inside = [iv for iv in ivs[:k] if iv.end >= ts]
        inside.sort(key=lambda iv: (iv.start, -iv.end))
        return [iv.name for iv in inside]

    per: dict[str, float] = defaultdict(float)
    certainty: dict[str, str] = {}
    busy: list[tuple[float, float]] = []
    for k in kernels:
        corr = (k.get("args") or {}).get("correlation")
        launch = launches.get(corr)
        cls = classify_stack(stack_at(*launch)) if launch else "other"
        per[cls] += k["dur"]
        certainty[cls] = "certain" if launch else "heuristic"
        busy.append((k["ts"], k["ts"] + k["dur"]))
    for c in copies:
        per["dma_transfer"] += c["dur"]
        certainty["dma_transfer"] = "certain"
        busy.append((c["ts"], c["ts"] + c["dur"]))
    span = (max(e for _, e in busy) - min(s for s, _ in busy)) if busy else 0.0
    glue = max(span - _union_length(busy), 0.0)
    per["host_glue"] += glue
    certainty["host_glue"] = "certain"
    return _report(per, certainty, span, unit="us", engines=None)


NEURON_FIELDS = {
    "list": "instructions",  # top-level key holding the instruction list (or a bare list)
    "engine": "engine",
    "start": "start_ns",
    "duration": "duration_ns",
    "label": "label",
}
_ENGINE_ALIASES = {
    "pe": "tensor", "tensor": "tensor", "tensorengine": "tensor",
    "dve": "vector", "vector": "vector", "vectorengine": "vector",
    "act": "scalar", "scalar": "scalar", "activation": "scalar", "scalarengine": "scalar",
    "pool": "gpsimd", "gpsimd": "gpsimd",
    "dma": "dma", "qsync": "dma",
}  # fmt: skip
_MODULE_IN_LABEL = re.compile(
    r"(TriangleAttention\w*|TriangleMultiplication\w*|MSAModule|MSALayer|PairformerModule|"
    r"PairformerNoSeqModule|Transition|DiffusionTransformer|DiffusionModule|AtomTransformer|"
    r"AtomAttentionEncoder|AtomAttentionDecoder)"
)
# Op-name heuristics used only when no module name is present in the label.
_HEURISTICS = (
    (re.compile(r"tri(angle)?_?att", re.I), "triangle_attention"),
    (re.compile(r"tri(angle)?_?mul", re.I), "triangle_multiplication"),
    (re.compile(r"msa", re.I), "msa_module"),
    (re.compile(r"atom", re.I), "windowed_atom_attention"),
    (
        re.compile(r"diffusion|score_model|structure_module", re.I),
        "diffusion_token_transformer",
    ),
)


def _engine(raw: str) -> str:
    key = re.sub(r"[^a-z]", "", raw.lower())
    for prefix, eng in _ENGINE_ALIASES.items():
        if key.startswith(prefix):
            return eng
    return "other"


def classify_label(label: str) -> tuple[str, str]:
    """(class, certainty) for one Neuron instruction label."""
    names = _MODULE_IN_LABEL.findall(label)
    if names:
        stack = ["DiffusionModule" if n == "DiffusionTransformer" else n for n in names]
        if "DiffusionTransformer" in names:
            stack = ["DiffusionModule", *names]
        if "Transition" in names and not set(names) & PAIRFORMER:
            stack = ["PairformerModule", *names]
        return classify_stack(stack), "certain"
    for pat, cls in _HEURISTICS:
        if pat.search(label):
            return cls, "heuristic"
    return "other", "heuristic"


def classify_neuron_profile(
    doc: Any, fields: dict[str, str] = NEURON_FIELDS
) -> dict[str, Any]:
    """Classify neuron-profile instructions (times in ns)."""
    insts = doc[fields["list"]] if isinstance(doc, dict) else doc
    per: dict[str, float] = defaultdict(float)
    eng_busy: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    certainty: dict[str, str] = {}
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    busy: list[tuple[float, float]] = []
    for ins in insts:
        s = float(ins[fields["start"]])
        e = s + float(ins[fields["duration"]])
        eng = _engine(str(ins[fields["engine"]]))
        if eng == "dma":
            cls, cert = "dma_transfer", "certain"
        else:
            cls, cert = classify_label(str(ins.get(fields["label"], "")))
        if certainty.get(cls) != "heuristic":
            certainty[cls] = cert
        eng_busy[cls][eng].append((s, e))
        spans[cls].append((s, e))
        busy.append((s, e))
    for cls, ivs in spans.items():
        per[cls] = _union_length(ivs)
    span = (max(e for _, e in busy) - min(s for s, _ in busy)) if busy else 0.0
    per["host_glue"] = max(span - _union_length(busy), 0.0)
    certainty["host_glue"] = "certain"
    engines = {
        cls: {
            eng: (_union_length(eng_busy[cls][eng]) / per[cls]) if per[cls] else 0.0
            for eng in ENGINES
        }
        for cls in spans
    }
    return _report(per, certainty, span, unit="ns", engines=engines)


def _report(
    per: dict[str, float],
    certainty: dict[str, str],
    span: float,
    unit: str,
    engines: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    total = sum(per.values())
    rows = []
    for cls in CLASSES:
        if cls not in per:
            continue
        row: dict[str, Any] = {
            "class": cls,
            "time": per[cls],
            "fraction": per[cls] / total if total else 0.0,
            "attribution": certainty.get(cls, "heuristic"),
        }
        if engines is not None and cls in engines:
            row["engine_busy_fraction"] = engines[cls]
        rows.append(row)
    return {"unit": unit, "span": span, "total_attributed": total, "classes": rows}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="harness.profile_classify")
    sub = ap.add_subparsers(dest="kind", required=True)
    g = sub.add_parser("gpu")
    g.add_argument("--trace", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    n = sub.add_parser("neuron")
    n.add_argument("--profile", type=Path, required=True)
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--fields", type=Path, help="JSON overriding NEURON_FIELDS")
    ns = ap.parse_args(argv)
    if ns.kind == "gpu":
        res = classify_gpu_trace(json.loads(ns.trace.read_text()))
    else:
        fields = {
            **NEURON_FIELDS,
            **(json.loads(ns.fields.read_text()) if ns.fields else {}),
        }
        res = classify_neuron_profile(json.loads(ns.profile.read_text()), fields)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(res, indent=2) + "\n")
    for r in res["classes"]:
        print(f"{r['class']:28s} {r['fraction']:6.1%}  {r['attribution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
