# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Fixed schemas for per-run results and the cost table.

The schemas are fixed from the first commit. Later phases may only append optional fields
(bump the minor part of the schema id); they never rename or remove a field. Validation is
hand-rolled so the harness has no schema-library dependency.
"""

from __future__ import annotations

from typing import Any

RESULT_SCHEMA = "sp-result/1.0"
COST_SCHEMA = "sp-cost/1.0"

BACKENDS = ("neuron", "gpu", "openfold-trt", "cpu")
STATUSES = ("measured", "dry-run", "failed", "did-not-run")

# field -> (allowed types, required). None in the type tuple means the value may be null.
_NUM = (int, float)
RESULT_FIELDS: dict[str, tuple[tuple[Any, ...], bool]] = {
    "schema": ((str,), True),
    "status": ((str,), True),
    "target_id": ((str,), True),
    "backend": ((str,), True),
    "seed": ((int,), True),
    "diffusion_samples": ((int,), True),
    "recycling_steps": ((int,), True),
    "sampling_steps": ((int,), True),
    "tokens": ((int, None), True),
    "lnc": ((int, None), True),
    "atom_graph_break": ((int, None), True),
    "worker_index": ((int, None), True),
    "workers": ((int, None), True),
    "run_group": ((str,), True),
    "wall_cold_s": ((*_NUM, None), True),
    "wall_warm_s": ((*_NUM, None), True),
    "model_s": ((*_NUM, None), True),
    "peak_accel_mem_bytes": ((int, None), True),
    "peak_accel_mem_source": ((str, None), True),
    "predictions": ((list,), True),
    "profile_path": ((str, None), True),
    "command": ((list,), True),
    "returncode": ((int, None), True),
    "error": ((str, None), True),
    "manifest_path": ((str, None), True),
}

COST_COLUMNS = (
    "schema",
    "phase",
    "target_id",
    "tokens",
    "platform",
    "instance_type",
    "backend",
    "lnc",
    "workers",
    "diffusion_samples",
    "seeds",
    "dl_s_per_structure",
    "msa_s",
    "msa_host",
    "price_od_usd_per_h",
    "price_spot_usd_per_h",
    "msa_price_od_usd_per_h",
    "msa_price_spot_usd_per_h",
    "dl_usd_per_structure_od",
    "dl_usd_per_structure_spot",
    "e2e_usd_per_structure_od",
    "e2e_usd_per_structure_spot",
    "ran",
    "ceiling_tokens",
    "note",
)


class SchemaError(ValueError):
    """A result document does not match the fixed schema."""


def _type_ok(value: Any, types: tuple[Any, ...]) -> bool:
    if value is None:
        return None in types
    if isinstance(value, bool):  # bool is an int subclass; never a valid number here
        return False
    return isinstance(value, tuple(t for t in types if t is not None))


def validate_result(doc: dict[str, Any]) -> None:
    """Raise SchemaError if `doc` is not a valid result document.

    Beyond types, enforces the honesty rules: a non-measured result carries no measured numbers,
    and a measured result has its wall times (warm as well as cold, except for the single-run CPU
    oracle) and a manifest.
    """
    missing = [k for k, (_, req) in RESULT_FIELDS.items() if req and k not in doc]
    if missing:
        raise SchemaError(f"missing fields: {missing}")
    for key, (types, _) in RESULT_FIELDS.items():
        if key in doc and not _type_ok(doc[key], types):
            raise SchemaError(f"field {key!r} has invalid value {doc[key]!r}")
    if doc["schema"] != RESULT_SCHEMA:
        raise SchemaError(f"schema {doc['schema']!r} != {RESULT_SCHEMA!r}")
    if doc["status"] not in STATUSES:
        raise SchemaError(f"status {doc['status']!r} not in {STATUSES}")
    if doc["backend"] not in BACKENDS:
        raise SchemaError(f"backend {doc['backend']!r} not in {BACKENDS}")
    measured = ("wall_cold_s", "wall_warm_s", "model_s", "peak_accel_mem_bytes")
    if doc["status"] != "measured" and any(doc[k] is not None for k in measured):
        raise SchemaError(f"status {doc['status']!r} must not carry measured values")
    if doc["status"] == "measured":
        if doc["wall_warm_s"] is None and doc["backend"] != "cpu":
            raise SchemaError("a measured accelerator result needs wall_warm_s")
        if doc["wall_cold_s"] is None:
            raise SchemaError("a measured result needs wall_cold_s")
        if doc["manifest_path"] is None:
            raise SchemaError("a measured result needs a manifest")


def empty_result(**fields: Any) -> dict[str, Any]:
    """A result document with every field present and no measured values."""
    doc: dict[str, Any] = dict.fromkeys(RESULT_FIELDS)
    doc.update(
        schema=RESULT_SCHEMA,
        predictions=[],
        command=[],
        status="dry-run",
        run_group="",
    )
    doc.update(fields)
    return doc
