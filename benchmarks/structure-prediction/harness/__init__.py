# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Phase A measurement harness: AF3-class structure prediction on Trainium2 vs G7e.

Modules:
    schema: fixed result and cost-table schemas (versioned; later phases append).
    manifest: per-run environment manifest written next to every result.
    backends: command construction for the neuron, gpu, openfold-trt and cpu backends.
    entry: in-process wrapper adding timing/memory capture (and fp32 for the oracle).
    run: the harness CLI.
    score: accuracy metrics and the work plan's section-2 gate.
    profile_classify: attribute profiled time to operation classes.
    cost_table: $/structure tables from result JSONs.
"""
