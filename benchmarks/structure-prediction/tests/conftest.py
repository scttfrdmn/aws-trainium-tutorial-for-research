# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the structure-prediction harness tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from sp_testutil import AA, FakeEC2, FakeHost, FakePricing  # noqa: E402


@pytest.fixture
def seq30() -> str:
    """A 30-residue sequence."""
    return "".join(AA[i % 20] for i in range(30))


@pytest.fixture
def fake_manifest_kwargs() -> dict[str, Any]:
    """Keyword arguments that make capture_manifest hermetic."""
    host = FakeHost("neuron")
    return {
        "run": host.run,
        "http": host.http,
        "imds_token": lambda: "tok",
        "pricing_client": FakePricing(),
        "ec2_client": FakeEC2(),
        "environ": {
            "NEURON_RT_VISIBLE_CORES": "0",
            "BOLTZ_NEURON_ATOM_GRAPH_BREAK": "1",
            "HOME": "/x",
        },
        "meminfo": "MemTotal:       130023424 kB\n",
    }
