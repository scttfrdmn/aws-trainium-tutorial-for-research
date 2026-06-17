"""Hardware validation harness for the AWS Trainium & Inferentia tutorial.

This package exists to keep one promise: **every code example in the tutorial is proven to run
on real AWS Neuron hardware**, with the proof captured as a machine-generated provenance artifact
(instance type, Neuron SDK version, git commit, date, metrics, and raw log) rather than a
hand-typed claim.

Layout:
    registry.py        - the catalog of validatable examples + pass thresholds
    provenance.py      - the result schema + environment capture (SDK versions, instance, commit)
    launcher.py        - spawn|awscli launcher abstraction (with boto3 fallback)
    run_on_hardware.py - orchestrator: runs examples on the current instance, writes artifacts
    render_status.py   - aggregates artifacts into VALIDATED.md + README badge

See ../docs/REVAMP_PLAN.md for the full design and the spawn/lagotto tooling decision.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
