"""Unit tests for the hardware-validation harness.

These run without AWS, Neuron, or torch -- they check the harness *contract* and provenance logic
so CI can guard the machinery that guards the examples. The actual on-hardware runs are exercised
separately by `validation/run_on_hardware.py --in-instance` on a Trainium instance.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from validation import registry
from validation.provenance import (
    ValidationResult,
    capture_environment,
    meets_thresholds,
)


def test_registry_nonempty_and_unique_keys():
    """The registry must have entries and unique keys (the harness indexes by key)."""
    assert registry.EXAMPLES, "registry should declare at least one example"
    keys = [e.key for e in registry.EXAMPLES]
    assert len(keys) == len(set(keys)), f"duplicate registry keys: {keys}"


@pytest.mark.parametrize(
    "example",
    [*registry.EXAMPLES, *registry.TORCHRUN_EXAMPLES],
    ids=lambda e: e.key,
)
def test_registered_module_imports_and_has_entrypoint(example):
    """Every registered example (incl. torchrun-only) must import and expose its entrypoint.

    Imports are lazy in the examples, so this works without torch/transformers/neuron installed.
    """
    module = importlib.import_module(example.module)
    entry = getattr(module, example.entrypoint, None)
    assert callable(entry), f"{example.module}.{example.entrypoint} is not callable"


@pytest.mark.parametrize("example", registry.EXAMPLES, ids=lambda e: e.key)
def test_thresholds_are_sane(example):
    """Thresholds must be present and within a plausible range (catches typos like 75 vs 0.75)."""
    assert example.thresholds, f"{example.key} declares no thresholds"
    for name, value in example.thresholds.items():
        assert 0.0 <= value <= 1.0 or value > 1.0, (
            name
        )  # ratios in [0,1]; counts/throughput >1


def test_meets_thresholds_pass_and_fail():
    """Threshold checking: passes when met, fails (with reasons) when not or when missing."""
    ok, fails = meets_thresholds({"eval_f1": 0.81}, {"eval_f1": 0.75})
    assert ok and not fails

    ok, fails = meets_thresholds({"eval_f1": 0.70}, {"eval_f1": 0.75})
    assert not ok and any("eval_f1" in f for f in fails)

    ok, fails = meets_thresholds({}, {"eval_f1": 0.75})
    assert not ok and any("missing" in f for f in fails)


def test_result_roundtrip(tmp_path: Path):
    """A ValidationResult writes valid JSON keyed by a filesystem-safe slug."""
    r = ValidationResult(
        example="examples/use_cases/biomedical_ner.py", status="passed"
    )
    r = capture_environment(r, clock="2026-06-16T00:00:00Z")
    out = r.write(tmp_path)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["example"] == "examples/use_cases/biomedical_ner.py"
    assert data["timestamp"] == "2026-06-16T00:00:00Z"
    # Off-hardware, Neuron fields must be None rather than fabricated.
    assert data["versions"]["torch_neuronx"] is None


def test_write_qualifies_filename_by_instance(tmp_path: Path):
    """When instance_type is set, the artifact filename is qualified with it, so results for the
    same example on different instances coexist instead of overwriting."""
    r1 = ValidationResult(
        example="examples.use_cases.biomedical_ner",
        status="passed",
        instance_type="trn1.2xlarge",
    )
    r2 = ValidationResult(
        example="examples.use_cases.biomedical_ner",
        status="passed",
        instance_type="trn2.48xlarge",
    )
    p1 = r1.write(tmp_path)
    p2 = r2.write(tmp_path)
    assert p1.name == "examples.use_cases.biomedical_ner@trn1.2xlarge.json"
    assert p2.name == "examples.use_cases.biomedical_ner@trn2.48xlarge.json"
    assert p1 != p2 and p1.exists() and p2.exists()  # neither overwrote the other


def test_render_status_groups_by_instance(tmp_path: Path, monkeypatch):
    """render_status renders one row per (example, instance) and counts multi-instance coverage."""
    from validation import render_status

    ex = registry.EXAMPLES[0]  # a real registered example (its module path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for inst in ("trn1.2xlarge", "trn2.48xlarge"):
        ValidationResult(example=ex.module, status="passed", instance_type=inst).write(
            results_dir
        )

    monkeypatch.setattr(render_status, "RESULTS_DIR", results_dir)
    loaded = render_status._load_results()
    assert len(loaded[ex.module]) == 2  # both instances grouped under the one example
    assert render_status._passed_on_any(loaded[ex.module])

    out = render_status.render(clock="2026-08-18T00:00:00Z")
    # Two data rows for this example (one per instance), and the multi-instance note is present.
    assert out.count(f"| `{ex.key}` |") == 2
    assert "trn1.2xlarge" in out and "trn2.48xlarge" in out
    assert "more than one instance" in out


def test_awscli_user_data_is_valid_bash():
    """The generated user-data must be syntactically valid bash, even with embedded quotes in the
    command -- a parse error aborts the whole script (including its self-terminate shutdown), which
    is exactly how a validation box can leak cost. Guards the shlex.quote (not repr) embedding."""
    import shutil
    import subprocess

    from validation.launcher import _awscli_user_data

    bash = shutil.which("bash")
    if not bash:  # pragma: no cover - CI runners have bash
        pytest.skip("bash not available")
    # A command containing single quotes is the case that broke repr-based embedding.
    ud = _awscli_user_data(
        "export X='s3://b/c'; echo \"it's fine\" | tee ~/log", max_hours=2.0
    )
    proc = subprocess.run([bash, "-n"], input=ud, text=True, capture_output=True)
    assert proc.returncode == 0, f"user-data failed bash -n:\n{proc.stderr}\n---\n{ud}"
    # Cost backstop must be present (detached TTL killer) so a broken command can't strand the box.
    assert "shutdown -h now" in ud and "sleep" in ud


def test_build_plan_image_id_override():
    """--image-id pins a specific AMI (e.g. a private/pre-release Neuron image) instead of the
    SSM-resolved public DLAMI, and it flows into both the plan and the awscli argv."""
    from validation.launcher import build_plan

    plan = build_plan(
        "trn1.2xlarge",
        "sa-east-1",
        name="t",
        remote_command="echo hi",
        prefer="awscli",
        image_id="ami-0deadbeef0",
    )
    assert plan.ami_id == "ami-0deadbeef0"
    assert (
        "ami-0deadbeef0" in plan.command
    )  # appears as --image-id value in run-instances


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"venv": "/root/xla29"},
        {
            "venv": "/root/xla 29",  # a space must survive quoting
            "create_venv": True,
            "venv_packages": "torch-neuronx==2.9.0.2.15.32035 neuronx-cc==2.28.7631.0",
        },
    ],
    ids=["glob", "venv", "create-venv"],
)
def test_bootstrap_prefix_guards_stack_and_is_valid_bash(kwargs):
    """Every bootstrap variant must be valid bash and must run the XLA-stack guard before the harness,
    so a DLAMI whose venv is TorchNeuron/vLLM (not torch-neuronx 2.9) stops instead of mis-validating."""
    import shutil
    import subprocess

    from validation.launcher import _awscli_user_data
    from validation.run_on_hardware import _XLA_STACK_GUARD, _bootstrap_prefix

    prefix = _bootstrap_prefix("https://github.com/o/r", "main", **kwargs)
    assert _XLA_STACK_GUARD in prefix
    assert prefix.index(_XLA_STACK_GUARD) < prefix.index("git clone")
    if kwargs.get("create_venv"):
        assert (
            "python3 -m venv" in prefix and "torch-neuronx==2.9.0.2.15.32035" in prefix
        )
        assert "download.pytorch.org/whl/cpu" in prefix  # avoid CUDA torch wheels
    if kwargs.get("venv"):
        assert (
            "aws_neuronx_venv_pytorch*" not in prefix
        )  # explicit venv replaces the glob
    bash = shutil.which("bash")
    if not bash:  # pragma: no cover
        pytest.skip("bash not available")
    proc = subprocess.run(
        [bash, "-n"],
        input=_awscli_user_data(prefix + "python3 -m x", max_hours=1.0),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_run_local_uploads_logs_from_exit_trap(capsys):
    """With --results-s3 the upload must run from an EXIT trap (fires even if bootstrap aborts under
    set -e), and the dry run must not launch anything."""
    from validation.run_on_hardware import main

    rc = main(
        [
            "--instance",
            "trn1.2xlarge",
            "--region",
            "us-east-2",
            "--example",
            "ner_biomedical",
            "--launcher",
            "awscli",
            "--image-id",
            "ami-0test",
            "--bootstrap",
            "--venv",
            "/root/xla29",
            "--create-venv",
            "--results-s3",
            "s3://bucket/run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "DRY RUN" in out
    assert "trap " in out and "EXIT" in out
    assert "s3://bucket/run/logs/ner_biomedical-run.log" in out


def test_run_in_instance_isolates_each_example(monkeypatch):
    """Multiple examples run one child process each (device memory is released between them), the
    exit code reflects child failures, and a single example stays in-process."""
    import subprocess

    from validation import run_on_hardware as roh

    calls = []

    class Done:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, cwd=None, check=False):
        calls.append(cmd)
        return Done(1 if "antibody_affinity_slm" in cmd else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    keys = ["ner_biomedical", "antibody_affinity_slm", "crystal_cif_slm"]
    rc = roh.run_in_instance(keys, smoke=True, clock="2026-10-20T00:00:00Z")
    assert rc == 1  # one child failed
    assert [c[c.index("--example") + 1] for c in calls] == keys
    assert all("--in-instance" in c and "--smoke" in c for c in calls)

    # single example: no child process
    calls.clear()
    monkeypatch.setattr(
        roh,
        "_run_one",
        lambda ex, smoke, clock: roh.ValidationResult(
            example=ex.module, status="passed"
        ),
    )
    monkeypatch.setattr(
        roh.ValidationResult, "write", lambda self, d: Path(d) / "x.json"
    )
    assert roh.run_in_instance(["ner_biomedical"], smoke=True, clock=None) == 0
    assert calls == []


def test_torchneuron_smoke_records_failure_off_hardware(tmp_path, monkeypatch):
    """Off Trn2 the smoke check must record a failed artifact (never a fabricated pass)."""
    from validation import torchneuron_smoke as tn

    monkeypatch.setattr(tn, "RESULTS_DIR", tmp_path)
    assert tn.main([]) == 1
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["example"] == "validation.torchneuron_smoke"
    assert data["status"] == "failed" and data["error"]


def test_post_command_runs_after_harness(capsys):
    """--post-command is appended after the harness invocation in the remote command."""
    from validation.run_on_hardware import main

    assert (
        main(
            [
                "--instance",
                "trn2.3xlarge",
                "--region",
                "sa-east-1",
                "--example",
                "ner_biomedical",
                "--launcher",
                "awscli",
                "--image-id",
                "ami-0test",
                "--post-command",
                "echo POSTCHECK",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.index("validation.run_on_hardware --in-instance") < out.index(
        "POSTCHECK"
    )


def test_capture_environment_is_offline_safe():
    """capture_environment must never raise off-hardware (no EC2, maybe no git)."""
    r = capture_environment(
        ValidationResult(example="x", status="skipped"), clock="2026-01-01T00:00:00Z"
    )
    assert r.timestamp == "2026-01-01T00:00:00Z"
    assert isinstance(r.versions, dict)
