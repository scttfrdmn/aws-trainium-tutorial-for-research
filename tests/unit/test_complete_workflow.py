"""Unit tests for the train-on-Trainium / serve-on-Inferentia workflow example.

These exercise only the generated script strings and the dry-run CLI; nothing touches AWS or
torch. ``boto3.client`` is stubbed by the autouse fixture in ``tests/conftest.py``.
"""

import ast
import sys
from unittest.mock import MagicMock, patch

import pytest
from complete_workflow import trainium_to_inferentia_pipeline as twf


@pytest.fixture
def pipeline() -> twf.TrainiumToInferentiaPipeline:
    """Construct the pipeline with stubbed boto3 clients."""
    return twf.TrainiumToInferentiaPipeline("unit-test", "unit-test-bucket")


@pytest.mark.unit
def test_training_script_parses(pipeline):
    """The generated training script is valid Python."""
    script = pipeline._generate_training_script(
        "AutoModelForSequenceClassification", "s3://bucket/data.tar.gz", {}
    )
    ast.parse(script)


@pytest.mark.unit
def test_inference_script_parses(pipeline):
    """The generated inference server script is valid Python."""
    ast.parse(pipeline._generate_inference_script("s3://bucket/model.pt", {}))


@pytest.mark.unit
def test_inference_script_matches_traced_graph(pipeline):
    """Server pads to the traced (1, 512) shape and indexes the traced tuple output."""
    script = pipeline._generate_inference_script("s3://bucket/model.pt", {})
    assert 'padding="max_length"' in script
    assert "max_length=512" in script
    assert "truncation=True" in script
    assert "outputs[0]" in script
    assert "padding=True" not in script
    assert ".logits" not in script


@pytest.mark.unit
def test_inference_script_rate_injected_from_table(pipeline):
    """The server's $/hr comes from ILLUSTRATIVE_HOURLY_USD, not a hardcoded literal."""
    script = pipeline._generate_inference_script("s3://bucket/model.pt", {})
    assert "__INF2_HOURLY_USD__" not in script
    assert "0.227" not in script.replace(
        repr(twf.ILLUSTRATIVE_HOURLY_USD["inf2.xlarge"]), ""
    )
    tree = ast.parse(script)
    rates = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "HOURLY_RATE_USD" for t in node.targets
        )
    ]
    assert len(rates) == 1
    assert ast.literal_eval(rates[0]) == twf.ILLUSTRATIVE_HOURLY_USD["inf2.xlarge"]


@pytest.mark.unit
def test_inference_monitor_rate_from_table(pipeline):
    """The inference cost monitor also uses the table's inf2.xlarge rate."""
    script = pipeline._get_inference_monitor_script()
    ast.parse(script)
    rate = twf.ILLUSTRATIVE_HOURLY_USD["inf2.xlarge"]
    assert f"self.hourly_rate = {rate!r}" in script


@pytest.mark.unit
def test_dry_run_main_launches_nothing(capsys):
    """main() with no flags is a dry run: no instances, no Trn2 trace-serving claim."""
    ec2 = MagicMock()
    with (
        patch("boto3.client", return_value=ec2),
        patch.object(sys, "argv", ["trainium_to_inferentia_pipeline.py"]),
    ):
        assert twf.main() is None
    ec2.run_instances.assert_not_called()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "Inf2/Trn1" in out
    assert "vLLM" not in out
    assert "Trn2/Trn3 serving use" not in out
