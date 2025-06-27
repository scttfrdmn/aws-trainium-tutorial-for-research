"""Integration tests for cost tracking functionality"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))


@pytest.mark.integration
@pytest.mark.aws
def test_full_cost_tracking_workflow(mock_aws_credentials):
    """Test complete cost tracking workflow"""
    from cost_monitor import generate_cost_report
    from ephemeral_instance import EphemeralMLInstance
    from setup_budget import create_research_budget

    # Mock all AWS calls
    with patch("boto3.client") as mock_client:
        mock_budgets = MagicMock()
        mock_ce = MagicMock()
        mock_ec2 = MagicMock()

        # Configure different clients
        def client_side_effect(service_name, **kwargs):
            if service_name == "budgets":
                return mock_budgets
            elif service_name == "ce":
                return mock_ce
            elif service_name == "ec2":
                return mock_ec2
            return MagicMock()

        mock_client.side_effect = client_side_effect

        # Configure mock responses
        mock_budgets.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_budgets.create_budget.return_value = {
            "ResponseMetadata": {"HTTPStatusCode": 200}
        }

        mock_ce.get_cost_and_usage.return_value = {
            "ResultsByTime": [
                {
                    "TimePeriod": {"Start": "2025-01-01", "End": "2025-01-02"},
                    "Groups": [
                        {
                            "Keys": ["trn1.2xlarge"],
                            "Metrics": {
                                "UnblendedCost": {"Amount": "5.50", "Unit": "USD"}
                            },
                        }
                    ],
                }
            ]
        }

        mock_ec2.run_instances.return_value = {
            "Instances": [{"InstanceId": "i-1234567890abcdef0"}]
        }

        # Test the workflow
        # 1. Set up budget
        budget_result = create_research_budget(
            monthly_limit=200, email="test@example.com"
        )
        assert budget_result is not None

        # 2. Launch ephemeral instance
        instance = EphemeralMLInstance(instance_type="trn1.2xlarge", max_hours=2)
        instance_id = instance.launch("integration-test")
        assert instance_id == "i-1234567890abcdef0"

        # 3. Generate cost report
        with patch("matplotlib.pyplot.savefig"):
            generate_cost_report()  # Should not raise exceptions

        # Verify all components were called
        mock_budgets.create_budget.assert_called_once()
        mock_ec2.run_instances.assert_called_once()
        mock_ce.get_cost_and_usage.assert_called_once()


@pytest.mark.integration
def test_cost_estimation_accuracy():
    """Test cost estimation accuracy across different scenarios"""
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "../../examples/domain_specific")
    )

    # Mock the dependencies
    with patch.dict(
        sys.modules,
        {
            "torch_xla": MagicMock(),
            "torch_xla.core": MagicMock(),
            "torch_xla.core.xla_model": MagicMock(),
            "torch_neuronx": MagicMock(),
        },
    ):
        # Test different scenarios
        scenarios = [
            {
                "instance_type": "trn1.2xlarge",
                "hours": 1,
                "expected_min": 0.30,
                "expected_max": 0.50,
            },
            {
                "instance_type": "trn1.32xlarge",
                "hours": 2,
                "expected_min": 10.0,
                "expected_max": 15.0,
            },
            {
                "instance_type": "inf2.xlarge",
                "hours": 4,
                "expected_min": 0.8,
                "expected_max": 1.2,
            },
        ]

        from ephemeral_instance import EphemeralMLInstance

        for scenario in scenarios:
            instance = EphemeralMLInstance(
                instance_type=scenario["instance_type"], max_hours=scenario["hours"]
            )

            estimated_cost = instance.get_hourly_rate() * scenario["hours"]

            assert (
                scenario["expected_min"] <= estimated_cost <= scenario["expected_max"]
            ), f"Cost estimate {estimated_cost} not in range [{scenario['expected_min']}, {scenario['expected_max']}] for {scenario['instance_type']}"


@pytest.mark.integration
def test_emergency_shutdown_workflow(mock_aws_credentials):
    """Test emergency shutdown functionality"""
    from emergency_shutdown import emergency_shutdown_all

    with patch("boto3.client") as mock_client, patch(
        "builtins.input", return_value="EMERGENCY"
    ):
        mock_ec2 = MagicMock()
        mock_client.return_value = mock_ec2

        # Mock running instances with ML tags
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-ml-instance-1",
                            "Tags": [
                                {"Key": "Name", "Value": "ML-Experiment-test"},
                                {"Key": "AutoTerminate", "Value": "true"},
                            ],
                        },
                        {
                            "InstanceId": "i-ml-instance-2",
                            "Tags": [{"Key": "Name", "Value": "ML-Training-job"}],
                        },
                    ]
                }
            ]
        }

        # Run emergency shutdown
        emergency_shutdown_all()

        # Verify instances were terminated
        mock_ec2.terminate_instances.assert_called_once()
        call_args = mock_ec2.terminate_instances.call_args[1]
        assert "i-ml-instance-1" in call_args["InstanceIds"]
        assert "i-ml-instance-2" in call_args["InstanceIds"]


@pytest.mark.integration
@pytest.mark.slow
def test_model_training_cost_tracking():
    """Test cost tracking during model training simulation"""
    import time
    from datetime import datetime

    # Simulate a training session with cost tracking
    class MockCostTracker:
        def __init__(self):
            self.start_time = datetime.now()
            self.costs = []
            self.hourly_rate = 6.45  # trn1.32xlarge spot

        def log_epoch_cost(self, epoch):
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            current_cost = elapsed_hours * self.hourly_rate
            self.costs.append(
                {"epoch": epoch, "elapsed_hours": elapsed_hours, "cost": current_cost}
            )

        def get_total_cost(self):
            return self.costs[-1]["cost"] if self.costs else 0

    tracker = MockCostTracker()

    # Simulate training epochs
    for epoch in range(5):
        time.sleep(0.1)  # Simulate training time
        tracker.log_epoch_cost(epoch)

    # Verify cost tracking
    assert len(tracker.costs) == 5
    assert tracker.costs[0]["cost"] < tracker.costs[-1]["cost"]  # Cost should increase
    assert tracker.get_total_cost() > 0

    # Verify cost progression is reasonable
    for i in range(1, len(tracker.costs)):
        assert tracker.costs[i]["elapsed_hours"] > tracker.costs[i - 1]["elapsed_hours"]
        assert tracker.costs[i]["cost"] > tracker.costs[i - 1]["cost"]
