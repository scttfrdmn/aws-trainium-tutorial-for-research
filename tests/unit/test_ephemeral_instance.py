"""Unit tests for ephemeral instance management"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

from ephemeral_instance import EphemeralMLInstance


@pytest.mark.unit
def test_ephemeral_instance_initialization():
    """Test EphemeralMLInstance initialization"""
    instance = EphemeralMLInstance(instance_type="trn1.32xlarge", max_hours=6)

    assert instance.instance_type == "trn1.32xlarge"
    assert instance.max_hours == 6
    assert instance.instance_id is None


@pytest.mark.unit
def test_get_hourly_rate():
    """Test hourly rate calculation"""
    instance = EphemeralMLInstance()

    # Test known instance types
    assert instance.get_hourly_rate() == 0.40  # Default trn1.2xlarge

    instance.instance_type = "trn1.32xlarge"
    assert instance.get_hourly_rate() == 6.45

    instance.instance_type = "inf2.xlarge"
    assert instance.get_hourly_rate() == 0.227

    # Test unknown instance type
    instance.instance_type = "unknown.type"
    assert instance.get_hourly_rate() == 1.0


@pytest.mark.unit
def test_launch_instance_success(mock_aws_credentials, mock_boto3_client):
    """Test successful instance launch"""
    # Setup mock response
    mock_response = {
        "Instances": [
            {
                "InstanceId": "i-1234567890abcdef0",
                "State": {"Name": "pending"},
            }
        ]
    }
    mock_boto3_client.run_instances.return_value = mock_response

    with patch("boto3.client") as mock_client:
        mock_client.return_value = mock_boto3_client

        instance = EphemeralMLInstance(instance_type="trn1.2xlarge", max_hours=4)
        instance_id = instance.launch("test-experiment")

        assert instance_id == "i-1234567890abcdef0"
        assert instance.instance_id == "i-1234567890abcdef0"

        # Verify run_instances was called with correct parameters
        call_args = mock_boto3_client.run_instances.call_args[1]
        assert call_args["InstanceType"] == "trn1.2xlarge"
        assert call_args["MinCount"] == 1
        assert call_args["MaxCount"] == 1

        # Check tags
        tags = call_args["TagSpecifications"][0]["Tags"]
        tag_dict = {tag["Key"]: tag["Value"] for tag in tags}
        assert tag_dict["Name"] == "ML-Experiment-test-experiment"
        assert tag_dict["AutoTerminate"] == "true"
        assert tag_dict["MaxHours"] == "4"

        # Check spot instance configuration
        assert call_args["InstanceMarketOptions"]["MarketType"] == "spot"


@pytest.mark.unit
def test_launch_instance_user_data_generation():
    """Test user data script generation"""
    with patch("boto3.client") as mock_client:
        mock_boto3_client = MagicMock()
        mock_response = {"Instances": [{"InstanceId": "i-1234567890abcdef0"}]}
        mock_boto3_client.run_instances.return_value = mock_response
        mock_client.return_value = mock_boto3_client

        instance = EphemeralMLInstance(max_hours=2)
        instance.launch("test-exp")

        call_args = mock_boto3_client.run_instances.call_args[1]
        user_data = call_args["UserData"]

        # Check that user data contains expected elements
        assert "test-exp" in user_data
        assert "2 hours" in user_data
        assert "torch-neuronx" in user_data
        assert "pip install" in user_data


@pytest.mark.unit
def test_get_neuron_ami():
    """Test Neuron AMI selection"""
    instance = EphemeralMLInstance()
    ami_id = instance.get_neuron_ami()

    # Should return a valid AMI ID format
    assert ami_id.startswith("ami-")
    assert len(ami_id) == 21  # Standard AMI ID length


@pytest.mark.unit
def test_instance_pricing_coverage():
    """Test that all expected instance types have pricing"""
    instance = EphemeralMLInstance()

    expected_types = [
        "trn1.2xlarge",
        "trn1.32xlarge",
        "trn2.48xlarge",
        "inf2.xlarge",
        "inf2.48xlarge",
    ]

    for instance_type in expected_types:
        instance.instance_type = instance_type
        rate = instance.get_hourly_rate()
        assert rate > 0, f"No pricing for {instance_type}"
