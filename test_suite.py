# tests/conftest.py
"""Pytest configuration and fixtures for AWS Trainium/Inferentia tutorial tests"""

import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch


@pytest.fixture
def mock_aws_config():
    """Mock AWS configuration for testing without actual AWS calls"""
    with patch("boto3.client") as mock_client:
        mock_ec2 = Mock()
        mock_s3 = Mock()
        mock_ce = Mock()

        # Configure mock responses
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-1234567890abcdef0",
                            "InstanceType": "trn1.2xlarge",
                            "State": {"Name": "running"},
                            "LaunchTime": "2025-01-01T00:00:00Z",
                            "Tags": [{"Key": "Name", "Value": "test-instance"}],
                        }
                    ]
                }
            ]
        }

        mock_ce.get_cost_and_usage.return_value = {
            "ResultsByTime": [
                {
                    "TimePeriod": {"Start": "2025-01-01", "End": "2025-01-02"},
                    "Groups": [
                        {
                            "Keys": ["trn1.2xlarge"],
                            "Metrics": {"UnblendedCost": {"Amount": "10.50"}},
                        }
                    ],
                }
            ]
        }

        def client_factory(service_name, **kwargs):
            if service_name == "ec2":
                return mock_ec2
            elif service_name == "s3":
                return mock_s3
            elif service_name == "ce":
                return mock_ce
            else:
                return Mock()

        mock_client.side_effect = client_factory
        yield mock_client


@pytest.fixture
def sample_climate_data():
    """Generate sample climate data for testing"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")

    data = {
        "date": dates,
        "temperature": 20
        + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        + np.random.normal(0, 2, len(dates)),
        "humidity": 50 + 20 * np.random.random(len(dates)),
        "pressure": 1013 + 10 * np.random.random(len(dates)),
        "wind_speed": 5 + 10 * np.random.random(len(dates)),
        "precipitation": np.random.exponential(2, len(dates)),
        "cloud_cover": 50 * np.random.random(len(dates)),
        "solar_radiation": 200
        + 100
        * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        * np.random.random(len(dates)),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_molecular_data():
    """Generate sample molecular data for testing"""
    smiles = [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "C1=CC=C(C=C1)C2=CC=CC=C2",  # Biphenyl
    ]

    properties = [
        [0.5, 0.1, 0.8, 0.9],  # [solubility, toxicity, bioavailability, drug_likeness]
        [0.3, 0.2, 0.7, 0.8],
        [0.4, 0.1, 0.9, 0.6],
        [0.2, 0.15, 0.85, 0.7],
        [0.1, 0.3, 0.4, 0.2],
    ]

    return smiles, properties


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_neuron_device():
    """Mock Neuron device for testing without actual hardware"""
    with patch("torch_xla.core.xla_model.xla_device") as mock_device:
        mock_device.return_value = torch.device("cpu")  # Use CPU for testing
        yield mock_device


@pytest.fixture
def mock_neuron_trace():
    """Mock Neuron model tracing for testing"""
    with patch("torch_neuronx.trace") as mock_trace:

        def trace_side_effect(model, inputs, **kwargs):
            # Return the original model for testing
            return model

        mock_trace.side_effect = trace_side_effect
        yield mock_trace


# tests/test_core.py
"""Tests for core functionality"""

from unittest.mock import Mock, patch

import pytest
import torch

from aws_trainium_tutorial.core import (
    CostTracker,
    EphemeralMLInstance,
    NeuronMigrationHelper,
)


class TestEphemeralMLInstance:
    """Test ephemeral ML instance management"""

    def test_initialization(self):
        instance = EphemeralMLInstance(instance_type="trn1.2xlarge", max_hours=4)
        assert instance.instance_type == "trn1.2xlarge"
        assert instance.max_hours == 4
        assert instance.instance_id is None

    def test_get_hourly_rate(self):
        instance = EphemeralMLInstance()
        rate = instance.get_hourly_rate()
        assert rate > 0
        assert isinstance(rate, float)

    @patch("boto3.client")
    def test_launch_instance(self, mock_boto_client):
        # Mock EC2 client
        mock_ec2 = Mock()
        mock_ec2.run_instances.return_value = {
            "Instances": [{"InstanceId": "i-1234567890abcdef0"}]
        }
        mock_boto_client.return_value = mock_ec2

        instance = EphemeralMLInstance(instance_type="trn1.2xlarge", max_hours=2)
        instance_id = instance.launch("test-experiment")

        assert instance_id == "i-1234567890abcdef0"
        assert instance.instance_id == "i-1234567890abcdef0"
        mock_ec2.run_instances.assert_called_once()


class TestCostTracker:
    """Test cost tracking functionality"""

    def test_initialization(self):
        tracker = CostTracker(experiment_name="test-experiment")
        assert tracker.experiment_name == "test-experiment"
        assert len(tracker.cost_events) == 0

    def test_log_cost_event(self):
        tracker = CostTracker(experiment_name="test")

        event = tracker.log_cost_event(
            event_type="training_start",
            cost=0.40,
            metadata={"instance_type": "trn1.2xlarge"},
        )

        assert len(tracker.cost_events) == 1
        assert event["event_type"] == "training_start"
        assert event["cost"] == 0.40
        assert event["metadata"]["instance_type"] == "trn1.2xlarge"

    def test_get_total_cost(self):
        tracker = CostTracker(experiment_name="test")

        tracker.log_cost_event("event1", 10.0)
        tracker.log_cost_event("event2", 5.50)
        tracker.log_cost_event("event3", 2.25)

        total = tracker.get_total_cost()
        assert total == 17.75


class TestNeuronMigrationHelper:
    """Test CUDA to Neuron migration utilities"""

    def test_initialization(self, mock_neuron_device):
        helper = NeuronMigrationHelper()
        assert helper.device is not None

    def test_get_device(self, mock_neuron_device):
        helper = NeuronMigrationHelper()
        device = helper.get_device()
        assert device is not None

    def test_migrate_model(self, mock_neuron_device, mock_neuron_trace):
        helper = NeuronMigrationHelper()

        # Create simple test model
        model = torch.nn.Linear(10, 5)

        # Test migration
        migrated_model = helper.migrate_model(model, compile_for_neuron=True)
        assert migrated_model is not None


# tests/test_examples.py
"""Tests for domain-specific examples"""

import pandas as pd
import pytest
import torch

from aws_trainium_tutorial.examples import ClimateTransformer, MolecularTransformer


class TestClimateTransformer:
    """Test climate science transformer model"""

    def test_model_initialization(self):
        model = ClimateTransformer(
            input_dim=9,
            d_model=256,
            nhead=4,
            num_layers=2,
            sequence_length=365,
            prediction_horizon=30,
        )
        assert model is not None
        assert model.input_dim == 9
        assert model.d_model == 256

    def test_forward_pass(self):
        model = ClimateTransformer(
            input_dim=9,
            d_model=128,
            nhead=4,
            num_layers=2,
            sequence_length=100,
            prediction_horizon=10,
        )

        # Create test input
        batch_size = 2
        seq_len = 100
        input_dim = 9

        x = torch.randn(batch_size, seq_len, input_dim)

        # Forward pass
        predictions = model(x)

        assert "mean" in predictions
        assert "variance" in predictions
        assert "trend" in predictions

        assert predictions["mean"].shape == (batch_size, 10)
        assert predictions["variance"].shape == (batch_size, 10)
        assert predictions["trend"].shape == (batch_size, 10)


class TestMolecularTransformer:
    """Test molecular property prediction model"""

    def test_model_initialization(self):
        model = MolecularTransformer(vocab_size=20, d_model=256, nhead=4, num_layers=2)
        assert model is not None

    def test_forward_pass(self):
        model = MolecularTransformer(
            vocab_size=20, d_model=128, nhead=4, num_layers=2, max_length=100
        )

        batch_size = 2
        seq_len = 100
        descriptor_dim = 6

        sequence = torch.randint(0, 20, (batch_size, seq_len))
        descriptors = torch.randn(batch_size, descriptor_dim)

        predictions, attention_weights = model(sequence, descriptors)

        assert "solubility" in predictions
        assert "toxicity" in predictions
        assert "bioavailability" in predictions
        assert "drug_likeness" in predictions

        for prop_name, pred in predictions.items():
            assert pred.shape == (batch_size, 1)

        assert attention_weights is not None


# tests/test_cost_estimation.py
"""Tests for cost estimation utilities"""

import pytest

from aws_trainium_tutorial.utils import (
    estimate_training_cost,
    get_optimal_instance_type,
)


class TestCostEstimation:
    """Test cost estimation functions"""

    def test_estimate_training_cost_basic(self):
        estimate = estimate_training_cost(
            model_size_params=110_000_000,  # BERT-base
            dataset_size_samples=50_000,
            epochs=3,
            instance_type="trn1.2xlarge",
            use_spot=True,
        )

        assert "estimated_cost_usd" in estimate
        assert "estimated_time_hours" in estimate
        assert "gpu_comparisons" in estimate
        assert "recommendations" in estimate

        assert estimate["estimated_cost_usd"] > 0
        assert estimate["estimated_time_hours"] > 0
        assert len(estimate["gpu_comparisons"]) > 0

    def test_estimate_training_cost_large_model(self):
        estimate = estimate_training_cost(
            model_size_params=7_000_000_000,  # 7B model
            dataset_size_samples=1_000_000,
            epochs=1,
            instance_type="trn2.48xlarge",
            use_spot=True,
        )

        # Large model should cost more
        assert estimate["estimated_cost_usd"] > 50

        # Should show significant savings vs GPU
        gpu_savings = estimate["gpu_comparisons"]["p5.48xlarge"]["savings_vs_gpu"]
        assert gpu_savings > 100

    def test_get_optimal_instance_type(self):
        recommendations = get_optimal_instance_type(
            model_size_params=1_000_000_000,  # 1B parameters
            budget_usd=100,
            time_constraint_hours=24,
        )

        assert len(recommendations) > 0

        # First recommendation should meet constraints
        best_rec = recommendations[0]
        assert best_rec["meets_constraints"] == True
        assert best_rec["estimated_cost"] <= 100
        assert best_rec["estimated_time"] <= 24

    def test_get_optimal_instance_type_no_constraints(self):
        recommendations = get_optimal_instance_type(
            model_size_params=500_000_000,
            budget_usd=10000,  # Large budget
            time_constraint_hours=None,
        )

        assert len(recommendations) > 0

        # All recommendations should meet budget
        for rec in recommendations:
            if rec["meets_constraints"]:
                assert rec["estimated_cost"] <= 10000


# tests/test_integration.py
"""Integration tests that require more setup"""

import tempfile

import pandas as pd
import pytest

from aws_trainium_tutorial.examples.domain_specific.climate_science import (
    ClimateDataset,
)


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_climate_dataset_creation(self, sample_climate_data, temp_directory):
        # Save sample data to temporary file
        data_path = f"{temp_directory}/climate_data.csv"
        sample_climate_data.to_csv(data_path, index=False)

        # Create dataset
        dataset = ClimateDataset(
            data_path=data_path, sequence_length=30, prediction_horizon=7
        )

        assert len(dataset) > 0

        # Test getting an item
        sequence, target = dataset[0]
        assert sequence.shape == (30, len(dataset.features))
        assert target.shape == (7,)

    @pytest.mark.slow
    def test_model_training_simulation(self, mock_neuron_device, mock_neuron_trace):
        """Simulate model training without actual AWS resources"""

        # This would be a longer integration test
        # that simulates the entire training pipeline

        from aws_trainium_tutorial.examples import ClimateTransformer

        # Create small model for testing
        model = ClimateTransformer(
            input_dim=7,
            d_model=64,
            nhead=2,
            num_layers=1,
            sequence_length=10,
            prediction_horizon=3,
        )

        # Simulate training data
        batch_size = 2
        x = torch.randn(batch_size, 10, 7)
        y = torch.randn(batch_size, 3)

        # Forward pass
        predictions = model(x)

        # Simulate loss calculation
        loss = torch.nn.functional.mse_loss(predictions["mean"], y)

        # Backward pass should work
        loss.backward()

        assert loss.item() > 0

    def test_cost_tracking_integration(self, mock_aws_config):
        """Test cost tracking with mocked AWS services"""

        from aws_trainium_tutorial.core import CostTracker

        tracker = CostTracker(experiment_name="integration-test")

        # Log several events
        tracker.log_cost_event("training_start", 0.0)
        tracker.log_cost_event("epoch_1", 0.40)
        tracker.log_cost_event("epoch_2", 0.40)
        tracker.log_cost_event("training_complete", 0.0)

        # Generate report
        report = tracker.generate_cost_report()

        assert "total_cost" in report
        assert "experiment_name" in report
        assert report["total_cost"] == 0.80


# tests/test_benchmarks.py
"""Tests for benchmarking and performance measurement"""

import time

import pytest
import torch

from aws_trainium_tutorial.advanced import NeuronOptimizedAttention


class TestBenchmarks:
    """Test benchmarking functionality"""

    def test_attention_benchmark(self):
        """Test attention mechanism benchmarking"""

        # Small model for testing
        attention = NeuronOptimizedAttention(d_model=128, num_heads=4)

        batch_size = 2
        seq_len = 64
        x = torch.randn(batch_size, seq_len, 128)

        # Time the forward pass
        start_time = time.time()
        output = attention(x)
        end_time = time.time()

        latency = end_time - start_time

        # Basic assertions
        assert output.shape == x.shape
        assert latency > 0
        assert latency < 1.0  # Should be fast for small input

    def test_memory_usage_estimation(self):
        """Test memory usage estimation"""

        from aws_trainium_tutorial.utils.cost_estimation import estimate_training_cost

        # Small model memory usage
        small_estimate = estimate_training_cost(
            model_size_params=100_000,
            dataset_size_samples=1000,
            epochs=1,
            instance_type="trn1.2xlarge",
        )

        # Large model memory usage
        large_estimate = estimate_training_cost(
            model_size_params=1_000_000_000,
            dataset_size_samples=1000,
            epochs=1,
            instance_type="trn1.2xlarge",
        )

        # Large model should take longer and cost more
        assert (
            large_estimate["estimated_time_hours"]
            > small_estimate["estimated_time_hours"]
        )
        assert (
            large_estimate["estimated_cost_usd"] > small_estimate["estimated_cost_usd"]
        )


# tests/test_utils.py
"""Tests for utility functions"""

import json

import pytest

from aws_trainium_tutorial.utils import setup_aws_environment


class TestUtils:
    """Test utility functions"""

    @pytest.mark.integration
    def test_setup_aws_environment(self, mock_aws_config):
        """Test AWS environment setup"""

        # This would test the AWS setup process
        # In a real scenario, this would create IAM roles, S3 buckets, etc.

        result = setup_aws_environment()

        # In mock mode, this should not fail
        assert result is not None

    def test_configuration_validation(self):
        """Test configuration file validation"""

        valid_config = {
            "experiment_name": "test-experiment",
            "instance_type": "trn1.2xlarge",
            "max_hours": 4,
            "batch_size": 32,
            "epochs": 10,
        }

        # This should not raise any exceptions
        assert valid_config["instance_type"] in [
            "trn1.2xlarge",
            "trn1.32xlarge",
            "trn2.48xlarge",
        ]
        assert valid_config["max_hours"] > 0
        assert valid_config["batch_size"] > 0
        assert valid_config["epochs"] > 0
