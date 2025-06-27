"""Unit tests for climate science example"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

# Add examples to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../examples/domain_specific")
)

# Mock torch_xla and torch_neuronx before importing
sys.modules["torch_xla"] = MagicMock()
sys.modules["torch_xla.core"] = MagicMock()
sys.modules["torch_xla.core.xla_model"] = MagicMock()
sys.modules["torch_neuronx"] = MagicMock()

from domain_specific_examples import ClimateDataset, ClimateTransformer


@pytest.mark.unit
def test_climate_dataset_initialization(sample_climate_data, temp_dir):
    """Test ClimateDataset initialization"""
    # Save sample data to temporary file
    data_path = os.path.join(temp_dir, "climate_data.csv")
    sample_climate_data.to_csv(data_path, index=False)

    dataset = ClimateDataset(
        data_path=data_path, sequence_length=10, prediction_horizon=5
    )

    assert len(dataset.features) > 0
    assert "temperature" in dataset.features
    assert "day_of_year" in dataset.features
    assert "month" in dataset.features
    assert dataset.sequence_length == 10
    assert dataset.prediction_horizon == 5


@pytest.mark.unit
def test_climate_dataset_length(sample_climate_data, temp_dir):
    """Test ClimateDataset length calculation"""
    data_path = os.path.join(temp_dir, "climate_data.csv")
    sample_climate_data.to_csv(data_path, index=False)

    dataset = ClimateDataset(
        data_path=data_path, sequence_length=30, prediction_horizon=7
    )

    expected_length = len(sample_climate_data) - 30 - 7
    assert len(dataset) == expected_length


@pytest.mark.unit
def test_climate_dataset_getitem(sample_climate_data, temp_dir):
    """Test ClimateDataset item retrieval"""
    data_path = os.path.join(temp_dir, "climate_data.csv")
    sample_climate_data.to_csv(data_path, index=False)

    dataset = ClimateDataset(
        data_path=data_path, sequence_length=10, prediction_horizon=5
    )

    sequence, target = dataset[0]

    # Check tensor shapes
    assert sequence.shape == (10, len(dataset.features))
    assert target.shape == (5,)

    # Check tensor types
    assert isinstance(sequence, torch.FloatTensor)
    assert isinstance(target, torch.FloatTensor)


@pytest.mark.unit
def test_climate_dataset_normalization(sample_climate_data, temp_dir):
    """Test feature normalization in ClimateDataset"""
    data_path = os.path.join(temp_dir, "climate_data.csv")
    sample_climate_data.to_csv(data_path, index=False)

    dataset = ClimateDataset(data_path=data_path)

    # Check that normalization statistics are stored
    assert "temperature" in dataset.means
    assert "temperature" in dataset.stds
    assert dataset.stds["temperature"] > 0  # Should have non-zero std


@pytest.mark.unit
def test_climate_transformer_initialization():
    """Test ClimateTransformer model initialization"""
    model = ClimateTransformer(
        input_dim=9,
        d_model=256,
        nhead=8,
        num_layers=4,
        sequence_length=30,
        prediction_horizon=7,
    )

    assert model.input_dim == 9
    assert model.d_model == 256
    assert model.prediction_horizon == 7

    # Check that all prediction heads exist
    expected_heads = ["mean", "variance", "trend"]
    for head in expected_heads:
        assert head in model.prediction_heads


@pytest.mark.unit
def test_climate_transformer_forward():
    """Test ClimateTransformer forward pass"""
    model = ClimateTransformer(
        input_dim=9,
        d_model=128,
        nhead=4,
        num_layers=2,
        sequence_length=10,
        prediction_horizon=5,
    )

    # Create sample input
    batch_size = 2
    sequence_length = 10
    input_dim = 9

    x = torch.randn(batch_size, sequence_length, input_dim)

    with torch.no_grad():
        predictions = model(x)

    # Check output structure
    expected_heads = ["mean", "variance", "trend"]
    for head in expected_heads:
        assert head in predictions
        assert predictions[head].shape == (batch_size, 5)  # prediction_horizon = 5

    # Check that variance outputs are positive
    assert torch.all(predictions["variance"] > 0)


@pytest.mark.unit
def test_climate_transformer_different_sequence_lengths():
    """Test ClimateTransformer with different sequence lengths"""
    model = ClimateTransformer(
        input_dim=9,
        d_model=128,
        nhead=4,
        num_layers=2,
        sequence_length=20,
        prediction_horizon=5,
    )

    # Test with shorter sequence (should still work due to positional encoding)
    x_short = torch.randn(1, 15, 9)

    with torch.no_grad():
        predictions = model(x_short)

    assert predictions["mean"].shape == (1, 5)


@pytest.mark.unit
def test_climate_model_parameter_count():
    """Test that model has reasonable parameter count"""
    model = ClimateTransformer(input_dim=9, d_model=256, nhead=8, num_layers=4)

    param_count = sum(p.numel() for p in model.parameters())

    # Should have reasonable number of parameters (not too small, not too large)
    assert 100_000 < param_count < 10_000_000


@pytest.mark.unit
def test_climate_inference_service_mock():
    """Test ClimateInferenceService initialization (mocked)"""
    from domain_specific_examples import ClimateInferenceService

    # Mock the model loading
    with patch("torch.jit.load") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        feature_stats = {
            "means": {"temperature": 20.0},
            "stds": {"temperature": 10.0},
            "features": ["temperature", "humidity", "pressure"],
        }

        service = ClimateInferenceService("mock_model.pt", feature_stats)

        assert service.model == mock_model
        assert service.feature_stats == feature_stats
        assert service.request_count == 0


@pytest.mark.unit
def test_temporal_features_generation(sample_climate_data, temp_dir):
    """Test that temporal features are correctly generated"""
    data_path = os.path.join(temp_dir, "climate_data.csv")
    sample_climate_data.to_csv(data_path, index=False)

    dataset = ClimateDataset(data_path=data_path)

    # Check that temporal features were added
    assert "day_of_year" in dataset.features
    assert "month" in dataset.features

    # Check that day_of_year is normalized to [0, 1]
    day_of_year_values = dataset.data["day_of_year"].values
    assert np.all(day_of_year_values >= 0)
    assert np.all(day_of_year_values <= 1)

    # Check that month is normalized to [0, 1]
    month_values = dataset.data["month"].values
    assert np.all(month_values >= 0)
    assert np.all(month_values <= 1)
