"""Pytest configuration and fixtures"""

import os
import shutil
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_aws_credentials() -> None:
    """Mock AWS credentials for testing"""
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
            "AWS_SECURITY_TOKEN": "testing",
            "AWS_SESSION_TOKEN": "testing",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    ):
        yield


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client"""
    with patch("boto3.client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing"""
    return {
        "instance_type": "trn1.2xlarge",
        "max_hours": 2,
        "budget_limit": 100,
        "email": "test@example.com",
        "region": "us-east-1",
        "experiment_name": "test-experiment",
    }


@pytest.fixture
def mock_neuron_available():
    """Mock Neuron SDK availability"""
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.return_value = MagicMock()
        yield


@pytest.fixture
def mock_torch():
    """Mock PyTorch"""
    with patch("torch.randn") as mock_randn:
        mock_randn.return_value = MagicMock()
        yield mock_randn


@pytest.fixture
def sample_smiles_data():
    """Sample SMILES data for molecular testing"""
    return [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]


@pytest.fixture
def sample_protein_sequences():
    """Sample protein sequences for testing"""
    return [
        "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPVNGFNLSALFLIRKMFPEVKEKCSAAPDPSIMVGFHVICDNHQPEVKDKCTKHMGFHYQLICNQDTYKDLFECDTPPVLRVSRSEKTSCDQDMDKQRAVCEKAGSKGSLSRMSKCCDIQTIQGICDSTHLCDKEQSDQTSCPACPNGSFNSRKSGTLRYMDCNRQQDKLQDLQAKLAKVCDNNKSCDFLTHQCLCGQPPQGCKPQTASVDKKDLQDHQACCDVCSEYQCKCKRTNQCCLDGSGHMCGGPLPQPGPQFDYQCSCVFPKTKDTACSSGPVCPKTFGGRKVLVHCKCKDLQQCLPYCADPKDVQCR",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ]


@pytest.fixture
def sample_climate_data():
    """Sample climate data for testing"""
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd

    # Generate sample climate data
    dates = [datetime.now() - timedelta(days=i) for i in range(365, 0, -1)]
    np.random.seed(42)  # For reproducible tests

    data = {
        "date": dates,
        "temperature": np.random.normal(20, 10, 365),
        "humidity": np.random.uniform(30, 90, 365),
        "pressure": np.random.normal(1013, 20, 365),
        "wind_speed": np.random.exponential(5, 365),
        "precipitation": np.random.exponential(2, 365),
        "cloud_cover": np.random.uniform(0, 100, 365),
        "solar_radiation": np.random.uniform(100, 1000, 365),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_social_media_data():
    """Sample social media data for testing"""
    import pandas as pd

    data = {
        "text": [
            "I love this new technology! It's amazing.",
            "This is terrible and makes me angry.",
            "Climate change is a serious issue we must address.",
            "Great work on the latest research findings.",
            "This misinformation needs to be stopped immediately.",
        ],
        "sentiment": [2, 0, 1, 2, 0],  # 0=negative, 1=neutral, 2=positive
        "toxicity": [0.1, 0.8, 0.2, 0.1, 0.7],
        "emotion": [5, 1, 3, 5, 1],  # 0-7 emotion classes
        "political_stance": [2, 1, 1, 2, 0],  # 0=liberal, 1=conservative, 2=neutral
        "misinformation": [0.1, 0.2, 0.1, 0.1, 0.9],
        "urgency": [0.2, 0.8, 0.7, 0.3, 0.9],
    }

    return pd.DataFrame(data)


@pytest.fixture
def mock_cost_explorer_response():
    """Mock AWS Cost Explorer response"""
    return {
        "ResultsByTime": [
            {
                "TimePeriod": {"Start": "2025-01-01", "End": "2025-01-02"},
                "Groups": [
                    {
                        "Keys": ["trn1.2xlarge"],
                        "Metrics": {
                            "UnblendedCost": {"Amount": "10.50", "Unit": "USD"}
                        },
                    }
                ],
            }
        ]
    }
