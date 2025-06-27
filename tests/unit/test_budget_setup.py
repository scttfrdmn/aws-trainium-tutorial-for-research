"""Unit tests for budget setup script"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

from setup_budget import create_research_budget


@pytest.mark.unit
def test_create_research_budget_success(mock_aws_credentials, mock_boto3_client):
    """Test successful budget creation"""
    # Setup mock responses
    mock_boto3_client.get_caller_identity.return_value = {"Account": "123456789012"}
    mock_boto3_client.create_budget.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200}
    }

    with patch("boto3.client") as mock_client:
        mock_client.return_value = mock_boto3_client

        result = create_research_budget(monthly_limit=500, email="test@example.com")

        # Verify budget creation was called
        mock_boto3_client.create_budget.assert_called_once()
        call_args = mock_boto3_client.create_budget.call_args[1]

        assert call_args["Budget"]["BudgetName"] == "ML-Research-Budget"
        assert call_args["Budget"]["BudgetLimit"]["Amount"] == "500"
        assert call_args["Budget"]["BudgetLimit"]["Unit"] == "USD"
        assert len(call_args["NotificationsWithSubscribers"]) == 3

        # Check notification thresholds
        thresholds = [
            notif["Notification"]["Threshold"]
            for notif in call_args["NotificationsWithSubscribers"]
        ]
        assert set(thresholds) == {50, 80, 100}


@pytest.mark.unit
def test_create_research_budget_failure(mock_aws_credentials, mock_boto3_client):
    """Test budget creation failure handling"""
    mock_boto3_client.get_caller_identity.return_value = {"Account": "123456789012"}
    mock_boto3_client.create_budget.side_effect = Exception("Access denied")

    with patch("boto3.client") as mock_client:
        mock_client.return_value = mock_boto3_client

        result = create_research_budget(monthly_limit=100, email="test@example.com")

        assert result is None


@pytest.mark.unit
def test_create_research_budget_default_values(mock_aws_credentials, mock_boto3_client):
    """Test budget creation with default values"""
    mock_boto3_client.get_caller_identity.return_value = {"Account": "123456789012"}
    mock_boto3_client.create_budget.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200}
    }

    with patch("boto3.client") as mock_client:
        mock_client.return_value = mock_boto3_client

        result = create_research_budget()

        call_args = mock_boto3_client.create_budget.call_args[1]
        assert call_args["Budget"]["BudgetLimit"]["Amount"] == "100"

        # Check default email in subscriber
        subscribers = call_args["NotificationsWithSubscribers"][0]["Subscribers"]
        assert subscribers[0]["Address"] == "your-email@university.edu"


@pytest.mark.unit
def test_create_research_budget_cost_filters(mock_aws_credentials, mock_boto3_client):
    """Test that budget has correct cost filters"""
    mock_boto3_client.get_caller_identity.return_value = {"Account": "123456789012"}
    mock_boto3_client.create_budget.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200}
    }

    with patch("boto3.client") as mock_client:
        mock_client.return_value = mock_boto3_client

        create_research_budget()

        call_args = mock_boto3_client.create_budget.call_args[1]
        cost_filters = call_args["Budget"]["CostFilters"]

        assert "Service" in cost_filters
        assert "Amazon Elastic Compute Cloud - Compute" in cost_filters["Service"]
