"""Unit tests for cost monitoring functionality"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

from cost_monitor import generate_cost_report


@pytest.mark.unit
def test_generate_cost_report_success(
    mock_aws_credentials, mock_cost_explorer_response
):
    """Test successful cost report generation"""
    with patch("boto3.client") as mock_client, patch(
        "matplotlib.pyplot.savefig"
    ) as mock_savefig, patch("matplotlib.pyplot.plot") as mock_plot:
        mock_ce = MagicMock()
        mock_ce.get_cost_and_usage.return_value = mock_cost_explorer_response
        mock_client.return_value = mock_ce

        # Should not raise any exceptions
        generate_cost_report()

        # Verify Cost Explorer was called
        mock_ce.get_cost_and_usage.assert_called_once()
        call_args = mock_ce.get_cost_and_usage.call_args[1]

        # Check that correct parameters were used
        assert call_args["Granularity"] == "DAILY"
        assert "UnblendedCost" in call_args["Metrics"]
        assert call_args["Filter"]["Dimensions"]["Key"] == "SERVICE"

        # Verify plot was saved
        mock_savefig.assert_called_once_with("ml_costs.png")


@pytest.mark.unit
def test_generate_cost_report_no_data(mock_aws_credentials):
    """Test cost report generation with no data"""
    empty_response = {"ResultsByTime": []}

    with patch("boto3.client") as mock_client, patch(
        "matplotlib.pyplot.savefig"
    ) as mock_savefig:
        mock_ce = MagicMock()
        mock_ce.get_cost_and_usage.return_value = empty_response
        mock_client.return_value = mock_ce

        # Should handle empty data gracefully
        generate_cost_report()

        mock_savefig.assert_called_once_with("ml_costs.png")


@pytest.mark.unit
def test_generate_cost_report_date_range(mock_aws_credentials):
    """Test that cost report uses correct date range"""
    with patch("boto3.client") as mock_client, patch("matplotlib.pyplot.savefig"):
        mock_ce = MagicMock()
        mock_ce.get_cost_and_usage.return_value = {"ResultsByTime": []}
        mock_client.return_value = mock_ce

        generate_cost_report()

        call_args = mock_ce.get_cost_and_usage.call_args[1]
        time_period = call_args["TimePeriod"]

        # Should have Start and End dates
        assert "Start" in time_period
        assert "End" in time_period

        # Dates should be in YYYY-MM-DD format
        import re

        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        assert re.match(date_pattern, time_period["Start"])
        assert re.match(date_pattern, time_period["End"])


@pytest.mark.unit
def test_generate_cost_report_service_filter(mock_aws_credentials):
    """Test that cost report filters for EC2 compute"""
    with patch("boto3.client") as mock_client, patch("matplotlib.pyplot.savefig"):
        mock_ce = MagicMock()
        mock_ce.get_cost_and_usage.return_value = {"ResultsByTime": []}
        mock_client.return_value = mock_ce

        generate_cost_report()

        call_args = mock_ce.get_cost_and_usage.call_args[1]
        service_filter = call_args["Filter"]["Dimensions"]

        assert service_filter["Key"] == "SERVICE"
        assert "Amazon Elastic Compute Cloud - Compute" in service_filter["Values"]


@pytest.mark.unit
def test_generate_cost_report_grouping(mock_aws_credentials):
    """Test that cost report groups by instance type"""
    with patch("boto3.client") as mock_client, patch("matplotlib.pyplot.savefig"):
        mock_ce = MagicMock()
        mock_ce.get_cost_and_usage.return_value = {"ResultsByTime": []}
        mock_client.return_value = mock_ce

        generate_cost_report()

        call_args = mock_ce.get_cost_and_usage.call_args[1]
        group_by = call_args["GroupBy"][0]

        assert group_by["Type"] == "DIMENSION"
        assert group_by["Key"] == "INSTANCE_TYPE"


@pytest.mark.unit
def test_cost_calculation(mock_cost_explorer_response):
    """Test cost calculation logic"""
    # Extract expected total from mock response
    expected_total = 10.50

    # Parse the response like the actual function does
    costs_by_day = {}
    for result in mock_cost_explorer_response["ResultsByTime"]:
        date = result["TimePeriod"]["Start"]
        for group in result["Groups"]:
            instance_type = group["Keys"][0]
            cost = float(group["Metrics"]["UnblendedCost"]["Amount"])

            if instance_type not in costs_by_day:
                costs_by_day[instance_type] = {}
            costs_by_day[instance_type][date] = cost

    # Calculate total like the function does
    total_cost = sum(sum(daily.values()) for daily in costs_by_day.values())

    assert total_cost == expected_total
