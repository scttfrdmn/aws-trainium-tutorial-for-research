"""Unit tests for the JSON cost report (scripts/cost_monitor.py --json) and the
static dashboard that renders it (monitoring/cost_monitoring_dashboard.html)."""

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import cost_monitor
import pytest
from cost_monitor import build_cost_report, is_ml_instance, main

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "monitoring" / "cost_monitoring_dashboard.html"

NOW = datetime(2025, 3, 15, 12, 0, tzinfo=timezone.utc)


def _monthly_response():
    def group(key, amount):
        return {
            "Keys": [key],
            "Metrics": {"UnblendedCost": {"Amount": amount, "Unit": "USD"}},
        }

    return {
        "ResultsByTime": [
            {
                "TimePeriod": {"Start": "2025-03-01", "End": "2025-03-15"},
                "Groups": [
                    group("trn1.2xlarge", "40.00"),
                    group("inf2.xlarge", "10.50"),
                    group("m5.large", "2.00"),
                    group("NoInstanceType", "3.25"),
                ],
            }
        ]
    }


def _daily_response(days=14):
    start = NOW.date() - timedelta(days=days)
    return {
        "ResultsByTime": [
            {
                "TimePeriod": {
                    "Start": (start + timedelta(days=i)).isoformat(),
                    "End": (start + timedelta(days=i + 1)).isoformat(),
                },
                "Total": {"UnblendedCost": {"Amount": "4.00", "Unit": "USD"}},
            }
            for i in range(days)
        ]
    }


def _instance(instance_id, instance_type, tags, hours_ago=2.0, **extra):
    return {
        "InstanceId": instance_id,
        "InstanceType": instance_type,
        "LaunchTime": NOW - timedelta(hours=hours_ago),
        "State": {"Name": "running"},
        "Tags": [{"Key": k, "Value": v} for k, v in tags.items()],
        **extra,
    }


def _ec2_response():
    return {
        "Reservations": [
            {
                "Instances": [
                    _instance(
                        "i-trn", "trn1.2xlarge", {"Name": "ML-bert"}, hours_ago=3.5
                    ),
                    _instance(
                        "i-inf",
                        "inf2.xlarge",
                        {"Name": "serve", "AutoTerminate": "true"},
                        InstanceLifecycle="spot",
                    ),
                    _instance("i-web", "m5.large", {"Name": "web-server"}),
                    _instance("i-untagged", "trn1.32xlarge", {}),
                ]
            }
        ]
    }


@pytest.fixture
def mocked_clients():
    ce = MagicMock()
    ce.get_cost_and_usage.side_effect = lambda **kw: (
        _monthly_response() if kw["Granularity"] == "MONTHLY" else _daily_response()
    )
    ec2 = MagicMock()
    ec2.describe_instances.return_value = _ec2_response()
    clients = {"ce": ce, "ec2": ec2}
    with patch("boto3.client", side_effect=lambda name, **kw: clients[name]) as fn:
        yield clients, fn


@pytest.mark.unit
def test_is_ml_instance_matches_emergency_shutdown_rule():
    assert is_ml_instance({"Tags": [{"Key": "Name", "Value": "ML-x"}]})
    assert is_ml_instance({"Tags": [{"Key": "AutoTerminate", "Value": "true"}]})
    assert not is_ml_instance({"Tags": [{"Key": "AutoTerminate", "Value": "false"}]})
    assert not is_ml_instance({"Tags": [{"Key": "Name", "Value": "my-ML-box"}]})
    assert not is_ml_instance({})


@pytest.mark.unit
def test_build_cost_report_schema(mocked_clients):
    clients, client_fn = mocked_clients
    report = build_cost_report(days=14, region="us-west-2", now=NOW)

    assert set(report) == {
        "schema_version",
        "generated_at",
        "region",
        "currency",
        "month_to_date",
        "daily_costs",
        "running_instances",
        "summary",
    }
    assert report["schema_version"] == cost_monitor.REPORT_SCHEMA_VERSION
    assert report["generated_at"] == "2025-03-15T12:00:00+00:00"
    assert report["region"] == "us-west-2"
    assert report["currency"] == "USD"
    client_fn.assert_any_call("ec2", region_name="us-west-2")

    mtd = report["month_to_date"]
    assert mtd["start"] == "2025-03-01"
    assert mtd["end"] == "2025-03-15"
    assert mtd["total"] == pytest.approx(55.75)
    assert mtd["trainium"] == pytest.approx(40.0)
    assert mtd["inferentia"] == pytest.approx(10.5)
    assert mtd["other"] == pytest.approx(5.25)  # m5.large + NoInstanceType
    # Sorted by cost, and the non-instance bucket is not shown as an "instance type".
    assert list(mtd["by_instance_type"]) == ["trn1.2xlarge", "inf2.xlarge", "m5.large"]

    assert len(report["daily_costs"]) == 14
    assert report["daily_costs"][0] == {"date": "2025-03-01", "amount": 4.0}
    assert report["daily_costs"][-1]["date"] == "2025-03-14"

    instances = {i["instance_id"]: i for i in report["running_instances"]}
    assert set(instances) == {"i-trn", "i-inf"}  # tag filter, not instance family
    trn = instances["i-trn"]
    assert set(trn) == {
        "instance_id",
        "instance_type",
        "family",
        "name",
        "lifecycle",
        "launch_time",
        "uptime_hours",
        "auto_terminate",
    }
    assert trn["family"] == "trainium"
    assert trn["lifecycle"] == "on-demand"
    assert trn["uptime_hours"] == pytest.approx(3.5)
    assert trn["auto_terminate"] is False
    assert instances["i-inf"]["lifecycle"] == "spot"
    assert instances["i-inf"]["auto_terminate"] is True

    summary = report["summary"]
    assert summary["running_ml_instances"] == 2
    assert summary["trainium_instances"] == 1
    assert summary["inferentia_instances"] == 1
    assert summary["daily_average_7d"] == pytest.approx(4.0)
    # 55.75 billed through the 14th + 4.00/day for Mar 15..31 (17 days).
    assert summary["projected_month_total"] == pytest.approx(55.75 + 4.0 * 17)

    # Cost Explorer queries: month-to-date by instance type, then the daily trend.
    calls = [c.kwargs for c in clients["ce"].get_cost_and_usage.call_args_list]
    assert calls[0]["GroupBy"] == [{"Type": "DIMENSION", "Key": "INSTANCE_TYPE"}]
    assert calls[1]["TimePeriod"] == {"Start": "2025-03-01", "End": "2025-03-15"}
    filters = clients["ec2"].describe_instances.call_args.kwargs["Filters"]
    assert filters == [{"Name": "instance-state-name", "Values": ["running"]}]

    json.dumps(report)  # must be JSON-serializable as-is


@pytest.mark.unit
def test_build_cost_report_first_of_month_skips_mtd_query(mocked_clients):
    clients, _ = mocked_clients
    first = datetime(2025, 4, 1, 9, 0, tzinfo=timezone.utc)
    report = build_cost_report(days=7, region="us-east-1", now=first)
    granularities = [
        c.kwargs["Granularity"] for c in clients["ce"].get_cost_and_usage.call_args_list
    ]
    assert granularities == ["DAILY"]  # Start == End would be rejected by the API
    assert report["month_to_date"]["total"] == 0.0


@pytest.mark.unit
def test_running_instances_follow_next_token():
    ec2 = MagicMock()
    page2 = {
        "Reservations": [
            {"Instances": [_instance("i-2", "trn2.48xlarge", {"Name": "ML-b"})]}
        ]
    }
    ec2.describe_instances.side_effect = [
        {**_ec2_response(), "NextToken": "tok"},
        page2,
    ]
    found = cost_monitor._running_ml_instances(ec2, NOW)
    assert [i["instance_id"] for i in found] == ["i-trn", "i-inf", "i-2"]
    assert ec2.describe_instances.call_args.kwargs["NextToken"] == "tok"


@pytest.mark.unit
def test_cli_json_writes_report(mocked_clients, tmp_path):
    out = tmp_path / "nested" / "cost_report.json"
    with patch("cost_monitor.generate_cost_report") as png_report:
        rc = main(["--json", str(out), "--region", "us-east-1", "--days", "7"])
    assert rc == 0
    png_report.assert_not_called()  # --json mode doesn't make the matplotlib chart
    data = json.loads(out.read_text())
    assert data["region"] == "us-east-1"
    assert data["month_to_date"]["by_instance_type"]
    assert data["summary"]["running_ml_instances"] == 2


@pytest.mark.unit
def test_cli_default_keeps_png_report():
    with patch("cost_monitor.generate_cost_report", return_value={}) as png_report:
        assert main([]) == 0
    png_report.assert_called_once_with(days_back=30)


@pytest.mark.unit
def test_cli_reports_aws_errors(capsys):
    with patch("cost_monitor.build_cost_report", side_effect=RuntimeError("denied")):
        assert main(["--json", "unused.json"]) == 1
    assert "denied" in capsys.readouterr().out


@pytest.mark.unit
def test_dashboard_has_no_credentials_or_aws_sdk():
    html = DASHBOARD.read_text()
    lowered = html.lower()
    for forbidden in (
        "aws-sdk",
        "sdk.amazonaws.com",
        "accesskeyid",
        "secretaccesskey",
        "sessiontoken",
        "aws.config",
        "new aws.",
    ):
        assert forbidden not in lowered, forbidden
    assert 'type="password"' not in lowered
    # The only external script is Chart.js.
    srcs = re.findall(r"<script[^>]*\bsrc=\"([^\"]+)\"", html)
    assert srcs and all("chart.js" in s for s in srcs), srcs
    # It renders a local report and tells the user how to regenerate it.
    assert "cost_report.json" in html
    assert "scripts/cost_monitor.py --json" in html
    assert "setInterval" not in html  # no auto-refresh against AWS


@pytest.mark.unit
def test_dashboard_reads_the_fields_the_report_writes():
    html = DASHBOARD.read_text()
    for field in (
        "month_to_date",
        "daily_costs",
        "running_instances",
        "by_instance_type",
        "daily_average_7d",
        "projected_month_total",
        "generated_at",
        "uptime_hours",
    ):
        assert field in html, field
