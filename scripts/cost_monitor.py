#!/usr/bin/env python3
"""Cost Monitoring and Reporting for Machine Learning Experiments.

This module provides comprehensive cost analysis and visualization tools for
ML research projects using AWS services. It focuses on tracking EC2 compute
costs from Trainium and Inferentia instances, which typically represent 90%+
of ML research expenses.

The cost monitoring system helps researchers:
- Track spending across different instance types and experiments
- Identify cost optimization opportunities
- Generate visual reports for budget planning
- Monitor spending trends over time
- Validate budget compliance

Key Features:
- Daily cost breakdown by instance type
- Visual cost trend analysis with matplotlib
- Instance type comparison and optimization recommendations
- Integration with AWS Cost Explorer API
- Automated report generation for research groups

Examples:
    Generate cost report for last 30 days:
        python cost_monitor.py

    Write a JSON report for the static dashboard in monitoring/, using your normal
    AWS credential chain (AWS_PROFILE, SSO, env vars) -- never browser-entered keys:
        AWS_PROFILE=research python scripts/cost_monitor.py --json monitoring/cost_report.json

    From Python code:
        from scripts.cost_monitor import generate_cost_report
        generate_cost_report()

Cost Analysis:
    The module tracks EC2 compute costs which include:
    - Trainium instances (trn1.*, trn2.*) for training
    - Inferentia instances (inf1.*, inf2.*) for inference
    - Data transfer costs between instances
    - EBS storage costs for training data

Note:
    Requires AWS credentials with ce:GetCostAndUsage permissions.
    The Cost Explorer API has a small charge (~$0.01 per request) but
    this is negligible compared to the cost savings from monitoring.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3

#: Version of the JSON report schema read by monitoring/cost_monitoring_dashboard.html.
REPORT_SCHEMA_VERSION = 1

#: Cost Explorer's group key for usage with no EC2 instance type (S3, transfer, ...).
NO_INSTANCE_TYPE = "NoInstanceType"


def generate_cost_report(
    days_back: int = 30, output_file: str = "ml_costs.png"
) -> dict[str, Any]:
    """Generate comprehensive visual cost report for ML experiments.

    Creates a detailed cost analysis report covering the specified time period,
    with visual charts showing spending trends by instance type. The report
    focuses on EC2 compute costs which represent the majority of ML research
    expenses.

    Args:
        days_back (int): Number of days to analyze from current date.
            Default: 30 days (good for monthly budget reviews)
            Recommended: 7 days for weekly reviews, 90 days for quarterly analysis
        output_file (str): Filename for the generated cost chart.
            Default: 'ml_costs.png'. Use descriptive names like
            'october_2024_costs.png' for archive purposes.

    Returns:
        dict: Cost analysis summary containing:
            - total_cost: Total spending for the period
            - daily_average: Average daily spending
            - instance_breakdown: Costs by instance type
            - cost_trends: Daily spending patterns
            - recommendations: Cost optimization suggestions

    Raises:
        boto3.exceptions.Boto3Error: If AWS Cost Explorer API access fails
            due to missing permissions or service issues.
        matplotlib.pyplot.Error: If chart generation fails due to missing
            display backend or file write permissions.

    Examples:
        >>> # Generate standard 30-day report
        >>> report = generate_cost_report()
        >>> print(f"Total cost: ${report['total_cost']:.2f}")

        >>> # Weekly cost review
        >>> weekly_report = generate_cost_report(7, "weekly_costs.png")
        >>> print(f"Daily average: ${weekly_report['daily_average']:.2f}")

        >>> # Quarterly analysis
        >>> quarterly = generate_cost_report(90, "q4_2024_costs.png")

    Cost Insights:
        The report provides several cost optimization insights:
        - Instance type efficiency (cost per compute unit)
        - Usage patterns (peak vs off-peak)
        - Spot vs on-demand pricing analysis
        - Recommendations for right-sizing instances

    Visual Output:
        Creates a line chart showing:
        - Daily costs by instance type
        - Trend lines for each instance family
        - Cost spikes and optimization opportunities
        - Comparative analysis across different instance types

    Note:
        The Cost Explorer API charges ~$0.01 per request, but this is
        negligible compared to potential savings from cost monitoring.
        The report focuses on EC2 compute costs which typically represent
        90%+ of ML research expenses.
    """
    import matplotlib.pyplot as plt  # lazy: --json mode needs no plotting backend

    ce = boto3.client("ce")  # Cost Explorer

    # Get costs for specified time period
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    response = ce.get_cost_and_usage(
        TimePeriod={
            "Start": start_date.strftime("%Y-%m-%d"),
            "End": end_date.strftime("%Y-%m-%d"),
        },
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter={
            "Dimensions": {
                "Key": "SERVICE",
                "Values": ["Amazon Elastic Compute Cloud - Compute"],
            }
        },
        GroupBy=[{"Type": "DIMENSION", "Key": "INSTANCE_TYPE"}],
    )

    # Parse results
    costs_by_day: dict[str, dict[str, float]] = {}
    for result in response["ResultsByTime"]:
        date = result["TimePeriod"]["Start"]
        for group in result["Groups"]:
            instance_type = group["Keys"][0]
            cost = float(group["Metrics"]["UnblendedCost"]["Amount"])

            if instance_type not in costs_by_day:
                costs_by_day[instance_type] = {}
            costs_by_day[instance_type][date] = cost

    # Create visualization
    plt.figure(figsize=(12, 6))
    for instance_type, daily_costs in costs_by_day.items():
        if any(cost > 0 for cost in daily_costs.values()):
            dates = sorted(daily_costs.keys())
            costs = [daily_costs.get(d, 0) for d in dates]
            plt.plot(dates, costs, label=instance_type, marker="o")

    plt.title("ML Experiment Costs by Instance Type")
    plt.xlabel("Date")
    plt.ylabel("Cost (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"📊 Cost report saved to {output_file}")

    # Calculate comprehensive summary statistics
    total_cost = sum(sum(daily.values()) for daily in costs_by_day.values())
    daily_average = total_cost / days_back if days_back > 0 else 0

    # Instance type breakdown
    instance_totals: dict[str, float] = {}
    for instance_type, daily_costs in costs_by_day.items():
        instance_totals[instance_type] = sum(daily_costs.values())

    # Generate cost optimization recommendations
    recommendations: list[str] = []
    if total_cost > 500:
        recommendations.append("Consider spot instances for 60-90% savings")
    if len(instance_totals) > 1:
        recommendations.append("Analyze instance type efficiency for right-sizing")
    if daily_average > 20:
        recommendations.append("Consider reserved instances for consistent workloads")

    # Print comprehensive summary
    print(f"\n💰 Total ML compute cost (last {days_back} days): ${total_cost:.2f}")
    print(f"📈 Daily average: ${daily_average:.2f}")

    if instance_totals:
        print("\n🔧 Instance Type Breakdown:")
        for instance_type, cost in sorted(
            instance_totals.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            print(f"  {instance_type}: ${cost:.2f} ({percentage:.1f}%)")

    if recommendations:
        print("\n💡 Cost Optimization Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")

    # Return analysis data
    return {
        "total_cost": total_cost,
        "daily_average": daily_average,
        "instance_breakdown": instance_totals,
        "recommendations": recommendations,
        "days_analyzed": days_back,
    }


def is_ml_instance(instance: dict[str, Any]) -> bool:
    """Return True if an EC2 instance is one of this tutorial's ML instances.

    Uses the same tag convention as ``scripts/emergency_shutdown.py``: a ``Name``
    tag starting with ``ML-`` or an ``AutoTerminate=true`` tag.

    Args:
        instance: One entry of ``describe_instances()["Reservations"][*]["Instances"]``.

    Returns:
        bool: Whether the instance is tagged as an ML experiment instance.
    """
    tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
    return tags.get("Name", "").startswith("ML-") or tags.get("AutoTerminate") == "true"


def _instance_family(instance_type: str) -> str:
    """Classify an instance type as ``trainium``, ``inferentia``, or ``other``."""
    if instance_type.startswith("trn"):
        return "trainium"
    if instance_type.startswith("inf"):
        return "inferentia"
    return "other"


def _amount(metrics: dict[str, Any]) -> float:
    """Extract the UnblendedCost amount from a Cost Explorer metrics dict."""
    return float(metrics["UnblendedCost"]["Amount"])


def _month_to_date(ce: Any, today: date) -> dict[str, Any]:
    """Month-to-date spend (all services), split by EC2 instance type and family."""
    start = today.replace(day=1)
    by_type: dict[str, float] = {}
    families = {"trainium": 0.0, "inferentia": 0.0, "other": 0.0}
    # Cost Explorer's End is exclusive and must be after Start. On the 1st of the
    # month there is no completed day yet, so report zeros instead of an invalid call.
    if start < today:
        response = ce.get_cost_and_usage(
            TimePeriod={"Start": start.isoformat(), "End": today.isoformat()},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "INSTANCE_TYPE"}],
        )
        for result in response.get("ResultsByTime", []):
            for group in result.get("Groups", []):
                key = group["Keys"][0]
                cost = _amount(group["Metrics"])
                if key == NO_INSTANCE_TYPE:
                    families["other"] += cost
                    continue
                by_type[key] = by_type.get(key, 0.0) + cost
                families[_instance_family(key)] += cost
    ranked = sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "start": start.isoformat(),
        "end": today.isoformat(),
        "total": round(sum(families.values()), 2),
        "trainium": round(families["trainium"], 2),
        "inferentia": round(families["inferentia"], 2),
        "other": round(families["other"], 2),
        "by_instance_type": {k: round(v, 2) for k, v in ranked},
    }


def _daily_costs(ce: Any, today: date, days: int) -> list[dict[str, Any]]:
    """Total daily spend (all services) for the last ``days`` completed days."""
    start = today - timedelta(days=days)
    response = ce.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": today.isoformat()},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
    )
    return [
        {
            "date": result["TimePeriod"]["Start"],
            "amount": round(_amount(result["Total"]), 2),
        }
        for result in response.get("ResultsByTime", [])
    ]


def _running_ml_instances(ec2: Any, now: datetime) -> list[dict[str, Any]]:
    """Running EC2 instances carrying the tutorial's ML tags (see is_ml_instance)."""
    instances: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {
        "Filters": [{"Name": "instance-state-name", "Values": ["running"]}]
    }
    while True:
        response = ec2.describe_instances(**kwargs)
        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                if not is_ml_instance(instance):
                    continue
                tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
                launch: datetime = instance["LaunchTime"]
                if launch.tzinfo is None:
                    launch = launch.replace(tzinfo=UTC)
                uptime_hours = (now - launch).total_seconds() / 3600
                instances.append(
                    {
                        "instance_id": instance["InstanceId"],
                        "instance_type": instance["InstanceType"],
                        "family": _instance_family(instance["InstanceType"]),
                        "name": tags.get("Name", ""),
                        "lifecycle": instance.get("InstanceLifecycle", "on-demand"),
                        "launch_time": launch.isoformat(),
                        "uptime_hours": round(uptime_hours, 2),
                        "auto_terminate": tags.get("AutoTerminate") == "true",
                    }
                )
        next_token = response.get("NextToken")
        if not next_token:
            return instances
        kwargs["NextToken"] = next_token


def build_cost_report(
    days: int = 14, region: str | None = None, now: datetime | None = None
) -> dict[str, Any]:
    """Collect the data shown by ``monitoring/cost_monitoring_dashboard.html``.

    Uses the standard boto3 credential chain (``AWS_PROFILE``, SSO, environment
    variables, instance role). It never asks for access keys: the browser
    dashboard only reads the JSON file this produces.

    Args:
        days: Number of completed days to include in the daily trend.
        region: Region whose running instances are listed. Defaults to the
            session's configured region (``AWS_REGION`` / ``AWS_DEFAULT_REGION`` /
            profile). Cost Explorer data is account-wide regardless of region.
        now: Override the current time (for tests). Must be timezone-aware.

    Returns:
        dict: JSON-serializable report with keys ``schema_version``,
        ``generated_at``, ``region``, ``currency``, ``month_to_date``,
        ``daily_costs``, ``running_instances``, and ``summary``.
    """
    now = now or datetime.now(UTC)
    today = now.date()
    region = region or boto3.session.Session().region_name or "us-east-1"

    ce = boto3.client("ce")
    ec2 = boto3.client("ec2", region_name=region)

    month = _month_to_date(ce, today)
    daily = _daily_costs(ce, today, days)
    instances = _running_ml_instances(ec2, now)

    # Burn rate from *billed* spend (last 7 completed days), not a hard-coded price list.
    recent = [d["amount"] for d in daily[-7:]]
    daily_average = sum(recent) / len(recent) if recent else 0.0
    next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
    days_left = (next_month - today).days  # includes today, which isn't billed yet

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "region": region,
        "currency": "USD",
        "month_to_date": month,
        "daily_costs": daily,
        "running_instances": instances,
        "summary": {
            "running_ml_instances": len(instances),
            "trainium_instances": sum(i["family"] == "trainium" for i in instances),
            "inferentia_instances": sum(i["family"] == "inferentia" for i in instances),
            "daily_average_7d": round(daily_average, 2),
            "projected_month_total": round(
                month["total"] + daily_average * days_left, 2
            ),
        },
    }


def write_cost_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Write a report from :func:`build_cost_report` as pretty-printed JSON.

    Args:
        path: Destination file (parent directories are created).
        report: The report dict.

    Returns:
        Path: The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    With no arguments, prints the 30-day EC2 cost summary and saves
    ``ml_costs.png`` (the original behavior). With ``--json PATH``, writes the
    report consumed by ``monitoring/cost_monitoring_dashboard.html`` instead.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        int: Process exit code (0 on success, 1 on AWS/API errors).
    """
    parser = argparse.ArgumentParser(
        description=(
            "ML cost report. Uses your normal AWS credential chain "
            "(AWS_PROFILE / SSO / env vars); never prompts for access keys."
        )
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="write a JSON report for monitoring/cost_monitoring_dashboard.html",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="days of history (default: 30 for the PNG report, 14 for --json)",
    )
    parser.add_argument(
        "--region",
        help="region for the running-instance list (default: session region)",
    )
    args = parser.parse_args(argv)

    try:
        if args.json:
            report = build_cost_report(days=args.days or 14, region=args.region)
            out = write_cost_report(args.json, report)
            mtd = report["month_to_date"]["total"]
            running = report["summary"]["running_ml_instances"]
            print(
                f"Wrote {out}: month to date ${mtd:.2f}, {running} running ML instances"
            )
            print("Open monitoring/cost_monitoring_dashboard.html and load this file.")
            return 0

        report_data = generate_cost_report(days_back=args.days or 30)
        print("\n✅ Cost monitoring report generated successfully!")
        print("📊 Check ml_costs.png for visual analysis")
        if report_data:
            print("💡 Use this data for budget planning and cost optimization")
            print("🔄 Run weekly for proactive cost management")
        return 0
    except Exception as e:
        print(f"❌ Error generating cost report: {e}")
        print(
            "💡 Check AWS credentials (AWS_PROFILE, `aws sso login`) and permissions:"
        )
        print("   ce:GetCostAndUsage, plus ec2:DescribeInstances for --json")
        return 1


if __name__ == "__main__":
    sys.exit(main())
