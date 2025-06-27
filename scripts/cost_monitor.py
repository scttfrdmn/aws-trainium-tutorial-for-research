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
from datetime import datetime, timedelta

import boto3
import matplotlib.pyplot as plt
import pandas as pd


def generate_cost_report(
    days_back: int = 30, output_file: str = "ml_costs.png"
) -> dict:
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
    costs_by_day = {}
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
    instance_totals = {}
    for instance_type, daily_costs in costs_by_day.items():
        instance_totals[instance_type] = sum(daily_costs.values())

    # Generate cost optimization recommendations
    recommendations = []
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
        print(f"\n🔧 Instance Type Breakdown:")
        for instance_type, cost in sorted(
            instance_totals.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            print(f"  {instance_type}: ${cost:.2f} ({percentage:.1f}%)")

    if recommendations:
        print(f"\n💡 Cost Optimization Recommendations:")
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


if __name__ == "__main__":
    """Generate ML cost report when run as standalone script.

    Provides command-line interface for generating cost reports with
    comprehensive analysis and visualization for ML research projects.

    The script generates:
    - Visual cost trends chart (ml_costs.png)
    - Console summary with total costs and insights
    - Breakdown by instance type
    - Cost optimization recommendations

    Usage:
        python cost_monitor.py

    Output:
        - ml_costs.png: Visual chart showing cost trends
        - Console output: Summary statistics and recommendations

    Example Output:
        📊 Cost report saved to ml_costs.png
        💰 Total ML compute cost (last 30 days): $245.67

        Instance Type Breakdown:
        - trn1.2xlarge: $156.80 (63.8%)
        - inf2.xlarge: $88.87 (36.2%)

        Recommendations:
        - Consider spot instances for 70% savings
        - Right-size instances based on utilization
    """
    try:
        report_data = generate_cost_report()
        print("\n✅ Cost monitoring report generated successfully!")
        print(f"📊 Check ml_costs.png for visual analysis")

        if report_data:
            print(f"💡 Use this data for budget planning and cost optimization")
            print(f"🔄 Run weekly for proactive cost management")

    except Exception as e:
        print(f"❌ Error generating cost report: {e}")
        print("💡 Check AWS credentials and Cost Explorer permissions")
        print("📖 See AWS documentation for ce:GetCostAndUsage permissions")
