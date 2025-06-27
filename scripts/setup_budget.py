#!/usr/bin/env python3
"""AWS Budget Setup for Machine Learning Research.

This module provides tools for setting up AWS budget alerts to prevent runaway costs
during machine learning research projects. It's specifically designed for academic
researchers and students who need cost control mechanisms when working with expensive
ML services like Trainium and Inferentia instances.

The budget system creates multi-threshold alerts to give researchers early warning
before reaching their spending limits, helping prevent bill shock and enabling
better cost management for research projects.

Examples:
    Basic budget setup:
        python setup_budget.py --email researcher@university.edu --limit 500

    Custom budget with different limit:
        python setup_budget.py --email lab@university.edu --limit 1000

    From Python code:
        from scripts.setup_budget import create_research_budget
        create_research_budget(monthly_limit=300, email="student@edu")

Cost Considerations:
    - Budget API calls are free
    - Email notifications are free through AWS SNS
    - Helps prevent unexpected charges from forgotten instances
    - Essential for research environments with limited funding

Note:
    Requires AWS credentials with budgets:CreateBudget permissions.
    The script targets EC2 compute costs but can be extended to other services.
"""
import argparse
import json

import boto3


def create_research_budget(
    monthly_limit: int = 100, email: str = "your-email@university.edu"
) -> dict:
    """Create a comprehensive budget with progressive alerts for ML research.

    Sets up an AWS budget specifically designed for machine learning research with
    three-tier alert system (50%, 80%, 100%) to provide early warnings and prevent
    cost overruns. The budget focuses on EC2 compute costs which are typically the
    largest expense in ML workloads.

    Args:
        monthly_limit (int): Maximum monthly spending limit in USD. Default is $100,
            suitable for small research projects. Consider $500-1000 for active
            research involving multiple experiments.
        email (str): Email address for budget alerts. Should be actively monitored
            as alerts are time-sensitive for cost control.

    Returns:
        dict: AWS budget creation response containing budget metadata and status.
            Returns None if budget creation fails or if budget already exists.

    Raises:
        RuntimeError: If AWS client creation fails due to missing credentials or
            invalid configuration.
        Exception: For various AWS API errors including permission issues or
            service limits.

    Examples:
        >>> # Basic research budget setup
        >>> response = create_research_budget(500, "researcher@university.edu")
        >>> print(f"Budget created: {response is not None}")

        >>> # Budget for large-scale research project
        >>> create_research_budget(2000, "ml-lab@university.edu")

    Cost Analysis:
        - 50% alert ($limit * 0.5): Early warning for budget planning
        - 80% alert ($limit * 0.8): Critical warning to pause experiments
        - 100% alert ($limit): Maximum budget reached, immediate action required

    Note:
        The budget filters specifically for "Amazon Elastic Compute Cloud - Compute"
        which includes Trainium (trn1.*) and Inferentia (inf1.*, inf2.*) instances.
        Data transfer and storage costs are not included but typically represent
        <10% of ML research costs.
    """
    try:
        client = boto3.client("budgets")
    except Exception as e:
        print(f"❌ Failed to create AWS client: {e}")
        return None

    budget = {
        "BudgetName": "ML-Research-Budget",
        "BudgetLimit": {"Amount": str(monthly_limit), "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
        "CostFilters": {"Service": ["Amazon Elastic Compute Cloud - Compute"]},
    }

    # Email alerts at different thresholds
    notifications = []
    for threshold in [50, 80, 100]:
        notifications.append(
            {
                "Notification": {
                    "NotificationType": "ACTUAL",
                    "ComparisonOperator": "GREATER_THAN",
                    "Threshold": threshold,
                    "ThresholdType": "PERCENTAGE",
                    "NotificationState": "ALARM",
                },
                "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}],
            }
        )

    try:
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity()["Account"]

        response = client.create_budget(
            AccountId=account_id,
            Budget=budget,
            NotificationsWithSubscribers=notifications,
        )
        print(
            f"✅ Budget created! You'll get alerts at ${monthly_limit*0.5:.0f}, ${monthly_limit*0.8:.0f}, and ${monthly_limit}"
        )
        return response
    except client.exceptions.DuplicateRecordException:
        print("⚠️  Budget already exists. Use AWS console to modify existing budget.")
        return None
    except Exception as e:
        print(f"❌ Error creating budget: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Set up AWS budget alerts for machine learning research projects.

        This script creates a comprehensive budget monitoring system with three-tier
        alerts (50%, 80%, 100%) to help researchers manage costs when using expensive
        ML services like Trainium and Inferentia instances.

        The budget specifically tracks EC2 compute costs, which typically represent
        90%+ of ML research expenses. Early alerts help prevent bill shock and enable
        better resource planning for research projects.

        Examples:
            python setup_budget.py --email researcher@university.edu --limit 500
            python setup_budget.py --email lab@institution.edu --limit 2000
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="""Monthly budget limit in USD. Recommendations:
        - $100-300: Individual student projects
        - $500-1000: Active research with multiple experiments
        - $1000+: Large-scale research or lab-wide budgets""",
    )
    parser.add_argument(
        "--email",
        type=str,
        required=True,
        help="""Email address for budget alerts. Use an actively monitored email
        as cost alerts are time-sensitive. Consider using a shared lab email
        for research group projects.""",
    )

    args = parser.parse_args()
    create_research_budget(args.limit, args.email)
