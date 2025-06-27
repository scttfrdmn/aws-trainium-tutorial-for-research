#!/usr/bin/env python3
"""
AWS Budget Setup for ML Research
Creates budget alerts to prevent runaway costs
"""
import boto3
import json
import argparse

def create_research_budget(monthly_limit=100, email="your-email@university.edu"):
    """Create a budget with alerts at 50%, 80%, and 100%"""
    client = boto3.client('budgets')
    
    budget = {
        'BudgetName': 'ML-Research-Budget',
        'BudgetLimit': {
            'Amount': str(monthly_limit),
            'Unit': 'USD'
        },
        'TimeUnit': 'MONTHLY',
        'BudgetType': 'COST',
        'CostFilters': {
            'Service': ['Amazon Elastic Compute Cloud - Compute']
        }
    }
    
    # Email alerts at different thresholds
    notifications = []
    for threshold in [50, 80, 100]:
        notifications.append({
            'Notification': {
                'NotificationType': 'ACTUAL',
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': threshold,
                'ThresholdType': 'PERCENTAGE',
                'NotificationState': 'ALARM'
            },
            'Subscribers': [{
                'SubscriptionType': 'EMAIL',
                'Address': email
            }]
        })
    
    try:
        response = client.create_budget(
            AccountId=boto3.client('sts').get_caller_identity()['Account'],
            Budget=budget,
            NotificationsWithSubscribers=notifications
        )
        print(f"✅ Budget created! You'll get alerts at ${monthly_limit*0.5:.0f}, ${monthly_limit*0.8:.0f}, and ${monthly_limit}")
        return response
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up AWS budget alerts for ML research')
    parser.add_argument('--limit', type=int, default=500, help='Monthly budget limit in USD')
    parser.add_argument('--email', type=str, required=True, help='Email for alerts')
    
    args = parser.parse_args()
    create_research_budget(args.limit, args.email)