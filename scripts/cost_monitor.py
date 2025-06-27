#!/usr/bin/env python3
"""
Cost monitoring and reporting for ML experiments
"""
import boto3
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

def generate_cost_report():
    """Generate visual cost report for ML experiments"""
    ce = boto3.client('ce')  # Cost Explorer
    
    # Get costs for last 30 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start_date.strftime('%Y-%m-%d'),
            'End': end_date.strftime('%Y-%m-%d')
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        Filter={
            'Dimensions': {
                'Key': 'SERVICE',
                'Values': ['Amazon Elastic Compute Cloud - Compute']
            }
        },
        GroupBy=[{
            'Type': 'DIMENSION',
            'Key': 'INSTANCE_TYPE'
        }]
    )
    
    # Parse results
    costs_by_day = {}
    for result in response['ResultsByTime']:
        date = result['TimePeriod']['Start']
        for group in result['Groups']:
            instance_type = group['Keys'][0]
            cost = float(group['Metrics']['UnblendedCost']['Amount'])
            
            if instance_type not in costs_by_day:
                costs_by_day[instance_type] = {}
            costs_by_day[instance_type][date] = cost
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    for instance_type, daily_costs in costs_by_day.items():
        if any(cost > 0 for cost in daily_costs.values()):
            dates = sorted(daily_costs.keys())
            costs = [daily_costs.get(d, 0) for d in dates]
            plt.plot(dates, costs, label=instance_type, marker='o')
    
    plt.title('ML Experiment Costs by Instance Type')
    plt.xlabel('Date')
    plt.ylabel('Cost (USD)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ml_costs.png')
    print("📊 Cost report saved to ml_costs.png")
    
    # Print summary
    total_cost = sum(sum(daily.values()) for daily in costs_by_day.values())
    print(f"\n💰 Total ML compute cost (last 30 days): ${total_cost:.2f}")

if __name__ == "__main__":
    generate_cost_report()