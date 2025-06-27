#!/usr/bin/env python3
"""
Ephemeral ML Instance Manager
Launches self-terminating instances to prevent runaway costs
"""
import boto3
import time
import argparse
from datetime import datetime

class EphemeralMLInstance:
    """Self-terminating ML instance for experiments"""
    
    def __init__(self, instance_type='trn1.2xlarge', max_hours=4):
        self.ec2 = boto3.client('ec2')
        self.instance_type = instance_type
        self.max_hours = max_hours
        self.instance_id = None
        
    def launch(self, experiment_name):
        """Launch instance with auto-termination"""
        
        user_data = f"""#!/bin/bash
# Set up termination timer
echo "sudo shutdown -h +{self.max_hours*60}" | at now + {self.max_hours} hours

# Log experiment details
echo "Experiment: {experiment_name}" > /home/ubuntu/experiment.log
echo "Started: $(date)" >> /home/ubuntu/experiment.log
echo "Auto-terminates: $(date -d '+{self.max_hours} hours')" >> /home/ubuntu/experiment.log

# Install dependencies
pip install torch-neuronx --extra-index-url https://pip.repos.neuron.amazonaws.com
pip install psutil boto3

# Run cost tracker in background
nohup python3 /home/ubuntu/track_cost.py > /home/ubuntu/cost_log.txt 2>&1 &
"""
        
        # Launch instance
        response = self.ec2.run_instances(
            ImageId=self.get_neuron_ami(),
            InstanceType=self.instance_type,
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'ML-Experiment-{experiment_name}'},
                    {'Key': 'AutoTerminate', 'Value': 'true'},
                    {'Key': 'MaxHours', 'Value': str(self.max_hours)},
                    {'Key': 'ExperimentStart', 'Value': datetime.now().isoformat()}
                ]
            }],
            InstanceMarketOptions={
                'MarketType': 'spot',
                'SpotOptions': {
                    'SpotInstanceType': 'one-time',
                    'InstanceInterruptionBehavior': 'terminate'
                }
            }
        )
        
        self.instance_id = response['Instances'][0]['InstanceId']
        print(f"✅ Launched {self.instance_type} instance: {self.instance_id}")
        print(f"⏰ Will auto-terminate in {self.max_hours} hours")
        print(f"💰 Estimated max cost: ${self.get_hourly_rate() * self.max_hours:.2f}")
        
        return self.instance_id
    
    def get_hourly_rate(self):
        """Get current spot price for instance type"""
        pricing = {
            'trn1.2xlarge': 0.40,
            'trn1.32xlarge': 6.45,
            'trn2.48xlarge': 12.00,
            'inf2.xlarge': 0.227,
            'inf2.48xlarge': 3.89,
        }
        return pricing.get(self.instance_type, 1.0)
    
    def get_neuron_ami(self):
        """Get latest Deep Learning AMI with Neuron"""
        return 'ami-0c55b159cbfafe1f0'  # Update for your region

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch ephemeral ML instance')
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--instance-type', type=str, default='trn1.2xlarge', help='Instance type')
    parser.add_argument('--max-hours', type=int, default=4, help='Maximum hours to run')
    
    args = parser.parse_args()
    
    launcher = EphemeralMLInstance(args.instance_type, args.max_hours)
    launcher.launch(args.name)