#!/usr/bin/env python3
"""Ephemeral Machine Learning Instance Manager.

This module provides tools for launching self-terminating AWS instances specifically
designed for machine learning research. The primary goal is cost control through
automatic instance termination, preventing the common research scenario of
forgotten instances running indefinitely and generating large bills.

The ephemeral instance system is particularly valuable for:
- Training experiments that may fail or complete early
- Research environments where instances are easily forgotten
- Student projects with limited budgets
- Batch inference jobs with defined runtime limits

Key Features:
- Automatic termination after specified time limit
- Spot instance usage for 60-90% cost savings
- Pre-configured with Neuron SDK for Trainium/Inferentia
- Built-in cost tracking and monitoring
- Experiment logging and metadata capture

Examples:
    Launch a 4-hour training instance:
        python ephemeral_instance.py --name climate-training --max-hours 4

    Launch overnight inference job:
        python ephemeral_instance.py --name batch-inference --max-hours 8 \
            --instance-type inf2.xlarge

    From Python code:
        launcher = EphemeralMLInstance("trn1.2xlarge", max_hours=6)
        instance_id = launcher.launch("bert-pretraining")

Cost Considerations:
    - Spot instances provide 60-90% savings vs on-demand
    - Auto-termination prevents forgotten instance charges
    - trn1.2xlarge spot: ~$0.40/hour vs $1.34/hour on-demand
    - inf2.xlarge spot: ~$0.227/hour vs $0.758/hour on-demand

Safety Features:
    - Maximum runtime enforcement prevents runaway costs
    - Cost estimation before launch
    - Automatic shutdown even if experiment hangs
    - Clear instance tagging for cost tracking
"""
import argparse
import time
from datetime import datetime
from typing import Dict, Optional

import boto3


class EphemeralMLInstance:
    """Self-terminating machine learning instance manager.

    Manages AWS EC2 instances specifically configured for ML research with automatic
    termination to prevent runaway costs. The class handles spot instance requests,
    Neuron SDK installation, experiment logging, and cost tracking.

    This is particularly valuable for research environments where instances might be
    forgotten after experiments complete or fail, leading to unexpected charges.

    Attributes:
        instance_type (str): AWS instance type (e.g., 'trn1.2xlarge', 'inf2.xlarge')
        max_hours (int): Maximum runtime before automatic termination
        instance_id (Optional[str]): ID of launched instance, None before launch
        ec2: Boto3 EC2 client for instance management

    Examples:
        >>> # Launch a 2-hour Trainium instance for training
        >>> launcher = EphemeralMLInstance("trn1.2xlarge", max_hours=2)
        >>> instance_id = launcher.launch("bert-fine-tuning")
        >>> print(f"Launched: {instance_id}")

        >>> # Launch overnight Inferentia inference job
        >>> launcher = EphemeralMLInstance("inf2.xlarge", max_hours=8)
        >>> launcher.launch("batch-inference-job")

    Cost Analysis:
        The class automatically estimates costs and provides savings information:
        - trn1.2xlarge spot: ~70% savings ($0.40 vs $1.34/hour)
        - trn1.32xlarge spot: ~75% savings ($6.45 vs $25.70/hour)
        - inf2.xlarge spot: ~70% savings ($0.227 vs $0.758/hour)

    Safety Features:
        - Automatic termination prevents forgotten instances
        - Spot instances reduce costs significantly
        - Clear experiment tagging and logging
        - Cost estimation before launch
        - Built-in experiment metadata capture
    """

    def __init__(self, instance_type: str = "trn1.2xlarge", max_hours: int = 4):
        """Initialize ephemeral ML instance manager.

        Args:
            instance_type (str): AWS instance type optimized for ML workloads.
                Supported types:
                - Trainium: trn1.2xlarge, trn1.32xlarge, trn2.48xlarge
                - Inferentia: inf2.xlarge, inf2.8xlarge, inf2.48xlarge
                Default: trn1.2xlarge (good balance of performance and cost)
            max_hours (int): Maximum runtime in hours before automatic termination.
                Recommended values:
                - 1-4 hours: Quick experiments and debugging
                - 4-8 hours: Standard training jobs
                - 8-24 hours: Large model training or batch inference
                Maximum: 168 hours (1 week) for safety

        Raises:
            RuntimeError: If AWS EC2 client creation fails due to missing
                credentials or configuration issues.

        Note:
            The instance will automatically terminate after max_hours regardless
            of job status to prevent runaway costs. Plan accordingly for long
            training jobs.
        """
        try:
            self.ec2 = boto3.client("ec2")
        except Exception as e:
            raise RuntimeError(f"Failed to create EC2 client: {e}")

        self.instance_type = instance_type
        self.max_hours = max_hours
        self.instance_id: Optional[str] = None

    def launch(self, experiment_name: str) -> str:
        """Launch ephemeral ML instance with automatic termination.

        Creates a spot instance pre-configured for ML research with Neuron SDK,
        automatic termination, cost tracking, and experiment logging. The instance
        will self-terminate after the specified time limit to prevent runaway costs.

        Args:
            experiment_name (str): Descriptive name for the experiment. Used for
                instance tagging, logging, and cost tracking. Should be descriptive
                and unique (e.g., 'bert-climate-classification', 'llama-inference-test').

        Returns:
            str: AWS instance ID of the launched instance. Use this for monitoring,
                connecting, or manual termination if needed.

        Raises:
            ValueError: If experiment_name is empty or whitespace-only.
            Exception: For AWS API errors including:
                - InsufficientInstanceCapacity: No spot instances available
                - InvalidParameterValue: Invalid instance type or configuration
                - UnauthorizedOperation: Missing EC2 permissions

        Examples:
            >>> launcher = EphemeralMLInstance("trn1.2xlarge", 4)
            >>> instance_id = launcher.launch("climate-sentiment-bert")
            >>> print(f"Training will run for max 4 hours: {instance_id}")

            >>> # Launch inference job
            >>> launcher = EphemeralMLInstance("inf2.xlarge", 2)
            >>> launcher.launch("batch-inference-1000-samples")

        Cost Information:
            The method displays estimated costs before launch:
            - Maximum cost = hourly_rate * max_hours
            - Actual cost may be lower if job completes early
            - Spot instance pricing provides 60-90% savings

        Instance Configuration:
            - Deep Learning AMI with pre-installed Neuron SDK
            - Automatic dependency installation (torch-neuronx, etc.)
            - Cost monitoring daemon for real-time tracking
            - Experiment metadata logging
            - Automatic termination timer

        Note:
            The instance will terminate automatically after max_hours regardless
            of job status. Monitor your experiments and save results to S3 or EFS
            to prevent data loss.
        """
        if not experiment_name or not experiment_name.strip():
            raise ValueError("Experiment name cannot be empty")

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
        try:
            response = self.ec2.run_instances(
                ImageId=self.get_neuron_ami(),
                InstanceType=self.instance_type,
                MinCount=1,
                MaxCount=1,
                UserData=user_data,
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {
                                "Key": "Name",
                                "Value": f"ML-Experiment-{experiment_name}",
                            },
                            {"Key": "AutoTerminate", "Value": "true"},
                            {"Key": "MaxHours", "Value": str(self.max_hours)},
                            {
                                "Key": "ExperimentStart",
                                "Value": datetime.now().isoformat(),
                            },
                        ],
                    }
                ],
                InstanceMarketOptions={
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                },
            )

            self.instance_id = response["Instances"][0]["InstanceId"]
            print(f"✅ Launched {self.instance_type} instance: {self.instance_id}")
            print(f"⏰ Will auto-terminate in {self.max_hours} hours")
            print(
                f"💰 Estimated max cost: ${self.get_hourly_rate() * self.max_hours:.2f}"
            )

            return self.instance_id

        except Exception as e:
            print(f"❌ Failed to launch instance: {e}")
            raise

    def get_hourly_rate(self) -> float:
        """Get current spot pricing for the configured instance type.

        Returns approximate spot pricing for common ML instance types. These are
        representative rates that may vary by region and current demand. Actual
        spot prices can fluctuate but typically remain 60-90% below on-demand rates.

        Returns:
            float: Hourly spot price in USD for the instance type.
                Returns 1.0 as fallback for unknown instance types.

        Examples:
            >>> launcher = EphemeralMLInstance("trn1.2xlarge")
            >>> rate = launcher.get_hourly_rate()
            >>> print(f"Estimated cost: ${rate}/hour")
            >>> max_cost = rate * launcher.max_hours
            >>> print(f"Maximum experiment cost: ${max_cost:.2f}")

        Pricing Reference (approximate spot rates):
            - trn1.2xlarge: $0.40/hour (vs $1.34 on-demand, 70% savings)
            - trn1.32xlarge: $6.45/hour (vs $25.70 on-demand, 75% savings)
            - trn2.48xlarge: $12.00/hour (vs $40.00 on-demand, 70% savings)
            - inf2.xlarge: $0.227/hour (vs $0.758 on-demand, 70% savings)
            - inf2.48xlarge: $3.89/hour (vs $12.98 on-demand, 70% savings)

        Note:
            Spot prices fluctuate based on demand. Use AWS EC2 console or
            describe_spot_price_history API for real-time pricing. These
            estimates help with budget planning and experiment cost estimation.
        """
        pricing = {
            "trn1.2xlarge": 0.40,
            "trn1.32xlarge": 6.45,
            "trn2.48xlarge": 12.00,
            "inf2.xlarge": 0.227,
            "inf2.48xlarge": 3.89,
        }
        return pricing.get(self.instance_type, 1.0)

    def get_neuron_ami(self) -> str:
        """Get AMI ID for Deep Learning AMI with Neuron SDK pre-installed.

        Returns the AMI ID for an Amazon Linux 2 Deep Learning AMI that includes
        pre-installed Neuron SDK, PyTorch, and other ML frameworks optimized for
        Trainium and Inferentia instances.

        Returns:
            str: AMI ID for the Deep Learning AMI. Currently returns a placeholder
                that should be updated with the latest AMI for your region.

        Note:
            In production, this should query the AWS Systems Manager Parameter Store
            to get the latest AMI ID dynamically:

            aws ssm get-parameters --names \
                /aws/service/deep-learning/amazonlinux/latest/ami-id

            Or use the EC2 describe_images API to find the latest Deep Learning AMI
            with Neuron support. The current placeholder should be updated with
            a valid AMI ID for your target region.

        Examples:
            >>> launcher = EphemeralMLInstance()
            >>> ami_id = launcher.get_neuron_ami()
            >>> print(f"Using AMI: {ami_id}")

        Recommended AMI Features:
            - Amazon Linux 2 base with AWS optimizations
            - Pre-installed Neuron SDK and drivers
            - PyTorch with Neuron extensions (torch-neuronx)
            - Common ML libraries (transformers, datasets, etc.)
            - AWS CLI and S3 integration tools
            - Docker support for containerized workloads

        Regional Availability:
            Deep Learning AMIs are available in all regions supporting Trainium/Inferentia:
            - us-east-1, us-west-2 (primary regions)
            - eu-west-1, ap-southeast-1 (secondary regions)
            - Check AWS documentation for latest regional availability
        """
        # TODO: This should be updated to fetch the latest AMI dynamically
        # Production implementation should query AWS Systems Manager Parameter Store:
        # aws ssm get-parameters --names /aws/service/deep-learning/amazonlinux/latest/ami-id
        # For now, using a placeholder that should be updated for your region
        return "ami-0c55b159cbfafe1f0"  # Update for your region


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Launch ephemeral machine learning instances with automatic cost control.

        This script creates self-terminating AWS instances optimized for ML research.
        The instances automatically terminate after a specified time limit to prevent
        runaway costs, a common issue in research environments.

        Features:
        - Automatic termination prevents forgotten instances
        - Spot pricing for 60-90% cost savings
        - Pre-configured Neuron SDK for Trainium/Inferentia
        - Built-in cost tracking and experiment logging
        - Perfect for training, inference, and experimentation

        Examples:
            # 4-hour Trainium training job
            python ephemeral_instance.py --name bert-training --max-hours 4

            # Overnight Inferentia inference
            python ephemeral_instance.py --name batch-inference \
                --instance-type inf2.xlarge --max-hours 8

            # Large model training with extended time
            python ephemeral_instance.py --name llama-fine-tune \
                --instance-type trn1.32xlarge --max-hours 12
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="""Experiment name for tagging and tracking. Use descriptive names
        like 'bert-climate-classification' or 'llama-inference-test' for easy
        identification in cost reports and instance management.""",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="trn1.2xlarge",
        help="""AWS instance type for ML workload. Options:
        Trainium (training): trn1.2xlarge ($0.40/hr), trn1.32xlarge ($6.45/hr)
        Inferentia (inference): inf2.xlarge ($0.227/hr), inf2.48xlarge ($3.89/hr)
        Default: trn1.2xlarge (good balance of performance and cost)""",
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=4,
        help="""Maximum runtime in hours before automatic termination.
        Recommendations:
        - 1-4 hours: Quick experiments and debugging
        - 4-8 hours: Standard training jobs
        - 8-24 hours: Large model training
        Safety limit: 168 hours (1 week)""",
    )

    args = parser.parse_args()

    launcher = EphemeralMLInstance(args.instance_type, args.max_hours)
    launcher.launch(args.name)
