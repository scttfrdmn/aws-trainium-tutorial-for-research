#!/usr/bin/env python3
"""
Emergency shutdown script for all ML instances
Use when you need to immediately stop all running experiments
"""
import boto3


def emergency_shutdown_all():
    """EMERGENCY: Stop all running ML instances"""
    ec2 = boto3.client("ec2")

    # Get all running instances
    response = ec2.describe_instances(
        Filters=[{"Name": "instance-state-name", "Values": ["running"]}]
    )

    instance_ids = []
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            # Only stop ML instances (safety check)
            tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
            if (
                tags.get("Name", "").startswith("ML-")
                or tags.get("AutoTerminate") == "true"
            ):
                instance_ids.append(instance["InstanceId"])

    if instance_ids:
        print(f"🛑 Found {len(instance_ids)} ML instances to stop:")
        for instance_id in instance_ids:
            print(f"  - {instance_id}")

        confirm = input("Type 'EMERGENCY' to confirm shutdown: ")
        if confirm == "EMERGENCY":
            ec2.terminate_instances(InstanceIds=instance_ids)
            print(f"🛑 Terminated {len(instance_ids)} instances")
        else:
            print("❌ Shutdown cancelled")
    else:
        print("✅ No ML instances running")


if __name__ == "__main__":
    emergency_shutdown_all()
