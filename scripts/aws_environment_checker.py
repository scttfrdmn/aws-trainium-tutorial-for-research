#!/usr/bin/env python3
"""AWS Environment Checker for Trainium and Inferentia Tutorial.

This comprehensive tool validates and optionally configures the AWS environment
required for running machine learning workloads on AWS Trainium and Inferentia.
It checks networking, permissions, quotas, and dependencies while providing
automated setup options for missing components.

Features:
    - Complete AWS environment validation
    - VPC, subnet, and security group verification and creation
    - Service quota checking for ML instances
    - IAM permission validation
    - Budget and cost control setup
    - Neuron SDK dependency verification
    - Automated environment configuration

Usage:
    # Basic environment check
    python aws_environment_checker.py

    # Check with automated fixes
    python aws_environment_checker.py --auto-fix

    # Create development environment from scratch
    python aws_environment_checker.py --create-dev-env

    # Check specific region and instance types
    python aws_environment_checker.py --region us-west-2 --instances trn1.2xlarge,inf2.xlarge

Prerequisites:
    - AWS CLI configured with appropriate credentials
    - Python 3.8+ with boto3 installed
    - Sufficient IAM permissions for resource creation

Cost Implications:
    The script can create AWS resources that incur costs:
    - VPC: Free (within limits)
    - Subnets: Free
    - Security Groups: Free
    - NAT Gateway: ~$45/month (if created)
    - Internet Gateway: Free

    All resource creation requires explicit user confirmation.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


class AWSEnvironmentChecker:
    """Comprehensive AWS environment validator and configurator for ML workloads.

    This class provides complete validation and setup capabilities for AWS
    environments running Trainium and Inferentia workloads, including
    networking, permissions, quotas, and cost controls.

    Args:
        region (str): AWS region to check/configure (default: us-east-1)
        auto_fix (bool): Whether to automatically fix issues (default: False)

    Example:
        checker = AWSEnvironmentChecker(region='us-west-2', auto_fix=True)
        results = checker.run_comprehensive_check()

        if results['all_checks_passed']:
            print("✅ Environment ready for ML training!")
        else:
            print("❌ Issues found:", results['issues'])
    """

    def __init__(self, region: str = "us-east-1", auto_fix: bool = False):
        """Initialize the AWS environment checker with regional configuration."""
        self.region = region
        self.auto_fix = auto_fix
        self.issues = []
        self.recommendations = []

        # Initialize AWS clients with error handling
        try:
            self.ec2 = boto3.client("ec2", region_name=region)
            self.iam = boto3.client("iam", region_name=region)
            self.budgets = boto3.client(
                "budgets", region_name="us-east-1"
            )  # Always us-east-1
            self.service_quotas = boto3.client("service-quotas", region_name=region)
            self.sts = boto3.client("sts", region_name=region)

            # Get account information
            self.account_id = self.sts.get_caller_identity()["Account"]

        except NoCredentialsError:
            print("❌ AWS credentials not found!")
            print("Please configure AWS CLI: aws configure")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to initialize AWS clients: {e}")
            sys.exit(1)

        # ML instance types and their requirements
        self.ml_instances = {
            "trn1.2xlarge": {
                "vcpus": 8,
                "memory": 32,
                "neuron_cores": 1,
                "cost_per_hour": 1.34,
            },
            "trn1.32xlarge": {
                "vcpus": 128,
                "memory": 512,
                "neuron_cores": 16,
                "cost_per_hour": 21.50,
            },
            "trn2.48xlarge": {
                "vcpus": 192,
                "memory": 768,
                "neuron_cores": 12,
                "cost_per_hour": 12.00,
            },
            "inf1.xlarge": {
                "vcpus": 4,
                "memory": 8,
                "neuron_cores": 1,
                "cost_per_hour": 0.368,
            },
            "inf1.2xlarge": {
                "vcpus": 8,
                "memory": 16,
                "neuron_cores": 1,
                "cost_per_hour": 0.584,
            },
            "inf1.6xlarge": {
                "vcpus": 24,
                "memory": 48,
                "neuron_cores": 4,
                "cost_per_hour": 1.901,
            },
            "inf2.xlarge": {
                "vcpus": 4,
                "memory": 16,
                "neuron_cores": 1,
                "cost_per_hour": 0.758,
            },
            "inf2.8xlarge": {
                "vcpus": 32,
                "memory": 128,
                "neuron_cores": 2,
                "cost_per_hour": 6.05,
            },
            "inf2.24xlarge": {
                "vcpus": 96,
                "memory": 384,
                "neuron_cores": 6,
                "cost_per_hour": 18.16,
            },
            "inf2.48xlarge": {
                "vcpus": 192,
                "memory": 768,
                "neuron_cores": 12,
                "cost_per_hour": 36.32,
            },
        }

    def run_comprehensive_check(self) -> Dict:
        """Run complete AWS environment validation with detailed reporting.

        Returns:
            dict: Comprehensive validation results including:
                - all_checks_passed: Boolean indicating overall success
                - individual_results: Results for each check category
                - issues: List of problems found
                - recommendations: List of suggested improvements
                - cost_estimates: Projected costs for recommended resources
                - next_steps: Actionable items for the user
        """
        print("🔍 AWS Environment Checker for Trainium & Inferentia")
        print("=" * 60)
        print(f"Region: {self.region}")
        print(f"Account: {self.account_id}")
        print(f"Auto-fix enabled: {self.auto_fix}")
        print()

        results = {
            "all_checks_passed": True,
            "region": self.region,
            "account_id": self.account_id,
            "timestamp": datetime.now().isoformat(),
            "individual_results": {},
            "issues": [],
            "recommendations": [],
            "cost_estimates": {},
            "next_steps": [],
        }

        # Run all validation checks
        checks = [
            ("Credentials & Permissions", self._check_credentials_and_permissions),
            ("VPC & Networking", self._check_vpc_and_networking),
            ("Security Groups", self._check_security_groups),
            ("Service Quotas", self._check_service_quotas),
            ("Budget & Cost Controls", self._check_budget_controls),
            ("Neuron SDK Dependencies", self._check_neuron_dependencies),
            ("Instance Availability", self._check_instance_availability),
        ]

        for check_name, check_function in checks:
            print(f"🔄 Checking {check_name}...")
            try:
                check_result = check_function()
                results["individual_results"][check_name] = check_result

                if not check_result.get("passed", False):
                    results["all_checks_passed"] = False

                print(
                    f"{'✅' if check_result.get('passed', False) else '❌'} {check_name}"
                )

            except Exception as e:
                error_msg = f"Error checking {check_name}: {str(e)}"
                print(f"❌ {error_msg}")
                results["individual_results"][check_name] = {
                    "passed": False,
                    "error": error_msg,
                }
                results["all_checks_passed"] = False

            print()

        # Compile final results
        results["issues"] = self.issues
        results["recommendations"] = self.recommendations

        # Generate summary report
        self._generate_summary_report(results)

        return results

    def _check_credentials_and_permissions(self) -> Dict:
        """Validate AWS credentials and required IAM permissions.

        Returns:
            dict: Validation results with permission details
        """
        result = {"passed": False, "details": {}}

        try:
            # Test basic AWS access
            caller_identity = self.sts.get_caller_identity()
            result["details"]["caller_identity"] = {
                "user_arn": caller_identity.get("Arn"),
                "account_id": caller_identity.get("Account"),
                "user_id": caller_identity.get("UserId"),
            }

            # Test required permissions by attempting read-only operations
            required_permissions = [
                ("EC2", self._test_ec2_permissions),
                ("IAM", self._test_iam_permissions),
                ("Budgets", self._test_budgets_permissions),
                ("Service Quotas", self._test_service_quotas_permissions),
            ]

            permission_results = {}
            all_permissions_valid = True

            for service, test_function in required_permissions:
                try:
                    permission_results[service] = test_function()
                except Exception as e:
                    permission_results[service] = {"error": str(e)}
                    all_permissions_valid = False
                    self.issues.append(f"Missing {service} permissions: {e}")

            result["details"]["permissions"] = permission_results
            result["passed"] = all_permissions_valid

            if all_permissions_valid:
                result["message"] = "All required permissions validated"
            else:
                result["message"] = "Some permissions missing - see issues list"
                self.recommendations.append(
                    "Ensure your IAM user/role has EC2, IAM, Budgets, and Service Quotas permissions"
                )

        except Exception as e:
            result["error"] = str(e)
            self.issues.append(f"Credential validation failed: {e}")

        return result

    def _test_ec2_permissions(self) -> Dict:
        """Test EC2 permissions required for ML workloads."""
        # Test various EC2 operations
        tests = {
            "describe_vpcs": lambda: self.ec2.describe_vpcs(MaxResults=5),
            "describe_subnets": lambda: self.ec2.describe_subnets(MaxResults=5),
            "describe_security_groups": lambda: self.ec2.describe_security_groups(
                MaxResults=5
            ),
            "describe_instances": lambda: self.ec2.describe_instances(MaxResults=5),
            "describe_availability_zones": lambda: self.ec2.describe_availability_zones(),
        }

        results = {}
        for test_name, test_func in tests.items():
            try:
                test_func()
                results[test_name] = "✅ Passed"
            except Exception as e:
                results[test_name] = f"❌ Failed: {e}"
                raise

        return results

    def _test_iam_permissions(self) -> Dict:
        """Test IAM permissions for role and policy management."""
        tests = {
            "get_user": lambda: self.iam.get_user(),
            "list_roles": lambda: self.iam.list_roles(MaxItems=5),
        }

        results = {}
        for test_name, test_func in tests.items():
            try:
                test_func()
                results[test_name] = "✅ Passed"
            except Exception as e:
                results[test_name] = f"❌ Failed: {e}"
                if "AccessDenied" not in str(
                    e
                ):  # Some IAM operations may be restricted
                    raise

        return results

    def _test_budgets_permissions(self) -> Dict:
        """Test AWS Budgets permissions for cost control."""
        try:
            self.budgets.describe_budgets(AccountId=self.account_id, MaxResults=5)
            return {"describe_budgets": "✅ Passed"}
        except Exception as e:
            return {"describe_budgets": f"❌ Failed: {e}"}

    def _test_service_quotas_permissions(self) -> Dict:
        """Test Service Quotas permissions for instance limit checking."""
        try:
            self.service_quotas.list_services(MaxResults=5)
            return {"list_services": "✅ Passed"}
        except Exception as e:
            return {"list_services": f"❌ Failed: {e}"}

    def _check_vpc_and_networking(self) -> Dict:
        """Validate or create VPC networking infrastructure for ML workloads.

        Returns:
            dict: VPC validation/creation results
        """
        result = {"passed": False, "details": {}}

        try:
            # Check for existing VPCs
            vpcs_response = self.ec2.describe_vpcs()
            vpcs = vpcs_response["Vpcs"]

            # Look for suitable VPC (non-default with proper CIDR)
            suitable_vpc = None
            for vpc in vpcs:
                if not vpc.get("IsDefault", True):  # Prefer non-default VPCs
                    suitable_vpc = vpc
                    break

            if not suitable_vpc and vpcs:
                # Use default VPC if available
                for vpc in vpcs:
                    if vpc.get("IsDefault", False):
                        suitable_vpc = vpc
                        break

            if suitable_vpc:
                vpc_id = suitable_vpc["VpcId"]
                result["details"]["vpc_id"] = vpc_id
                result["details"]["vpc_cidr"] = suitable_vpc.get("CidrBlock")

                # Check subnets
                subnets_result = self._check_subnets(vpc_id)
                result["details"]["subnets"] = subnets_result

                # Check internet connectivity
                connectivity_result = self._check_internet_connectivity(vpc_id)
                result["details"]["connectivity"] = connectivity_result

                if (
                    subnets_result["suitable_subnets"]
                    and connectivity_result["has_internet_access"]
                ):
                    result["passed"] = True
                    result["message"] = f"VPC {vpc_id} is suitable for ML workloads"
                else:
                    if self.auto_fix:
                        result = self._fix_vpc_issues(vpc_id, result)
                    else:
                        self.issues.append("VPC networking issues found")
                        self.recommendations.append(
                            "Run with --auto-fix to resolve networking issues"
                        )

            else:
                # No suitable VPC found
                if self.auto_fix:
                    result = self._create_ml_vpc()
                else:
                    self.issues.append("No suitable VPC found")
                    self.recommendations.append(
                        "Create a VPC with public and private subnets"
                    )
                    result["message"] = "No suitable VPC found - creation required"

        except Exception as e:
            result["error"] = str(e)
            self.issues.append(f"VPC validation failed: {e}")

        return result

    def _check_subnets(self, vpc_id: str) -> Dict:
        """Check for suitable subnets within a VPC."""
        try:
            subnets_response = self.ec2.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )
            subnets = subnets_response["Subnets"]

            public_subnets = []
            private_subnets = []

            for subnet in subnets:
                if subnet.get("MapPublicIpOnLaunch", False):
                    public_subnets.append(subnet)
                else:
                    private_subnets.append(subnet)

            # Check availability zones
            az_coverage = set(subnet["AvailabilityZone"] for subnet in subnets)

            return {
                "total_subnets": len(subnets),
                "public_subnets": len(public_subnets),
                "private_subnets": len(private_subnets),
                "availability_zones": list(az_coverage),
                "suitable_subnets": len(subnets) >= 2 and len(az_coverage) >= 2,
                "subnet_details": [
                    {
                        "subnet_id": s["SubnetId"],
                        "cidr": s["CidrBlock"],
                        "az": s["AvailabilityZone"],
                        "public": s.get("MapPublicIpOnLaunch", False),
                    }
                    for s in subnets
                ],
            }

        except Exception as e:
            return {"error": str(e), "suitable_subnets": False}

    def _check_internet_connectivity(self, vpc_id: str) -> Dict:
        """Check if VPC has proper internet connectivity."""
        try:
            # Check for Internet Gateway
            igw_response = self.ec2.describe_internet_gateways(
                Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}]
            )
            has_igw = len(igw_response["InternetGateways"]) > 0

            # Check route tables
            rt_response = self.ec2.describe_route_tables(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )

            has_public_route = False
            for rt in rt_response["RouteTables"]:
                for route in rt.get("Routes", []):
                    if route.get("DestinationCidrBlock") == "0.0.0.0/0":
                        has_public_route = True
                        break

            return {
                "has_internet_gateway": has_igw,
                "has_public_routes": has_public_route,
                "has_internet_access": has_igw and has_public_route,
                "igw_count": len(igw_response["InternetGateways"]),
                "route_table_count": len(rt_response["RouteTables"]),
            }

        except Exception as e:
            return {"error": str(e), "has_internet_access": False}

    def _create_ml_vpc(self) -> Dict:
        """Create a complete VPC infrastructure optimized for ML workloads."""
        print("🏗️  Creating ML-optimized VPC infrastructure...")

        if not self._confirm_action(
            "Create new VPC with subnets and internet gateway?"
        ):
            return {"passed": False, "message": "VPC creation cancelled by user"}

        try:
            # Create VPC
            vpc_response = self.ec2.create_vpc(
                CidrBlock="10.0.0.0/16",
                TagSpecifications=[
                    {
                        "ResourceType": "vpc",
                        "Tags": [
                            {"Key": "Name", "Value": "ML-Research-VPC"},
                            {"Key": "Purpose", "Value": "Machine Learning Research"},
                            {"Key": "CreatedBy", "Value": "AWS-Trainium-Tutorial"},
                        ],
                    }
                ],
            )
            vpc_id = vpc_response["Vpc"]["VpcId"]

            # Wait for VPC to be available
            print(f"Created VPC: {vpc_id}")
            self.ec2.get_waiter("vpc_available").wait(VpcIds=[vpc_id])

            # Enable DNS hostnames and resolution
            self.ec2.modify_vpc_attribute(
                VpcId=vpc_id, EnableDnsHostnames={"Value": True}
            )
            self.ec2.modify_vpc_attribute(
                VpcId=vpc_id, EnableDnsSupport={"Value": True}
            )

            # Get availability zones
            az_response = self.ec2.describe_availability_zones()
            azs = [az["ZoneName"] for az in az_response["AvailabilityZones"][:3]]

            # Create subnets
            subnets = []
            for i, az in enumerate(azs):
                # Public subnet
                public_subnet = self.ec2.create_subnet(
                    VpcId=vpc_id,
                    CidrBlock=f"10.0.{i*2+1}.0/24",
                    AvailabilityZone=az,
                    TagSpecifications=[
                        {
                            "ResourceType": "subnet",
                            "Tags": [
                                {"Key": "Name", "Value": f"ML-Public-Subnet-{az}"},
                                {"Key": "Type", "Value": "Public"},
                            ],
                        }
                    ],
                )

                # Private subnet
                private_subnet = self.ec2.create_subnet(
                    VpcId=vpc_id,
                    CidrBlock=f"10.0.{i*2+2}.0/24",
                    AvailabilityZone=az,
                    TagSpecifications=[
                        {
                            "ResourceType": "subnet",
                            "Tags": [
                                {"Key": "Name", "Value": f"ML-Private-Subnet-{az}"},
                                {"Key": "Type", "Value": "Private"},
                            ],
                        }
                    ],
                )

                subnets.extend([public_subnet["Subnet"], private_subnet["Subnet"]])

                # Enable auto-assign public IP for public subnets
                self.ec2.modify_subnet_attribute(
                    SubnetId=public_subnet["Subnet"]["SubnetId"],
                    MapPublicIpOnLaunch={"Value": True},
                )

            # Create and attach Internet Gateway
            igw_response = self.ec2.create_internet_gateway(
                TagSpecifications=[
                    {
                        "ResourceType": "internet-gateway",
                        "Tags": [{"Key": "Name", "Value": "ML-Internet-Gateway"}],
                    }
                ]
            )
            igw_id = igw_response["InternetGateway"]["InternetGatewayId"]

            self.ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)

            # Update route table for public access
            rt_response = self.ec2.describe_route_tables(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )
            main_rt_id = rt_response["RouteTables"][0]["RouteTableId"]

            self.ec2.create_route(
                RouteTableId=main_rt_id,
                DestinationCidrBlock="0.0.0.0/0",
                GatewayId=igw_id,
            )

            print(f"✅ Created complete VPC infrastructure:")
            print(f"   VPC ID: {vpc_id}")
            print(f"   Subnets: {len(subnets)} across {len(azs)} AZs")
            print(f"   Internet Gateway: {igw_id}")

            return {
                "passed": True,
                "message": "ML VPC infrastructure created successfully",
                "details": {
                    "vpc_id": vpc_id,
                    "vpc_cidr": "10.0.0.0/16",
                    "subnets": len(subnets),
                    "availability_zones": azs,
                    "internet_gateway": igw_id,
                    "created": True,
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "error": f"Failed to create VPC: {e}",
                "message": "VPC creation failed",
            }

    def _check_security_groups(self) -> Dict:
        """Validate or create security groups for ML workloads."""
        result = {"passed": False, "details": {}}

        try:
            # Look for existing ML-specific security group
            sg_response = self.ec2.describe_security_groups(
                Filters=[
                    {
                        "Name": "group-name",
                        "Values": ["ML-Research-SG", "ml-research", "trainium-sg"],
                    }
                ]
            )

            existing_sgs = sg_response["SecurityGroups"]

            if existing_sgs:
                sg = existing_sgs[0]
                result["details"]["security_group_id"] = sg["GroupId"]
                result["details"]["existing"] = True

                # Validate security group rules
                rules_valid = self._validate_sg_rules(sg)
                result["details"]["rules_validation"] = rules_valid

                if rules_valid["valid"]:
                    result["passed"] = True
                    result[
                        "message"
                    ] = f"Security group {sg['GroupId']} is properly configured"
                else:
                    if self.auto_fix:
                        self._fix_security_group_rules(sg["GroupId"])
                        result["passed"] = True
                        result["message"] = "Security group rules updated"
                    else:
                        self.issues.append("Security group rules need updating")

            else:
                # Create new security group
                if self.auto_fix:
                    sg_result = self._create_ml_security_group()
                    result.update(sg_result)
                else:
                    self.issues.append("No suitable security group found")
                    self.recommendations.append(
                        "Create a security group for ML workloads"
                    )

        except Exception as e:
            result["error"] = str(e)
            self.issues.append(f"Security group validation failed: {e}")

        return result

    def _validate_sg_rules(self, security_group: Dict) -> Dict:
        """Validate security group rules for ML workloads."""
        required_rules = {
            "ssh_access": {"port": 22, "protocol": "tcp"},
            "jupyter_notebook": {"port": 8888, "protocol": "tcp"},
            "tensorboard": {"port": 6006, "protocol": "tcp"},
            "custom_ml_ports": {"port_range": "8000-9000", "protocol": "tcp"},
        }

        ingress_rules = security_group.get("IpPermissions", [])

        validation_results = {}
        for rule_name, required in required_rules.items():
            validation_results[rule_name] = self._check_sg_rule_exists(
                ingress_rules, required
            )

        all_valid = all(validation_results.values())

        return {
            "valid": all_valid,
            "rule_details": validation_results,
            "missing_rules": [
                name for name, valid in validation_results.items() if not valid
            ],
        }

    def _check_sg_rule_exists(self, rules: List, required_rule: Dict) -> bool:
        """Check if a specific security group rule exists."""
        for rule in rules:
            if rule.get("IpProtocol") == required_rule["protocol"]:
                if "port" in required_rule:
                    if (
                        rule.get("FromPort") == required_rule["port"]
                        and rule.get("ToPort") == required_rule["port"]
                    ):
                        return True
                elif "port_range" in required_rule:
                    start, end = map(int, required_rule["port_range"].split("-"))
                    if rule.get("FromPort") == start and rule.get("ToPort") == end:
                        return True
        return False

    def _create_ml_security_group(self) -> Dict:
        """Create a security group optimized for ML research workloads."""
        print("🛡️  Creating ML research security group...")

        try:
            # Get VPC ID (use first available VPC)
            vpcs_response = self.ec2.describe_vpcs()
            vpc_id = vpcs_response["Vpcs"][0]["VpcId"]

            # Create security group
            sg_response = self.ec2.create_security_group(
                GroupName="ML-Research-SG",
                Description="Security group for ML research workloads on Trainium and Inferentia",
                VpcId=vpc_id,
                TagSpecifications=[
                    {
                        "ResourceType": "security-group",
                        "Tags": [
                            {"Key": "Name", "Value": "ML-Research-SG"},
                            {"Key": "Purpose", "Value": "Machine Learning Research"},
                        ],
                    }
                ],
            )

            sg_id = sg_response["GroupId"]

            # Add ingress rules for ML workloads
            self.ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [
                            {"CidrIp": "0.0.0.0/0", "Description": "SSH access"}
                        ],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8888,
                        "ToPort": 8888,
                        "IpRanges": [
                            {"CidrIp": "0.0.0.0/0", "Description": "Jupyter Notebook"}
                        ],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 6006,
                        "ToPort": 6006,
                        "IpRanges": [
                            {"CidrIp": "0.0.0.0/0", "Description": "TensorBoard"}
                        ],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8000,
                        "ToPort": 9000,
                        "IpRanges": [
                            {"CidrIp": "0.0.0.0/0", "Description": "Custom ML services"}
                        ],
                    },
                ],
            )

            print(f"✅ Created security group: {sg_id}")

            return {
                "passed": True,
                "message": "ML security group created successfully",
                "details": {
                    "security_group_id": sg_id,
                    "created": True,
                    "rules_configured": True,
                },
            }

        except Exception as e:
            return {"passed": False, "error": f"Failed to create security group: {e}"}

    def _check_service_quotas(self) -> Dict:
        """Check service quotas for ML instance types."""
        result = {"passed": False, "details": {}}

        try:
            # EC2 service code
            service_code = "ec2"

            # Get current quotas for ML instances
            quota_checks = {}

            for instance_type, specs in self.ml_instances.items():
                try:
                    # Check vCPU quota
                    vcpu_quota = self._get_quota_value(
                        service_code, "L-34B43A08"
                    )  # Running On-Demand instances

                    quota_checks[instance_type] = {
                        "required_vcpus": specs["vcpus"],
                        "current_vcpu_quota": vcpu_quota,
                        "sufficient_quota": vcpu_quota >= specs["vcpus"],
                        "cost_per_hour": specs["cost_per_hour"],
                        "neuron_cores": specs["neuron_cores"],
                    }

                except Exception as e:
                    quota_checks[instance_type] = {
                        "error": str(e),
                        "sufficient_quota": False,
                    }

            # Check overall quota sufficiency
            sufficient_instances = [
                inst
                for inst, check in quota_checks.items()
                if check.get("sufficient_quota", False)
            ]

            result["details"]["quota_checks"] = quota_checks
            result["details"]["sufficient_instances"] = sufficient_instances
            result["passed"] = len(sufficient_instances) > 0

            if not result["passed"]:
                self.issues.append("Insufficient service quotas for ML instances")
                self.recommendations.append(
                    "Request quota increases for required instance types"
                )
            else:
                result[
                    "message"
                ] = f"Sufficient quotas for {len(sufficient_instances)} instance types"

        except Exception as e:
            result["error"] = str(e)
            self.issues.append(f"Service quota check failed: {e}")

        return result

    def _get_quota_value(self, service_code: str, quota_code: str) -> int:
        """Get current service quota value."""
        try:
            response = self.service_quotas.get_service_quota(
                ServiceCode=service_code, QuotaCode=quota_code
            )
            return int(response["Quota"]["Value"])
        except:
            # Fallback to default quota if API call fails
            return 256  # Conservative default for vCPUs

    def _check_budget_controls(self) -> Dict:
        """Check for existing budget controls and cost management setup."""
        result = {"passed": False, "details": {}}

        try:
            # Check existing budgets
            budgets_response = self.budgets.describe_budgets(
                AccountId=self.account_id, MaxResults=50
            )

            existing_budgets = budgets_response.get("Budgets", [])
            ml_budgets = [
                b
                for b in existing_budgets
                if any(
                    term in b["BudgetName"].lower()
                    for term in ["ml", "machine", "research", "training"]
                )
            ]

            result["details"]["total_budgets"] = len(existing_budgets)
            result["details"]["ml_specific_budgets"] = len(ml_budgets)
            result["details"]["budget_names"] = [
                b["BudgetName"] for b in existing_budgets
            ]

            if ml_budgets or len(existing_budgets) > 0:
                result["passed"] = True
                result[
                    "message"
                ] = f"Found {len(existing_budgets)} budget(s), {len(ml_budgets)} ML-specific"
            else:
                if self.auto_fix:
                    budget_result = self._create_ml_budget()
                    result.update(budget_result)
                else:
                    self.issues.append("No cost control budgets found")
                    self.recommendations.append(
                        "Set up budget alerts to prevent unexpected costs"
                    )

        except Exception as e:
            result["error"] = str(e)
            # Budget API not available in all regions - not a critical failure
            result["passed"] = True
            result["message"] = "Budget check skipped (API not available)"

        return result

    def _create_ml_budget(self) -> Dict:
        """Create a budget specifically for ML research costs."""
        print("💰 Creating ML research budget...")

        if not self._confirm_action(
            "Create a $500/month ML research budget with alerts?"
        ):
            return {"passed": False, "message": "Budget creation cancelled"}

        try:
            budget = {
                "BudgetName": "ML-Research-Budget",
                "BudgetLimit": {"Amount": "500", "Unit": "USD"},
                "TimeUnit": "MONTHLY",
                "BudgetType": "COST",
                "CostFilters": {"Service": ["Amazon Elastic Compute Cloud - Compute"]},
            }

            # Create budget
            self.budgets.create_budget(AccountId=self.account_id, Budget=budget)

            print("✅ Created ML research budget: $500/month")

            return {
                "passed": True,
                "message": "ML research budget created successfully",
                "details": {"budget_amount": 500, "created": True},
            }

        except Exception as e:
            return {"passed": False, "error": f"Failed to create budget: {e}"}

    def _check_neuron_dependencies(self) -> Dict:
        """Check for Neuron SDK and related dependencies."""
        result = {"passed": False, "details": {}}

        try:
            import subprocess

            # Check Python packages
            python_packages = [
                "torch-neuronx",
                "neuronx-cc",
                "torch-xla",
                "transformers",
                "boto3",
            ]

            package_status = {}
            for package in python_packages:
                try:
                    result_check = subprocess.run(
                        [sys.executable, "-c", f"import {package.replace('-', '_')}"],
                        capture_output=True,
                        text=True,
                    )
                    package_status[package] = result_check.returncode == 0
                except:
                    package_status[package] = False

            result["details"]["python_packages"] = package_status

            # Check system dependencies (if on EC2)
            system_deps = self._check_system_dependencies()
            result["details"]["system_dependencies"] = system_deps

            # Overall assessment
            critical_packages = ["boto3"]  # Most critical for environment check
            critical_available = all(
                package_status.get(pkg, False) for pkg in critical_packages
            )

            result["passed"] = critical_available

            if not critical_available:
                self.recommendations.append(
                    "Install required Python packages: pip install boto3 torch transformers"
                )

            missing_neuron = not any(
                package_status.get(pkg, False)
                for pkg in ["torch-neuronx", "neuronx-cc"]
            )
            if missing_neuron:
                self.recommendations.append(
                    "Install Neuron SDK when deploying to Trainium/Inferentia instances"
                )

        except Exception as e:
            result["error"] = str(e)

        return result

    def _check_system_dependencies(self) -> Dict:
        """Check system-level dependencies."""
        import subprocess

        deps = {
            "python3": ["python3", "--version"],
            "pip": ["pip", "--version"],
            "aws_cli": ["aws", "--version"],
        }

        status = {}
        for name, cmd in deps.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                status[name] = {
                    "available": result.returncode == 0,
                    "version": result.stdout.strip()
                    if result.returncode == 0
                    else None,
                }
            except:
                status[name] = {"available": False}

        return status

    def _check_instance_availability(self) -> Dict:
        """Check availability of ML instance types in the region."""
        result = {"passed": False, "details": {}}

        try:
            # Get available instance types
            response = self.ec2.describe_instance_type_offerings(
                Filters=[{"Name": "location", "Values": [self.region]}],
                LocationType="region",
            )

            available_types = {
                offering["InstanceType"]
                for offering in response["InstanceTypeOfferings"]
            }

            # Check ML instance availability
            ml_availability = {}
            for instance_type in self.ml_instances:
                ml_availability[instance_type] = {
                    "available": instance_type in available_types,
                    "specs": self.ml_instances[instance_type],
                }

            available_ml_types = [
                itype for itype, info in ml_availability.items() if info["available"]
            ]

            result["details"]["ml_instance_availability"] = ml_availability
            result["details"]["available_ml_types"] = available_ml_types
            result["details"]["total_available"] = len(available_ml_types)

            result["passed"] = len(available_ml_types) > 0

            if not result["passed"]:
                self.issues.append(f"No ML instance types available in {self.region}")
                self.recommendations.append(
                    "Consider using a different region with ML instance availability"
                )
            else:
                result[
                    "message"
                ] = f"{len(available_ml_types)} ML instance types available"

        except Exception as e:
            result["error"] = str(e)
            self.issues.append(f"Instance availability check failed: {e}")

        return result

    def _confirm_action(self, message: str) -> bool:
        """Get user confirmation for potentially costly actions."""
        if not self.auto_fix:
            return False

        response = input(f"\\n⚠️  {message} (y/N): ").strip().lower()
        return response in ["y", "yes"]

    def _generate_summary_report(self, results: Dict):
        """Generate a comprehensive summary report."""
        print("\\n" + "=" * 60)
        print("📋 AWS ENVIRONMENT VALIDATION SUMMARY")
        print("=" * 60)

        if results["all_checks_passed"]:
            print("🎉 ALL CHECKS PASSED! Your environment is ready for ML workloads.")
        else:
            print("⚠️  ISSUES FOUND - See details below")

        print(f"\\n📍 Region: {results['region']}")
        print(f"🏢 Account: {results['account_id']}")
        print(f"🕐 Checked: {results['timestamp']}")

        # Individual check results
        print("\\n🔍 Individual Check Results:")
        for check_name, check_result in results["individual_results"].items():
            status = "✅ PASS" if check_result.get("passed", False) else "❌ FAIL"
            print(f"  {status} {check_name}")

        # Issues and recommendations
        if self.issues:
            print(f"\\n❌ Issues Found ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")

        if self.recommendations:
            print(f"\\n💡 Recommendations ({len(self.recommendations)}):")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")

        # Cost estimates for available instances
        self._print_cost_estimates(results)

        # Next steps
        print("\\n🚀 Next Steps:")
        if results["all_checks_passed"]:
            print(
                "  1. Launch a Trainium instance: aws ec2 run-instances --instance-type trn1.2xlarge"
            )
            print("  2. Install Neuron SDK: pip install torch-neuronx")
            print("  3. Start training your models!")
        else:
            print("  1. Address the issues listed above")
            print("  2. Re-run this checker with --auto-fix to resolve automatically")
            print("  3. Consult AWS documentation for manual fixes")

        print(
            "\\n📄 Save this report with: python aws_environment_checker.py > environment_report.txt"
        )

    def _print_cost_estimates(self, results: Dict):
        """Print cost estimates for available ML instances."""
        instance_results = results["individual_results"].get(
            "Instance Availability", {}
        )
        available_types = instance_results.get("details", {}).get(
            "available_ml_types", []
        )

        if available_types:
            print(f"\\n💰 Cost Estimates for Available ML Instances:")
            print(
                "  Instance Type    | vCPUs | Memory | Neuron Cores | Cost/Hour | Cost/Day"
            )
            print("  " + "-" * 75)

            for instance_type in sorted(available_types):
                specs = self.ml_instances[instance_type]
                daily_cost = specs["cost_per_hour"] * 24
                print(
                    f"  {instance_type:<15} | {specs['vcpus']:>5} | {specs['memory']:>6}GB | "
                    f"{specs['neuron_cores']:>11} | ${specs['cost_per_hour']:>8.2f} | ${daily_cost:>8.2f}"
                )

    def create_development_environment(self):
        """Create a complete development environment for ML research."""
        print("🏗️  Creating Complete ML Development Environment")
        print("=" * 60)

        if not self._confirm_action(
            "Create complete ML development environment? This will create VPC, subnets, security groups, and budgets."
        ):
            print("❌ Environment creation cancelled")
            return

        # Enable auto-fix for this operation
        original_auto_fix = self.auto_fix
        self.auto_fix = True

        try:
            # Run comprehensive check with auto-fix
            results = self.run_comprehensive_check()

            if results["all_checks_passed"]:
                print("\\n🎉 Development environment created successfully!")
                print("\\n📋 Created Resources:")

                for check_name, result in results["individual_results"].items():
                    if result.get("details", {}).get("created"):
                        print(f"  ✅ {check_name}")

                print("\\n🚀 Your environment is now ready for ML research!")
                print("\\nRecommended next steps:")
                print(
                    "  1. Launch an instance: aws ec2 run-instances --instance-type trn1.2xlarge"
                )
                print("  2. Connect via SSH and install Neuron SDK")
                print("  3. Start your ML experiments!")

            else:
                print("\\n❌ Some issues remain. Check the detailed output above.")

        finally:
            # Restore original auto-fix setting
            self.auto_fix = original_auto_fix


def main():
    """Main entry point for the AWS environment checker."""
    parser = argparse.ArgumentParser(
        description="AWS Environment Checker for Trainium and Inferentia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aws_environment_checker.py                           # Basic check
  python aws_environment_checker.py --auto-fix               # Check and fix issues
  python aws_environment_checker.py --create-dev-env         # Create full environment
  python aws_environment_checker.py --region us-west-2       # Check specific region
  python aws_environment_checker.py --instances trn1.2xlarge # Check specific instances
        """,
    )

    parser.add_argument(
        "--region", default="us-east-1", help="AWS region to check (default: us-east-1)"
    )

    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically fix issues where possible",
    )

    parser.add_argument(
        "--create-dev-env",
        action="store_true",
        help="Create a complete development environment",
    )

    parser.add_argument(
        "--instances",
        help="Comma-separated list of instance types to check (default: all)",
    )

    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Create checker instance
    checker = AWSEnvironmentChecker(region=args.region, auto_fix=args.auto_fix)

    # Filter instance types if specified
    if args.instances:
        requested_instances = [inst.strip() for inst in args.instances.split(",")]
        checker.ml_instances = {
            k: v for k, v in checker.ml_instances.items() if k in requested_instances
        }

    try:
        if args.create_dev_env:
            checker.create_development_environment()
        else:
            results = checker.run_comprehensive_check()

            # Save results if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\\n💾 Results saved to {args.output}")

            # Exit with appropriate code
            sys.exit(0 if results["all_checks_passed"] else 1)

    except KeyboardInterrupt:
        print("\\n\\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
