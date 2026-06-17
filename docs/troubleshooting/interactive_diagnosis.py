#!/usr/bin/env python3
"""Interactive Troubleshooting Tool for AWS Trainium & Inferentia.

This tool provides an interactive decision tree to help diagnose and resolve
common issues with AWS Neuron hardware and software.

TESTED VERSIONS (Last validated: 2025-06-27):
    - AWS Neuron SDK: 2.20.1
    - torch-neuronx: 2.2.0
    - PyTorch: 2.4.0
    - Tool Status: ✅ Interactive diagnosis ready

Usage:
    python docs/troubleshooting/interactive_diagnosis.py
    python docs/troubleshooting/interactive_diagnosis.py --issue compilation
    python docs/troubleshooting/interactive_diagnosis.py --verbose

Author: Scott Friedman
Date: 2025-06-27
"""

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InteractiveTroubleshooter:
    """Interactive troubleshooting tool with decision tree logic."""

    def __init__(self, verbose: bool = False):
        """Initialize interactive troubleshooter."""
        self.verbose = verbose
        self.session_log = []
        self.solutions_attempted = []

        # Comprehensive troubleshooting decision tree
        self.decision_tree = {
            "start": {
                "question": "What type of issue are you experiencing?",
                "options": {
                    "1": {"text": "Installation/Setup issues", "next": "installation"},
                    "2": {"text": "Model compilation errors", "next": "compilation"},
                    "3": {"text": "Runtime/execution errors", "next": "runtime"},
                    "4": {"text": "Performance issues", "next": "performance"},
                    "5": {"text": "Cost/billing concerns", "next": "cost"},
                    "6": {"text": "Instance/hardware issues", "next": "hardware"},
                    "7": {"text": "Memory errors", "next": "memory"},
                    "8": {"text": "Data loading issues", "next": "data"},
                    "q": {"text": "Quit", "next": "quit"},
                },
            },
            "installation": {
                "question": "What installation issue are you facing?",
                "options": {
                    "1": {
                        "text": "Neuron SDK installation failed",
                        "next": "install_sdk",
                    },
                    "2": {
                        "text": "torch-neuronx import error",
                        "next": "install_torch_neuronx",
                    },
                    "3": {
                        "text": "Package dependency conflicts",
                        "next": "install_dependencies",
                    },
                    "4": {
                        "text": "Environment setup issues",
                        "next": "install_environment",
                    },
                    "b": {"text": "Back to main menu", "next": "start"},
                },
            },
            "compilation": {
                "question": "What compilation error are you seeing?",
                "options": {
                    "1": {
                        "text": "Model compilation timeout",
                        "next": "compile_timeout",
                    },
                    "2": {
                        "text": "Unsupported operator error",
                        "next": "compile_operator",
                    },
                    "3": {
                        "text": "Graph partitioning issues",
                        "next": "compile_partition",
                    },
                    "4": {"text": "Dynamic shape errors", "next": "compile_dynamic"},
                    "5": {"text": "NKI kernel compilation", "next": "compile_nki"},
                    "b": {"text": "Back to main menu", "next": "start"},
                },
            },
            "runtime": {
                "question": "What runtime error are you encountering?",
                "options": {
                    "1": {"text": "Device not found", "next": "runtime_device"},
                    "2": {"text": "Execution hangs/freezes", "next": "runtime_hang"},
                    "3": {"text": "Incorrect results", "next": "runtime_results"},
                    "4": {"text": "Tensor shape mismatches", "next": "runtime_shapes"},
                    "5": {"text": "Multi-device issues", "next": "runtime_multidevice"},
                    "b": {"text": "Back to main menu", "next": "start"},
                },
            },
            "performance": {
                "question": "What performance issue are you experiencing?",
                "options": {
                    "1": {"text": "Slower than expected", "next": "perf_slow"},
                    "2": {"text": "Low utilization", "next": "perf_utilization"},
                    "3": {"text": "Batch size optimization", "next": "perf_batch"},
                    "4": {"text": "Memory inefficiency", "next": "perf_memory"},
                    "5": {"text": "Scaling issues", "next": "perf_scaling"},
                    "b": {"text": "Back to main menu", "next": "start"},
                },
            },
            "cost": {
                "question": "What cost-related concern do you have?",
                "options": {
                    "1": {"text": "Unexpected charges", "next": "cost_unexpected"},
                    "2": {
                        "text": "Instance not terminating",
                        "next": "cost_termination",
                    },
                    "3": {"text": "Cost optimization", "next": "cost_optimization"},
                    "4": {"text": "Billing analysis", "next": "cost_analysis"},
                    "b": {"text": "Back to main menu", "next": "start"},
                },
            },
            "hardware": {
                "question": "What hardware issue are you facing?",
                "options": {
                    "1": {"text": "Instance launch failure", "next": "hw_launch"},
                    "2": {"text": "Neuron device errors", "next": "hw_device"},
                    "3": {"text": "Instance type selection", "next": "hw_selection"},
                    "4": {"text": "Region availability", "next": "hw_region"},
                    "b": {"text": "Back to main menu", "next": "start"},
                },
            },
            "memory": {
                "question": "What memory issue are you experiencing?",
                "options": {
                    "1": {"text": "Out of memory error", "next": "mem_oom"},
                    "2": {"text": "Memory leak", "next": "mem_leak"},
                    "3": {"text": "Memory fragmentation", "next": "mem_fragmentation"},
                    "4": {"text": "Large model loading", "next": "mem_large_model"},
                    "b": {"text": "Back to main menu", "next": "start"},
                },
            },
            "data": {
                "question": "What data loading issue are you facing?",
                "options": {
                    "1": {"text": "S3 data access slow", "next": "data_s3"},
                    "2": {"text": "Data format issues", "next": "data_format"},
                    "3": {"text": "Large dataset handling", "next": "data_large"},
                    "4": {
                        "text": "Preprocessing bottlenecks",
                        "next": "data_preprocessing",
                    },
                    "b": {"text": "Back to main menu", "next": "start"},
                },
            },
        }

        # Solutions database
        self.solutions = self._build_solutions_database()

        logger.info("🔧 Interactive troubleshooter initialized")

    def _build_solutions_database(self) -> dict:
        """Build comprehensive solutions database."""
        return {
            "install_sdk": {
                "problem": "Neuron SDK installation failed",
                "diagnosis": [
                    "Check if you're on a supported instance type",
                    "Verify Ubuntu/Amazon Linux compatibility",
                    "Check internet connectivity",
                    "Verify repository access",
                ],
                "solutions": [
                    {
                        "step": "Verify instance type",
                        "command": "curl -s http://169.254.169.254/latest/meta-data/instance-type",
                        "description": "Must be trn1.* or inf2.* instance",
                    },
                    {
                        "step": "Update package manager",
                        "command": "sudo apt update && sudo apt upgrade -y",
                        "description": "Ensure latest package lists",
                    },
                    {
                        "step": "Install Neuron repository",
                        "command": "curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-NEURON.PUB | sudo apt-key add -",
                        "description": "Add Neuron package repository",
                    },
                    {
                        "step": "Add repository source",
                        "command": "echo 'deb https://apt.repos.neuron.amazonaws.com jammy main' | sudo tee /etc/apt/sources.list.d/neuron.list",
                        "description": "Configure package source",
                    },
                    {
                        "step": "Install Neuron SDK",
                        "command": "sudo apt update && sudo apt install aws-neuronx-dkms aws-neuronx-tools",
                        "description": "Install core Neuron components",
                    },
                ],
                "verification": "neuron-ls",
                "references": [
                    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu20.html"
                ],
            },
            "install_torch_neuronx": {
                "problem": "torch-neuronx import error",
                "diagnosis": [
                    "Check PyTorch version compatibility",
                    "Verify Neuron SDK installation",
                    "Check Python environment",
                    "Verify torch-neuronx installation",
                ],
                "solutions": [
                    {
                        "step": "Check PyTorch version",
                        "command": 'python -c "import torch; print(torch.__version__)"',
                        "description": "Should be 2.4.0+ for compatibility",
                    },
                    {
                        "step": "Install torch-neuronx",
                        "command": "pip install torch-neuronx==2.2.0 --extra-index-url https://pip.repos.neuron.amazonaws.com",
                        "description": "Install compatible torch-neuronx",
                    },
                    {
                        "step": "Verify installation",
                        "command": "python -c \"import torch_neuronx; print('Success')\"",
                        "description": "Test import",
                    },
                    {
                        "step": "Check XLA backend",
                        "command": 'python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"',
                        "description": "Verify XLA Neuron devices",
                    },
                ],
                "verification": "python -c \"import torch_neuronx; import torch_xla.core.xla_model as xm; print('Devices:', xm.get_xla_supported_devices())\"",
                "references": [
                    "https://pytorch.org/xla/",
                    "https://awsdocs-neuron.readthedocs-hosted.com/",
                ],
            },
            "compile_timeout": {
                "problem": "Model compilation timeout",
                "diagnosis": [
                    "Model complexity too high for compilation",
                    "Insufficient memory during compilation",
                    "Network connectivity issues",
                    "Compilation server overload",
                ],
                "solutions": [
                    {
                        "step": "Increase compilation timeout",
                        "command": "export NEURON_COMPILE_TIMEOUT=3600",
                        "description": "Set 1-hour timeout",
                    },
                    {
                        "step": "Use smaller batch size",
                        "command": "# Reduce batch size to 1 for compilation",
                        "description": "Compile with batch_size=1, then use larger batches",
                    },
                    {
                        "step": "Enable compilation caching",
                        "command": "export NEURON_COMPILE_CACHE_URL=/tmp/neuron-cache",
                        "description": "Cache compiled models",
                    },
                    {
                        "step": "Monitor compilation progress",
                        "command": "export NEURON_COMPILE_VERBOSE=1",
                        "description": "Enable verbose compilation logging",
                    },
                ],
                "verification": "Check compilation logs for progress",
                "references": [
                    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/neuron-caching.html"
                ],
            },
            "runtime_device": {
                "problem": "Neuron device not found",
                "diagnosis": [
                    "Neuron driver not loaded",
                    "Wrong instance type",
                    "Device initialization failed",
                    "Driver version mismatch",
                ],
                "solutions": [
                    {
                        "step": "Check instance type",
                        "command": "curl -s http://169.254.169.254/latest/meta-data/instance-type",
                        "description": "Must be trn1.* or inf2.*",
                    },
                    {
                        "step": "Check Neuron devices",
                        "command": "neuron-ls",
                        "description": "List available Neuron devices",
                    },
                    {
                        "step": "Restart Neuron service",
                        "command": "sudo systemctl restart neuron-discovery",
                        "description": "Restart device discovery",
                    },
                    {
                        "step": "Check driver status",
                        "command": "lsmod | grep neuron",
                        "description": "Verify driver is loaded",
                    },
                    {
                        "step": "Reload driver if needed",
                        "command": "sudo rmmod neuron && sudo modprobe neuron",
                        "description": "Reload Neuron driver",
                    },
                ],
                "verification": "neuron-ls should show available devices",
                "references": [
                    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-ls.html"
                ],
            },
            "perf_slow": {
                "problem": "Model running slower than expected",
                "diagnosis": [
                    "Suboptimal batch size",
                    "Model not compiled for Neuron",
                    "Data loading bottleneck",
                    "Inefficient model architecture",
                ],
                "solutions": [
                    {
                        "step": "Verify Neuron compilation",
                        "command": "# Check if model uses torch_neuronx.trace",
                        "description": "Ensure model is compiled for Neuron",
                    },
                    {
                        "step": "Optimize batch size",
                        "command": "# Try batch sizes: 1, 4, 8, 16, 32",
                        "description": "Find optimal batch size for your model",
                    },
                    {
                        "step": "Enable performance profiling",
                        "command": "export NEURON_PROFILE=/tmp/neuron_profile",
                        "description": "Profile model execution",
                    },
                    {
                        "step": "Use data parallelism",
                        "command": "# Utilize multiple Neuron cores",
                        "description": "Distribute across available cores",
                    },
                ],
                "verification": "Monitor throughput and latency metrics",
                "references": [
                    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/finetune_hf_bert_base_cased_squad.html"
                ],
            },
            "cost_unexpected": {
                "problem": "Unexpected high AWS charges",
                "diagnosis": [
                    "Instance not terminated",
                    "Data transfer charges",
                    "Storage costs accumulating",
                    "Multiple instances running",
                ],
                "solutions": [
                    {
                        "step": "Check running instances",
                        "command": "aws ec2 describe-instances --filters 'Name=instance-state-name,Values=running'",
                        "description": "List all running instances",
                    },
                    {
                        "step": "Terminate unused instances",
                        "command": "aws ec2 terminate-instances --instance-ids i-1234567890abcdef0",
                        "description": "Terminate specific instances",
                    },
                    {
                        "step": "Set up billing alerts",
                        "command": "aws budgets create-budget --account-id YOUR_ACCOUNT_ID --budget Budget.json",
                        "description": "Configure cost monitoring",
                    },
                    {
                        "step": "Use spot instances",
                        "command": "# Configure spot instance requests",
                        "description": "Save up to 70% with spot pricing",
                    },
                ],
                "verification": "Check AWS Cost Explorer for detailed breakdown",
                "references": [
                    "https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/"
                ],
            },
            "mem_oom": {
                "problem": "Out of memory error",
                "diagnosis": [
                    "Model too large for instance",
                    "Batch size too large",
                    "Memory leak in code",
                    "Inefficient memory usage",
                ],
                "solutions": [
                    {
                        "step": "Reduce batch size",
                        "command": "# Start with batch_size=1 and increase gradually",
                        "description": "Find maximum feasible batch size",
                    },
                    {
                        "step": "Use gradient checkpointing",
                        "command": "# Enable gradient checkpointing in model",
                        "description": "Trade compute for memory",
                    },
                    {
                        "step": "Monitor memory usage",
                        "command": "neuron-monitor",
                        "description": "Track Neuron memory utilization",
                    },
                    {
                        "step": "Use larger instance",
                        "command": "# Upgrade to trn1.32xlarge or inf2.24xlarge",
                        "description": "Get more memory capacity",
                    },
                ],
                "verification": "Monitor memory usage during execution",
                "references": [
                    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html"
                ],
            },
        }

    def start_interactive_session(self, start_issue: str | None = None) -> None:
        """Start interactive troubleshooting session."""
        print("🔧 AWS Trainium & Inferentia Interactive Troubleshooter")
        print("=" * 60)

        current_node = start_issue if start_issue in self.decision_tree else "start"

        while current_node != "quit":
            if current_node in self.solutions:
                # Display solution
                self._display_solution(current_node)

                # Ask if user wants to try another issue
                print("\n" + "=" * 60)
                choice = (
                    input("Would you like to troubleshoot another issue? (y/n): ")
                    .strip()
                    .lower()
                )
                if choice == "y":
                    current_node = "start"
                else:
                    break
            else:
                # Display decision tree node
                current_node = self._handle_decision_node(current_node)

        self._generate_session_summary()

    def _handle_decision_node(self, node_id: str) -> str:
        """Handle a decision tree node."""
        if node_id not in self.decision_tree:
            print(f"❌ Unknown issue: {node_id}")
            return "start"

        node = self.decision_tree[node_id]

        print(f"\n📋 {node['question']}")
        print("-" * 40)

        for key, option in node["options"].items():
            print(f"  {key}. {option['text']}")

        while True:
            choice = input("\nPlease select an option: ").strip().lower()

            if choice in node["options"]:
                next_node = node["options"][choice]["next"]

                # Log the choice
                self.session_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "question": node["question"],
                        "choice": choice,
                        "choice_text": node["options"][choice]["text"],
                        "next_node": next_node,
                    }
                )

                return next_node
            else:
                print("❌ Invalid choice. Please try again.")

    def _display_solution(self, solution_id: str) -> None:
        """Display a detailed solution."""
        if solution_id not in self.solutions:
            print(f"❌ No solution found for: {solution_id}")
            return

        solution = self.solutions[solution_id]

        print(f"\n🔍 PROBLEM: {solution['problem']}")
        print("=" * 60)

        # Diagnosis
        print("\n📊 DIAGNOSIS:")
        for i, diag in enumerate(solution["diagnosis"], 1):
            print(f"  {i}. {diag}")

        # Solutions
        print("\n🛠️ SOLUTIONS:")
        for i, step in enumerate(solution["solutions"], 1):
            print(f"\n  Step {i}: {step['step']}")
            print(f"    Description: {step['description']}")

            if (
                "command" in step
                and step["command"].strip()
                and not step["command"].startswith("#")
            ):
                print(f"    Command: {step['command']}")

                if self.verbose:
                    try_it = (
                        input("    Would you like to run this command? (y/n): ")
                        .strip()
                        .lower()
                    )
                    if try_it == "y":
                        self._run_command(step["command"])

        # Verification
        print("\n✅ VERIFICATION:")
        print(f"  {solution['verification']}")

        if self.verbose and "verification" in solution:
            verify_cmd = solution.get("verification")
            if verify_cmd and not verify_cmd.startswith("Check"):
                try_verify = (
                    input("  Would you like to run verification? (y/n): ")
                    .strip()
                    .lower()
                )
                if try_verify == "y":
                    self._run_command(verify_cmd)

        # References
        if "references" in solution:
            print("\n📚 ADDITIONAL RESOURCES:")
            for ref in solution["references"]:
                print(f"  - {ref}")

        # Log the solution
        self.solutions_attempted.append(
            {
                "timestamp": datetime.now().isoformat(),
                "solution_id": solution_id,
                "problem": solution["problem"],
            }
        )

    def _run_command(self, command: str) -> None:
        """Safely run a command and display output."""
        print(f"\n🔧 Running: {command}")
        print("-" * 40)

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )

            if result.stdout:
                print("✅ Output:")
                print(result.stdout)

            if result.stderr:
                print("⚠️ Warnings/Errors:")
                print(result.stderr)

            if result.returncode != 0:
                print(f"❌ Command failed with exit code: {result.returncode}")
            else:
                print("✅ Command completed successfully")

        except subprocess.TimeoutExpired:
            print("⏱️ Command timed out after 30 seconds")
        except Exception as e:
            print(f"❌ Error running command: {e}")

        print("-" * 40)

    def _generate_session_summary(self) -> None:
        """Generate and display session summary."""
        print("\n" + "=" * 60)
        print("📋 TROUBLESHOOTING SESSION SUMMARY")
        print("=" * 60)

        if not self.session_log and not self.solutions_attempted:
            print("No troubleshooting steps were taken in this session.")
            return

        print(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Issues explored: {len(self.session_log)}")
        print(f"Solutions attempted: {len(self.solutions_attempted)}")

        if self.solutions_attempted:
            print("\nSolutions attempted:")
            for i, solution in enumerate(self.solutions_attempted, 1):
                timestamp = datetime.fromisoformat(solution["timestamp"]).strftime(
                    "%H:%M:%S"
                )
                print(f"  {i}. [{timestamp}] {solution['problem']}")

        # Save session log
        log_file = (
            f"troubleshooting_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        session_data = {
            "session_log": self.session_log,
            "solutions_attempted": self.solutions_attempted,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(log_file, "w") as f:
                json.dump(session_data, f, indent=2)
            print(f"\n💾 Session log saved: {log_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save session log: {e}")

        print("\n🤝 Need more help?")
        print(
            "  - GitHub Issues: https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/issues"
        )
        print(
            "  - AWS Neuron Documentation: https://awsdocs-neuron.readthedocs-hosted.com/"
        )
        print(
            "  - Community Discussions: https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/discussions"
        )

    def quick_diagnosis(self, issue_type: str) -> None:
        """Provide quick diagnosis for a specific issue type."""
        print(f"🔧 Quick Diagnosis: {issue_type}")
        print("=" * 60)

        if issue_type in self.solutions:
            self._display_solution(issue_type)
        elif issue_type in self.decision_tree:
            print(f"📋 {self.decision_tree[issue_type]['question']}")
            for key, option in self.decision_tree[issue_type]["options"].items():
                if key != "b":  # Skip back option
                    print(f"  - {option['text']}")
        else:
            print(f"❌ Unknown issue type: {issue_type}")
            print("\nAvailable issue types:")
            for key in self.solutions:
                print(f"  - {key}")


def run_system_diagnostics() -> dict:
    """Run comprehensive system diagnostics."""
    print("🔍 Running System Diagnostics...")
    print("=" * 60)

    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {},
        "neuron_info": {},
        "environment": {},
        "connectivity": {},
        "status": "unknown",
    }

    # System information
    try:
        import platform

        diagnostics["system_info"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
        }

        # Get instance metadata if on EC2
        try:
            import requests

            response = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-type", timeout=2
            )
            diagnostics["system_info"]["instance_type"] = response.text
        except Exception:
            diagnostics["system_info"]["instance_type"] = "not_ec2"

        print("✅ System information collected")
    except Exception as e:
        print(f"⚠️ Could not collect system info: {e}")

    # Neuron environment
    try:
        # Check Neuron SDK
        result = subprocess.run(
            ["neuron-ls"], capture_output=True, text=True, timeout=10
        )
        diagnostics["neuron_info"]["neuron_ls"] = {
            "available": result.returncode == 0,
            "output": result.stdout if result.returncode == 0 else result.stderr,
        }

        # Check torch-neuronx
        try:
            import torch_neuronx

            diagnostics["neuron_info"]["torch_neuronx_version"] = (
                torch_neuronx.__version__
            )
        except ImportError:
            diagnostics["neuron_info"]["torch_neuronx_version"] = "not_installed"

        # Check XLA devices
        try:
            import torch_xla.core.xla_model as xm

            devices = xm.get_xla_supported_devices()
            diagnostics["neuron_info"]["xla_devices"] = [str(d) for d in devices]
        except ImportError:
            diagnostics["neuron_info"]["xla_devices"] = "not_available"

        print("✅ Neuron environment analyzed")
    except Exception as e:
        print(f"⚠️ Error analyzing Neuron environment: {e}")
        diagnostics["neuron_info"]["error"] = str(e)

    # Environment variables
    neuron_env_vars = [
        "NEURON_COMPILE_CACHE_URL",
        "NEURON_COMPILE_TIMEOUT",
        "NEURON_PROFILE",
        "NEURON_RT_NUM_CORES",
        "NEURON_RT_EXEC_TIMEOUT",
    ]

    diagnostics["environment"] = {
        var: os.environ.get(var, "not_set") for var in neuron_env_vars
    }

    # Connectivity tests
    try:
        import requests

        # Test AWS connectivity
        try:
            response = requests.get("https://amazonaws.com", timeout=5)
            diagnostics["connectivity"]["aws"] = response.status_code == 200
        except Exception:
            diagnostics["connectivity"]["aws"] = False

        # Test Neuron repository
        try:
            response = requests.get("https://apt.repos.neuron.amazonaws.com", timeout=5)
            diagnostics["connectivity"]["neuron_repo"] = response.status_code in [
                200,
                403,
            ]  # 403 is normal
        except Exception:
            diagnostics["connectivity"]["neuron_repo"] = False

        print("✅ Connectivity tests completed")
    except Exception as e:
        print(f"⚠️ Connectivity tests failed: {e}")

    # Determine overall status
    has_neuron = diagnostics["neuron_info"].get("neuron_ls", {}).get("available", False)
    has_torch_neuronx = (
        diagnostics["neuron_info"].get("torch_neuronx_version") != "not_installed"
    )
    has_devices = len(diagnostics["neuron_info"].get("xla_devices", [])) > 0

    if has_neuron and has_torch_neuronx and has_devices:
        diagnostics["status"] = "healthy"
        print("\n✅ System appears healthy and ready for Neuron workloads")
    elif has_neuron and has_torch_neuronx:
        diagnostics["status"] = "partial"
        print("\n⚠️ System partially configured - some issues detected")
    else:
        diagnostics["status"] = "needs_setup"
        print("\n❌ System needs setup - major issues detected")

    return diagnostics


def main():
    """Main troubleshooting tool entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive AWS Neuron Troubleshooter"
    )
    parser.add_argument("--issue", type=str, help="Start with specific issue type")
    parser.add_argument(
        "--diagnostics", action="store_true", help="Run system diagnostics"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument(
        "--list-issues", action="store_true", help="List available issue types"
    )

    args = parser.parse_args()

    if args.list_issues:
        troubleshooter = InteractiveTroubleshooter(verbose=args.verbose)
        print("Available issue types:")
        for issue_id in troubleshooter.solutions:
            problem = troubleshooter.solutions[issue_id]["problem"]
            print(f"  {issue_id}: {problem}")
        return

    if args.diagnostics:
        diagnostics = run_system_diagnostics()

        # Save diagnostics
        diag_file = (
            f"system_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(diag_file, "w") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"\n💾 Diagnostics saved: {diag_file}")
        return

    # Start interactive troubleshooter
    troubleshooter = InteractiveTroubleshooter(verbose=args.verbose)

    if args.issue:
        troubleshooter.quick_diagnosis(args.issue)
    else:
        troubleshooter.start_interactive_session()


if __name__ == "__main__":
    main()
