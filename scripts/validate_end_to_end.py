#!/usr/bin/env python3
"""End-to-End Tutorial Validation Suite.

This script validates that all tutorial examples work correctly from start to
finish, ensuring the complete learning experience functions as designed.

Validation Categories:
    - Basic examples (hello_trainium, data processing)
    - Framework integration (PyTorch, TensorFlow, JAX)
    - Advanced features (NKI optimization, distributed training)
    - Production workflows (deployment, monitoring)
    - Sister tutorial integration (benchmarking, comparisons)

TESTED VERSIONS (Last validated: 2025-06-27):
    - Python: 3.11.7
    - PyTorch: 2.4.0
    - torch-neuronx: 2.2.0
    - AWS Neuron SDK: 2.20.1
    - Test Status: ✅ Comprehensive end-to-end validation ready

Usage:
    # Validate all examples
    python scripts/validate_end_to_end.py --all

    # Validate specific category
    python scripts/validate_end_to_end.py --category basic

    # Quick smoke test
    python scripts/validate_end_to_end.py --smoke-test

Author: Scott Friedman
Date: 2025-06-27
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EndToEndValidator:
    """Comprehensive end-to-end validation for the tutorial."""

    def __init__(self, tutorial_root: Optional[str] = None):
        """Initialize end-to-end validator."""
        self.tutorial_root = (
            Path(tutorial_root) if tutorial_root else Path(__file__).parent.parent
        )
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "tutorial_root": str(self.tutorial_root),
            "python_executable": sys.executable,
            "categories": {},
            "overall_status": "unknown",
        }

        # Define test categories and their examples
        self.test_categories = {
            "basic": {
                "description": "Basic tutorial examples",
                "examples": [
                    "examples/basic/hello_trainium.py",
                    "examples/datasets/aws_open_data.py",
                ],
                "required": True,
            },
            "frameworks": {
                "description": "ML framework integration",
                "examples": ["examples/frameworks/neuron_library_support.py"],
                "required": True,
            },
            "advanced": {
                "description": "Advanced NKI and optimization",
                "examples": ["examples/advanced/nki_optimization.py"],
                "required": False,
            },
            "production": {
                "description": "Production deployment patterns",
                "examples": ["examples/deployment/inferentia_serving.py"],
                "required": False,
            },
            "integration": {
                "description": "MLOps and enterprise integration",
                "examples": [
                    "examples/integration/mlflow_neuron_integration.py",
                    "examples/integration/kubeflow_neuron_pipeline.py",
                ],
                "required": False,
            },
            "benchmarking": {
                "description": "Performance benchmarking",
                "examples": ["examples/benchmarking/neuron_vs_nvidia_comparison.py"],
                "required": True,
            },
            "enterprise": {
                "description": "Enterprise security and compliance",
                "examples": ["examples/enterprise/security_compliance_patterns.py"],
                "required": False,
            },
            "easter_eggs": {
                "description": "Creative computing examples",
                "examples": [
                    "examples/easter_eggs/creative_showcase.py",
                    "examples/easter_eggs/precision_emulation.py",
                ],
                "required": False,
            },
        }

        logger.info(f"🔍 End-to-End Validator initialized")
        logger.info(f"   Tutorial root: {self.tutorial_root}")
        logger.info(f"   Categories: {len(self.test_categories)}")

    def validate_environment(self) -> Dict:
        """Validate the tutorial environment setup."""
        logger.info("🌍 Validating environment setup...")

        env_results = {
            "python_version": sys.version,
            "tutorial_exists": self.tutorial_root.exists(),
            "required_files": {},
            "python_packages": {},
            "status": "unknown",
        }

        # Check required files exist
        required_files = [
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            "VERSION_MATRIX.md",
            "examples/basic/hello_trainium.py",
        ]

        for file_path in required_files:
            full_path = self.tutorial_root / file_path
            env_results["required_files"][file_path] = full_path.exists()

        # Check Python packages
        required_packages = ["torch", "numpy", "pandas", "boto3"]

        for package in required_packages:
            try:
                __import__(package)
                env_results["python_packages"][package] = "available"
            except ImportError:
                env_results["python_packages"][package] = "missing"

        # Check Neuron packages
        try:
            import torch_neuronx

            env_results["python_packages"]["torch_neuronx"] = torch_neuronx.__version__
        except ImportError:
            env_results["python_packages"]["torch_neuronx"] = "missing"

        # Determine status
        files_ok = all(env_results["required_files"].values())
        packages_ok = all(
            status != "missing" for status in env_results["python_packages"].values()
        )

        if files_ok and packages_ok:
            env_results["status"] = "passed"
        else:
            env_results["status"] = "failed"

        return env_results

    def validate_example(self, example_path: str, timeout: int = 120) -> Dict:
        """Validate a single tutorial example."""
        full_path = self.tutorial_root / example_path

        result = {
            "example": example_path,
            "exists": full_path.exists(),
            "execution_time": 0,
            "status": "unknown",
            "output": "",
            "error": "",
        }

        if not full_path.exists():
            result["status"] = "failed"
            result["error"] = "Example file not found"
            return result

        logger.info(f"   Testing: {example_path}")

        try:
            # Run the example with a timeout
            start_time = time.time()

            # Create environment with tutorial root in Python path
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.tutorial_root)

            process = subprocess.run(
                [sys.executable, str(full_path)],
                cwd=str(self.tutorial_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            end_time = time.time()
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["return_code"] = process.returncode

            if process.returncode == 0:
                result["status"] = "passed"
            else:
                result["status"] = "failed"

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["error"] = f"Example exceeded {timeout} second timeout"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def validate_category(self, category: str) -> Dict:
        """Validate all examples in a category."""
        if category not in self.test_categories:
            return {"status": "failed", "error": f"Unknown category: {category}"}

        category_info = self.test_categories[category]
        logger.info(
            f"🧪 Validating category: {category} - {category_info['description']}"
        )

        category_results = {
            "category": category,
            "description": category_info["description"],
            "required": category_info["required"],
            "examples": {},
            "summary": {
                "total": len(category_info["examples"]),
                "passed": 0,
                "failed": 0,
                "timeout": 0,
                "error": 0,
            },
            "status": "unknown",
        }

        # Validate each example
        for example in category_info["examples"]:
            result = self.validate_example(example)
            category_results["examples"][example] = result

            # Update summary
            status = result["status"]
            if status in category_results["summary"]:
                category_results["summary"][status] += 1

        # Determine category status
        passed = category_results["summary"]["passed"]
        total = category_results["summary"]["total"]

        if passed == total:
            category_results["status"] = "passed"
        elif passed > 0:
            category_results["status"] = "partial"
        else:
            category_results["status"] = "failed"

        logger.info(f"   Category {category}: {passed}/{total} examples passed")
        return category_results

    def run_smoke_test(self) -> Dict:
        """Run quick smoke test of essential examples."""
        logger.info("💨 Running smoke test...")

        # Essential examples for smoke test
        smoke_examples = [
            "examples/basic/hello_trainium.py",
            "examples/datasets/aws_open_data.py",
        ]

        smoke_results = {"test_type": "smoke_test", "examples": {}, "status": "unknown"}

        all_passed = True
        for example in smoke_examples:
            result = self.validate_example(
                example, timeout=60
            )  # Shorter timeout for smoke test
            smoke_results["examples"][example] = result

            if result["status"] != "passed":
                all_passed = False

        smoke_results["status"] = "passed" if all_passed else "failed"
        return smoke_results

    def run_full_validation(self) -> Dict:
        """Run complete end-to-end validation."""
        logger.info("🚀 Starting full end-to-end validation...")

        # Validate environment first
        env_results = self.validate_environment()
        self.validation_results["environment"] = env_results

        if env_results["status"] != "passed":
            logger.error("Environment validation failed. Stopping validation.")
            self.validation_results["overall_status"] = "failed"
            return self.validation_results

        # Validate each category
        for category in self.test_categories:
            category_results = self.validate_category(category)
            self.validation_results["categories"][category] = category_results

        # Calculate overall status
        self._calculate_overall_status()

        logger.info(
            f"✅ Full validation completed: {self.validation_results['overall_status']}"
        )
        return self.validation_results

    def run_category_validation(self, category: str) -> Dict:
        """Run validation for a specific category."""
        logger.info(f"🎯 Running validation for category: {category}")

        # Validate environment first
        env_results = self.validate_environment()
        self.validation_results["environment"] = env_results

        # Validate the specific category
        if category in self.test_categories:
            category_results = self.validate_category(category)
            self.validation_results["categories"][category] = category_results
            self.validation_results["overall_status"] = category_results["status"]
        else:
            self.validation_results["overall_status"] = "failed"
            self.validation_results["error"] = f"Unknown category: {category}"

        return self.validation_results

    def _calculate_overall_status(self):
        """Calculate overall validation status."""
        required_categories = [
            cat for cat, info in self.test_categories.items() if info["required"]
        ]
        optional_categories = [
            cat for cat, info in self.test_categories.items() if not info["required"]
        ]

        # Check required categories
        required_passed = 0
        required_total = len(required_categories)

        for category in required_categories:
            if category in self.validation_results["categories"]:
                if (
                    self.validation_results["categories"][category]["status"]
                    == "passed"
                ):
                    required_passed += 1

        # Check optional categories
        optional_passed = 0
        optional_total = len(optional_categories)

        for category in optional_categories:
            if category in self.validation_results["categories"]:
                if (
                    self.validation_results["categories"][category]["status"]
                    == "passed"
                ):
                    optional_passed += 1

        # Determine overall status
        if required_passed == required_total:
            if optional_passed == optional_total:
                self.validation_results["overall_status"] = "passed"
            elif optional_passed > 0:
                self.validation_results["overall_status"] = "mostly_passed"
            else:
                self.validation_results["overall_status"] = "required_passed"
        elif required_passed > 0:
            self.validation_results["overall_status"] = "partial"
        else:
            self.validation_results["overall_status"] = "failed"

        # Add summary
        self.validation_results["summary"] = {
            "required_categories": {"passed": required_passed, "total": required_total},
            "optional_categories": {"passed": optional_passed, "total": optional_total},
        }

    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        report_lines = []
        report_lines.append("# End-to-End Tutorial Validation Report")
        report_lines.append(
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(
            f"**Tutorial Root**: {self.validation_results['tutorial_root']}"
        )
        report_lines.append(
            f"**Overall Status**: {self.validation_results.get('overall_status', 'unknown')}"
        )

        # Environment section
        if "environment" in self.validation_results:
            env = self.validation_results["environment"]
            report_lines.append("\n## Environment Validation")
            report_lines.append(f"- **Status**: {env.get('status', 'unknown')}")
            report_lines.append(
                f"- **Python**: {env.get('python_version', 'unknown').split()[0]}"
            )

            # Required files
            files = env.get("required_files", {})
            missing_files = [f for f, exists in files.items() if not exists]
            if missing_files:
                report_lines.append(f"- **Missing Files**: {', '.join(missing_files)}")
            else:
                report_lines.append("- **Required Files**: All present ✅")

            # Python packages
            packages = env.get("python_packages", {})
            missing_packages = [
                p for p, status in packages.items() if status == "missing"
            ]
            if missing_packages:
                report_lines.append(
                    f"- **Missing Packages**: {', '.join(missing_packages)}"
                )
            else:
                report_lines.append("- **Python Packages**: All available ✅")

        # Summary section
        if "summary" in self.validation_results:
            summary = self.validation_results["summary"]
            report_lines.append("\n## Validation Summary")

            req = summary["required_categories"]
            opt = summary["optional_categories"]

            report_lines.append(
                f"- **Required Categories**: {req['passed']}/{req['total']} passed"
            )
            report_lines.append(
                f"- **Optional Categories**: {opt['passed']}/{opt['total']} passed"
            )

        # Category details
        if "categories" in self.validation_results:
            report_lines.append("\n## Category Results")

            for category, results in self.validation_results["categories"].items():
                status_emoji = {
                    "passed": "✅",
                    "partial": "⚠️",
                    "failed": "❌",
                    "timeout": "⏱️",
                    "error": "💥",
                }.get(results.get("status", "unknown"), "❓")

                report_lines.append(f"\n### {category.title()} {status_emoji}")
                report_lines.append(
                    f"**Description**: {results.get('description', 'N/A')}"
                )
                report_lines.append(f"**Required**: {results.get('required', False)}")

                summary = results.get("summary", {})
                total = summary.get("total", 0)
                passed = summary.get("passed", 0)
                report_lines.append(f"**Examples**: {passed}/{total} passed")

                # Example details
                examples = results.get("examples", {})
                for example, result in examples.items():
                    status = result.get("status", "unknown")
                    execution_time = result.get("execution_time", 0)

                    status_symbol = {
                        "passed": "✅",
                        "failed": "❌",
                        "timeout": "⏱️",
                        "error": "💥",
                    }.get(status, "❓")

                    report_lines.append(
                        f"- {example} {status_symbol} ({execution_time:.1f}s)"
                    )

                    if status != "passed" and result.get("error"):
                        report_lines.append(f"  Error: {result['error'][:100]}...")

        # Recommendations
        report_lines.append("\n## Recommendations")
        overall_status = self.validation_results.get("overall_status", "unknown")

        if overall_status == "passed":
            report_lines.append(
                "🎉 All validations passed! The tutorial is ready for use."
            )
        elif overall_status == "mostly_passed":
            report_lines.append(
                "✅ All required examples passed. Optional examples have some issues."
            )
            report_lines.append("- Review failed optional examples for improvements")
            report_lines.append("- Tutorial is suitable for production use")
        elif overall_status == "required_passed":
            report_lines.append(
                "⚠️ Required examples passed but optional examples failed."
            )
            report_lines.append("- Focus on fixing optional example issues")
            report_lines.append("- Core tutorial functionality is working")
        elif overall_status == "partial":
            report_lines.append("❌ Some required examples failed.")
            report_lines.append("- Fix required example issues before release")
            report_lines.append("- Check dependencies and environment setup")
        else:
            report_lines.append("🚨 Critical validation failures detected.")
            report_lines.append("- Review environment setup and dependencies")
            report_lines.append("- Check error messages for specific issues")
            report_lines.append("- Ensure running on compatible hardware")

        return "\n".join(report_lines)


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="End-to-End Tutorial Validation")
    parser.add_argument("--all", action="store_true", help="Run full validation suite")
    parser.add_argument("--category", type=str, help="Validate specific category")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run quick smoke test"
    )
    parser.add_argument(
        "--list-categories", action="store_true", help="List available categories"
    )
    parser.add_argument(
        "--tutorial-root", type=str, help="Path to tutorial root directory"
    )
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument(
        "--report", action="store_true", help="Generate validation report"
    )

    args = parser.parse_args()

    # Initialize validator
    validator = EndToEndValidator(args.tutorial_root)

    # List categories if requested
    if args.list_categories:
        print("Available validation categories:")
        for category, info in validator.test_categories.items():
            required = "Required" if info["required"] else "Optional"
            print(f"  {category}: {info['description']} ({required})")
        return

    # Determine what to run
    if not any([args.all, args.category, args.smoke_test]):
        args.smoke_test = True  # Default to smoke test

    # Run validation
    if args.all:
        results = validator.run_full_validation()
    elif args.category:
        results = validator.run_category_validation(args.category)
    elif args.smoke_test:
        results = validator.run_smoke_test()
        validator.validation_results.update(results)
        results = validator.validation_results

    # Generate report if requested
    if args.report or args.output:
        report = validator.generate_report()

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
        else:
            print(report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"end_to_end_validation_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nValidation results saved to: {results_file}")

    # Print summary
    overall_status = results.get("overall_status", "unknown")
    print(f"Overall status: {overall_status}")

    if "summary" in results:
        summary = results["summary"]
        req = summary["required_categories"]
        opt = summary["optional_categories"]
        print(f"Required categories: {req['passed']}/{req['total']} passed")
        print(f"Optional categories: {opt['passed']}/{opt['total']} passed")

    # Exit with appropriate code
    if overall_status in ["passed", "mostly_passed", "required_passed"]:
        sys.exit(0)
    elif overall_status == "partial":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
