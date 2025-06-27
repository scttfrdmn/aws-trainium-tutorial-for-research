#!/usr/bin/env python3
"""Sister Tutorial Comparison Testing Framework.

This script validates the comparison framework between the AWS Neuron tutorial
and the NVIDIA GPU sister tutorial, ensuring fair and accurate benchmarking
across both platforms.

Validation Categories:
    - Sister tutorial availability and structure
    - Identical model architectures verification
    - Dataset compatibility and consistency
    - Benchmarking framework synchronization
    - Cross-platform result comparison accuracy

TESTED VERSIONS (Last validated: 2025-06-27):
    - Neuron Tutorial: 2025.1.0
    - GPU Tutorial: 2025.1.0
    - Comparison Framework: v1.0
    - Test Status: ✅ Sister tutorial integration validated

Usage:
    # Test complete comparison framework
    python scripts/test_sister_tutorial_comparison.py --full

    # Quick compatibility check
    python scripts/test_sister_tutorial_comparison.py --quick

    # Generate comparison report
    python scripts/test_sister_tutorial_comparison.py --generate-report

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

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SisterTutorialTester:
    """Comprehensive testing for sister tutorial comparison framework."""

    def __init__(
        self,
        neuron_tutorial_root: Optional[str] = None,
        gpu_tutorial_root: Optional[str] = None,
    ):
        """Initialize sister tutorial tester."""
        # Set default paths
        if neuron_tutorial_root is None:
            neuron_tutorial_root = Path(__file__).parent.parent
        if gpu_tutorial_root is None:
            gpu_tutorial_root = (
                Path(__file__).parent.parent.parent / "nvidia-gpu-tutorial-for-research"
            )

        self.neuron_root = Path(neuron_tutorial_root)
        self.gpu_root = Path(gpu_tutorial_root)

        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "neuron_tutorial_root": str(self.neuron_root),
            "gpu_tutorial_root": str(self.gpu_root),
            "tests": {},
            "overall_status": "unknown",
        }

        # Define comparison test matrix
        self.comparison_tests = {
            "structure": {
                "description": "Directory structure and file organization",
                "critical": True,
            },
            "models": {
                "description": "Model architecture consistency",
                "critical": True,
            },
            "datasets": {
                "description": "Dataset compatibility and format",
                "critical": True,
            },
            "benchmarking": {
                "description": "Benchmarking framework synchronization",
                "critical": True,
            },
            "documentation": {
                "description": "Documentation consistency and completeness",
                "critical": False,
            },
            "version_alignment": {
                "description": "Software version compatibility",
                "critical": True,
            },
        }

        logger.info(f"🔍 Sister Tutorial Tester initialized")
        logger.info(f"   Neuron tutorial: {self.neuron_root}")
        logger.info(f"   GPU tutorial: {self.gpu_root}")

    def test_tutorial_availability(self) -> Dict:
        """Test that both tutorials are available and accessible."""
        logger.info("📂 Testing tutorial availability...")

        availability_results = {
            "neuron_tutorial": {
                "exists": self.neuron_root.exists(),
                "readable": False,
                "key_files": {},
            },
            "gpu_tutorial": {
                "exists": self.gpu_root.exists(),
                "readable": False,
                "key_files": {},
            },
            "status": "unknown",
        }

        # Check Neuron tutorial
        if availability_results["neuron_tutorial"]["exists"]:
            availability_results["neuron_tutorial"]["readable"] = os.access(
                self.neuron_root, os.R_OK
            )

            # Check key files
            key_files = [
                "README.md",
                "requirements.txt",
                "VERSION_MATRIX.md",
                "pyproject.toml",
            ]
            for file in key_files:
                file_path = self.neuron_root / file
                availability_results["neuron_tutorial"]["key_files"][
                    file
                ] = file_path.exists()

        # Check GPU tutorial
        if availability_results["gpu_tutorial"]["exists"]:
            availability_results["gpu_tutorial"]["readable"] = os.access(
                self.gpu_root, os.R_OK
            )

            # Check key files
            key_files = [
                "README.md",
                "requirements.txt",
                "VERSION_MATRIX.md",
                "pyproject.toml",
            ]
            for file in key_files:
                file_path = self.gpu_root / file
                availability_results["gpu_tutorial"]["key_files"][
                    file
                ] = file_path.exists()

        # Determine status
        neuron_ok = (
            availability_results["neuron_tutorial"]["exists"]
            and availability_results["neuron_tutorial"]["readable"]
        )
        gpu_ok = (
            availability_results["gpu_tutorial"]["exists"]
            and availability_results["gpu_tutorial"]["readable"]
        )

        if neuron_ok and gpu_ok:
            availability_results["status"] = "passed"
        elif neuron_ok:
            availability_results["status"] = "neuron_only"
        elif gpu_ok:
            availability_results["status"] = "gpu_only"
        else:
            availability_results["status"] = "failed"

        return availability_results

    def test_structure_compatibility(self) -> Dict:
        """Test directory structure and organization compatibility."""
        logger.info("🏗️ Testing structure compatibility...")

        structure_results = {
            "directory_comparison": {},
            "file_comparison": {},
            "missing_in_gpu": [],
            "missing_in_neuron": [],
            "status": "unknown",
        }

        # Define critical directories that should exist in both tutorials
        critical_dirs = [
            "examples/basic",
            "examples/datasets",
            "examples/frameworks",
            "examples/benchmarking",
            "examples/deployment",
            "examples/integration",
            "examples/enterprise",
            "docs",
            "tests",
        ]

        # Check directory structure
        for dir_path in critical_dirs:
            neuron_dir = self.neuron_root / dir_path
            gpu_dir = self.gpu_root / dir_path

            structure_results["directory_comparison"][dir_path] = {
                "neuron_exists": neuron_dir.exists(),
                "gpu_exists": gpu_dir.exists(),
                "both_exist": neuron_dir.exists() and gpu_dir.exists(),
            }

            if neuron_dir.exists() and not gpu_dir.exists():
                structure_results["missing_in_gpu"].append(dir_path)
            elif gpu_dir.exists() and not neuron_dir.exists():
                structure_results["missing_in_neuron"].append(dir_path)

        # Check critical files
        critical_files = [
            "examples/benchmarking/neuron_vs_nvidia_comparison.py",
            "examples/benchmarking/gpu_vs_neuron_comparison.py",  # In GPU tutorial
            "examples/datasets/aws_open_data.py",
            "examples/datasets/gpu_open_data.py",  # In GPU tutorial
            "VERSION_MATRIX.md",
        ]

        for file_path in critical_files:
            neuron_file = self.neuron_root / file_path
            gpu_file = self.gpu_root / file_path.replace(
                "neuron_vs_nvidia", "gpu_vs_neuron"
            ).replace("aws_open_data", "gpu_open_data")

            structure_results["file_comparison"][file_path] = {
                "neuron_exists": neuron_file.exists(),
                "gpu_exists": gpu_file.exists(),
            }

        # Determine status
        critical_matches = sum(
            1
            for comp in structure_results["directory_comparison"].values()
            if comp["both_exist"]
        )
        total_critical = len(critical_dirs)

        if critical_matches == total_critical:
            structure_results["status"] = "passed"
        elif critical_matches >= total_critical * 0.8:  # 80% match
            structure_results["status"] = "mostly_compatible"
        else:
            structure_results["status"] = "incompatible"

        structure_results["compatibility_score"] = critical_matches / total_critical

        return structure_results

    def test_model_consistency(self) -> Dict:
        """Test that model architectures are identical between tutorials."""
        logger.info("🧠 Testing model consistency...")

        model_results = {
            "model_definitions": {},
            "parameter_counts": {},
            "architecture_hashes": {},
            "status": "unknown",
        }

        # Test models to compare
        test_models = ["bert-base", "gpt2-small", "resnet-50", "transformer-encoder"]

        try:
            # Import model suites from both tutorials
            sys.path.insert(0, str(self.neuron_root))
            sys.path.insert(0, str(self.gpu_root))

            # Try to import both model suites
            try:
                from examples.benchmarking.neuron_vs_nvidia_comparison import (
                    StandardModelSuite as NeuronModels,
                )

                neuron_models_available = True
            except ImportError as e:
                neuron_models_available = False
                logger.warning(f"Could not import Neuron models: {e}")

            try:
                from examples.benchmarking.gpu_vs_neuron_comparison import (
                    StandardModelSuite as GPUModels,
                )

                gpu_models_available = True
            except ImportError as e:
                gpu_models_available = False
                logger.warning(f"Could not import GPU models: {e}")

            if not (neuron_models_available and gpu_models_available):
                model_results["status"] = "failed"
                model_results[
                    "error"
                ] = "Could not import model suites from both tutorials"
                return model_results

            # Compare models
            neuron_suite = NeuronModels()
            gpu_suite = GPUModels()

            for model_name in test_models:
                try:
                    # Get models from both suites
                    if hasattr(neuron_suite, f"get_{model_name.replace('-', '_')}"):
                        neuron_model = getattr(
                            neuron_suite, f"get_{model_name.replace('-', '_')}"
                        )()
                        gpu_model = getattr(
                            gpu_suite, f"get_{model_name.replace('-', '_')}"
                        )()

                        # Compare parameter counts
                        neuron_params = sum(
                            p.numel() for p in neuron_model.parameters()
                        )
                        gpu_params = sum(p.numel() for p in gpu_model.parameters())

                        model_results["parameter_counts"][model_name] = {
                            "neuron": neuron_params,
                            "gpu": gpu_params,
                            "match": neuron_params == gpu_params,
                        }

                        # Compare model string representations (architecture)
                        neuron_str = str(neuron_model)
                        gpu_str = str(gpu_model)

                        model_results["architecture_hashes"][model_name] = {
                            "neuron_hash": hash(neuron_str),
                            "gpu_hash": hash(gpu_str),
                            "match": neuron_str == gpu_str,
                        }

                        model_results["model_definitions"][model_name] = "compared"

                except Exception as e:
                    model_results["model_definitions"][model_name] = f"error: {e}"

            # Determine overall status
            param_matches = sum(
                1
                for model in model_results["parameter_counts"].values()
                if model["match"]
            )
            arch_matches = sum(
                1
                for model in model_results["architecture_hashes"].values()
                if model["match"]
            )
            total_models = len(
                [
                    m
                    for m in model_results["model_definitions"].values()
                    if m == "compared"
                ]
            )

            if param_matches == total_models and arch_matches == total_models:
                model_results["status"] = "passed"
            elif param_matches == total_models:
                model_results["status"] = "parameter_match"
            else:
                model_results["status"] = "failed"

        except Exception as e:
            model_results["status"] = "error"
            model_results["error"] = str(e)
        finally:
            # Clean up sys.path
            if str(self.neuron_root) in sys.path:
                sys.path.remove(str(self.neuron_root))
            if str(self.gpu_root) in sys.path:
                sys.path.remove(str(self.gpu_root))

        return model_results

    def test_dataset_compatibility(self) -> Dict:
        """Test dataset format and structure compatibility."""
        logger.info("📊 Testing dataset compatibility...")

        dataset_results = {
            "dataset_managers": {},
            "dataset_formats": {},
            "data_samples": {},
            "status": "unknown",
        }

        try:
            # Test dataset manager imports
            sys.path.insert(0, str(self.neuron_root))
            sys.path.insert(0, str(self.gpu_root))

            try:
                from examples.datasets.aws_open_data import (
                    AWSOpenDataManager as NeuronDataManager,
                )

                neuron_data_available = True
            except ImportError as e:
                neuron_data_available = False
                logger.warning(f"Could not import Neuron data manager: {e}")

            try:
                from examples.datasets.gpu_open_data import (
                    GPUOpenDataManager as GPUDataManager,
                )

                gpu_data_available = True
            except ImportError as e:
                gpu_data_available = False
                logger.warning(f"Could not import GPU data manager: {e}")

            if not (neuron_data_available and gpu_data_available):
                dataset_results["status"] = "failed"
                dataset_results[
                    "error"
                ] = "Could not import data managers from both tutorials"
                return dataset_results

            # Create temporary directories for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                neuron_cache = Path(temp_dir) / "neuron_cache"
                gpu_cache = Path(temp_dir) / "gpu_cache"

                neuron_manager = NeuronDataManager(cache_dir=str(neuron_cache))
                gpu_manager = GPUDataManager(cache_dir=str(gpu_cache))

                # Compare available datasets
                neuron_datasets = neuron_manager.list_available_datasets()
                gpu_datasets = gpu_manager.list_available_datasets()

                common_datasets = set(neuron_datasets.keys()) & set(gpu_datasets.keys())
                dataset_results["common_datasets"] = list(common_datasets)
                dataset_results["neuron_only"] = list(
                    set(neuron_datasets.keys()) - set(gpu_datasets.keys())
                )
                dataset_results["gpu_only"] = list(
                    set(gpu_datasets.keys()) - set(neuron_datasets.keys())
                )

                # Test a sample dataset
                if "nasa_climate" in common_datasets:
                    try:
                        # Download samples from both managers
                        neuron_sample = neuron_manager.download_dataset_sample(
                            "nasa_climate", "small"
                        )
                        gpu_sample = gpu_manager.download_dataset_sample(
                            "nasa_climate", "small"
                        )

                        # Load and compare
                        with open(neuron_sample, "r") as f:
                            neuron_data = json.load(f)
                        with open(gpu_sample, "r") as f:
                            gpu_data = json.load(f)

                        # Compare structure
                        neuron_keys = set(neuron_data.keys())
                        gpu_keys = set(gpu_data.keys())

                        dataset_results["data_samples"]["nasa_climate"] = {
                            "neuron_keys": list(neuron_keys),
                            "gpu_keys": list(gpu_keys),
                            "common_keys": list(neuron_keys & gpu_keys),
                            "structure_match": neuron_keys == gpu_keys,
                        }

                        # Compare sample counts
                        neuron_samples = neuron_data.get("samples", 0)
                        gpu_samples = gpu_data.get("samples", 0)

                        dataset_results["data_samples"]["nasa_climate"][
                            "sample_counts"
                        ] = {
                            "neuron": neuron_samples,
                            "gpu": gpu_samples,
                            "match": neuron_samples == gpu_samples,
                        }

                    except Exception as e:
                        dataset_results["data_samples"]["nasa_climate"] = {
                            "error": str(e)
                        }

            # Determine status
            if len(common_datasets) >= 3:  # At least 3 common datasets
                dataset_results["status"] = "passed"
            elif len(common_datasets) >= 1:
                dataset_results["status"] = "partial"
            else:
                dataset_results["status"] = "failed"

        except Exception as e:
            dataset_results["status"] = "error"
            dataset_results["error"] = str(e)
        finally:
            # Clean up sys.path
            if str(self.neuron_root) in sys.path:
                sys.path.remove(str(self.neuron_root))
            if str(self.gpu_root) in sys.path:
                sys.path.remove(str(self.gpu_root))

        return dataset_results

    def test_benchmarking_framework(self) -> Dict:
        """Test benchmarking framework compatibility and synchronization."""
        logger.info("⚡ Testing benchmarking framework...")

        benchmark_results = {
            "config_compatibility": {},
            "result_format_compatibility": {},
            "benchmark_methods": {},
            "status": "unknown",
        }

        try:
            sys.path.insert(0, str(self.neuron_root))
            sys.path.insert(0, str(self.gpu_root))

            # Import benchmark configurations
            try:
                from examples.benchmarking.neuron_vs_nvidia_comparison import (
                    BenchmarkConfig as NeuronConfig,
                )
                from examples.benchmarking.neuron_vs_nvidia_comparison import (
                    BenchmarkResult as NeuronResult,
                )

                neuron_bench_available = True
            except ImportError as e:
                neuron_bench_available = False
                logger.warning(f"Could not import Neuron benchmark classes: {e}")

            try:
                from examples.benchmarking.gpu_vs_neuron_comparison import (
                    BenchmarkConfig as GPUConfig,
                )
                from examples.benchmarking.gpu_vs_neuron_comparison import (
                    BenchmarkResult as GPUResult,
                )

                gpu_bench_available = True
            except ImportError as e:
                gpu_bench_available = False
                logger.warning(f"Could not import GPU benchmark classes: {e}")

            if not (neuron_bench_available and gpu_bench_available):
                benchmark_results["status"] = "failed"
                benchmark_results[
                    "error"
                ] = "Could not import benchmark classes from both tutorials"
                return benchmark_results

            # Compare BenchmarkConfig fields
            neuron_config_fields = set(NeuronConfig.__dataclass_fields__.keys())
            gpu_config_fields = set(GPUConfig.__dataclass_fields__.keys())

            benchmark_results["config_compatibility"] = {
                "neuron_fields": list(neuron_config_fields),
                "gpu_fields": list(gpu_config_fields),
                "common_fields": list(neuron_config_fields & gpu_config_fields),
                "match": neuron_config_fields == gpu_config_fields,
            }

            # Compare BenchmarkResult fields
            neuron_result_fields = set(NeuronResult.__dataclass_fields__.keys())
            gpu_result_fields = set(GPUResult.__dataclass_fields__.keys())

            benchmark_results["result_format_compatibility"] = {
                "neuron_fields": list(neuron_result_fields),
                "gpu_fields": list(gpu_result_fields),
                "common_fields": list(neuron_result_fields & gpu_result_fields),
                "match": neuron_result_fields == gpu_result_fields,
            }

            # Test if we can create compatible configs
            try:
                neuron_config = NeuronConfig(
                    model_name="bert-base",
                    framework="pytorch",
                    task_type="inference",
                    platform="neuron",
                    instance_type="inf2.xlarge",
                )

                gpu_config = GPUConfig(
                    model_name="bert-base",
                    framework="pytorch",
                    task_type="inference",
                    platform="gpu",
                    instance_type="g5.xlarge",
                )

                benchmark_results["benchmark_methods"]["config_creation"] = "success"

            except Exception as e:
                benchmark_results["benchmark_methods"][
                    "config_creation"
                ] = f"failed: {e}"

            # Determine status
            config_match = benchmark_results["config_compatibility"]["match"]
            result_match = benchmark_results["result_format_compatibility"]["match"]

            if config_match and result_match:
                benchmark_results["status"] = "passed"
            elif len(benchmark_results["config_compatibility"]["common_fields"]) >= 5:
                benchmark_results["status"] = "mostly_compatible"
            else:
                benchmark_results["status"] = "incompatible"

        except Exception as e:
            benchmark_results["status"] = "error"
            benchmark_results["error"] = str(e)
        finally:
            # Clean up sys.path
            if str(self.neuron_root) in sys.path:
                sys.path.remove(str(self.neuron_root))
            if str(self.gpu_root) in sys.path:
                sys.path.remove(str(self.gpu_root))

        return benchmark_results

    def test_version_alignment(self) -> Dict:
        """Test software version alignment between tutorials."""
        logger.info("📋 Testing version alignment...")

        version_results = {
            "version_matrices": {},
            "framework_versions": {},
            "alignment_score": 0,
            "status": "unknown",
        }

        try:
            # Read version matrices from both tutorials
            neuron_version_file = self.neuron_root / "VERSION_MATRIX.md"
            gpu_version_file = self.gpu_root / "VERSION_MATRIX.md"

            if neuron_version_file.exists():
                with open(neuron_version_file, "r") as f:
                    neuron_content = f.read()
                version_results["version_matrices"]["neuron"] = "available"
            else:
                version_results["version_matrices"]["neuron"] = "missing"
                neuron_content = ""

            if gpu_version_file.exists():
                with open(gpu_version_file, "r") as f:
                    gpu_content = f.read()
                version_results["version_matrices"]["gpu"] = "available"
            else:
                version_results["version_matrices"]["gpu"] = "missing"
                gpu_content = ""

            # Extract key framework versions from content
            frameworks_to_check = ["PyTorch", "TensorFlow", "JAX", "Python"]

            for framework in frameworks_to_check:
                neuron_version = self._extract_version_from_content(
                    neuron_content, framework
                )
                gpu_version = self._extract_version_from_content(gpu_content, framework)

                version_results["framework_versions"][framework] = {
                    "neuron": neuron_version,
                    "gpu": gpu_version,
                    "compatible": self._versions_compatible(
                        neuron_version, gpu_version
                    ),
                }

            # Calculate alignment score
            compatible_count = sum(
                1
                for fw in version_results["framework_versions"].values()
                if fw["compatible"]
            )
            total_frameworks = len(frameworks_to_check)
            version_results["alignment_score"] = compatible_count / total_frameworks

            # Determine status
            if version_results["alignment_score"] >= 0.8:
                version_results["status"] = "aligned"
            elif version_results["alignment_score"] >= 0.6:
                version_results["status"] = "mostly_aligned"
            else:
                version_results["status"] = "misaligned"

        except Exception as e:
            version_results["status"] = "error"
            version_results["error"] = str(e)

        return version_results

    def _extract_version_from_content(
        self, content: str, framework: str
    ) -> Optional[str]:
        """Extract version number for a framework from content."""
        import re

        # Look for patterns like "PyTorch: 2.4.0" or "PyTorch** | 2.4.0"
        patterns = [
            rf"{framework}[:\s]*(\d+\.\d+\.\d+)",
            rf"\*\*{framework}\*\*[|\s]*(\d+\.\d+\.\d+)",
            rf"{framework}.*?(\d+\.\d+\.\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _versions_compatible(
        self, version1: Optional[str], version2: Optional[str]
    ) -> bool:
        """Check if two versions are compatible (same major.minor)."""
        if not version1 or not version2:
            return False

        try:
            v1_parts = version1.split(".")
            v2_parts = version2.split(".")

            # Check if major.minor versions match
            return v1_parts[0] == v2_parts[0] and v1_parts[1] == v2_parts[1]
        except:
            return False

    def run_full_comparison_test(self) -> Dict:
        """Run complete sister tutorial comparison test suite."""
        logger.info("🚀 Running full sister tutorial comparison test...")

        # Test 1: Tutorial availability
        self.test_results["tests"]["availability"] = self.test_tutorial_availability()

        # Only continue if both tutorials are available
        if self.test_results["tests"]["availability"]["status"] != "passed":
            self.test_results["overall_status"] = "failed"
            self.test_results["error"] = "Sister tutorial not available for comparison"
            return self.test_results

        # Test 2: Structure compatibility
        self.test_results["tests"]["structure"] = self.test_structure_compatibility()

        # Test 3: Model consistency
        self.test_results["tests"]["models"] = self.test_model_consistency()

        # Test 4: Dataset compatibility
        self.test_results["tests"]["datasets"] = self.test_dataset_compatibility()

        # Test 5: Benchmarking framework
        self.test_results["tests"]["benchmarking"] = self.test_benchmarking_framework()

        # Test 6: Version alignment
        self.test_results["tests"]["version_alignment"] = self.test_version_alignment()

        # Calculate overall status
        self._calculate_overall_status()

        logger.info(
            f"✅ Sister tutorial comparison test completed: {self.test_results['overall_status']}"
        )
        return self.test_results

    def run_quick_comparison_test(self) -> Dict:
        """Run quick sister tutorial compatibility check."""
        logger.info("⚡ Running quick sister tutorial comparison...")

        # Quick tests
        self.test_results["tests"]["availability"] = self.test_tutorial_availability()
        self.test_results["tests"]["structure"] = self.test_structure_compatibility()

        # Simple overall status
        if self.test_results["tests"]["availability"][
            "status"
        ] == "passed" and self.test_results["tests"]["structure"]["status"] in [
            "passed",
            "mostly_compatible",
        ]:
            self.test_results["overall_status"] = "compatible"
        else:
            self.test_results["overall_status"] = "incompatible"

        return self.test_results

    def _calculate_overall_status(self):
        """Calculate overall comparison test status."""
        critical_tests = [
            name for name, info in self.comparison_tests.items() if info["critical"]
        ]

        test_statuses = []
        for test_name in critical_tests:
            if test_name in self.test_results["tests"]:
                status = self.test_results["tests"][test_name].get("status", "unknown")
                test_statuses.append(status)

        # Count passed tests
        passed_count = sum(
            1 for status in test_statuses if status in ["passed", "aligned"]
        )
        mostly_passed_count = sum(
            1
            for status in test_statuses
            if status in ["passed", "aligned", "mostly_compatible", "mostly_aligned"]
        )
        total_tests = len(test_statuses)

        if passed_count == total_tests:
            self.test_results["overall_status"] = "fully_compatible"
        elif mostly_passed_count >= total_tests * 0.8:
            self.test_results["overall_status"] = "mostly_compatible"
        elif mostly_passed_count >= total_tests * 0.5:
            self.test_results["overall_status"] = "partially_compatible"
        else:
            self.test_results["overall_status"] = "incompatible"

        # Add summary
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_count,
            "mostly_passed_tests": mostly_passed_count,
            "compatibility_score": mostly_passed_count / total_tests
            if total_tests > 0
            else 0,
        }

    def generate_comparison_report(self) -> str:
        """Generate comprehensive sister tutorial comparison report."""
        report_lines = []
        report_lines.append("# Sister Tutorial Comparison Test Report")
        report_lines.append(
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(
            f"**Neuron Tutorial**: {self.test_results['neuron_tutorial_root']}"
        )
        report_lines.append(
            f"**GPU Tutorial**: {self.test_results['gpu_tutorial_root']}"
        )
        report_lines.append(
            f"**Overall Status**: {self.test_results.get('overall_status', 'unknown')}"
        )

        # Summary
        if "summary" in self.test_results:
            summary = self.test_results["summary"]
            report_lines.append(f"\n## Summary")
            report_lines.append(
                f"- **Compatibility Score**: {summary['compatibility_score']:.2%}"
            )
            report_lines.append(
                f"- **Tests Passed**: {summary['passed_tests']}/{summary['total_tests']}"
            )
            report_lines.append(
                f"- **Tests Mostly Passed**: {summary['mostly_passed_tests']}/{summary['total_tests']}"
            )

        # Detailed results
        for test_name, test_result in self.test_results.get("tests", {}).items():
            status = test_result.get("status", "unknown")
            status_emoji = {
                "passed": "✅",
                "mostly_compatible": "⚠️",
                "compatible": "✅",
                "failed": "❌",
                "incompatible": "❌",
                "error": "💥",
            }.get(status, "❓")

            report_lines.append(f"\n## {test_name.title()} Test {status_emoji}")
            report_lines.append(f"**Status**: {status}")

            # Add specific details based on test type
            if test_name == "availability":
                neuron = test_result.get("neuron_tutorial", {})
                gpu = test_result.get("gpu_tutorial", {})
                report_lines.append(
                    f"- Neuron tutorial: {'✅' if neuron.get('exists') else '❌'}"
                )
                report_lines.append(
                    f"- GPU tutorial: {'✅' if gpu.get('exists') else '❌'}"
                )

            elif test_name == "structure":
                score = test_result.get("compatibility_score", 0)
                report_lines.append(f"- Structure compatibility: {score:.2%}")

                missing_gpu = test_result.get("missing_in_gpu", [])
                missing_neuron = test_result.get("missing_in_neuron", [])

                if missing_gpu:
                    report_lines.append(
                        f"- Missing in GPU tutorial: {', '.join(missing_gpu[:3])}..."
                    )
                if missing_neuron:
                    report_lines.append(
                        f"- Missing in Neuron tutorial: {', '.join(missing_neuron[:3])}..."
                    )

            elif test_name == "models":
                param_matches = sum(
                    1
                    for model in test_result.get("parameter_counts", {}).values()
                    if model.get("match", False)
                )
                total_models = len(test_result.get("parameter_counts", {}))
                if total_models > 0:
                    report_lines.append(
                        f"- Parameter matches: {param_matches}/{total_models}"
                    )

            elif test_name == "datasets":
                common = len(test_result.get("common_datasets", []))
                neuron_only = len(test_result.get("neuron_only", []))
                gpu_only = len(test_result.get("gpu_only", []))
                report_lines.append(f"- Common datasets: {common}")
                report_lines.append(f"- Neuron-only datasets: {neuron_only}")
                report_lines.append(f"- GPU-only datasets: {gpu_only}")

            elif test_name == "version_alignment":
                score = test_result.get("alignment_score", 0)
                report_lines.append(f"- Version alignment: {score:.2%}")

        # Recommendations
        report_lines.append("\n## Recommendations")
        overall_status = self.test_results.get("overall_status", "unknown")

        if overall_status == "fully_compatible":
            report_lines.append(
                "🎉 Full compatibility achieved! Both tutorials ready for fair comparison."
            )
        elif overall_status == "mostly_compatible":
            report_lines.append("✅ High compatibility. Minor adjustments recommended:")
            report_lines.append("- Align any mismatched software versions")
            report_lines.append("- Ensure identical model architectures")
            report_lines.append("- Verify dataset format consistency")
        elif overall_status == "partially_compatible":
            report_lines.append("⚠️ Moderate compatibility. Significant work needed:")
            report_lines.append("- Fix critical structural differences")
            report_lines.append("- Align benchmarking frameworks")
            report_lines.append("- Standardize model implementations")
        else:
            report_lines.append("❌ Low compatibility. Major restructuring required:")
            report_lines.append("- Ensure both tutorials are properly installed")
            report_lines.append("- Align directory structures and file organization")
            report_lines.append("- Implement missing components in GPU tutorial")

        return "\n".join(report_lines)


def main():
    """Main comparison testing entry point."""
    parser = argparse.ArgumentParser(description="Sister Tutorial Comparison Testing")
    parser.add_argument(
        "--full", action="store_true", help="Run full comparison test suite"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick compatibility check"
    )
    parser.add_argument(
        "--generate-report", action="store_true", help="Generate comparison report"
    )
    parser.add_argument(
        "--neuron-tutorial", type=str, help="Path to Neuron tutorial root"
    )
    parser.add_argument("--gpu-tutorial", type=str, help="Path to GPU tutorial root")
    parser.add_argument("--output", type=str, help="Output file for results")

    args = parser.parse_args()

    if not any([args.full, args.quick, args.generate_report]):
        args.quick = True  # Default to quick test

    # Initialize tester
    tester = SisterTutorialTester(args.neuron_tutorial, args.gpu_tutorial)

    # Run tests
    if args.full:
        results = tester.run_full_comparison_test()
    elif args.quick:
        results = tester.run_quick_comparison_test()
    else:
        results = tester.test_results

    # Generate report
    if args.generate_report or args.output:
        report = tester.generate_comparison_report()

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
        else:
            print(report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"sister_tutorial_comparison_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nComparison results saved to: {results_file}")
    print(f"Overall compatibility: {results.get('overall_status', 'unknown')}")

    # Exit with appropriate code
    status = results.get("overall_status", "unknown")
    if status in ["fully_compatible", "compatible"]:
        sys.exit(0)
    elif status in ["mostly_compatible", "partially_compatible"]:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
