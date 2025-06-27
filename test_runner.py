#!/usr/bin/env python3
"""
Comprehensive test runner for AWS Trainium & Inferentia Tutorial
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time
        print(f"✅ {description} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.1f}s")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False


def check_dependencies() -> bool:
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")

    required_packages = [
        "pytest",
        "black",
        "flake8",
        "mypy",
        "isort",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Run: make install-dev")
        return False

    print("✅ All dependencies available")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run tests and quality checks")
    parser.add_argument("--skip-lint", action="store_true", help="Skip linting checks")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--unit-only", action="store_true", help="Run unit tests only")
    parser.add_argument("--fix", action="store_true", help="Auto-fix formatting issues")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("🚀 AWS Trainium & Inferentia Tutorial - Test Runner")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    all_passed = True

    # Formatting and linting
    if not args.skip_lint:
        if args.fix:
            # Auto-fix formatting
            all_passed &= run_command(
                [
                    "python",
                    "-m",
                    "black",
                    "scripts/",
                    "examples/",
                    "advanced/",
                    "tests/",
                ],
                "Auto-formatting code with Black",
            )
            all_passed &= run_command(
                [
                    "python",
                    "-m",
                    "isort",
                    "scripts/",
                    "examples/",
                    "advanced/",
                    "tests/",
                ],
                "Auto-sorting imports with isort",
            )

        # Check formatting
        all_passed &= run_command(
            [
                "python",
                "-m",
                "black",
                "--check",
                "scripts/",
                "examples/",
                "advanced/",
                "tests/",
            ],
            "Checking code formatting",
        )

        # Check import sorting
        all_passed &= run_command(
            [
                "python",
                "-m",
                "isort",
                "--check-only",
                "scripts/",
                "examples/",
                "advanced/",
                "tests/",
            ],
            "Checking import sorting",
        )

        # Lint with flake8
        all_passed &= run_command(
            ["python", "-m", "flake8", "scripts/", "examples/", "advanced/", "tests/"],
            "Running flake8 linting",
        )

        # Type checking with mypy (only for scripts)
        all_passed &= run_command(
            ["python", "-m", "mypy", "scripts/", "--ignore-missing-imports"],
            "Running mypy type checking",
        )

    # Tests
    if not args.skip_tests:
        if args.unit_only:
            # Unit tests only
            pytest_cmd = [
                "python",
                "-m",
                "pytest",
                "tests/unit/",
                "-v",
                "-m",
                "not aws and not neuron",
            ]
            if args.verbose:
                pytest_cmd.extend(["--tb=long", "-s"])

            all_passed &= run_command(pytest_cmd, "Running unit tests")
        else:
            # All tests (excluding AWS and slow tests by default)
            pytest_cmd = [
                "python",
                "-m",
                "pytest",
                "tests/",
                "-v",
                "-m",
                "not aws and not slow",
            ]
            if args.verbose:
                pytest_cmd.extend(["--tb=long", "-s"])

            all_passed &= run_command(pytest_cmd, "Running fast tests")

    # Security check (if bandit is available)
    try:
        import bandit

        all_passed &= run_command(
            ["python", "-m", "bandit", "-r", "scripts/", "examples/", "advanced/"],
            "Running security checks",
        )
    except ImportError:
        print("ℹ️  Bandit not available, skipping security checks")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed!")
        print("\nNext steps:")
        print("  • Run 'make test-integration' to test with AWS credentials")
        print("  • Run 'make install-neuron' to test Neuron SDK integration")
        print("  • Check examples work: 'make run-climate-example'")
        sys.exit(0)
    else:
        print("❌ Some checks failed!")
        print("\nTo fix issues:")
        print("  • Run with --fix to auto-format code")
        print("  • Check the error messages above")
        print("  • Run 'make format' to fix formatting")
        sys.exit(1)


if __name__ == "__main__":
    main()
