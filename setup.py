"""
AWS Trainium & Inferentia Tutorial Package Setup
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aws-trainium-inferentia-tutorial",
    version="3.0.0",
    author="AWS ML Research Community",
    author_email="your-email@example.com",
    description="Complete tutorial and tools for academic research using AWS Trainium and Inferentia chips",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aws-trainium-inferentia-tutorial",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
            "pre-commit>=2.15",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12",
        ],
        "benchmarking": [
            "matplotlib>=3.5",
            "seaborn>=0.11",
            "plotly>=5.0",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aws-ml-setup=scripts.setup_aws_environment:main",
            "aws-ml-monitor=scripts.cost_monitor:main",
            "aws-ml-emergency=scripts.emergency_shutdown:main",
        ],
    },
    include_package_data=True,
)