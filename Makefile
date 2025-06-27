.PHONY: install install-dev install-neuron test lint format clean docs build help

# Default Python version
PYTHON := python3

help: ## Show this help message
	@echo "AWS Trainium & Inferentia Tutorial - Available Commands:"
	@echo "======================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

install: ## Install package in development mode
	$(PYTHON) -m pip install -e .

install-dev: ## Install with development dependencies
	$(PYTHON) -m pip install -e ".[dev,neuron,science,notebooks]"
	$(PYTHON) -m pre-commit install

install-neuron: ## Install Neuron SDK components
	@echo "Installing Neuron SDK..."
	$(PYTHON) -m pip install torch-neuronx neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com
	@echo "✅ Neuron SDK installed"

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --cov=scripts --cov=examples --cov=advanced --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v -m "not aws and not neuron"

test-integration: ## Run integration tests (requires AWS)
	$(PYTHON) -m pytest tests/integration/ -v -m "aws"

test-fast: ## Run fast tests only
	$(PYTHON) -m pytest tests/ -v -m "not slow and not aws"

lint: ## Run all linting checks
	$(PYTHON) -m black --check scripts/ examples/ advanced/ tests/
	$(PYTHON) -m isort --check-only scripts/ examples/ advanced/ tests/
	$(PYTHON) -m flake8 scripts/ examples/ advanced/ tests/
	$(PYTHON) -m mypy scripts/ --ignore-missing-imports

format: ## Format all code
	$(PYTHON) -m black scripts/ examples/ advanced/ tests/
	$(PYTHON) -m isort scripts/ examples/ advanced/ tests/

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

setup-aws: ## Set up AWS environment
	$(PYTHON) scripts/setup_aws_environment.py

monitor-costs: ## Monitor current AWS costs
	$(PYTHON) scripts/cost_monitor.py

emergency-shutdown: ## Emergency shutdown of all ML instances
	$(PYTHON) scripts/emergency_shutdown.py

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	$(PYTHON) -m build

# Development shortcuts
run-climate-example: ## Run climate science example
	cd examples/domain_specific && $(PYTHON) domain_specific_examples.py

run-rag-example: ## Run RAG pipeline example
	cd examples/rag_pipeline && $(PYTHON) modern_rag_example.py

run-workflow-example: ## Run complete Trainium→Inferentia workflow
	cd examples/complete_workflow && $(PYTHON) trainium_to_inferentia_pipeline.py
