.PHONY: install install-dev install-neuron test test-unit test-integration test-fast lint format pre-commit clean build help

# Use uv when available, falling back to python3 -m pip. uv is the recommended workflow.
PYTHON := python3
UV := $(shell command -v uv 2>/dev/null)

help: ## Show this help message
	@echo "AWS Trainium & Inferentia Tutorial - Available Commands:"
	@echo "======================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

install: ## Install package in development mode
ifeq ($(UV),)
	$(PYTHON) -m pip install -e .
else
	uv pip install -e .
endif

install-dev: ## Install with development dependencies (+ pre-commit hooks)
ifeq ($(UV),)
	$(PYTHON) -m pip install -e ".[dev,neuron,science,notebooks]"
else
	uv venv
	uv pip install -e ".[dev,neuron,science,notebooks]"
endif
	pre-commit install

install-neuron: ## Install Neuron SDK components (run on a Neuron instance/DLAMI)
	@echo "Installing Neuron SDK from the AWS Neuron pip index..."
ifeq ($(UV),)
	$(PYTHON) -m pip install torch-neuronx neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com
else
	uv pip install torch-neuronx neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com
endif
	@echo "✅ Neuron SDK installed"

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --cov=scripts --cov=examples --cov=advanced --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v -m "not aws and not neuron"

test-integration: ## Run integration tests (requires AWS)
	$(PYTHON) -m pytest tests/integration/ -v -m "aws"

test-fast: ## Run fast tests only
	$(PYTHON) -m pytest tests/ -v -m "not slow and not aws"

lint: ## Run all linting checks (ruff + mypy)
	$(PYTHON) -m ruff check scripts/ examples/ advanced/ tests/
	$(PYTHON) -m ruff format --check scripts/ examples/ advanced/ tests/
	$(PYTHON) -m mypy scripts/ --ignore-missing-imports

format: ## Format all code and auto-fix lint issues (ruff)
	$(PYTHON) -m ruff check --fix scripts/ examples/ advanced/ tests/
	$(PYTHON) -m ruff format scripts/ examples/ advanced/ tests/

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
