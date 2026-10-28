.PHONY: install install-dev install-neuron test test-unit test-integration test-fast coverage lint format pre-commit clean build help

# Use uv when available, falling back to python3 -m pip. uv is the recommended workflow.
PYTHON := python3
UV := $(shell command -v uv 2>/dev/null)
# Tool runner: `uv run --extra dev <tool>` under uv, else `python3 -m <tool>` (dev deps installed).
ifeq ($(UV),)
RUN := $(PYTHON) -m
else
RUN := uv run --extra dev
endif

help: ## Show this help message
	@echo "AWS Trainium & Inferentia Tutorial - Available Commands:"
	@echo "======================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

test: ## Run the test suite off-hardware (same selection as CI)
	$(RUN) pytest -m "not aws and not neuron"

test-unit: ## Run unit tests only
	$(RUN) pytest tests/unit/ -v -m "not aws and not neuron"

test-integration: ## Run integration tests (requires AWS)
	$(RUN) pytest tests/integration/ -v -m "aws"

test-fast: ## Run fast tests only
	$(RUN) pytest -m "not slow and not aws and not neuron"

coverage: ## Coverage report (HTML + terminal); informational, not gated
	$(RUN) pytest -m "not aws and not neuron" --cov=scripts --cov=examples --cov=advanced --cov=validation --cov-report=html --cov-report=term --cov-fail-under=0

lint: ## Run all linting checks (same as CI: ruff + ruff format + mypy)
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy scripts/ --ignore-missing-imports

format: ## Format all code and auto-fix lint issues (ruff)
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

setup-aws: ## Check (read-only) the AWS environment; pass ARGS=--auto-fix to fix issues
	$(PYTHON) scripts/aws_environment_checker.py $(ARGS)

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

# Development shortcuts (the validated / real examples)
run-ner-smoke: ## Run the validated NER example on CPU (smoke test, free)
	NER_SMOKE=1 $(PYTHON) examples/use_cases/biomedical_ner.py

run-debug-walkthrough: ## Run the debugging walkthrough (CPU)
	$(PYTHON) examples/debugging/diagnose_common_failures.py

run-workflow-example: ## Run complete Trainium→Inferentia workflow (requires AWS)
	cd examples/complete_workflow && $(PYTHON) trainium_to_inferentia_pipeline.py
