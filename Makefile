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
	$(PYTHON) -m pip install -e ".[dev,docs,benchmarking]"

install-neuron: ## Install Neuron SDK components
	@echo "Installing Neuron SDK..."
	$(PYTHON) -m pip install torch-neuronx neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com
	@echo "✅ Neuron SDK installed"

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v

lint: ## Run linting checks
	$(PYTHON) -m flake8 scripts/ examples/ advanced/
	$(PYTHON) -m black --check scripts/ examples/ advanced/

format: ## Format code with black
	$(PYTHON) -m black scripts/ examples/ advanced/

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