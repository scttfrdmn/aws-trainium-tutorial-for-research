# Quick Start Guide

## Prerequisites

1. **AWS Account**: With access to Trainium/Inferentia instances
2. **Python 3.8+**: Recommended version for compatibility
3. **Basic ML Knowledge**: Familiarity with PyTorch and transformers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aws-trainium-inferentia-tutorial
cd aws-trainium-inferentia-tutorial

# Install dependencies
make install-dev

# Install Neuron SDK
make install-neuron
```

## First Steps

### 1. Set Up Cost Controls

```bash
# Create budget alerts (IMPORTANT!)
python scripts/setup_budget.py --limit 500 --email your-email@university.edu
```

### 2. Configure AWS

```bash
# Set up your AWS credentials
aws configure

# Test your setup
python scripts/cost_monitor.py
```

### 3. Launch Your First Experiment

```bash
# Launch a small test instance (auto-terminates in 2 hours)
python scripts/ephemeral_instance.py --name "test-experiment" --max-hours 2
```

## Example Workflows

### Climate Science Research
```bash
make run-climate-example
```

### RAG Pipeline
```bash
make run-rag-example
```

### Complete Training → Inference Pipeline
```bash
make run-workflow-example
```

## Cost Monitoring

Monitor your spending in real-time:

```bash
# Generate cost report
make monitor-costs

# Emergency shutdown (if needed)
make emergency-shutdown
```

## Next Steps

1. Read the [complete tutorial](docs/main_tutorial_doc.md)
2. Explore [domain-specific examples](examples/domain_specific/)
3. Try the [advanced patterns](advanced/)
4. Set up [cost monitoring dashboard](monitoring/)

## Getting Help

- Check the [troubleshooting guide](docs/troubleshooting.md)
- Review [common issues](docs/faq.md)
- Ask questions in [GitHub Discussions](../../discussions)