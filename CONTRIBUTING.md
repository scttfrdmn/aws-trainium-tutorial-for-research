# Contributing to AWS Trainium & Inferentia Tutorial

We welcome contributions from the community! This guide outlines how to contribute to the tutorial.

## 🎯 Types of Contributions

We welcome several types of contributions:

1. **New Domain Examples**: Research examples for specific academic fields
2. **Performance Optimizations**: Improvements to existing code or new optimization techniques
3. **Bug Fixes**: Corrections to existing code or documentation
4. **Documentation**: Improvements to tutorials, docstrings, or guides
5. **Tools and Utilities**: New helper scripts or monitoring tools
6. **Cost Optimization**: New strategies for reducing AWS costs

## 🚀 Getting Started

### Prerequisites

- Python 3.10–3.13 (the repo pins 3.12 via `.python-version`)
- [uv](https://docs.astral.sh/uv/) (recommended) for environment management
- AWS Account with Trainium/Inferentia access (only for hardware-dependent work)
- Basic familiarity with PyTorch and transformers

### Setup Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/scttfrdmn/aws-trainium-tutorial-for-research
   cd aws-trainium-tutorial-for-research
   ```

2. **Install in development mode** (uv recommended):
   ```bash
   make install-dev      # uv venv + uv pip install -e ".[dev,...]" + pre-commit install
   make install-neuron   # only on a Neuron instance/DLAMI
   ```

3. **Run the checks**:
   ```bash
   make lint   # ruff check + ruff format --check + mypy
   make test   # pytest (use `-m "not aws and not neuron"` off-hardware)
   ```

## 📋 Project tracking, versioning & changelog

**All project work lives on GitHub — not in local files.** Do not add `PROJECT_STATUS.md`,
`ROADMAP.md`, `NEXT_ACTIONS.md`, or similar planning docs to the repo; they go stale and clutter
the tree.

- **Plan/track** with [GitHub milestones, issues, and labels](https://github.com/scttfrdmn/aws-trainium-tutorial-for-research/milestones).
  Labels follow a `type:` / `area:` / `status:` taxonomy. Each release is a milestone.
- **Versioning:** [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html). Pre-1.0, the
  tutorial is evolving; `0.MINOR.PATCH`. `1.0.0` = every example hardware-validated.
- **Changelog:** keep [`CHANGELOG.md`](CHANGELOG.md) in [Keep a Changelog](https://keepachangelog.com/)
  format. Add an entry under `## [Unreleased]` in the same PR as your change; releases move the
  Unreleased section under a version heading and tag `vMAJOR.MINOR.PATCH`.
- **Commits/PRs:** reference the issue (`#123`) and use clear, conventional summaries.

## 📝 Development Guidelines

### Code Style

- Use [Ruff](https://docs.astral.sh/ruff/) for formatting **and** linting: `make format` (auto-fix) / `make lint` (check). Ruff replaces black, isort, flake8, and pydocstyle.
- Follow [PEP 8](https://pep8.org/); modern syntax (PEP 585/604 type hints, f-strings) is enforced by ruff's `UP` rules.
- Use type hints where possible; `mypy` runs in CI.
- Write docstrings for public functions/classes in `scripts/` (the maintained tooling). Docstring rules are relaxed for teaching examples under `examples/`.

### Testing

- Write tests for new functionality
- Ensure all tests pass: `make test`
- Include both unit tests and integration tests where appropriate

### Cost Considerations

- Always include cost estimates in examples
- Use spot instances by default
- Include auto-termination mechanisms
- Test cost tracking functionality

## 🔬 Adding New Domain Examples

We especially welcome examples from new research domains. Follow `examples/use_cases/biomedical_ner.py` as the template (real data, `run(config)` entrypoint, harness-validatable).

## 🧪 Testing Guidelines

Create tests in `tests/` directory following the existing patterns.

## 📊 Performance Benchmarking

When adding new optimizations, include benchmarks in the `benchmarks/` directory.

## 🔍 Pull Request Process

### Before Submitting

- [ ] Code passes `make lint` (ruff + mypy) and `make format` leaves no changes
- [ ] Tests pass (`make test`)
- [ ] Documentation updated
- [ ] Cost estimates included **and labeled as estimates** (don't claim "verified" without a cited run)
- [ ] Any new performance numbers state the instance type and Neuron SDK version used

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] New domain example

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Tested on AWS instances
- [ ] Cost estimates verified

## Cost Impact
Estimated cost per experiment: $X.XX
Savings vs alternative: X%
```

## 💡 Ideas for Contributions

### High Priority
- **New domain examples**: Physics, astronomy, economics, linguistics
- **Advanced NKI kernels**: Custom operations for specific domains
- **Cost optimization tools**: Better spot instance management

### Medium Priority  
- **Monitoring improvements**: Better dashboards and alerting
- **Educational content**: Jupyter notebooks and tutorials
- **Integration examples**: With popular ML frameworks

## 🤝 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion

### Communication Guidelines

- Be respectful and inclusive
- Focus on constructive feedback
- Share knowledge and learn from others

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to making ML research more accessible and cost-effective! 🚀**