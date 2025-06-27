# Contributing to AWS Trainium & Inferentia Tutorial

Thank you for your interest in contributing to this tutorial! This guide will help you get started.

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

- Python 3.8+ 
- AWS Account with Trainium/Inferentia access
- Basic familiarity with PyTorch and transformers

### Setup Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/aws-trainium-inferentia-tutorial
   cd aws-trainium-inferentia-tutorial
   ```

2. **Install in development mode**:
   ```bash
   make install-dev
   make install-neuron
   ```

3. **Run tests to verify setup**:
   ```bash
   make test
   ```

## 📝 Development Guidelines

### Code Style

- Use [Black](https://black.readthedocs.io/) for code formatting: `make format`
- Follow [PEP 8](https://pep8.org/) guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes

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

We especially welcome examples from new research domains. Follow the existing examples in `examples/domain_specific/` as templates.

## 🧪 Testing Guidelines

Create tests in `tests/` directory following the existing patterns.

## 📊 Performance Benchmarking

When adding new optimizations, include benchmarks in the `benchmarks/` directory.

## 🔍 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines (`make lint`)
- [ ] Tests pass (`make test`)
- [ ] Documentation updated
- [ ] Cost estimates included
- [ ] Examples tested on actual AWS instances

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