# Contributing to BMW Agents

Thank you for your interest in contributing to the BMW Agents framework! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/bmw-agents.git
   cd bmw-agents
   ```
3. Install the project in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

### Coding Standards

We follow these coding standards:

- Use [Black](https://github.com/psf/black) for code formatting with a line length of 100
- Sort imports with [isort](https://pycqa.github.io/isort/) (Black compatible)
- Use [mypy](http://mypy-lang.org/) for type checking
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines

The project includes configurations for these tools in the `pyproject.toml` file.

### Running Tests

Run tests using pytest:

```bash
pytest
```

For more verbose output:

```bash
pytest -v
```

### Adding New Features

When adding new features, please follow these guidelines:

1. **Documentation**: Add docstrings to all public classes and methods
2. **Tests**: Write unit tests for your code
3. **Type Hints**: Include proper type hints for all functions and methods
4. **Examples**: Consider adding an example to demonstrate your feature

### Adding New Dependencies

If your contribution requires new dependencies:

1. Add them to the `dependencies` or `optional-dependencies` section in `pyproject.toml`
2. Update `setup.py` if applicable
3. Explain why the dependency is needed in your PR description

## Pull Request Process

1. Ensure your code passes all tests and linting checks
2. Update documentation, including docstrings and the README if applicable
3. Submit a pull request with a clear description of the changes and any relevant issue numbers
4. Wait for review and address any comments

## Prompt Strategy Contributions

When adding new prompt strategies:

1. Place them in `bmw_agents/core/prompt_strategies/`
2. Include associated prompt templates in `bmw_agents/configs/prompt_templates/`
3. Ensure they follow the base class interfaces
4. Add example usage in the `examples` directory

## LLM Provider Contributions

When adding support for new LLM providers:

1. Extend the provider classes in `bmw_agents/utils/llm_providers.py`
2. Implement all required methods, including token counting
3. Add appropriate error handling
4. Document usage requirements (API keys, etc.)

## License

By contributing to BMW Agents, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

If you have questions about contributing, please open an issue or reach out to the maintainers.

Thank you for your contributions to making BMW Agents better! 