# Contributing

Contributions are welcome and greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways.

## Types of Contributions

### Report Bugs

Report bugs at:
https://github.com/francois-durand/svvamp/issues

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might help reproduce the problem.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with **"bug"** and **"help wanted"** is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with **"enhancement"** and **"help wanted"** is open to contributors.

### Write Documentation

SVVAMP can always use more documentation. This includes:

* improvements to the official documentation
* better docstrings
* tutorials, blog posts, or external guides

### Submit Feedback

The best way to send feedback is to open an issue:

https://github.com/francois-durand/svvamp/issues

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible.
* Remember that this is a volunteer-driven project and contributions are welcome.

---

# Get Started

Ready to contribute? Here's how to set up **svvamp** for local development.

## 1. Fork the repository

Fork the repository on GitHub.

## 2. Clone your fork

```bash
git clone git@github.com:your_name_here/svvamp.git
cd svvamp
```

## 3. Create the development environment

This project uses **uv** to manage environments and dependencies.

Create a virtual environment:

```bash
uv venv
```

Install the package in editable mode with development dependencies:

```bash
uv pip install -e ".[dev]"
```

## 4. Create a branch

Create a branch for your contribution:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

---

# Running Tests

Run the test suite with:

```bash
uv run pytest
```

To run a subset of tests:

```bash
uv run pytest tests/test_svvamp.py
```

---

# Code Quality

Lint the code using **ruff**:

```bash
uv run ruff check
```

You can also automatically fix some issues:

```bash
uv run ruff check --fix
```

---

# Pull Request Guidelines

Before submitting a pull request, please check that:

1. The pull request includes tests when appropriate.
2. If new functionality is added, the documentation is updated.
3. The test suite passes.

Continuous integration automatically tests the project on multiple Python versions.

---

# Documentation

To build the documentation locally:

```bash
uv pip install -e ".[docs]"
uv run sphinx-build docs docs/_build
```

---

# Release Process (Maintainers)

Versions are managed with **bump-my-version**.

To bump the version:

```bash
bump-my-version patch
```

Then build the package:

```bash
uv run python -m build
```

---

Thank you for contributing to SVVAMP!
