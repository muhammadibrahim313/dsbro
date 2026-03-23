# Contributing to dsbro

## How to Contribute

1. Fork the repository.
2. Create a feature branch:
   `git checkout -b feature/your-feature-name`
3. Make your changes.
4. Run tests:
   `pytest tests/ -v`
5. Run the linter:
   `ruff check dsbro/ tests/`
6. Commit with a clear message:
   `git commit -m "feat: add new function xyz"`
7. Push to your fork:
   `git push origin feature/your-feature-name`
8. Open a Pull Request against the `main` branch.

## Commit Message Convention

Use these prefixes:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `style:` formatting, no code change
- `refactor:` code restructuring
- `test:` adding tests
- `chore:` maintenance tasks

## Code Standards

- Python 3.9+ compatible
- Type hints on all public functions
- Google-style docstrings with `Args`, `Returns`, `Example`
- All public functions need at least one test
- `ruff check` must pass with zero errors
- `pytest` must pass with zero failures

## Adding a New Function

1. Add the function to the appropriate module:
   `io.py`, `eda.py`, `prep.py`, `viz.py`, `ml.py`, `metrics.py`, `utils.py`, or `text.py`
2. Follow existing patterns:
   smart defaults, never modify input in-place, return useful objects
3. Add a test in the corresponding test file
4. Update the module docstring if needed
5. Run `pytest` and `ruff check`

## Reporting Bugs

- Open a GitHub Issue
- Include:
  Python version, dsbro version, minimal code to reproduce, full error traceback

## Suggesting Features

- Open a GitHub Issue with a title starting with `[Feature Request]`
- Describe the use case:
  what problem does it solve?
- Show a code example of how you would want to use it
