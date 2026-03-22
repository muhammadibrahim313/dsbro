# dsbro

Your Data Science Bro. One import away.

`dsbro` is a lightweight Python toolkit for notebook-heavy data science work. It aims to reduce
the repeated setup, file handling, and utility code that shows up in Kaggle notebooks, Colab
experiments, and local Jupyter workflows.

## Status

This repository is being built in phases. The current foundation includes:

- Packaging and project scaffold
- Shared helpers and plotting themes
- `dsbro.utils`
- `dsbro.io`
- Pytest coverage for the foundation modules

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Example

```python
import dsbro

dsbro.setup()
train = dsbro.io.load("train.csv")
preview = dsbro.io.peek("train.csv", n=3)
```

## Roadmap

The planned modules are:

- `io`
- `eda`
- `prep`
- `viz`
- `ml`
- `metrics`
- `utils`
- `text`

## Development

```bash
pytest tests/ -v
ruff check dsbro/ tests/
ruff format dsbro/ tests/
python -m build
```

