<!-- logo placeholder -->
<p align="center">
  <img src="https://raw.githubusercontent.com/muhammadibrahim313/dsbro/main/imgs/dsbro.png" alt="dsbro logo" width="320">
</p>

# dsbro

[![CI](https://github.com/muhammadibrahim313/dsbro/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammadibrahim313/dsbro/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/dsbro.svg)](https://pypi.org/project/dsbro/)
[![Downloads](https://img.shields.io/pypi/dm/dsbro.svg)](https://pypistats.org/packages/dsbro)

Your Data Science Bro. One import away.

`dsbro` is an all-in-one Python library for notebook-first data science. It pulls together setup, file I/O, EDA, preprocessing, visualization, metrics, ML baselines, and text helpers into one import with smart defaults and dark-theme visuals.

## Installation

```bash
pip install dsbro
pip install dsbro[ml]
pip install dsbro[all]
```

## Quick Start

```python
import dsbro
dsbro.setup()
print(dsbro.version())
dsbro.about()
```

```python
from dsbro import eda
eda.overview(df)
```

```python
from dsbro import prep
df_small = prep.reduce_memory(df)
df_clean = prep.fill_missing(df)
```

```python
from dsbro import viz
viz.set_theme("dark")
viz.heatmap(df.corr(numeric_only=True))
```

```python
from dsbro import ml
results = ml.compare(df, target="target", cv=3)
results.head()
```

```python
from dsbro import metrics
metrics.regression_report(y_true, y_pred)
```

## Modules

| Module | What it does |
| --- | --- |
| `utils` | Notebook setup, seeding, timers, system info, and environment helpers |
| `io` | File loading, saving, peeking, searching, and submission utilities |
| `eda` | Dataset overview, missing values, correlation, outliers, drift, and profiling |
| `prep` | Encoding, scaling, missing-value handling, feature engineering, and memory reduction |
| `viz` | Dark-theme charts for tabular analysis and model evaluation |
| `metrics` | Quick regression and classification metrics in one place |
| `ml` | Model comparison, training, tuning, blending, stacking, and OOF utilities |
| `text` | Text cleaning, tokenization, n-grams, word frequency, and TF-IDF features |

## Why dsbro?

- You stop copy-pasting the same notebook boilerplate for setup, missing values, scaling, and memory reduction.
- You get cleaner charts without writing styling code every time.
- You avoid scattered imports across pandas, seaborn, sklearn, and utility snippets.
- You can benchmark baseline models in one line instead of wiring cross-validation by hand.
- You keep common Kaggle and Colab workflows in one small, consistent package.

## For Kaggle Users

Use `dsbro` to replace the usual notebook starter blocks:

```python
!pip install dsbro[all] -q

import dsbro
dsbro.setup()
```

Useful first calls:

```python
from dsbro import eda, prep, ml
eda.profile(train_df, target="target")
train_small = prep.reduce_memory(train_df)
leaderboard = ml.compare(train_small, target="target", cv=3)
```

## For Colab Users

Install in the first cell, then keep the rest of the notebook clean:

```python
!pip install dsbro[all] -q

import dsbro
dsbro.setup()
```

Colab-friendly flow:

```python
from dsbro import io, eda, viz
data = io.load("/content/train.csv")
eda.overview(data)
viz.hist(data, col="target")
```

## Dependencies

Core dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Optional extras:

- `dsbro[ml]`: `lightgbm`, `xgboost`, `catboost`, `optuna`
- `dsbro[plotly]`: `plotly`
- `dsbro[all]`: all optional extras together

## Notebook Example

The project includes a proper tutorial notebook:

- [examples/quickstart.ipynb](https://github.com/muhammadibrahim313/dsbro/blob/main/examples/quickstart.ipynb)

## Development

```bash
pytest tests/ -v
ruff check dsbro/ tests/
python -m build
```

## Contributing

We welcome contributions. Read [CONTRIBUTING.md](https://github.com/muhammadibrahim313/dsbro/blob/main/CONTRIBUTING.md) before opening a PR.

## License

MIT. See [LICENSE](https://github.com/muhammadibrahim313/dsbro/blob/main/LICENSE).

## Author

Muhammad Ibrahim Qasmi  
[Website](https://ibrahimqasmi.com)  
[GitHub](https://github.com/muhammadibrahim313/dsbro)
