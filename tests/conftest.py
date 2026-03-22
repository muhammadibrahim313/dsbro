"""Shared pytest fixtures for dsbro tests."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a small mixed-type DataFrame for tests."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "feature_num": [10.5, 12.0, 9.5, 14.0],
            "feature_cat": ["a", "b", "a", "c"],
            "target": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def submission_df() -> pd.DataFrame:
    """Return a simple Kaggle-style submission DataFrame."""
    return pd.DataFrame({"id": [1, 2], "target": [0.1, 0.9]})


@pytest.fixture
def temp_data_dir(tmp_path, sample_df: pd.DataFrame):
    """Create a temporary file tree with common sample files."""
    root = tmp_path / "data"
    nested = root / "nested"
    nested.mkdir(parents=True)

    sample_df.to_csv(root / "sample.csv", index=False)
    sample_df.to_csv(root / "sample_a.csv", index=False)
    sample_df.head(2).to_csv(nested / "sample_b.csv", index=False)
    sample_df.to_csv(root / "sample.tsv", sep="\t", index=False)
    sample_df.to_json(root / "sample.json", orient="records")
    (root / "notes.txt").write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    (nested / "script.py").write_text("print('hello')\n", encoding="utf-8")

    return root
