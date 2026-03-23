"""Tests for dsbro.eda."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from dsbro.eda import (
    cardinality,
    categorical_summary,
    compare,
    correlate,
    describe_plus,
    distribution,
    drift,
    duplicates,
    missing,
    numeric_summary,
    outliers,
    overview,
    profile,
    target_analysis,
    value_counts_plot,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close matplotlib figures between tests."""
    yield
    plt.close("all")


@pytest.fixture
def eda_df() -> pd.DataFrame:
    """Return a mixed-type DataFrame with missing values and duplicates."""
    return pd.DataFrame(
        {
            "num_a": [1, 2, 2, 4, 100, 2],
            "num_b": [10, 11, 11, 14, 40, 11],
            "cat_a": ["a", "b", "b", "c", None, "b"],
            "cat_b": ["x", "x", "y", "y", "z", "y"],
            "target_class": [0, 1, 1, 0, 1, 1],
            "target_reg": [0.1, 1.2, 1.2, 0.4, 2.5, 1.2],
        }
    )


@pytest.fixture
def compare_df() -> pd.DataFrame:
    """Return a second DataFrame for comparison and drift checks."""
    return pd.DataFrame(
        {
            "num_a": [2, 3, 3, 5, 6, 7],
            "num_b": [12, 13, 13, 15, 16, 18],
            "cat_a": ["a", "b", "c", "c", "d", "d"],
            "cat_b": ["x", "y", "y", "z", "z", "z"],
            "target_class": [0, 1, 0, 0, 1, 0],
            "target_reg": [0.2, 1.0, 0.9, 0.5, 1.5, 1.7],
        }
    )


def test_overview_returns_shape_and_column_summary(eda_df: pd.DataFrame):
    result = overview(eda_df, sample_size=3)

    assert result["shape"] == (6, 6)
    assert "column_summary" in result
    assert len(result["sample"]) == 3


def test_overview_prints_formatted_report(eda_df: pd.DataFrame, capsys):
    overview(eda_df, sample_size=3)

    captured = capsys.readouterr()
    assert "Dataset Overview" in captured.out
    assert "--- Column Summary ---" in captured.out


def test_describe_plus_includes_extra_metrics(eda_df: pd.DataFrame):
    summary = describe_plus(eda_df)

    assert "missing_pct" in summary.columns
    assert "skew" in summary.columns
    assert summary.loc["num_a", "count"] == 6


def test_missing_returns_summary_without_plot(eda_df: pd.DataFrame):
    summary = missing(eda_df, plot=False)

    assert summary.loc["cat_a", "missing"] == 1


def test_distribution_uses_dark_theme(eda_df: pd.DataFrame):
    figure, axes = distribution(eda_df[["num_a", "cat_a"]], show=False)

    assert figure.get_facecolor() == mcolors.to_rgba("#1a1a2e")
    assert axes.size >= 2


def test_correlate_supports_mixed_types(eda_df: pd.DataFrame):
    matrix = correlate(eda_df[["num_a", "cat_a", "target_class"]], plot=False)

    assert matrix.shape == (3, 3)
    assert matrix.loc["num_a", "num_a"] == 1.0


def test_outliers_detects_extreme_values(eda_df: pd.DataFrame):
    summary = outliers(eda_df[["num_a", "num_b"]], plot=False)

    num_a_row = summary.loc[summary["column"] == "num_a"].iloc[0]
    assert int(num_a_row["outlier_count"]) >= 1


def test_compare_summarizes_shared_columns(
    eda_df: pd.DataFrame,
    compare_df: pd.DataFrame,
):
    summary = compare(eda_df, compare_df, names=("train", "test"), plot=False)

    assert "mean_train" in summary.columns
    assert "cat_a" in summary["column"].tolist()


def test_target_analysis_returns_feature_scores(eda_df: pd.DataFrame):
    summary = target_analysis(eda_df, target="target_class", plot=False)

    assert "association" in summary.columns
    assert "num_a" in summary["feature"].tolist()


def test_cardinality_flags_high_cardinality(eda_df: pd.DataFrame):
    summary = cardinality(eda_df[["num_a", "cat_a"]], threshold=0.4)

    num_a_row = summary.loc[summary["column"] == "num_a"].iloc[0]
    assert bool(num_a_row["high_cardinality"]) is True


def test_duplicates_returns_examples(eda_df: pd.DataFrame):
    result = duplicates(eda_df, subset=["num_a", "num_b", "cat_a", "cat_b"])

    assert result["duplicate_count"] >= 1
    assert not result["examples"].empty


def test_duplicates_prints_summary(eda_df: pd.DataFrame, capsys):
    duplicates(eda_df, subset=["num_a", "num_b", "cat_a", "cat_b"])

    captured = capsys.readouterr()
    assert "Duplicates" in captured.out
    assert "Percentage" in captured.out


def test_value_counts_plot_returns_counts_and_axis(eda_df: pd.DataFrame):
    counts, figure, axis = value_counts_plot(eda_df, "cat_b", show=False)

    assert figure is not None
    assert axis.get_title() == "Value Counts: cat_b"
    assert counts.iloc[0]["count"] >= 1


def test_numeric_summary_returns_statistics(eda_df: pd.DataFrame):
    summary = numeric_summary(eda_df)

    assert summary.loc["num_b", "max"] == 40


def test_categorical_summary_returns_top_values(eda_df: pd.DataFrame):
    summary = categorical_summary(eda_df)

    assert summary.loc["cat_b", "unique"] == 3
    assert summary.loc["cat_a", "missing"] == 1


def test_drift_returns_psi_and_ks(
    eda_df: pd.DataFrame,
    compare_df: pd.DataFrame,
):
    summary = drift(eda_df, compare_df, cols=["num_a", "cat_a"])

    assert "psi" in summary.columns
    assert summary["column"].tolist()[0] in {"num_a", "cat_a"}


def test_profile_returns_bundle(eda_df: pd.DataFrame):
    result = profile(eda_df, target="target_class", show=False)

    assert "overview" in result
    assert "correlation" in result
    assert result["target_analysis"] is not None
