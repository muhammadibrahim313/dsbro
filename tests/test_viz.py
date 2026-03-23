"""Tests for dsbro.viz."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from dsbro.viz import (
    bar,
    box,
    confusion_matrix,
    countplot,
    feature_importance,
    heatmap,
    hist,
    learning_curve,
    line,
    pairplot,
    pie,
    plotly_bar,
    plotly_line,
    plotly_scatter,
    precision_recall_curve,
    residual_plot,
    roc_curve,
    save_plot,
    scatter,
    set_theme,
    subplot_grid,
)

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None


@pytest.fixture(autouse=True)
def close_figures():
    """Close matplotlib figures between tests."""
    yield
    plt.close("all")


@pytest.fixture
def viz_df() -> pd.DataFrame:
    """Return a small DataFrame for plotting tests."""
    return pd.DataFrame(
        {
            "category": ["a", "b", "c", "a"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "value2": [4.0, 3.0, 2.0, 1.0],
            "group": ["g1", "g1", "g2", "g2"],
        }
    )


@pytest.fixture
def iris_model():
    """Return a fitted model and data for model-based plots."""
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_set_theme_returns_dark_defaults():
    config = set_theme("dark")

    assert config["figure.facecolor"] == "#1a1a2e"


def test_bar_returns_dark_themed_plot(viz_df: pd.DataFrame):
    figure, axis = bar(viz_df, "category", "value", show=False)

    assert figure.get_facecolor() == mcolors.to_rgba("#1a1a2e")
    assert axis.get_title() == "Bar Plot"


def test_line_returns_axis(viz_df: pd.DataFrame):
    _, axis = line(viz_df, "category", "value", show=False)

    assert axis.get_title() == "Line Plot"


def test_scatter_returns_axis(viz_df: pd.DataFrame):
    _, axis = scatter(viz_df, "value", "value2", hue="group", show=False)

    assert axis.get_title() == "Scatter Plot"


def test_hist_returns_axis(viz_df: pd.DataFrame):
    _, axis = hist(viz_df, "value", show=False)

    assert axis.get_title() == "Histogram: value"


def test_box_returns_axis(viz_df: pd.DataFrame):
    _, axis = box(viz_df, x="group", y="value", show=False)

    assert axis.get_title() == "Box Plot"


def test_heatmap_returns_axis():
    _, axis = heatmap(np.array([[1, 2], [3, 4]]), show=False)

    assert axis.get_title() == "Heatmap"


def test_pie_returns_axis():
    _, axis = pie([1, 2, 3], labels=["a", "b", "c"], show=False)

    assert axis.get_title() == "Pie Chart"


def test_countplot_returns_axis(viz_df: pd.DataFrame):
    _, axis = countplot(viz_df, "category", show=False)

    assert axis.get_title() == "Count Plot: category"


def test_pairplot_returns_figure_and_axes(viz_df: pd.DataFrame):
    figure, axes = pairplot(viz_df, cols=["value", "value2"], hue="group", show=False)

    assert figure is not None
    assert axes.size >= 1


def test_feature_importance_returns_axis(iris_model):
    model, _, _ = iris_model
    feature_names = ["f1", "f2", "f3", "f4"]
    _, axis = feature_importance(model, feature_names, show=False)

    assert axis.get_title() == "Feature Importance"


def test_confusion_matrix_returns_axis():
    _, axis = confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0], show=False)

    assert axis.get_title() == "Confusion Matrix"


def test_roc_curve_returns_axis():
    _, axis = roc_curve([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], show=False)

    assert axis.get_title() == "ROC Curve"


def test_precision_recall_curve_returns_axis():
    _, axis = precision_recall_curve([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], show=False)

    assert axis.get_title() == "Precision-Recall Curve"


def test_learning_curve_returns_axis():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=500)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    _, axis = learning_curve(model, X, y, cv=cv, show=False)

    assert axis.get_title() == "Learning Curve"


def test_residual_plot_returns_axis():
    _, axis = residual_plot([1.0, 2.0, 3.0], [0.9, 2.1, 2.8], show=False)

    assert axis.get_title() == "Residual Plot"


def test_plotly_bar_optional_dep(viz_df: pd.DataFrame):
    if not PLOTLY_AVAILABLE:
        with pytest.raises(ImportError):
            plotly_bar(viz_df, "category", "value", show=False)
        return

    figure, axis = plotly_bar(viz_df, "category", "value", show=False)
    assert axis is None
    assert figure is not None


def test_plotly_scatter_optional_dep(viz_df: pd.DataFrame):
    if not PLOTLY_AVAILABLE:
        with pytest.raises(ImportError):
            plotly_scatter(viz_df, "value", "value2", show=False)
        return

    figure, axis = plotly_scatter(viz_df, "value", "value2", show=False)
    assert axis is None
    assert figure is not None


def test_plotly_line_optional_dep(viz_df: pd.DataFrame):
    if not PLOTLY_AVAILABLE:
        with pytest.raises(ImportError):
            plotly_line(viz_df, "category", "value", show=False)
        return

    figure, axis = plotly_line(viz_df, "category", "value", show=False)
    assert axis is None
    assert figure is not None


def test_subplot_grid_returns_axes():
    def draw(axis):
        axis.plot([0, 1], [0, 1], color="#00d4ff")
        axis.set_title("Mini Plot")

    figure, axes = subplot_grid([draw, draw], ncols=2, show=False)

    assert figure is not None
    assert axes.size == 2


def test_save_plot_writes_file(viz_df: pd.DataFrame, tmp_path: Path):
    figure, _ = bar(viz_df, "category", "value", show=False)
    target = save_plot(figure, tmp_path / "plot.png")

    assert target.exists()
