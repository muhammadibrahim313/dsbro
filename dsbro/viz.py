"""Visualization helpers for dsbro."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)
from sklearn.metrics import (
    precision_recall_curve as sk_precision_recall_curve,
)
from sklearn.metrics import (
    roc_curve as sk_roc_curve,
)
from sklearn.model_selection import learning_curve as sk_learning_curve

from dsbro._themes import DEFAULT_THEME, apply_matplotlib_theme, get_theme

_ACTIVE_THEME = DEFAULT_THEME
_PLOTLY_COLORS = ["#00d4ff", "#00ff88", "#ff6b6b", "#ffd93d", "#c084fc"]


def _validate_dataframe(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """Validate that an object is a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame for {name}, got {type(df).__name__}")
    return df.copy()


def _validate_column(df: pd.DataFrame, column: str) -> None:
    """Validate that a column exists in a DataFrame."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")


def _apply_theme(style: str | None = None) -> dict[str, Any]:
    """Apply and return the active plotting theme."""
    theme_name = style or _ACTIVE_THEME
    theme = apply_matplotlib_theme(theme_name)
    sns.set_theme(style="darkgrid" if theme_name != "light" else "whitegrid", rc=theme)
    return theme


def _create_figure_axis(
    figsize: tuple[float, float] = (8.0, 5.0),
    *,
    style: str | None = None,
) -> tuple[Figure, Axes]:
    """Create a themed matplotlib figure and axis."""
    theme = _apply_theme(style)
    figure, axis = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor(theme["figure.facecolor"])
    axis.set_facecolor(theme["axes.facecolor"])
    return figure, axis


def _finalize_plot(figure: Figure, show: bool) -> None:
    """Finalize a matplotlib figure."""
    figure.tight_layout()
    if show:
        plt.show()


def _add_vertical_bar_labels(axis: Axes) -> None:
    """Add labels above vertical bars."""
    for patch in axis.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        axis.annotate(
            f"{height:.2f}" if not float(height).is_integer() else f"{height:.0f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#e0e0e0",
            xytext=(0, 3),
            textcoords="offset points",
        )


def _add_horizontal_bar_labels(axis: Axes) -> None:
    """Add labels to horizontal bars."""
    for patch in axis.patches:
        width = patch.get_width()
        if np.isnan(width):
            continue
        axis.annotate(
            f"{width:.2f}" if not float(width).is_integer() else f"{width:.0f}",
            (width, patch.get_y() + patch.get_height() / 2),
            ha="left",
            va="center",
            fontsize=9,
            color="#e0e0e0",
            xytext=(4, 0),
            textcoords="offset points",
        )


def _resolve_plotly() -> Any:
    """Import plotly.graph_objects lazily."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Plotly required. Install with: pip install dsbro[plotly]") from exc
    return go


def set_theme(style: str = DEFAULT_THEME) -> dict[str, Any]:
    """Set the default dsbro plotting theme.

    Args:
        style: Theme name such as ``dark``, ``light``, ``paper``, ``kaggle``, or ``neon``.

    Returns:
        The applied matplotlib rcParams dictionary.

    Example:
        >>> from dsbro.viz import set_theme
        >>> config = set_theme("dark")
        >>> config["figure.facecolor"]
        '#1a1a2e'
    """
    global _ACTIVE_THEME
    get_theme(style)
    _ACTIVE_THEME = style
    return _apply_theme(style)


def bar(
    df: pd.DataFrame,
    x: str,
    y: str | None = None,
    horizontal: bool = False,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a bar plot with dsbro styling.

    Args:
        df: Input DataFrame.
        x: Category column name.
        y: Optional value column. When omitted, counts are plotted.
        horizontal: Whether to render a horizontal bar chart.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axis.

    Example:
        >>> import pandas as pd
        >>> from dsbro.viz import bar
        >>> fig, ax = bar(pd.DataFrame({"x": ["a", "b"]}), "x", show=False)
        >>> fig is not None
        True
    """
    data = _validate_dataframe(df)
    _validate_column(data, x)
    if y is not None:
        _validate_column(data, y)

    figure, axis = _create_figure_axis()
    if y is None:
        counts = data[x].astype("object").fillna("Missing").value_counts().reset_index()
        counts.columns = [x, "count"]
        if horizontal:
            sns.barplot(data=counts, y=x, x="count", ax=axis, color="#00d4ff")
            _add_horizontal_bar_labels(axis)
        else:
            sns.barplot(data=counts, x=x, y="count", ax=axis, color="#00d4ff")
            axis.tick_params(axis="x", rotation=45)
            _add_vertical_bar_labels(axis)
        axis.set_ylabel("Count")
    else:
        if horizontal:
            sns.barplot(data=data, y=x, x=y, ax=axis, color="#00d4ff")
            _add_horizontal_bar_labels(axis)
        else:
            sns.barplot(data=data, x=x, y=y, ax=axis, color="#00d4ff")
            axis.tick_params(axis="x", rotation=45)
            _add_vertical_bar_labels(axis)
        axis.set_ylabel(y)
    axis.set_title("Bar Plot")
    axis.set_xlabel(x if not horizontal else y or "count")
    _finalize_plot(figure, show)
    return figure, axis


def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a line plot with markers.

    Args:
        df: Input DataFrame.
        x: X-axis column name.
        y: Y-axis column name.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axis.

    Example:
        >>> import pandas as pd
        >>> from dsbro.viz import line
        >>> fig, ax = line(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "x", "y", show=False)
        >>> ax.get_title() == "Line Plot"
        True
    """
    data = _validate_dataframe(df)
    _validate_column(data, x)
    _validate_column(data, y)

    figure, axis = _create_figure_axis()
    sns.lineplot(data=data, x=x, y=y, ax=axis, marker="o", color="#00d4ff")
    axis.set_title("Line Plot")
    _finalize_plot(figure, show)
    return figure, axis


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    size: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a scatter plot with optional hue and size encodings.

    Args:
        df: Input DataFrame.
        x: X-axis column name.
        y: Y-axis column name.
        hue: Optional hue column name.
        size: Optional size column name.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axis.

    Example:
        >>> import pandas as pd
        >>> from dsbro.viz import scatter
        >>> fig, ax = scatter(pd.DataFrame({"x": [1], "y": [2]}), "x", "y", show=False)
        >>> fig is not None
        True
    """
    data = _validate_dataframe(df)
    for column in [x, y, hue, size]:
        if column is not None:
            _validate_column(data, column)

    figure, axis = _create_figure_axis()
    sns.scatterplot(data=data, x=x, y=y, hue=hue, size=size, ax=axis)
    axis.set_title("Scatter Plot")
    _finalize_plot(figure, show)
    return figure, axis


def hist(
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a histogram with KDE overlay.

    Args:
        df: Input DataFrame.
        col: Numeric column to plot.
        bins: Number of histogram bins.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axis.
    """
    data = _validate_dataframe(df)
    _validate_column(data, col)

    figure, axis = _create_figure_axis()
    sns.histplot(data[col].dropna(), bins=bins, kde=True, ax=axis, color="#00d4ff")
    axis.set_title(f"Histogram: {col}")
    axis.set_xlabel(col)
    _finalize_plot(figure, show)
    return figure, axis


def box(
    df: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a box plot.

    Args:
        df: Input DataFrame.
        x: Optional grouping column.
        y: Optional numeric column.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axis.
    """
    data = _validate_dataframe(df)
    if x is not None:
        _validate_column(data, x)
    if y is not None:
        _validate_column(data, y)

    figure, axis = _create_figure_axis()
    sns.boxplot(data=data, x=x, y=y, ax=axis, color="#00d4ff")
    axis.set_title("Box Plot")
    if x is not None:
        axis.tick_params(axis="x", rotation=45)
    _finalize_plot(figure, show)
    return figure, axis


def heatmap(
    data: pd.DataFrame | np.ndarray,
    annot: bool = True,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create an annotated heatmap.

    Args:
        data: 2D data structure plotted as a heatmap.
        annot: Whether to annotate values inside cells.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axis.
    """
    frame = pd.DataFrame(data)
    figure, axis = _create_figure_axis(figsize=(7.0, 5.5))
    sns.heatmap(frame, annot=annot, cmap="mako", ax=axis)
    axis.set_title("Heatmap")
    _finalize_plot(figure, show)
    return figure, axis


def pie(
    data: pd.Series | list[float] | np.ndarray,
    labels: list[str] | None = None,
    explode: list[float] | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a pie chart.

    Args:
        data: Slice sizes.
        labels: Optional labels for each slice.
        explode: Optional explode offsets.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axis.
    """
    values = pd.Series(data)
    figure, axis = _create_figure_axis()
    axis.pie(
        values,
        labels=labels or values.index.astype(str).tolist(),
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "#e0e0e0"},
    )
    axis.set_title("Pie Chart")
    _finalize_plot(figure, show)
    return figure, axis


def countplot(
    df: pd.DataFrame,
    col: str,
    top_n: int = 20,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a count plot with value labels.

    Args:
        df: Input DataFrame.
        col: Categorical column to count.
        top_n: Maximum number of categories to show.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axis.
    """
    data = _validate_dataframe(df)
    _validate_column(data, col)

    counts = data[col].astype("object").fillna("Missing").value_counts().head(top_n).reset_index()
    counts.columns = [col, "count"]
    figure, axis = _create_figure_axis()
    sns.barplot(data=counts, x=col, y="count", ax=axis, color="#00ff88")
    axis.tick_params(axis="x", rotation=45)
    _add_vertical_bar_labels(axis)
    axis.set_title(f"Count Plot: {col}")
    _finalize_plot(figure, show)
    return figure, axis


def pairplot(
    df: pd.DataFrame,
    hue: str | None = None,
    cols: list[str] | tuple[str, ...] | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Create a seaborn pair plot.

    Args:
        df: Input DataFrame.
        hue: Optional hue column.
        cols: Optional subset of columns.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axes array.
    """
    data = _validate_dataframe(df)
    selected = list(cols) if cols is not None else None
    if selected is not None:
        for column in selected:
            _validate_column(data, column)
    if hue is not None:
        _validate_column(data, hue)

    _apply_theme()
    pair_grid = sns.pairplot(data, vars=selected, hue=hue, corner=False)
    pair_grid.fig.patch.set_facecolor(get_theme(_ACTIVE_THEME)["figure.facecolor"])
    pair_grid.fig.tight_layout()
    if show:
        plt.show()
    return pair_grid.fig, np.asarray(pair_grid.axes)


def feature_importance(
    model_or_array: Any,
    feature_names: list[str] | tuple[str, ...],
    top_n: int = 20,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot feature importances from a model or array.

    Args:
        model_or_array: Fitted model with ``feature_importances_`` or ``coef_``, or a raw array.
        feature_names: Feature names aligned with the importance values.
        top_n: Maximum number of features to show.
        show: Whether to display the figure.

    Returns:
        A matplotlib figure and axis.
    """
    if hasattr(model_or_array, "feature_importances_"):
        importances = np.asarray(model_or_array.feature_importances_)
    elif hasattr(model_or_array, "coef_"):
        importances = np.asarray(model_or_array.coef_)
        importances = (
            np.abs(importances).mean(axis=0) if importances.ndim > 1 else np.abs(importances)
        )
    else:
        importances = np.asarray(model_or_array)

    summary = pd.DataFrame({"feature": list(feature_names), "importance": importances})
    summary = summary.sort_values("importance", ascending=False).head(top_n)

    figure, axis = _create_figure_axis(figsize=(8.0, 6.0))
    sns.barplot(data=summary, y="feature", x="importance", ax=axis, color="#00d4ff")
    _add_horizontal_bar_labels(axis)
    axis.set_title("Feature Importance")
    _finalize_plot(figure, show)
    return figure, axis


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: list[Any] | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create an annotated confusion matrix heatmap."""
    matrix = sk_confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = _create_figure_axis(figsize=(6.0, 5.0))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="mako", ax=axis, cbar=False)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    _finalize_plot(figure, show)
    return figure, axis


def roc_curve(
    y_true: Any,
    y_prob: Any,
    title: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create an ROC curve with AUC annotation."""
    fpr, tpr, _ = sk_roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    figure, axis = _create_figure_axis()
    axis.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color="#00d4ff")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#ff6b6b")
    axis.set_title(title or "ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()
    _finalize_plot(figure, show)
    return figure, axis


def precision_recall_curve(
    y_true: Any,
    y_prob: Any,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a precision-recall curve with average precision annotation."""
    precision, recall, _ = sk_precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    figure, axis = _create_figure_axis()
    axis.plot(recall, precision, color="#00ff88", label=f"AP = {ap_score:.3f}")
    axis.set_title("Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    _finalize_plot(figure, show)
    return figure, axis


def learning_curve(
    model: Any,
    X: Any,
    y: Any,
    cv: int = 5,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot train and validation learning curves for an estimator."""
    train_sizes, train_scores, valid_scores = sk_learning_curve(
        model,
        X,
        y,
        cv=cv,
        shuffle=True,
        random_state=42,
        train_sizes=np.linspace(0.2, 1.0, 5),
    )
    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)

    figure, axis = _create_figure_axis()
    axis.plot(train_sizes, train_mean, marker="o", label="Train", color="#00d4ff")
    axis.plot(train_sizes, valid_mean, marker="o", label="Validation", color="#00ff88")
    axis.set_title("Learning Curve")
    axis.set_xlabel("Training Examples")
    axis.set_ylabel("Score")
    axis.legend()
    _finalize_plot(figure, show)
    return figure, axis


def residual_plot(
    y_true: Any,
    y_pred: Any,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Create a residual plot for regression predictions."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    figure, axis = _create_figure_axis()
    axis.scatter(y_pred, residuals, color="#00d4ff")
    axis.axhline(0, linestyle="--", color="#ff6b6b")
    axis.set_title("Residual Plot")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Residual")
    _finalize_plot(figure, show)
    return figure, axis


def plotly_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    show: bool = True,
) -> tuple[Any, None]:
    """Create an interactive Plotly bar chart."""
    data = _validate_dataframe(df)
    _validate_column(data, x)
    _validate_column(data, y)
    go = _resolve_plotly()
    figure = go.Figure(
        data=[
            go.Bar(
                x=data[x],
                y=data[y],
                marker_color=_PLOTLY_COLORS[0],
            )
        ]
    )
    figure.update_layout(template="plotly_dark", title="Plotly Bar Chart")
    if show:
        figure.show()
    return figure, None


def plotly_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    show: bool = True,
) -> tuple[Any, None]:
    """Create an interactive Plotly scatter chart."""
    data = _validate_dataframe(df)
    _validate_column(data, x)
    _validate_column(data, y)
    if color is not None:
        _validate_column(data, color)
    go = _resolve_plotly()
    marker = {"size": 10, "color": data[color] if color else _PLOTLY_COLORS[0]}
    figure = go.Figure(data=[go.Scatter(x=data[x], y=data[y], mode="markers", marker=marker)])
    figure.update_layout(template="plotly_dark", title="Plotly Scatter Chart")
    if show:
        figure.show()
    return figure, None


def plotly_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    show: bool = True,
) -> tuple[Any, None]:
    """Create an interactive Plotly line chart."""
    data = _validate_dataframe(df)
    _validate_column(data, x)
    _validate_column(data, y)
    if color is not None:
        _validate_column(data, color)
    go = _resolve_plotly()
    line_color = _PLOTLY_COLORS[0]
    if color is not None:
        line_color = data[color]
    figure = go.Figure(
        data=[go.Scatter(x=data[x], y=data[y], mode="lines+markers", line={"color": line_color})]
    )
    figure.update_layout(template="plotly_dark", title="Plotly Line Chart")
    if show:
        figure.show()
    return figure, None


def subplot_grid(
    plot_funcs: list[Callable[[Axes], Any]],
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Render multiple axis-level plot functions in a shared grid."""
    if ncols <= 0:
        raise ValueError("ncols must be greater than 0")
    rows = int(np.ceil(len(plot_funcs) / ncols)) or 1
    theme = _apply_theme()
    figure, axes = plt.subplots(
        rows,
        ncols,
        figsize=figsize or (6 * ncols, 4 * rows),
        squeeze=False,
    )
    figure.patch.set_facecolor(theme["figure.facecolor"])
    for axis in axes.flat:
        axis.set_facecolor(theme["axes.facecolor"])

    for axis, func in zip(axes.flat, plot_funcs):
        func(axis)
    for axis in axes.flat[len(plot_funcs) :]:
        axis.set_visible(False)

    _finalize_plot(figure, show)
    return figure, axes


def save_plot(fig: Figure | None, path: str | Path, dpi: int = 300) -> Path:
    """Save a matplotlib figure to disk."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = fig or plt.gcf()
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    return target


__all__ = [
    "bar",
    "box",
    "confusion_matrix",
    "countplot",
    "feature_importance",
    "heatmap",
    "hist",
    "learning_curve",
    "line",
    "pairplot",
    "pie",
    "plotly_bar",
    "plotly_line",
    "plotly_scatter",
    "precision_recall_curve",
    "residual_plot",
    "roc_curve",
    "save_plot",
    "scatter",
    "set_theme",
    "subplot_grid",
]
