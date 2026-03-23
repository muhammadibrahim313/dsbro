"""Exploratory data analysis tools for dsbro."""

from __future__ import annotations

from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import chi2_contingency, ks_2samp

from dsbro._helpers import (
    _format_size,
    _print_dataframe,
    _print_divider,
    _print_header,
    _print_kv,
    _print_sub_header,
)
from dsbro._themes import apply_matplotlib_theme


def _validate_dataframe(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """Validate that an object is a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame for {name}, got {type(df).__name__}")
    return df.copy()


def _resolve_columns(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    *,
    exclude: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Resolve requested DataFrame columns with validation."""
    available = list(df.columns)
    excluded = set(exclude or [])
    if cols is None:
        return [column for column in available if column not in excluded]

    missing_columns = [column for column in cols if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Columns not found in DataFrame: {missing_text}")
    return [column for column in cols if column not in excluded]


def _numeric_columns(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    """Return numeric columns from a DataFrame."""
    selected = _resolve_columns(df, cols)
    return [column for column in selected if is_numeric_dtype(df[column])]


def _categorical_columns(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    """Return categorical-like columns from a DataFrame."""
    selected = _resolve_columns(df, cols)
    return [
        column
        for column in selected
        if not is_numeric_dtype(df[column]) or is_bool_dtype(df[column])
    ]


def _is_categorical(series: pd.Series) -> bool:
    """Return whether a Series should be treated as categorical."""
    return bool(not is_numeric_dtype(series) or is_bool_dtype(series))


def _prepare_plot(
    n_plots: int,
    *,
    ncols: int = 2,
    figsize_per_plot: tuple[float, float] = (6.0, 4.0),
) -> tuple[Figure, np.ndarray]:
    """Create a themed subplot grid."""
    theme = apply_matplotlib_theme("dark")
    sns.set_theme(style="darkgrid", rc=theme)
    rows = max(1, ceil(max(1, n_plots) / ncols))
    figure, axes = plt.subplots(
        rows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * rows),
        squeeze=False,
    )
    figure.patch.set_facecolor(theme["figure.facecolor"])
    for axis in axes.flat:
        axis.set_facecolor(theme["axes.facecolor"])
    return figure, axes


def _finalize_plot(figure: Figure, show: bool) -> None:
    """Finalize a matplotlib figure."""
    for axis in figure.axes:
        _style_axis(axis)
    plt.tight_layout()
    if show:
        plt.show()


def _hide_unused_axes(axes: np.ndarray, used: int) -> None:
    """Hide unused subplot axes."""
    for axis in axes.flat[used:]:
        axis.set_visible(False)


def _add_bar_labels(axis: Axes) -> None:
    """Add value labels to bar patches."""
    for patch in axis.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        axis.annotate(
            f"{height:.0f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#e0e0e0",
            xytext=(0, 3),
            textcoords="offset points",
            clip_on=False,
        )
    axis.margins(y=0.15)


def _style_axis(axis: Axes) -> None:
    """Apply consistent dsbro axis styling."""
    title = axis.get_title()
    xlabel = axis.get_xlabel()
    ylabel = axis.get_ylabel()
    if title:
        axis.set_title(title, fontsize=14, fontweight="bold", color="#e0e0e0")
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=11, color="#cccccc")
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=11, color="#cccccc")
    axis.tick_params(axis="both", colors="#cccccc", labelsize=10)

    legend = axis.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("#1a1a2e")
        legend.get_frame().set_edgecolor("#444444")
        legend.get_title().set_color("#e0e0e0")
        for text in legend.get_texts():
            text.set_color("#e0e0e0")

    for spine in axis.spines.values():
        spine.set_color("#555555")


def _series_to_categories(series: pd.Series) -> pd.Series:
    """Convert a Series into string categories with missing values filled."""
    return series.astype("object").fillna("Missing").astype(str)


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Compute Cramer's V for two categorical Series."""
    observed = pd.crosstab(_series_to_categories(left), _series_to_categories(right))
    if observed.empty or observed.shape[0] < 2 or observed.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(observed)
    total = observed.to_numpy().sum()
    if total == 0:
        return 0.0
    phi2 = chi2 / total
    rows, cols = observed.shape
    correction = ((cols - 1) * (rows - 1)) / max(total - 1, 1)
    phi2_corrected = max(0.0, phi2 - correction)
    rows_corrected = rows - ((rows - 1) ** 2) / max(total - 1, 1)
    cols_corrected = cols - ((cols - 1) ** 2) / max(total - 1, 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2_corrected / denominator))


def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Compute the correlation ratio for categorical-numeric pairs."""
    valid = pd.DataFrame({"category": categories, "value": measurements}).dropna()
    if valid.empty:
        return 0.0
    grand_mean = valid["value"].mean()
    grouped = valid.groupby("category")["value"]
    numerator = sum(len(group) * (group.mean() - grand_mean) ** 2 for _, group in grouped)
    denominator = float(((valid["value"] - grand_mean) ** 2).sum())
    if denominator == 0:
        return 0.0
    return float(np.sqrt(numerator / denominator))


def _association_score(left: pd.Series, right: pd.Series) -> float:
    """Compute an association score for two Series using a smart default."""
    if left.name == right.name:
        return 1.0

    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if pair.empty:
        return np.nan

    left_values = pair["left"]
    right_values = pair["right"]

    if is_numeric_dtype(left_values) and is_numeric_dtype(right_values):
        if left_values.nunique() <= 1 or right_values.nunique() <= 1:
            return 0.0
        return float(left_values.corr(right_values))

    if _is_categorical(left_values) and _is_categorical(right_values):
        return _cramers_v(left_values, right_values)

    if _is_categorical(left_values):
        return _correlation_ratio(_series_to_categories(left_values), pd.to_numeric(right_values))

    return _correlation_ratio(_series_to_categories(right_values), pd.to_numeric(left_values))


def _population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute PSI for numeric or categorical distributions."""
    valid = pd.DataFrame({"expected": expected, "actual": actual})
    left = valid["expected"]
    right = valid["actual"]

    if is_numeric_dtype(left) and is_numeric_dtype(right):
        combined = pd.concat([left, right]).dropna()
        if combined.nunique() <= 1:
            return 0.0
        try:
            edges = np.unique(np.quantile(combined, np.linspace(0, 1, bins + 1)))
        except ValueError:
            edges = np.array([combined.min(), combined.max()])
        if len(edges) < 2:
            return 0.0
        left_counts = (
            pd.cut(left, bins=edges, include_lowest=True, duplicates="drop")
            .value_counts(normalize=True, sort=False)
            .to_numpy()
        )
        right_counts = (
            pd.cut(right, bins=edges, include_lowest=True, duplicates="drop")
            .value_counts(normalize=True, sort=False)
            .to_numpy()
        )
    else:
        categories = sorted(set(_series_to_categories(left)) | set(_series_to_categories(right)))
        left_counts = (
            _series_to_categories(left)
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0)
            .to_numpy()
        )
        right_counts = (
            _series_to_categories(right)
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0)
            .to_numpy()
        )

    epsilon = 1e-6
    left_safe = np.clip(left_counts, epsilon, None)
    right_safe = np.clip(right_counts, epsilon, None)
    psi = np.sum((right_safe - left_safe) * np.log(right_safe / left_safe))
    return float(psi)


def overview(df: pd.DataFrame, sample_size: int = 5) -> dict[str, Any]:
    """Summarize the overall structure of a DataFrame.

    Args:
        df: Input DataFrame.
        sample_size: Number of sample rows to include in the output.

    Returns:
        A dictionary containing shape, memory, duplicates, and per-column summary data.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import overview
        >>> result = overview(pd.DataFrame({"x": [1, 2, 3]}))
        >>> result["shape"]
        (3, 1)
    """
    data = _validate_dataframe(df)
    if not isinstance(sample_size, int):
        raise TypeError(f"Expected int for sample_size, got {type(sample_size).__name__}")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")

    missing_counts = data.isna().sum()
    column_summary = pd.DataFrame(
        {
            "dtype": data.dtypes.astype(str),
            "non_null": data.notna().sum(),
            "missing": missing_counts,
            "missing_pct": (missing_counts / len(data) * 100).round(2) if len(data) else 0.0,
            "unique": data.nunique(dropna=True),
        }
    )

    duplicate_count = int(data.duplicated().sum())
    result = {
        "shape": data.shape,
        "rows": len(data),
        "columns": len(data.columns),
        "memory_bytes": int(data.memory_usage(deep=True).sum()),
        "memory_mb": float(data.memory_usage(deep=True).sum() / (1024**2)),
        "duplicate_count": duplicate_count,
        "duplicate_pct": float((duplicate_count / len(data) * 100) if len(data) else 0.0),
        "column_summary": column_summary,
        "sample": data.head(sample_size),
    }
    _print_header("Dataset Overview")
    _print_kv("Rows", result["rows"])
    _print_kv("Columns", result["columns"])
    _print_kv("Memory", _format_size(result["memory_bytes"]))
    _print_kv(
        "Duplicates",
        f"{result['duplicate_count']} ({result['duplicate_pct']:.1f}%)",
    )
    _print_divider()
    _print_sub_header("Column Summary")
    _print_dataframe(result["column_summary"])
    _print_divider()
    return result


def describe_plus(df: pd.DataFrame) -> pd.DataFrame:
    """Return an extended descriptive summary for all DataFrame columns.

    Args:
        df: Input DataFrame.

    Returns:
        A DataFrame with descriptive statistics plus missingness and cardinality metrics.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import describe_plus
        >>> describe_plus(pd.DataFrame({"x": [1, 2, 3]})).loc["x", "mean"]
        2.0
    """
    data = _validate_dataframe(df)
    numeric_cols = _numeric_columns(data)
    summary = pd.DataFrame(index=data.columns)
    summary["dtype"] = data.dtypes.astype(str)
    summary["count"] = data.count()
    summary["missing"] = data.isna().sum()
    summary["missing_pct"] = (summary["missing"] / len(data) * 100).round(2) if len(data) else 0.0
    summary["unique"] = data.nunique(dropna=True)

    zeros_pct: list[float] = []
    for column in data.columns:
        if column in numeric_cols and len(data):
            zeros_pct.append(float((data[column] == 0).mean() * 100))
        else:
            zeros_pct.append(np.nan)
    summary["zeros_pct"] = zeros_pct

    if numeric_cols:
        describe = data[numeric_cols].describe().T
        summary.loc[numeric_cols, describe.columns] = describe
        summary.loc[numeric_cols, "median"] = data[numeric_cols].median()
        summary.loc[numeric_cols, "skew"] = data[numeric_cols].skew(numeric_only=True)
        summary.loc[numeric_cols, "kurtosis"] = data[numeric_cols].kurt(numeric_only=True)

    _print_header("Describe Plus")
    _print_dataframe(summary)
    return summary


def missing(
    df: pd.DataFrame,
    plot: bool = True,
    show: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, Figure, Axes]:
    """Analyze missing values in a DataFrame.

    Args:
        df: Input DataFrame.
        plot: Whether to create a missing-value bar chart.
        show: Whether to display the plot when ``plot`` is ``True``.

    Returns:
        A missing-value summary DataFrame, optionally followed by a figure and axis.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import missing
        >>> summary = missing(pd.DataFrame({"x": [1, None]}), plot=False)
        >>> summary.loc["x", "missing"]
        1
    """
    data = _validate_dataframe(df)
    summary = pd.DataFrame(
        {
            "missing": data.isna().sum(),
            "missing_pct": (data.isna().mean() * 100).round(2),
        }
    ).sort_values(["missing", "missing_pct"], ascending=False)

    _print_header("Missing Values")
    _print_dataframe(summary)

    if not plot:
        return summary

    figure, axes = _prepare_plot(1, ncols=1)
    axis = axes.flat[0]
    missing_only = summary[summary["missing"] > 0]

    if missing_only.empty:
        axis.text(
            0.5,
            0.5,
            "No missing values detected",
            ha="center",
            va="center",
            fontsize=14,
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    else:
        sns.barplot(
            x=missing_only.index,
            y=missing_only["missing"],
            ax=axis,
            color="#00d4ff",
        )
        _add_bar_labels(axis)
        axis.tick_params(axis="x", rotation=45)
        axis.set_xlabel("Column")
        axis.set_ylabel("Missing Values")
    axis.set_title("Missing Value Summary")
    _finalize_plot(figure, show)
    return summary, figure, axis


def distribution(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    max_cols: int = 6,
    bins: int = 30,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot feature distributions for numeric and categorical columns.

    Args:
        df: Input DataFrame.
        cols: Optional subset of columns to plot.
        max_cols: Maximum number of columns to visualize.
        bins: Number of bins used for numeric histograms.
        show: Whether to display the generated figure.

    Returns:
        A matplotlib figure and axes array.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import distribution
        >>> fig, axes = distribution(pd.DataFrame({"x": [1, 2, 3]}), show=False)
        >>> fig is not None
        True
    """
    data = _validate_dataframe(df)
    if not isinstance(max_cols, int):
        raise TypeError(f"Expected int for max_cols, got {type(max_cols).__name__}")
    if max_cols <= 0:
        raise ValueError("max_cols must be greater than 0")

    selected = _resolve_columns(data, list(cols) if cols is not None else None)[:max_cols]
    if not selected:
        raise ValueError("No columns available for distribution plotting")

    figure, axes = _prepare_plot(len(selected))
    for index, column in enumerate(selected):
        axis = axes.flat[index]
        series = data[column]
        if is_numeric_dtype(series):
            sns.histplot(
                series.dropna(),
                bins=bins,
                kde=True,
                ax=axis,
                color="#00d4ff",
                line_kws={"linewidth": 2},
            )
            axis.set_ylabel("Count")
        else:
            counts = _series_to_categories(series).value_counts().head(15)
            sns.barplot(x=counts.index, y=counts.values, ax=axis, color="#00ff88")
            _add_bar_labels(axis)
            axis.tick_params(axis="x", rotation=45)
            axis.set_ylabel("Count")
        axis.set_title(f"Distribution: {column}")
        axis.set_xlabel(column)

    _hide_unused_axes(axes, len(selected))
    figure.subplots_adjust(hspace=0.4, wspace=0.3)
    _finalize_plot(figure, show)
    return figure, axes


def correlate(
    df: pd.DataFrame,
    method: str = "auto",
    cols: list[str] | tuple[str, ...] | None = None,
    plot: bool = True,
    show: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, Figure, Axes]:
    """Compute feature associations with smart defaults.

    Args:
        df: Input DataFrame.
        method: Correlation method. Use ``auto`` for mixed-type associations.
        cols: Optional subset of columns.
        plot: Whether to create a heatmap.
        show: Whether to display the plot when ``plot`` is ``True``.

    Returns:
        A correlation matrix, optionally followed by a heatmap figure and axis.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import correlate
        >>> corr = correlate(pd.DataFrame({"x": [1, 2], "y": [2, 3]}), plot=False)
        >>> corr.loc["x", "y"] > 0
        True
    """
    data = _validate_dataframe(df)
    selected = _resolve_columns(data, list(cols) if cols is not None else None)
    if not selected:
        raise ValueError("No columns available for correlation analysis")

    normalized_method = method.lower()
    valid_methods = {"auto", "pearson", "spearman", "kendall"}
    if normalized_method not in valid_methods:
        valid_text = ", ".join(sorted(valid_methods))
        raise ValueError(f"Unknown method '{method}'. Available methods: {valid_text}")

    if normalized_method == "auto":
        matrix = pd.DataFrame(index=selected, columns=selected, dtype=float)
        for left in selected:
            for right in selected:
                matrix.loc[left, right] = _association_score(data[left], data[right])
        matrix = matrix.astype(float)
    else:
        numeric_cols = _numeric_columns(data, selected)
        if not numeric_cols:
            raise ValueError("Correlation methods other than 'auto' require numeric columns")
        matrix = data[numeric_cols].corr(method=normalized_method)

    if not plot:
        return matrix

    figsize = (12.0, 10.0) if len(matrix.columns) >= 6 else (7.0, 5.0)
    figure, axes = _prepare_plot(1, ncols=1, figsize_per_plot=figsize)
    axis = axes.flat[0]
    sns.heatmap(
        matrix,
        annot=True,
        annot_kws={"size": 7},
        cmap="mako",
        vmin=-1,
        vmax=1,
        ax=axis,
    )
    axis.set_title("Feature Correlation")
    _finalize_plot(figure, show)
    return matrix, figure, axis


def outliers(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    method: str = "iqr",
    factor: float = 1.5,
    plot: bool = True,
    show: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, Figure, Axes]:
    """Detect outliers in numeric columns.

    Args:
        df: Input DataFrame.
        cols: Optional subset of columns.
        method: Detection method: ``iqr`` or ``zscore``.
        factor: Threshold multiplier used by the selected method.
        plot: Whether to create a boxplot summary.
        show: Whether to display the figure when ``plot`` is ``True``.

    Returns:
        An outlier summary DataFrame, optionally followed by a figure and axis.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import outliers
        >>> summary = outliers(pd.DataFrame({"x": [1, 2, 100]}), plot=False)
        >>> int(summary.loc[0, "outlier_count"]) >= 1
        True
    """
    data = _validate_dataframe(df)
    selected = _numeric_columns(data, list(cols) if cols is not None else None)
    if not selected:
        raise ValueError("Outlier detection requires at least one numeric column")

    normalized_method = method.lower()
    if normalized_method not in {"iqr", "zscore"}:
        raise ValueError("method must be 'iqr' or 'zscore'")

    results: list[dict[str, Any]] = []
    for column in selected:
        series = data[column].dropna()
        if series.empty:
            results.append(
                {
                    "column": column,
                    "outlier_count": 0,
                    "outlier_pct": 0.0,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                }
            )
            continue

        if normalized_method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            mask = (series < lower_bound) | (series > upper_bound)
        else:
            mean_value = series.mean()
            std_value = series.std(ddof=0)
            threshold = factor * std_value if std_value else 0.0
            lower_bound = mean_value - threshold
            upper_bound = mean_value + threshold
            mask = (series < lower_bound) | (series > upper_bound)

        results.append(
            {
                "column": column,
                "outlier_count": int(mask.sum()),
                "outlier_pct": float(mask.mean() * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
        )

    summary = (
        pd.DataFrame(results)
        .sort_values("outlier_pct", ascending=False)
        .reset_index(drop=True)
    )
    if not plot:
        return summary

    figure, axes = _prepare_plot(1, ncols=1, figsize_per_plot=(12.0, 6.0))
    axis = axes.flat[0]
    melted = data[selected].melt(var_name="column", value_name="value").dropna()
    sns.boxplot(data=melted, x="value", y="column", ax=axis, color="#00d4ff")
    axis.set_title(f"Outlier Check ({normalized_method.upper()})")
    axis.set_xlabel("Value")
    axis.set_ylabel("Column")
    _finalize_plot(figure, show)
    return summary, figure, axis


def compare(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    names: tuple[str, str] | list[str] | None = None,
    cols: list[str] | tuple[str, ...] | None = None,
    max_cols: int = 4,
    plot: bool = True,
    show: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, Figure, np.ndarray]:
    """Compare two DataFrames column by column.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        names: Optional display names for the two datasets.
        cols: Optional subset of shared columns.
        max_cols: Maximum number of columns to visualize.
        plot: Whether to create comparison plots.
        show: Whether to display the figure when ``plot`` is ``True``.

    Returns:
        A comparison summary DataFrame, optionally followed by a figure and axes.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import compare
        >>> left = pd.DataFrame({"x": [1, 2, 3]})
        >>> right = pd.DataFrame({"x": [2, 3, 4]})
        >>> summary = compare(left, right, plot=False)
        >>> "x" in summary["column"].tolist()
        True
    """
    left = _validate_dataframe(df1, "df1")
    right = _validate_dataframe(df2, "df2")

    shared_columns = sorted(set(left.columns) & set(right.columns))
    if cols is not None:
        requested = list(cols)
        missing_columns = [column for column in requested if column not in shared_columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Columns are not shared between both DataFrames: {missing_text}")
        shared_columns = requested

    if not shared_columns:
        raise ValueError("No shared columns available for comparison")

    label_left, label_right = tuple(names) if names is not None else ("df1", "df2")

    rows: list[dict[str, Any]] = []
    for column in shared_columns:
        row: dict[str, Any] = {
            "column": column,
            f"dtype_{label_left}": str(left[column].dtype),
            f"dtype_{label_right}": str(right[column].dtype),
            f"missing_pct_{label_left}": float(left[column].isna().mean() * 100),
            f"missing_pct_{label_right}": float(right[column].isna().mean() * 100),
            f"unique_{label_left}": int(left[column].nunique(dropna=True)),
            f"unique_{label_right}": int(right[column].nunique(dropna=True)),
        }
        if is_numeric_dtype(left[column]) and is_numeric_dtype(right[column]):
            row[f"mean_{label_left}"] = float(left[column].mean())
            row[f"mean_{label_right}"] = float(right[column].mean())
            row["mean_delta"] = float(row[f"mean_{label_right}"] - row[f"mean_{label_left}"])
        else:
            row[f"top_{label_left}"] = _series_to_categories(left[column]).mode().iat[0]
            row[f"top_{label_right}"] = _series_to_categories(right[column]).mode().iat[0]
            row["mean_delta"] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not plot:
        return summary

    display_columns = shared_columns[:max_cols]
    figure, axes = _prepare_plot(len(display_columns))
    for index, column in enumerate(display_columns):
        axis = axes.flat[index]
        if is_numeric_dtype(left[column]) and is_numeric_dtype(right[column]):
            sns.kdeplot(left[column].dropna(), ax=axis, label=label_left, fill=True)
            sns.kdeplot(right[column].dropna(), ax=axis, label=label_right, fill=True)
            axis.set_ylabel("Density")
        else:
            left_counts = _series_to_categories(left[column]).value_counts(normalize=True)
            right_counts = _series_to_categories(right[column]).value_counts(normalize=True)
            compare_frame = pd.DataFrame(
                {label_left: left_counts, label_right: right_counts}
            ).fillna(0)
            compare_frame.head(10).plot(kind="bar", ax=axis)
            _add_bar_labels(axis)
            axis.tick_params(axis="x", rotation=45)
            axis.set_ylabel("Share")
        axis.set_title(f"Compare: {column}")
        axis.legend()

    _hide_unused_axes(axes, len(display_columns))
    _finalize_plot(figure, show)
    return summary, figure, axes


def target_analysis(
    df: pd.DataFrame,
    target: str,
    cols: list[str] | tuple[str, ...] | None = None,
    max_cols: int = 6,
    plot: bool = True,
    show: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, Figure, np.ndarray]:
    """Analyze feature relationships against a target column.

    Args:
        df: Input DataFrame.
        target: Target column name.
        cols: Optional feature subset. The target column is excluded automatically.
        max_cols: Maximum number of features to visualize.
        plot: Whether to create plots.
        show: Whether to display the generated figure.

    Returns:
        A feature-vs-target summary DataFrame, optionally followed by a figure and axes.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import target_analysis
        >>> frame = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        >>> summary = target_analysis(frame, target="y", plot=False)
        >>> "x" in summary["feature"].tolist()
        True
    """
    data = _validate_dataframe(df)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    features = _resolve_columns(data, list(cols) if cols is not None else None, exclude=[target])
    if not features:
        raise ValueError("No feature columns available for target analysis")

    target_series = data[target]
    target_type = "categorical" if _is_categorical(target_series) else "numeric"
    rows: list[dict[str, Any]] = []
    for feature in features:
        feature_series = data[feature]
        feature_type = "categorical" if _is_categorical(feature_series) else "numeric"
        score = _association_score(feature_series, target_series)
        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "target_type": target_type,
                "missing_pct": float(feature_series.isna().mean() * 100),
                "association": float(score) if pd.notna(score) else np.nan,
            }
        )

    summary = pd.DataFrame(rows).sort_values("association", ascending=False, na_position="last")
    if not plot:
        return summary.reset_index(drop=True)

    display_features = summary["feature"].head(max_cols).tolist()
    figure, axes = _prepare_plot(len(display_features))
    for index, feature in enumerate(display_features):
        axis = axes.flat[index]
        feature_series = data[feature]
        feature_type = "categorical" if _is_categorical(feature_series) else "numeric"

        if feature_type == "numeric" and target_type == "numeric":
            sns.scatterplot(data=data, x=feature, y=target, ax=axis, color="#00d4ff")
        elif feature_type == "numeric" and target_type == "categorical":
            sns.boxplot(data=data, x=target, y=feature, ax=axis, color="#00d4ff")
        elif feature_type == "categorical" and target_type == "numeric":
            grouped = (
                data.groupby(feature, dropna=False)[target]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            grouped[feature] = _series_to_categories(grouped[feature])
            sns.barplot(data=grouped, x=feature, y=target, ax=axis, color="#00ff88")
            _add_bar_labels(axis)
            axis.tick_params(axis="x", rotation=45)
        else:
            crosstab = pd.crosstab(
                _series_to_categories(feature_series),
                _series_to_categories(target_series),
                normalize="index",
            ).head(10)
            crosstab.plot(kind="bar", stacked=True, ax=axis)
            axis.tick_params(axis="x", rotation=45)
            axis.set_ylabel("Share")
        axis.set_title(f"{feature} vs {target}")

    _hide_unused_axes(axes, len(display_features))
    _finalize_plot(figure, show)
    return summary.reset_index(drop=True), figure, axes


def cardinality(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Summarize column cardinality and flag high-cardinality fields.

    Args:
        df: Input DataFrame.
        threshold: Ratio of unique values to rows used to flag high cardinality.

    Returns:
        A cardinality summary DataFrame.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import cardinality
        >>> summary = cardinality(pd.DataFrame({"x": ["a", "b", "c"]}))
        >>> bool(summary.loc[0, "high_cardinality"])
        True
    """
    data = _validate_dataframe(df)
    if threshold < 0:
        raise ValueError("threshold must be greater than or equal to 0")

    row_count = max(len(data), 1)
    summary = pd.DataFrame(
        {
            "column": data.columns,
            "unique_count": [int(data[column].nunique(dropna=True)) for column in data.columns],
            "unique_pct": [
                float(data[column].nunique(dropna=True) / row_count) for column in data.columns
            ],
        }
    )
    summary["high_cardinality"] = summary["unique_pct"] >= threshold
    summary = summary.sort_values("unique_count", ascending=False).reset_index(drop=True)
    _print_header("Cardinality")
    _print_dataframe(summary)
    return summary


def duplicates(
    df: pd.DataFrame,
    subset: list[str] | tuple[str, ...] | None = None,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Summarize duplicate rows in a DataFrame.

    Args:
        df: Input DataFrame.
        subset: Optional columns used to determine duplicates.
        sample_size: Number of duplicate examples to include.

    Returns:
        A dictionary containing duplicate counts, percentages, and sample rows.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import duplicates
        >>> result = duplicates(pd.DataFrame({"x": [1, 1, 2]}))
        >>> result["duplicate_count"]
        1
    """
    data = _validate_dataframe(df)
    if not isinstance(sample_size, int):
        raise TypeError(f"Expected int for sample_size, got {type(sample_size).__name__}")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")

    selected_subset = list(subset) if subset is not None else None
    if selected_subset is not None:
        _resolve_columns(data, selected_subset)

    mask = data.duplicated(subset=selected_subset, keep="first")
    duplicate_rows = data.loc[mask].head(sample_size)
    duplicate_count = int(mask.sum())
    result = {
        "duplicate_count": duplicate_count,
        "duplicate_pct": float((duplicate_count / len(data) * 100) if len(data) else 0.0),
        "subset": selected_subset,
        "examples": duplicate_rows,
    }
    _print_header("Duplicates")
    _print_kv("Count", result["duplicate_count"])
    _print_kv("Percentage", f"{result['duplicate_pct']:.1f}%")
    _print_divider()
    return result


def value_counts_plot(
    df: pd.DataFrame,
    col: str,
    top_n: int = 20,
    normalize: bool = False,
    show: bool = True,
) -> tuple[pd.DataFrame, Figure, Axes]:
    """Plot value counts for a categorical column.

    Args:
        df: Input DataFrame.
        col: Column to summarize.
        top_n: Maximum number of categories to display.
        normalize: Whether to show proportions instead of raw counts.
        show: Whether to display the figure.

    Returns:
        A value-count summary DataFrame plus the generated figure and axis.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import value_counts_plot
        >>> frame = pd.DataFrame({"x": ["a", "a", "b"]})
        >>> counts, fig, ax = value_counts_plot(frame, "x", show=False)
        >>> counts.iloc[0]["count"]
        2
    """
    data = _validate_dataframe(df)
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    counts = (
        _series_to_categories(data[col])
        .value_counts(normalize=normalize)
        .head(top_n)
        .rename("count")
        .reset_index()
        .rename(columns={"index": col})
    )

    figure, axes = _prepare_plot(1, ncols=1)
    axis = axes.flat[0]
    sns.barplot(data=counts, x=col, y="count", ax=axis, color="#00ff88")
    _add_bar_labels(axis)
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("Share" if normalize else "Count")
    axis.set_title(f"Value Counts: {col}")
    _finalize_plot(figure, show)
    return counts, figure, axis


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        A DataFrame with numeric summary statistics.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import numeric_summary
        >>> summary = numeric_summary(pd.DataFrame({"x": [1, 2, 3]}))
        >>> summary.loc["x", "mean"]
        2.0
    """
    data = _validate_dataframe(df)
    numeric_cols = _numeric_columns(data)
    if not numeric_cols:
        return pd.DataFrame()

    summary = pd.DataFrame(index=numeric_cols)
    summary["min"] = data[numeric_cols].min()
    summary["max"] = data[numeric_cols].max()
    summary["mean"] = data[numeric_cols].mean()
    summary["median"] = data[numeric_cols].median()
    summary["std"] = data[numeric_cols].std()
    summary["skew"] = data[numeric_cols].skew(numeric_only=True)
    summary["kurtosis"] = data[numeric_cols].kurt(numeric_only=True)
    summary["missing_pct"] = (data[numeric_cols].isna().mean() * 100).round(2)
    _print_header("Numeric Summary")
    _print_dataframe(summary)
    return summary


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for categorical columns.

    Args:
        df: Input DataFrame.

    Returns:
        A DataFrame with categorical summary statistics.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import categorical_summary
        >>> summary = categorical_summary(pd.DataFrame({"x": ["a", "b", "a"]}))
        >>> summary.loc["x", "unique"]
        2
    """
    data = _validate_dataframe(df)
    categorical_cols = _categorical_columns(data)
    if not categorical_cols:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for column in categorical_cols:
        value_counts = _series_to_categories(data[column]).value_counts(dropna=False)
        rows.append(
            {
                "column": column,
                "unique": int(data[column].nunique(dropna=True)),
                "top": value_counts.index[0] if not value_counts.empty else np.nan,
                "freq": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "missing": int(data[column].isna().sum()),
                "missing_pct": float(data[column].isna().mean() * 100),
            }
        )
    summary = pd.DataFrame(rows).set_index("column")
    _print_header("Categorical Summary")
    _print_dataframe(summary)
    return summary


def drift(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Estimate dataset drift between two DataFrames.

    Args:
        df1: Reference DataFrame.
        df2: Comparison DataFrame.
        cols: Optional subset of shared columns.
        bins: Number of bins used for PSI on numeric columns.

    Returns:
        A DataFrame with PSI and KS statistics per shared feature.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import drift
        >>> left = pd.DataFrame({"x": [1, 2, 3]})
        >>> right = pd.DataFrame({"x": [3, 4, 5]})
        >>> "psi" in drift(left, right).columns
        True
    """
    left = _validate_dataframe(df1, "df1")
    right = _validate_dataframe(df2, "df2")
    shared_columns = sorted(set(left.columns) & set(right.columns))
    if cols is not None:
        requested = list(cols)
        missing_columns = [column for column in requested if column not in shared_columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Columns are not shared between both DataFrames: {missing_text}")
        shared_columns = requested

    if not shared_columns:
        raise ValueError("No shared columns available for drift analysis")

    rows: list[dict[str, Any]] = []
    for column in shared_columns:
        left_series = left[column]
        right_series = right[column]
        psi = _population_stability_index(left_series, right_series, bins=bins)

        ks_stat = np.nan
        ks_pvalue = np.nan
        if is_numeric_dtype(left_series) and is_numeric_dtype(right_series):
            valid_left = left_series.dropna()
            valid_right = right_series.dropna()
            if not valid_left.empty and not valid_right.empty:
                ks_result = ks_2samp(valid_left, valid_right)
                ks_stat = float(ks_result.statistic)
                ks_pvalue = float(ks_result.pvalue)

        rows.append(
            {
                "column": column,
                "psi": psi,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue,
                "drift_flag": bool(psi >= 0.2 or (pd.notna(ks_pvalue) and ks_pvalue < 0.05)),
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def profile(
    df: pd.DataFrame,
    target: str | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Build a compact EDA profile bundle for a DataFrame.

    Args:
        df: Input DataFrame.
        target: Optional target column for supervised analysis.
        show: Whether to display generated figures.

    Returns:
        A dictionary containing summary tables and generated figures.

    Example:
        >>> import pandas as pd
        >>> from dsbro.eda import profile
        >>> result = profile(pd.DataFrame({"x": [1, 2, 3]}), show=False)
        >>> "overview" in result
        True
    """
    data = _validate_dataframe(df)
    overview_result = overview(data)
    describe_result = describe_plus(data)

    missing_result: pd.DataFrame | tuple[pd.DataFrame, Figure, Axes]
    if data.isna().any().any():
        missing_result = missing(data, plot=True, show=show)
        missing_summary = missing_result[0]
        missing_plot = missing_result
    else:
        missing_summary = missing(data, plot=False)
        missing_plot = None

    cardinality_result = cardinality(data)
    numeric_result = numeric_summary(data)
    categorical_result = categorical_summary(data)

    results: dict[str, Any] = {
        "overview": overview_result,
        "describe_plus": describe_result,
        "missing": missing_summary,
        "cardinality": cardinality_result,
        "numeric_summary": numeric_result,
        "categorical_summary": categorical_result,
    }

    if len(data.columns) > 1:
        results["correlation"] = correlate(data, plot=False)
        results["distribution_plot"] = distribution(data, show=show)
    else:
        results["correlation"] = pd.DataFrame()
        results["distribution_plot"] = None

    if target is not None:
        results["target_analysis"] = target_analysis(data, target=target, show=show)
    else:
        results["target_analysis"] = None

    results["missing_plot"] = missing_plot

    return results


__all__ = [
    "cardinality",
    "categorical_summary",
    "compare",
    "correlate",
    "describe_plus",
    "distribution",
    "drift",
    "duplicates",
    "missing",
    "numeric_summary",
    "outliers",
    "overview",
    "profile",
    "target_analysis",
    "value_counts_plot",
]
