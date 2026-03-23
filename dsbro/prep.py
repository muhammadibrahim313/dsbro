"""Preprocessing and feature engineering tools for dsbro."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OrdinalEncoder,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


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
    """Return categorical columns from a DataFrame."""
    selected = _resolve_columns(df, cols)
    return [column for column in selected if not is_numeric_dtype(df[column])]


def _mode_value(series: pd.Series) -> Any:
    """Return the most common non-null value in a Series."""
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return 0 if is_numeric_dtype(series) else "Missing"


def _is_classification_target(series: pd.Series) -> bool:
    """Infer whether a target Series represents a classification problem."""
    if not is_numeric_dtype(series):
        return True
    unique_count = series.nunique(dropna=True)
    return bool(unique_count <= 20 and pd.api.types.is_integer_dtype(series))


def encode(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    method: str = "label",
    target: str | None = None,
    drop_first: bool = False,
) -> pd.DataFrame:
    """Encode categorical columns with a one-line API.

    Args:
        df: Input DataFrame.
        cols: Columns to encode. Defaults to categorical columns.
        method: Encoding method: ``label``, ``ordinal``, ``onehot``, ``target``,
            ``frequency``, or ``binary``.
        target: Target column required when using ``target`` encoding.
        drop_first: Whether to drop the first level when using one-hot encoding.

    Returns:
        A transformed DataFrame copy.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import encode
        >>> frame = pd.DataFrame({"city": ["a", "b", "a"]})
        >>> encode(frame, method="label")["city"].dtype.kind in {"i", "u"}
        True
    """
    data = _validate_dataframe(df)
    selected = _categorical_columns(data, list(cols) if cols is not None else None)
    if not selected:
        return data

    normalized_method = method.lower()
    if normalized_method == "frequency":
        return frequency_encode(data, selected)
    if normalized_method == "target":
        if target is None:
            raise ValueError("target is required when method='target'")
        return target_encode(data, selected, target=target)
    if normalized_method == "onehot":
        working = data.copy()
        for column in selected:
            working[column] = working[column].astype("object").fillna("Missing")
        return pd.get_dummies(working, columns=selected, drop_first=drop_first)

    transformed = data.copy()
    if normalized_method == "label":
        for column in selected:
            filled = transformed[column].astype("object").fillna("Missing")
            codes, _ = pd.factorize(filled, sort=True)
            transformed[column] = codes.astype("int64")
        return transformed

    if normalized_method == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        filled = transformed[selected].astype("object").fillna("Missing")
        transformed[selected] = encoder.fit_transform(filled).astype("int64")
        return transformed

    if normalized_method == "binary":
        for column in selected:
            filled = transformed[column].astype("object").fillna("Missing")
            unique_values = sorted(filled.unique())
            if len(unique_values) != 2:
                raise ValueError(f"Column '{column}' is not binary and cannot use binary encoding")
            mapping = {value: index for index, value in enumerate(unique_values)}
            transformed[column] = filled.map(mapping).astype("int64")
        return transformed

    raise ValueError(f"Unknown encoding method: {method}")


def scale(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    method: str = "standard",
) -> pd.DataFrame:
    """Scale numeric columns with smart defaults.

    Args:
        df: Input DataFrame.
        cols: Columns to scale. Defaults to numeric columns.
        method: Scaling method: ``standard``, ``minmax``, ``robust``, ``quantile``,
            or ``log1p``.

    Returns:
        A scaled DataFrame copy.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import scale
        >>> frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> round(scale(frame)["x"].mean(), 6)
        0.0
    """
    data = _validate_dataframe(df)
    selected = _numeric_columns(data, list(cols) if cols is not None else None)
    if not selected:
        return data

    normalized_method = method.lower()
    if normalized_method == "log1p":
        return log_transform(data, selected)

    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "quantile": QuantileTransformer(
            n_quantiles=min(max(len(data), 2), 1000),
            output_distribution="normal",
            random_state=42,
        ),
    }
    if normalized_method not in scaler_map:
        raise ValueError(f"Unknown scaling method: {method}")

    transformed = data.copy()
    scaler = scaler_map[normalized_method]
    transformed[selected] = scaler.fit_transform(transformed[selected])
    return transformed


def fill_missing(
    df: pd.DataFrame,
    strategy: str = "smart",
    cols: list[str] | tuple[str, ...] | None = None,
    fill_value: Any | None = None,
) -> pd.DataFrame:
    """Fill missing values without mutating the input DataFrame.

    Args:
        df: Input DataFrame.
        strategy: Fill strategy: ``smart``, ``mean``, ``median``, ``mode``,
            ``constant``, ``knn``, or ``iterative``.
        cols: Optional subset of columns to fill.
        fill_value: Constant value used when ``strategy='constant'``.

    Returns:
        A DataFrame copy with missing values filled.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import fill_missing
        >>> frame = pd.DataFrame({"x": [1.0, None, 3.0]})
        >>> fill_missing(frame)["x"].isna().sum()
        0
    """
    data = _validate_dataframe(df)
    selected = _resolve_columns(data, list(cols) if cols is not None else None)
    if not selected:
        return data

    numeric_cols = [column for column in selected if is_numeric_dtype(data[column])]
    categorical_cols = [column for column in selected if column not in numeric_cols]
    normalized_strategy = strategy.lower()
    transformed = data.copy()

    if normalized_strategy == "smart":
        for column in numeric_cols:
            transformed[column] = transformed[column].fillna(transformed[column].median())
        for column in categorical_cols:
            transformed[column] = transformed[column].fillna(_mode_value(transformed[column]))
        return transformed

    if normalized_strategy in {"mean", "median", "mode"}:
        for column in selected:
            if normalized_strategy == "mean":
                if is_numeric_dtype(transformed[column]):
                    transformed[column] = transformed[column].fillna(transformed[column].mean())
            elif normalized_strategy == "median":
                if is_numeric_dtype(transformed[column]):
                    transformed[column] = transformed[column].fillna(transformed[column].median())
            else:
                transformed[column] = transformed[column].fillna(_mode_value(transformed[column]))
        return transformed

    if normalized_strategy == "constant":
        for column in selected:
            replacement = fill_value
            if replacement is None:
                replacement = 0 if is_numeric_dtype(transformed[column]) else "Missing"
            transformed[column] = transformed[column].fillna(replacement)
        return transformed

    if normalized_strategy == "knn":
        from sklearn.impute import KNNImputer

        if not numeric_cols:
            return transformed
        imputer = KNNImputer()
        transformed[numeric_cols] = imputer.fit_transform(transformed[numeric_cols])
        return transformed

    if normalized_strategy == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        if not numeric_cols:
            return transformed
        imputer = IterativeImputer(random_state=42)
        transformed[numeric_cols] = imputer.fit_transform(transformed[numeric_cols])
        return transformed

    raise ValueError(f"Unknown fill strategy: {strategy}")


def remove_outliers(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    method: str = "iqr",
    factor: float = 1.5,
) -> pd.DataFrame:
    """Remove rows flagged as outliers from numeric columns.

    Args:
        df: Input DataFrame.
        cols: Numeric columns used for outlier detection.
        method: Detection method: ``iqr`` or ``zscore``.
        factor: Threshold multiplier used by the selected method.

    Returns:
        A filtered DataFrame copy.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import remove_outliers
        >>> frame = pd.DataFrame({"x": [1, 2, 100]})
        >>> len(remove_outliers(frame)) < len(frame)
        True
    """
    data = _validate_dataframe(df)
    selected = _numeric_columns(data, list(cols) if cols is not None else None)
    if not selected:
        return data

    normalized_method = method.lower()
    if normalized_method not in {"iqr", "zscore"}:
        raise ValueError("method must be 'iqr' or 'zscore'")

    keep_mask = pd.Series(True, index=data.index)
    for column in selected:
        series = data[column]
        if normalized_method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
        else:
            mean_value = series.mean()
            std_value = series.std(ddof=0)
            lower_bound = mean_value - factor * std_value
            upper_bound = mean_value + factor * std_value
        keep_mask &= series.between(lower_bound, upper_bound) | series.isna()

    return data.loc[keep_mask].copy()


def clip_outliers(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Clip numeric columns to percentile bounds.

    Args:
        df: Input DataFrame.
        cols: Numeric columns to clip. Defaults to all numeric columns.
        lower: Lower quantile bound.
        upper: Upper quantile bound.

    Returns:
        A clipped DataFrame copy.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import clip_outliers
        >>> frame = pd.DataFrame({"x": [1, 2, 100]})
        >>> clip_outliers(frame)["x"].max() < 100
        True
    """
    data = _validate_dataframe(df)
    if not 0 <= lower < upper <= 1:
        raise ValueError("Expected 0 <= lower < upper <= 1")

    selected = _numeric_columns(data, list(cols) if cols is not None else None)
    transformed = data.copy()
    for column in selected:
        lower_bound = transformed[column].quantile(lower)
        upper_bound = transformed[column].quantile(upper)
        transformed[column] = transformed[column].clip(lower=lower_bound, upper=upper_bound)
    return transformed


def reduce_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage.

    Args:
        df: Input DataFrame.
        verbose: Whether to print a short memory reduction summary.

    Returns:
        A DataFrame copy with downcast dtypes where possible.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import reduce_memory
        >>> frame = pd.DataFrame({"x": [1, 2, 3]})
        >>> str(reduce_memory(frame)["x"].dtype).startswith("int")
        True
    """
    data = _validate_dataframe(df)
    transformed = data.copy()
    start_memory = transformed.memory_usage(deep=True).sum()

    for column in transformed.columns:
        series = transformed[column]
        if pd.api.types.is_integer_dtype(series):
            transformed[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            transformed[column] = pd.to_numeric(series, downcast="float")
        elif series.dtype == "object":
            unique_ratio = series.nunique(dropna=False) / max(len(series), 1)
            if unique_ratio <= 0.8:
                transformed[column] = series.astype("category")

    end_memory = transformed.memory_usage(deep=True).sum()
    transformed.attrs["memory_reduction_pct"] = float(
        ((start_memory - end_memory) / start_memory * 100) if start_memory else 0.0
    )
    if verbose:
        print(f"Memory reduced by {transformed.attrs['memory_reduction_pct']:.2f}%")
    return transformed


def drop_correlated(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop highly correlated numeric columns.

    Args:
        df: Input DataFrame.
        threshold: Absolute correlation threshold above which columns are dropped.

    Returns:
        A DataFrame copy without highly correlated numeric columns.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import drop_correlated
        >>> frame = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        >>> len(drop_correlated(frame).columns) < len(frame.columns)
        True
    """
    data = _validate_dataframe(df)
    numeric_cols = _numeric_columns(data)
    if len(numeric_cols) < 2:
        return data

    correlation = data[numeric_cols].corr().abs()
    upper_triangle = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper_triangle.columns if (upper_triangle[column] > threshold).any()
    ]
    return data.drop(columns=to_drop)


def drop_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Drop numeric columns with low variance.

    Args:
        df: Input DataFrame.
        threshold: Minimum variance required to keep a numeric column.

    Returns:
        A DataFrame copy without low-variance numeric columns.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import drop_low_variance
        >>> frame = pd.DataFrame({"x": [1, 1, 1], "y": [1, 2, 3]})
        >>> "x" not in drop_low_variance(frame).columns
        True
    """
    data = _validate_dataframe(df)
    numeric_cols = _numeric_columns(data)
    if not numeric_cols:
        return data

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(data[numeric_cols].fillna(0))
    keep_numeric = [column for column, keep in zip(numeric_cols, selector.get_support()) if keep]
    keep_columns = [
        column for column in data.columns if column not in numeric_cols or column in keep_numeric
    ]
    return data[keep_columns].copy()


def drop_high_cardinality(
    df: pd.DataFrame,
    threshold: float = 0.9,
    cols: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Drop columns whose unique-to-row ratio exceeds a threshold.

    Args:
        df: Input DataFrame.
        threshold: Unique-value ratio threshold used for dropping columns.
        cols: Optional subset of columns to evaluate.

    Returns:
        A DataFrame copy with high-cardinality columns removed.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import drop_high_cardinality
        >>> frame = pd.DataFrame({"id": [1, 2, 3], "x": ["a", "a", "b"]})
        >>> "id" not in drop_high_cardinality(frame, threshold=0.8).columns
        True
    """
    data = _validate_dataframe(df)
    if threshold < 0:
        raise ValueError("threshold must be greater than or equal to 0")

    selected = _resolve_columns(data, list(cols) if cols is not None else None)
    row_count = max(len(data), 1)
    to_drop = [
        column for column in selected if data[column].nunique(dropna=True) / row_count > threshold
    ]
    return data.drop(columns=to_drop)


def datetime_features(
    df: pd.DataFrame,
    col: str,
    prefix: str | None = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Expand a datetime column into common calendar features.

    Args:
        df: Input DataFrame.
        col: Datetime-like column to expand.
        prefix: Optional prefix for generated feature names.
        drop_original: Whether to drop the source column after expansion.

    Returns:
        A DataFrame copy with added datetime features.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import datetime_features
        >>> frame = pd.DataFrame({"date": ["2024-01-01"]})
        >>> "date_year" in datetime_features(frame, "date").columns
        True
    """
    data = _validate_dataframe(df)
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    transformed = data.copy()
    prefix_name = prefix or col
    series = transformed[col]
    if not is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors="coerce")

    transformed[f"{prefix_name}_year"] = series.dt.year
    transformed[f"{prefix_name}_month"] = series.dt.month
    transformed[f"{prefix_name}_day"] = series.dt.day
    transformed[f"{prefix_name}_hour"] = series.dt.hour
    transformed[f"{prefix_name}_minute"] = series.dt.minute
    transformed[f"{prefix_name}_dayofweek"] = series.dt.dayofweek
    transformed[f"{prefix_name}_is_weekend"] = series.dt.dayofweek.isin([5, 6]).astype("Int64")
    transformed[f"{prefix_name}_quarter"] = series.dt.quarter
    transformed[f"{prefix_name}_week_of_year"] = series.dt.isocalendar().week.astype("Int64")
    transformed[f"{prefix_name}_is_month_start"] = series.dt.is_month_start.astype("Int64")
    transformed[f"{prefix_name}_is_month_end"] = series.dt.is_month_end.astype("Int64")

    if drop_original:
        transformed = transformed.drop(columns=[col])
    return transformed


def text_features(
    df: pd.DataFrame,
    col: str,
    prefix: str | None = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Expand a text column into simple NLP-style features.

    Args:
        df: Input DataFrame.
        col: Text column to expand.
        prefix: Optional prefix for generated feature names.
        drop_original: Whether to drop the source column after expansion.

    Returns:
        A DataFrame copy with added text features.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import text_features
        >>> frame = pd.DataFrame({"text": ["Hello World"]})
        >>> "text_word_count" in text_features(frame, "text").columns
        True
    """
    data = _validate_dataframe(df)
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    transformed = data.copy()
    prefix_name = prefix or col
    text = transformed[col].fillna("").astype(str)
    words = text.str.findall(r"\b\w+\b")

    transformed[f"{prefix_name}_word_count"] = words.str.len()
    transformed[f"{prefix_name}_char_count"] = text.str.len()
    transformed[f"{prefix_name}_avg_word_length"] = (
        words.apply(lambda items: np.mean([len(item) for item in items]) if items else 0.0)
    )
    transformed[f"{prefix_name}_has_digits"] = text.str.contains(r"\d").astype("int64")
    transformed[f"{prefix_name}_has_special"] = text.str.contains(r"[^A-Za-z0-9\s]").astype("int64")
    transformed[f"{prefix_name}_uppercase_ratio"] = text.apply(
        lambda value: (
            sum(character.isupper() for character in value) / len(value) if value else 0.0
        )
    )

    if drop_original:
        transformed = transformed.drop(columns=[col])
    return transformed


def interaction_features(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...],
    operations: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Create pairwise interaction features for numeric columns.

    Args:
        df: Input DataFrame.
        cols: Numeric columns used to generate interactions.
        operations: Operations to apply. Defaults to multiply, add, subtract, divide.

    Returns:
        A DataFrame copy with added interaction features.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import interaction_features
        >>> frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> "x_multiply_y" in interaction_features(frame, ["x", "y"]).columns
        True
    """
    data = _validate_dataframe(df)
    selected = _numeric_columns(data, list(cols))
    if len(selected) < 2:
        raise ValueError("interaction_features requires at least two numeric columns")

    selected_ops = list(operations or ["multiply", "add", "subtract", "divide"])
    transformed = data.copy()
    for left, right in combinations(selected, 2):
        left_series = transformed[left]
        right_series = transformed[right]
        if "multiply" in selected_ops:
            transformed[f"{left}_multiply_{right}"] = left_series * right_series
        if "add" in selected_ops:
            transformed[f"{left}_add_{right}"] = left_series + right_series
        if "subtract" in selected_ops:
            transformed[f"{left}_subtract_{right}"] = left_series - right_series
        if "divide" in selected_ops:
            division = left_series / right_series.replace(0, np.nan)
            transformed[f"{left}_divide_{right}"] = division.replace([np.inf, -np.inf], np.nan)
    return transformed


def polynomial_features(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...],
    degree: int = 2,
    include_bias: bool = False,
) -> pd.DataFrame:
    """Generate polynomial features for selected numeric columns.

    Args:
        df: Input DataFrame.
        cols: Numeric columns to transform.
        degree: Polynomial degree.
        include_bias: Whether to include the bias term.

    Returns:
        A DataFrame copy with additional polynomial features.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import polynomial_features
        >>> frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> transformed = polynomial_features(frame, ["x", "y"])
        >>> any(column.startswith("x^2") for column in transformed.columns)
        True
    """
    data = _validate_dataframe(df)
    selected = _numeric_columns(data, list(cols))
    if not selected:
        return data

    transformer = PolynomialFeatures(degree=degree, include_bias=include_bias)
    values = transformer.fit_transform(data[selected].fillna(0))
    feature_names = transformer.get_feature_names_out(selected)
    transformed = data.copy()
    generated = pd.DataFrame(values, columns=feature_names, index=data.index)
    new_columns = [column for column in generated.columns if column not in selected]
    for column in new_columns:
        transformed[column] = generated[column]
    return transformed


def bin_numeric(
    df: pd.DataFrame,
    col: str,
    bins: int = 10,
    method: str = "equal_width",
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Bin a numeric column into discrete intervals.

    Args:
        df: Input DataFrame.
        col: Numeric column to bin.
        bins: Number of bins.
        method: Binning strategy: ``equal_width``, ``equal_freq``, or ``kmeans``.
        labels: Optional labels passed to pandas binning helpers.

    Returns:
        A DataFrame copy with a new ``{col}_bin`` column.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import bin_numeric
        >>> frame = pd.DataFrame({"x": [1, 2, 3, 4]})
        >>> "x_bin" in bin_numeric(frame, "x").columns
        True
    """
    data = _validate_dataframe(df)
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    if not is_numeric_dtype(data[col]):
        raise TypeError(f"Column '{col}' must be numeric for binning")

    transformed = data.copy()
    normalized_method = method.lower()
    new_column = f"{col}_bin"

    if normalized_method == "equal_width":
        transformed[new_column] = pd.cut(
            transformed[col],
            bins=bins,
            labels=labels,
            duplicates="drop",
        )
        return transformed
    if normalized_method == "equal_freq":
        transformed[new_column] = pd.qcut(
            transformed[col],
            q=bins,
            labels=labels,
            duplicates="drop",
        )
        return transformed
    if normalized_method == "kmeans":
        discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans")
        encoded = discretizer.fit_transform(transformed[[col]]).astype("int64").ravel()
        transformed[new_column] = encoded
        return transformed

    raise ValueError(f"Unknown binning method: {method}")


def target_encode(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...],
    target: str,
    cv: int = 5,
    smoothing: float = 10.0,
) -> pd.DataFrame:
    """Target-encode categorical columns with cross-validation.

    Args:
        df: Input DataFrame containing both features and target.
        cols: Columns to target-encode.
        target: Target column name.
        cv: Number of cross-validation folds.
        smoothing: Smoothing strength applied to category means.

    Returns:
        A DataFrame copy with encoded columns.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import target_encode
        >>> frame = pd.DataFrame({"x": ["a", "a", "b"], "y": [0, 1, 1]})
        >>> target_encode(frame, ["x"], target="y")["x"].dtype.kind == "f"
        True
    """
    data = _validate_dataframe(df)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    selected = _resolve_columns(data, list(cols), exclude=[target])
    transformed = data.copy()
    target_series = transformed[target]
    global_mean = float(target_series.mean())
    is_classification = _is_classification_target(target_series)
    effective_splits = min(cv, len(transformed))
    if is_classification:
        min_class_count = int(target_series.value_counts().min())
        effective_splits = min(effective_splits, min_class_count)

    if effective_splits < 2:
        for column in selected:
            filled = transformed[column].astype("object").fillna("Missing")
            stats = target_series.groupby(filled).mean()
            transformed[column] = filled.map(stats).fillna(global_mean).astype(float)
        return transformed

    splitter = (
        StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
        if is_classification
        else KFold(n_splits=effective_splits, shuffle=True, random_state=42)
    )

    for column in selected:
        encoded = pd.Series(index=transformed.index, dtype=float)
        filled = transformed[column].astype("object").fillna("Missing")
        for train_idx, valid_idx in splitter.split(transformed, target_series):
            train_categories = filled.iloc[train_idx]
            valid_categories = filled.iloc[valid_idx]
            train_target = target_series.iloc[train_idx]
            stats = train_target.groupby(train_categories).agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            encoded.iloc[valid_idx] = valid_categories.map(smooth).fillna(global_mean)
        transformed[column] = encoded.astype(float)

    return transformed


def frequency_encode(df: pd.DataFrame, cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Replace categories with their empirical frequency.

    Args:
        df: Input DataFrame.
        cols: Columns to frequency-encode.

    Returns:
        A DataFrame copy with encoded columns.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import frequency_encode
        >>> frame = pd.DataFrame({"x": ["a", "a", "b"]})
        >>> frequency_encode(frame, ["x"])["x"].max() <= 1
        True
    """
    data = _validate_dataframe(df)
    selected = _resolve_columns(data, list(cols))
    transformed = data.copy()
    for column in selected:
        filled = transformed[column].astype("object").fillna("Missing")
        frequencies = filled.value_counts(normalize=True)
        transformed[column] = filled.map(frequencies).astype(float)
    return transformed


def log_transform(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Apply a safe log1p transform to numeric columns.

    Args:
        df: Input DataFrame.
        cols: Numeric columns to transform. Defaults to all numeric columns.

    Returns:
        A transformed DataFrame copy.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import log_transform
        >>> frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> log_transform(frame)["x"].min() >= 0
        True
    """
    data = _validate_dataframe(df)
    selected = _numeric_columns(data, list(cols) if cols is not None else None)
    transformed = data.copy()
    for column in selected:
        series = transformed[column]
        minimum = series.min(skipna=True)
        shift = 0.0 if pd.isna(minimum) or minimum > -1 else abs(float(minimum)) + 1.0
        transformed[column] = np.log1p(series + shift)
    return transformed


def auto_preprocess(
    df: pd.DataFrame,
    target: str | None = None,
    encode_method: str = "onehot",
    scale_method: str = "standard",
    fill_strategy: str = "smart",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a lightweight preprocessing pipeline with smart defaults.

    Args:
        df: Input DataFrame.
        target: Optional target column left untouched by preprocessing.
        encode_method: Encoding strategy for categorical feature columns.
        scale_method: Scaling strategy for numeric feature columns.
        fill_strategy: Missing-value strategy applied before encoding and scaling.

    Returns:
        A tuple of ``(processed_df, report)``.

    Example:
        >>> import pandas as pd
        >>> from dsbro.prep import auto_preprocess
        >>> frame = pd.DataFrame({"x": [1.0, None], "city": ["a", "b"]})
        >>> processed, report = auto_preprocess(frame)
        >>> processed.isna().sum().sum() == 0
        True
    """
    data = _validate_dataframe(df)
    if target is not None and target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    feature_columns = [column for column in data.columns if column != target]
    features = fill_missing(data[feature_columns], strategy=fill_strategy)
    categorical_cols = _categorical_columns(features)

    if categorical_cols:
        if encode_method.lower() == "target":
            if target is None:
                raise ValueError("target is required when encode_method='target'")
            merged = pd.concat([features, data[[target]]], axis=1)
            features = target_encode(merged, categorical_cols, target=target)[feature_columns]
        else:
            features = encode(features, cols=categorical_cols, method=encode_method)

    numeric_cols = _numeric_columns(features)
    if scale_method:
        features = scale(features, cols=numeric_cols, method=scale_method)

    processed = features.copy()
    if target is not None:
        processed[target] = data[target]

    report = {
        "original_shape": data.shape,
        "processed_shape": processed.shape,
        "filled_columns": feature_columns,
        "encoded_columns": categorical_cols,
        "scaled_columns": _numeric_columns(features),
    }
    return processed, report


__all__ = [
    "auto_preprocess",
    "bin_numeric",
    "clip_outliers",
    "datetime_features",
    "drop_correlated",
    "drop_high_cardinality",
    "drop_low_variance",
    "encode",
    "fill_missing",
    "frequency_encode",
    "interaction_features",
    "log_transform",
    "polynomial_features",
    "reduce_memory",
    "remove_outliers",
    "scale",
    "target_encode",
    "text_features",
]
