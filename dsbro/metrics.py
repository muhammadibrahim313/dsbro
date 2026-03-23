"""Metric helpers for dsbro."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _to_numpy(values: Any, name: str) -> np.ndarray:
    """Convert inputs to 1D or 2D numpy arrays."""
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError(f"{name} must contain at least one value")
    return array


def _validate_shapes(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate that targets and predictions have matching lengths."""
    true = _to_numpy(y_true, "y_true")
    pred = _to_numpy(y_pred, "y_pred")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    return true, pred


def _prepare_probabilities(y_true: np.ndarray, y_prob: Any) -> np.ndarray | None:
    """Normalize probability-like predictions for metric computation."""
    if y_prob is None:
        return None
    probabilities = _to_numpy(y_prob, "y_prob")
    if len(probabilities) != len(y_true):
        raise ValueError("y_true and y_prob must have the same length")
    return probabilities


def classification_report(
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None = None,
) -> dict[str, float]:
    """Compute a compact classification metric report.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Optional probability scores or class-probability matrix.

    Returns:
        A dictionary of classification metrics.

    Example:
        >>> from dsbro.metrics import classification_report
        >>> report = classification_report([0, 1], [0, 1], [0.1, 0.9])
        >>> report["accuracy"]
        1.0
    """
    true, pred = _validate_shapes(y_true, y_pred)
    probabilities = _prepare_probabilities(true, y_prob)
    average = "binary" if np.unique(true).size <= 2 else "weighted"

    report = {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, average=average, zero_division=0)),
        "recall": float(recall_score(true, pred, average=average, zero_division=0)),
        "f1": float(f1_score(true, pred, average=average, zero_division=0)),
        "mcc": float(matthews_corrcoef(true, pred)),
        "cohen_kappa": float(cohen_kappa_score(true, pred)),
    }

    if probabilities is not None:
        try:
            if probabilities.ndim == 1:
                report["auc"] = float(roc_auc_score(true, probabilities))
                prob_for_loss = np.column_stack([1 - probabilities, probabilities])
            else:
                report["auc"] = float(roc_auc_score(true, probabilities, multi_class="ovr"))
                prob_for_loss = probabilities
            report["log_loss"] = float(log_loss(true, prob_for_loss))
        except ValueError:
            report["auc"] = np.nan
            report["log_loss"] = np.nan

    return report


def regression_report(
    y_true: Any,
    y_pred: Any,
) -> dict[str, float]:
    """Compute a compact regression metric report.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        A dictionary of regression metrics.

    Example:
        >>> from dsbro.metrics import regression_report
        >>> report = regression_report([1.0, 2.0], [1.0, 2.5])
        >>> "rmse" in report
        True
    """
    true, pred = _validate_shapes(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    r2 = float(r2_score(true, pred))

    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(true, pred)),
        "mape": float(mean_absolute_percentage_error(true, pred)),
        "r2": r2,
        "median_ae": float(median_absolute_error(true, pred)),
        "explained_variance": float(explained_variance_score(true, pred)),
    }


def metric(y_true: Any, y_pred: Any, name: str = "accuracy") -> float:
    """Compute a single named metric.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted labels, scores, or values depending on the metric.
        name: Metric name.

    Returns:
        The requested metric value.

    Example:
        >>> from dsbro.metrics import metric
        >>> round(metric([0, 1], [0.1, 0.9], name="auc"), 3)
        1.0
    """
    normalized = name.lower()
    true, pred = _validate_shapes(y_true, y_pred)

    regression_metrics = {
        "mse": lambda: float(mean_squared_error(true, pred)),
        "rmse": lambda: float(np.sqrt(mean_squared_error(true, pred))),
        "mae": lambda: float(mean_absolute_error(true, pred)),
        "mape": lambda: float(mean_absolute_percentage_error(true, pred)),
        "r2": lambda: float(r2_score(true, pred)),
        "median_ae": lambda: float(median_absolute_error(true, pred)),
        "explained_variance": lambda: float(explained_variance_score(true, pred)),
    }
    classification_metrics = {
        "accuracy": lambda: float(accuracy_score(true, pred)),
        "precision": lambda: float(
            precision_score(true, pred, average="weighted", zero_division=0)
        ),
        "recall": lambda: float(
            recall_score(true, pred, average="weighted", zero_division=0)
        ),
        "f1": lambda: float(f1_score(true, pred, average="weighted", zero_division=0)),
        "mcc": lambda: float(matthews_corrcoef(true, pred)),
        "cohen_kappa": lambda: float(cohen_kappa_score(true, pred)),
        "auc": lambda: float(roc_auc_score(true, pred)),
        "log_loss": lambda: float(log_loss(true, np.column_stack([1 - pred, pred]))),
    }

    if normalized in regression_metrics:
        return regression_metrics[normalized]()
    if normalized in classification_metrics:
        return classification_metrics[normalized]()

    available = ", ".join(sorted({*regression_metrics, *classification_metrics}))
    raise ValueError(f"Unknown metric '{name}'. Available metrics: {available}")


def all_metrics(y_true: Any, y_pred: Any, task: str = "classification") -> dict[str, float]:
    """Compute all relevant metrics for a task type.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        task: ``classification`` or ``regression``.

    Returns:
        A dictionary of metrics for the selected task.

    Example:
        >>> from dsbro.metrics import all_metrics
        >>> metrics = all_metrics([1.0, 2.0], [1.0, 2.2], task="regression")
        >>> "rmse" in metrics
        True
    """
    normalized = task.lower()
    if normalized == "classification":
        return classification_report(y_true, y_pred)
    if normalized == "regression":
        return regression_report(y_true, y_pred)
    raise ValueError("task must be 'classification' or 'regression'")


def competition_score(y_true: Any, y_pred: Any, metric: str = "rmse") -> float:
    """Alias for metric() using Kaggle-style naming.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values or scores.
        metric: Metric name.

    Returns:
        The requested metric value.

    Example:
        >>> from dsbro.metrics import competition_score
        >>> competition_score([1.0, 2.0], [1.0, 2.0], metric="rmse")
        0.0
    """
    return _metric_alias(y_true, y_pred, metric)


def _metric_alias(y_true: Any, y_pred: Any, metric_name: str) -> float:
    """Internal alias-safe wrapper around metric()."""
    return metric(y_true, y_pred, name=metric_name)


__all__ = [
    "all_metrics",
    "classification_report",
    "competition_score",
    "metric",
    "regression_report",
]
