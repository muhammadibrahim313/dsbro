"""Tests for dsbro.metrics."""

from __future__ import annotations

import numpy as np

from dsbro.metrics import (
    all_metrics,
    classification_report,
    competition_score,
    metric,
    regression_report,
)


def test_classification_report_returns_expected_keys():
    report = classification_report([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.8, 0.2])

    assert report["accuracy"] == 0.75
    assert "auc" in report
    assert "log_loss" in report


def test_classification_report_omits_probability_metrics_without_y_prob():
    report = classification_report([0, 1, 1, 0], [0, 1, 0, 0])

    assert "auc" not in report
    assert "log_loss" not in report


def test_regression_report_returns_expected_keys():
    report = regression_report([1.0, 2.0, 3.0], [1.0, 2.2, 2.8])

    assert "rmse" in report
    assert "median_ae" in report
    assert "explained_variance" in report


def test_metric_supports_default_accuracy_auc_rmse_and_mse():
    accuracy_value = metric([0, 1, 1, 0], [0, 1, 0, 0])
    auc_value = metric([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], name="auc")
    rmse_value = metric([1.0, 2.0], [1.0, 2.5], name="rmse")
    mse_value = metric([1.0, 2.0], [1.0, 2.5], name="mse")

    assert accuracy_value == 0.75
    assert round(auc_value, 4) == 1.0
    assert rmse_value > 0
    assert mse_value > 0


def test_all_metrics_returns_task_specific_dicts():
    cls_metrics = all_metrics([0, 1], [0, 1], task="classification")
    reg_metrics = all_metrics([1.0, 2.0], [1.0, 2.2], task="regression")

    assert "accuracy" in cls_metrics
    assert "rmse" in reg_metrics


def test_competition_score_aliases_metric():
    score = competition_score([1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.5]), metric="rmse")

    assert score > 0
