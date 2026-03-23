"""Tests for dsbro.ml."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

from dsbro.ml import (
    adversarial_validation,
    auto_train,
    blend,
    compare,
    cross_validate,
    feature_select,
    oof_predict,
    power_mean,
    pseudo_label,
    stack,
    train,
    tune,
)


@pytest.fixture
def cls_df() -> pd.DataFrame:
    """Return a small classification DataFrame."""
    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    frame["category"] = np.where(frame["f0"] > frame["f0"].median(), "high", "low")
    frame["target"] = y
    return frame


@pytest.fixture
def reg_df() -> pd.DataFrame:
    """Return a small regression DataFrame."""
    X, y = make_regression(
        n_samples=80,
        n_features=5,
        n_informative=4,
        noise=0.5,
        random_state=42,
    )
    frame = pd.DataFrame(X, columns=[f"r{i}" for i in range(X.shape[1])])
    frame["target"] = y
    return frame


def test_compare_returns_leaderboard(cls_df: pd.DataFrame, capsys):
    leaderboard = compare(cls_df, target="target", cv=2)

    captured = capsys.readouterr()
    assert "model" in leaderboard.columns
    assert (leaderboard["status"] == "ok").any()
    assert "Model Compare" in captured.out
    assert re.search(r"\b\d+\.\d{4}\b", captured.out)


def test_train_returns_model_predictions_and_summary(cls_df: pd.DataFrame):
    model, preds, summary = train(cls_df, target="target", model="logistic", cv=2)

    assert hasattr(model, "predict")
    assert len(preds) == len(cls_df)
    assert summary["task"] == "classification"


def test_cross_validate_returns_fold_scores(cls_df: pd.DataFrame):
    X = cls_df.drop(columns=["target"])
    y = cls_df["target"]
    result = cross_validate(LogisticRegression(max_iter=1000), X, y, cv=2)

    assert "fold_scores" in result
    assert len(result["fold_scores"]) == 2


def test_tune_gridsearch_returns_best_model(cls_df: pd.DataFrame):
    X = cls_df.drop(columns=["target"])
    y = cls_df["target"]
    result = tune("logistic", X, y, method="gridsearch")

    assert "best_params" in result
    assert hasattr(result["best_model"], "predict")


def test_blend_and_power_mean_work_on_predictions():
    preds = [np.array([0.1, 0.8]), np.array([0.2, 0.7]), np.array([0.15, 0.75])]
    blended = blend(preds, method="weighted")
    powered = power_mean(preds, p=2.0)

    assert blended.shape == (2,)
    assert powered.shape == (2,)


def test_stack_returns_meta_predictions():
    base_preds = [np.array([0.1, 0.9, 0.8, 0.2]), np.array([0.2, 0.8, 0.7, 0.3])]
    meta_model, stacked_preds = stack(base_preds, "logistic", [0, 1, 1, 0])

    assert hasattr(meta_model, "predict")
    assert len(stacked_preds) == 4


def test_oof_predict_matches_target_length(cls_df: pd.DataFrame):
    X = cls_df.drop(columns=["target"])
    y = cls_df["target"]
    preds = oof_predict("logistic", X, y, cv=2)

    assert len(preds) == len(y)


def test_feature_select_returns_ranked_features(cls_df: pd.DataFrame):
    X = cls_df.drop(columns=["target", "category"])
    y = cls_df["target"]
    summary = feature_select(X, y, method="importance", top_n=3)

    assert "feature" in summary.columns
    assert summary["selected"].sum() == 3


def test_pseudo_label_augments_training_data(cls_df: pd.DataFrame):
    train_frame = cls_df.iloc[:50]
    test_frame = cls_df.drop(columns=["target"]).iloc[50:]
    result = pseudo_label(
        "logistic",
        train_frame.drop(columns=["target"]),
        train_frame["target"],
        test_frame,
        threshold=0.5,
    )

    assert result["pseudo_count"] >= 0
    assert len(result["X_augmented"]) >= len(train_frame.drop(columns=["target"]))


def test_adversarial_validation_returns_auc(cls_df: pd.DataFrame):
    train_frame = cls_df.drop(columns=["target"]).iloc[:40]
    test_frame = cls_df.drop(columns=["target"]).iloc[40:].copy()
    test_frame["f0"] = test_frame["f0"] + 0.5
    result = adversarial_validation(train_frame, test_frame, cv=2)

    assert "auc_mean" in result


def test_auto_train_returns_best_model_bundle(cls_df: pd.DataFrame):
    result = auto_train(cls_df, target="target")

    assert "best_model_name" in result
    assert "leaderboard" in result


def test_train_supports_regression(reg_df: pd.DataFrame):
    model, preds, summary = train(reg_df, target="target", model=LinearRegression(), cv=2)

    assert hasattr(model, "predict")
    assert len(preds) == len(reg_df)
    assert summary["task"] == "regression"
