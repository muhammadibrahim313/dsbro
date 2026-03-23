"""Tests for dsbro.prep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsbro.prep import (
    auto_preprocess,
    bin_numeric,
    clip_outliers,
    datetime_features,
    drop_correlated,
    drop_high_cardinality,
    drop_low_variance,
    encode,
    fill_missing,
    frequency_encode,
    interaction_features,
    log_transform,
    polynomial_features,
    reduce_memory,
    remove_outliers,
    scale,
    target_encode,
    text_features,
)


@pytest.fixture
def prep_df() -> pd.DataFrame:
    """Return a DataFrame that exercises all prep helpers."""
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 100.0, 5.0],
            "num2": [10.0, 20.0, 30.0, 400.0, 50.0],
            "corr": [2.0, 4.0, np.nan, 200.0, 10.0],
            "constant": [1, 1, 1, 1, 1],
            "cat1": ["a", "b", None, "a", "c"],
            "cat2": ["x", "y", "y", "x", "z"],
            "date": ["2024-01-01", "2024-01-02", None, "2024-02-01", "2024-03-05"],
            "text": ["Hello 123", "bye!", "ALL CAPS", "mix3d text", None],
            "id_col": ["id1", "id2", "id3", "id4", "id5"],
            "target_cls": [0, 1, 1, 0, 1],
            "target_reg": [1.0, 2.0, 2.5, 4.0, 5.0],
        }
    )


def test_encode_onehot_returns_new_columns(prep_df: pd.DataFrame):
    transformed = encode(prep_df[["cat1", "cat2"]], method="onehot")

    assert "cat1_a" in transformed.columns
    assert prep_df["cat1"].isna().sum() == 1


def test_scale_standard_centers_numeric_data(prep_df: pd.DataFrame):
    transformed = scale(fill_missing(prep_df[["num1", "num2"]]), method="standard")

    assert round(float(transformed["num1"].mean()), 6) == 0.0


def test_fill_missing_smart_fills_nulls(prep_df: pd.DataFrame):
    transformed = fill_missing(prep_df[["num1", "cat1"]], strategy="smart")

    assert transformed.isna().sum().sum() == 0


def test_remove_outliers_drops_extreme_rows(prep_df: pd.DataFrame):
    transformed = remove_outliers(prep_df[["num1", "num2"]], cols=["num1"])

    assert len(transformed) < len(prep_df)


def test_clip_outliers_caps_values(prep_df: pd.DataFrame):
    transformed = clip_outliers(prep_df[["num1"]], lower=0.0, upper=0.8)

    assert transformed["num1"].max() < prep_df["num1"].max()


def test_reduce_memory_downcasts_numeric_types(prep_df: pd.DataFrame):
    transformed = reduce_memory(prep_df[["constant", "cat2"]], verbose=False)

    assert str(transformed["constant"].dtype) in {"int8", "int16", "int32", "int64"}
    assert str(transformed["cat2"].dtype) == "category"


def test_drop_correlated_removes_high_corr_columns(prep_df: pd.DataFrame):
    transformed = drop_correlated(fill_missing(prep_df[["num1", "corr"]]), threshold=0.95)

    assert len(transformed.columns) == 1


def test_drop_low_variance_removes_constant_column(prep_df: pd.DataFrame):
    transformed = drop_low_variance(prep_df[["constant", "num2"]])

    assert "constant" not in transformed.columns


def test_drop_high_cardinality_removes_id_like_columns(prep_df: pd.DataFrame):
    transformed = drop_high_cardinality(prep_df[["id_col", "cat1"]], threshold=0.8)

    assert "id_col" not in transformed.columns


def test_datetime_features_adds_calendar_columns(prep_df: pd.DataFrame):
    transformed = datetime_features(prep_df[["date"]], "date")

    assert "date_year" in transformed.columns
    assert "date_week_of_year" in transformed.columns


def test_text_features_adds_text_metrics(prep_df: pd.DataFrame):
    transformed = text_features(prep_df[["text"]], "text")

    assert "text_word_count" in transformed.columns
    assert "text_has_digits" in transformed.columns


def test_interaction_features_adds_pairwise_columns(prep_df: pd.DataFrame):
    base = fill_missing(prep_df[["num1", "num2"]])
    transformed = interaction_features(base, ["num1", "num2"])

    assert "num1_multiply_num2" in transformed.columns
    assert "num1_divide_num2" in transformed.columns


def test_polynomial_features_adds_generated_terms(prep_df: pd.DataFrame):
    base = fill_missing(prep_df[["num1", "num2"]])
    transformed = polynomial_features(base, ["num1", "num2"])

    assert any(
        column.startswith("num1^2") or column == "num1 num2" for column in transformed.columns
    )


def test_bin_numeric_adds_binned_column(prep_df: pd.DataFrame):
    transformed = bin_numeric(fill_missing(prep_df[["num2"]]), "num2", bins=3)

    assert "num2_bin" in transformed.columns


def test_target_encode_returns_float_column(prep_df: pd.DataFrame):
    transformed = target_encode(prep_df[["cat1", "target_cls"]], ["cat1"], target="target_cls")

    assert transformed["cat1"].dtype.kind == "f"


def test_frequency_encode_replaces_categories_with_ratios(prep_df: pd.DataFrame):
    transformed = frequency_encode(prep_df[["cat2"]], ["cat2"])

    assert transformed["cat2"].max() <= 1.0


def test_log_transform_is_non_negative_for_positive_data(prep_df: pd.DataFrame):
    transformed = log_transform(fill_missing(prep_df[["num2"]]), ["num2"])

    assert transformed["num2"].min() >= 0


def test_auto_preprocess_returns_processed_frame_and_report(prep_df: pd.DataFrame):
    transformed, report = auto_preprocess(
        prep_df[["num1", "cat1", "target_cls"]],
        target="target_cls",
    )

    assert transformed.isna().sum().sum() == 0
    assert "target_cls" in transformed.columns
    assert "processed_shape" in report
