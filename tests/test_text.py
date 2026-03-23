"""Tests for dsbro.text."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from dsbro.text import clean_text, ngrams, tfidf_features, tokenize, word_frequency


@pytest.fixture(autouse=True)
def close_figures():
    """Close matplotlib figures between tests."""
    yield
    plt.close("all")


def test_clean_text_strips_html_urls_and_specials():
    cleaned = clean_text("<b>Hello!</b> Visit https://example.com now.")

    assert cleaned == "hello visit now"


def test_tokenize_supports_word_and_sentence_modes():
    words = tokenize("Data science bro")
    sentences = tokenize("First sentence. Second sentence!", method="sentence")

    assert words == ["data", "science", "bro"]
    assert len(sentences) == 2


def test_ngrams_returns_word_pairs():
    pairs = ngrams("data science bro", n=2)

    assert pairs == [("data", "science"), ("science", "bro")]


def test_word_frequency_returns_dataframe_and_plot():
    summary = word_frequency(["data science", "science bro"], top_n=2)
    plotted, figure, axis = word_frequency(
        ["data science", "science bro"],
        top_n=2,
        plot=True,
        show=False,
    )

    assert summary.iloc[0]["word"] == "science"
    assert plotted.equals(summary)
    assert figure is not None
    assert axis.get_title() == "Word Frequency"


def test_tfidf_features_returns_feature_frame():
    frame = pd.DataFrame({"text": ["data science", "science bro", "bro data"]})
    features = tfidf_features(frame, "text", max_features=4)

    assert features.shape[0] == len(frame)
    assert all(column.startswith("text_tfidf_") for column in features.columns)
