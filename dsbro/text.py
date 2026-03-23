"""Text processing helpers for dsbro."""

from __future__ import annotations

import html
import re
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer

from dsbro._themes import apply_matplotlib_theme

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<[^>]+>")
_SPECIAL_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_WORD_PATTERN = re.compile(r"\b\w+\b")


def _ensure_text(value: Any) -> str:
    """Convert an arbitrary value into a safe text string."""
    if value is None:
        return ""
    return str(value)


def _prepare_plot() -> tuple[Figure, Axes]:
    """Create a themed figure for text plots."""
    theme = apply_matplotlib_theme("dark")
    sns.set_theme(style="darkgrid", rc=theme)
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    figure.patch.set_facecolor(theme["figure.facecolor"])
    axis.set_facecolor(theme["axes.facecolor"])
    return figure, axis


def clean_text(
    text: Any,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_special: bool = True,
) -> str:
    """Normalize free-form text into a cleaner string.

    Args:
        text: Input text-like value.
        lowercase: Whether to lowercase the result.
        remove_urls: Whether to strip URLs.
        remove_html: Whether to strip HTML tags and unescape entities.
        remove_special: Whether to remove punctuation and special characters.

    Returns:
        A cleaned text string.

    Example:
        >>> from dsbro.text import clean_text
        >>> clean_text("<b>Hello!</b> https://example.com")
        'hello'
    """
    cleaned = _ensure_text(text)
    if remove_html:
        cleaned = html.unescape(cleaned)
        cleaned = _HTML_PATTERN.sub(" ", cleaned)
    if remove_urls:
        cleaned = _URL_PATTERN.sub(" ", cleaned)
    if lowercase:
        cleaned = cleaned.lower()
    if remove_special:
        cleaned = _SPECIAL_PATTERN.sub(" ", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def tokenize(text: Any, method: str = "word") -> list[str]:
    """Tokenize text into words or sentences.

    Args:
        text: Input text-like value.
        method: Tokenization mode: ``word`` or ``sentence``.

    Returns:
        A list of tokens.

    Example:
        >>> from dsbro.text import tokenize
        >>> tokenize("One two three")
        ['one', 'two', 'three']
    """
    cleaned = clean_text(text, remove_special=False)
    normalized_method = method.lower()
    if normalized_method == "word":
        return _WORD_PATTERN.findall(cleaned.lower())
    if normalized_method == "sentence":
        sentences = re.split(r"(?<=[.!?])\s+", _ensure_text(text).strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    raise ValueError("method must be 'word' or 'sentence'")


def ngrams(text: Any, n: int = 2) -> list[tuple[str, ...]]:
    """Generate word n-grams from text.

    Args:
        text: Input text-like value.
        n: N-gram size.

    Returns:
        A list of n-gram tuples.

    Example:
        >>> from dsbro.text import ngrams
        >>> ngrams("data science bro", n=2)
        [('data', 'science'), ('science', 'bro')]
    """
    if not isinstance(n, int):
        raise TypeError(f"Expected int for n, got {type(n).__name__}")
    if n <= 0:
        raise ValueError("n must be greater than 0")

    tokens = tokenize(text, method="word")
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def word_frequency(
    texts: Any,
    top_n: int = 20,
    plot: bool = False,
    show: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, Figure, Axes]:
    """Compute token frequencies across one or more texts.

    Args:
        texts: A string or iterable of strings.
        top_n: Maximum number of tokens to return.
        plot: Whether to render a bar chart.
        show: Whether to display the figure when ``plot`` is ``True``.

    Returns:
        A frequency DataFrame, optionally followed by a figure and axis.

    Example:
        >>> from dsbro.text import word_frequency
        >>> freq = word_frequency(["data science", "science bro"])
        >>> freq.iloc[0]["word"]
        'science'
    """
    if not isinstance(top_n, int):
        raise TypeError(f"Expected int for top_n, got {type(top_n).__name__}")
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0")

    if isinstance(texts, (str, bytes)) or texts is None:
        collection = [_ensure_text(texts)]
    else:
        collection = [_ensure_text(item) for item in texts]

    counter: Counter[str] = Counter()
    for value in collection:
        counter.update(tokenize(value, method="word"))

    summary = pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])
    if not plot:
        return summary

    figure, axis = _prepare_plot()
    sns.barplot(data=summary, x="word", y="count", ax=axis, color="#00d4ff")
    axis.tick_params(axis="x", rotation=45)
    axis.set_title("Word Frequency")
    figure.tight_layout()
    if show:
        plt.show()
    return summary, figure, axis


def tfidf_features(
    df: pd.DataFrame,
    col: str,
    max_features: int = 100,
    ngram_range: tuple[int, int] = (1, 1),
) -> pd.DataFrame:
    """Create TF-IDF features from a text column.

    Args:
        df: Input DataFrame.
        col: Text column to vectorize.
        max_features: Maximum number of features to generate.
        ngram_range: Inclusive n-gram range used by ``TfidfVectorizer``.

    Returns:
        A DataFrame of TF-IDF features indexed like the input.

    Example:
        >>> import pandas as pd
        >>> from dsbro.text import tfidf_features
        >>> frame = pd.DataFrame({"text": ["data science", "science bro"]})
        >>> tfidf_features(frame, "text", max_features=3).shape[0]
        2
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    if not isinstance(max_features, int):
        raise TypeError(f"Expected int for max_features, got {type(max_features).__name__}")
    if max_features <= 0:
        raise ValueError("max_features must be greater than 0")

    texts = df[col].fillna("").astype(str).map(clean_text)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(texts)
    feature_names = [f"{col}_tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    return pd.DataFrame(matrix.toarray(), columns=feature_names, index=df.index)


__all__ = [
    "clean_text",
    "ngrams",
    "tfidf_features",
    "tokenize",
    "word_frequency",
]
