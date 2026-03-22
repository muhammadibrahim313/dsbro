"""Tests for dsbro.utils."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from dsbro.utils import (
    chunk,
    download,
    flatten,
    gpu_info,
    notebook_setup,
    parallelize,
    seed,
    setup,
    show_versions,
    suppress_warnings,
    system_info,
    timer,
)


def test_setup_sets_pandas_options():
    config = setup(dark=False, max_columns=20, max_rows=50, precision=3)

    assert config["theme"] == "light"
    assert pd.get_option("display.max_columns") == 20
    assert pd.get_option("display.max_rows") == 50
    assert pd.get_option("display.precision") == 3


def test_notebook_setup_aliases_setup():
    config = notebook_setup(dark=False)

    assert config["theme"] == "light"


def test_seed_makes_numpy_repeatable():
    seed(123)
    first = np.random.rand(4)
    seed(123)
    second = np.random.rand(4)

    assert np.allclose(first, second)


def test_timer_records_elapsed(capsys):
    with timer("benchmark") as result:
        time.sleep(0.01)

    captured = capsys.readouterr()
    assert "benchmark:" in captured.out
    assert result["elapsed"] >= 0.01


def test_gpu_info_returns_mapping():
    info = gpu_info()

    assert isinstance(info, dict)
    assert "available" in info


def test_system_info_contains_expected_keys():
    info = system_info()

    assert "python_version" in info
    assert "packages" in info
    assert "numpy" in info["packages"]


def test_show_versions_returns_versions(capsys):
    versions = show_versions()

    captured = capsys.readouterr()
    assert "python:" in captured.out
    assert "numpy" in versions


def test_suppress_warnings_sets_ignore_filter():
    warnings.resetwarnings()
    suppress_warnings()

    assert warnings.filters[0][0] == "ignore"


def test_flatten_flattens_nested_iterables():
    flattened = flatten([1, [2, (3, 4)], {5, 6}])

    assert set(flattened) == {1, 2, 3, 4, 5, 6}


def test_chunk_splits_iterable():
    batches = list(chunk([1, 2, 3, 4, 5], 2))

    assert batches == [[1, 2], [3, 4], [5]]


def test_parallelize_applies_function():
    results = parallelize(lambda value: value * 2, [1, 2, 3], n_jobs=1)

    assert results == [2, 4, 6]


def test_download_copies_local_file(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("hello dsbro", encoding="utf-8")
    target = tmp_path / "downloads" / "copied.txt"

    downloaded = download(source.resolve().as_uri(), target)

    assert downloaded == target
    assert downloaded.read_text(encoding="utf-8") == "hello dsbro"
