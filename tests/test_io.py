"""Tests for dsbro.io."""

from __future__ import annotations

import pandas as pd

from dsbro.io import (
    copy_structure,
    fileinfo,
    find,
    load,
    merge_csvs,
    peek,
    read_all,
    sample_files,
    save,
    to_kaggle_submission,
    tree,
)


def test_tree_renders_nested_structure(temp_data_dir, capsys):
    rendered = tree(temp_data_dir, depth=2)

    captured = capsys.readouterr()
    assert "data/" in rendered
    assert "nested/" in rendered
    assert "sample.csv" in captured.out


def test_load_reads_common_formats(temp_data_dir):
    csv_df = load(temp_data_dir / "sample.csv")
    json_df = load(temp_data_dir / "sample.json")
    text_value = load(temp_data_dir / "notes.txt")

    assert isinstance(csv_df, pd.DataFrame)
    assert isinstance(json_df, pd.DataFrame)
    assert "alpha" in text_value


def test_save_writes_dataframe_and_text(tmp_path, sample_df):
    csv_path = save(sample_df, tmp_path / "written.csv")
    text_path = save("hello", tmp_path / "written.txt")

    assert csv_path.exists()
    assert text_path.read_text(encoding="utf-8") == "hello"


def test_peek_returns_dataframe_head(temp_data_dir):
    preview = peek(temp_data_dir / "sample.csv", n=2)

    assert isinstance(preview, pd.DataFrame)
    assert len(preview) == 2


def test_find_supports_glob_and_regex(temp_data_dir):
    csv_matches = find(temp_data_dir, "*.csv")
    regex_matches = find(temp_data_dir, r"sample_[ab]\.csv")

    assert len(csv_matches) == 3
    assert len(regex_matches) == 2


def test_fileinfo_reports_tabular_metadata(temp_data_dir):
    info = fileinfo(temp_data_dir / "sample.csv")

    assert info["shape"] == (4, 4)
    assert info["encoding"] == "utf-8"


def test_peek_prints_header(temp_data_dir, capsys):
    peek(temp_data_dir / "sample.csv", n=2)

    captured = capsys.readouterr()
    assert "--- Peek: sample.csv (first 2 rows) ---" in captured.out


def test_fileinfo_prints_formatted_summary(temp_data_dir, capsys):
    fileinfo(temp_data_dir / "sample.csv")

    captured = capsys.readouterr()
    assert "File Info: sample.csv" in captured.out
    assert "Modified" in captured.out


def test_merge_csvs_combines_matches(temp_data_dir):
    merged = merge_csvs(temp_data_dir)

    assert len(merged) == 10


def test_copy_structure_creates_directories_without_files(temp_data_dir, tmp_path):
    destination = copy_structure(temp_data_dir, tmp_path / "copied")

    assert (destination / "nested").is_dir()
    assert not (destination / "sample.csv").exists()


def test_sample_files_returns_requested_count(temp_data_dir):
    sampled = sample_files(temp_data_dir, n=2)

    assert len(sampled) == 2
    assert all(path.is_file() for path in sampled)


def test_read_all_returns_mapping(temp_data_dir):
    loaded = read_all(temp_data_dir, "*.csv")

    assert len(loaded) == 3
    assert all(isinstance(value, pd.DataFrame) for value in loaded.values())


def test_to_kaggle_submission_saves_csv(tmp_path, submission_df):
    output = to_kaggle_submission(submission_df, tmp_path / "submission")

    assert output.suffix == ".csv"
    assert output.exists()
