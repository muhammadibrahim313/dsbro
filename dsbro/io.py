"""File and directory utilities for dsbro."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from dsbro._helpers import (
    _count_lines,
    _detect_text_encoding,
    _ensure_path,
    _format_size,
    _safe_import,
)

_TABULAR_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather", ".xlsx", ".xls"}
_TEXT_SUFFIXES = {".txt", ".log", ".md", ".py", ".json", ".yaml", ".yml"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}


def tree(path: str | Path, depth: int = 2) -> str:
    """Print a simple directory tree with file sizes.

    Args:
        path: Directory or file path to render.
        depth: Maximum depth to traverse relative to ``path``.

    Returns:
        The rendered tree as a string.

    Example:
        >>> from dsbro.io import tree
        >>> output = tree(".")
        >>> isinstance(output, str)
        True
    """
    root = _ensure_path(path)
    if not isinstance(depth, int):
        raise TypeError(f"Expected int for depth, got {type(depth).__name__}")
    if depth < 0:
        raise ValueError("depth must be greater than or equal to 0")

    lines: list[str] = []

    def _render(current: Path, prefix: str = "", level: int = 0) -> None:
        if level > depth:
            return
        if current.is_file():
            lines.append(f"{prefix}{current.name} ({_format_size(current.stat().st_size)})")
            return

        name = current.name or str(current)
        lines.append(f"{prefix}{name}/")
        if level == depth:
            return

        entries = sorted(current.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower()))
        for index, entry in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            child_prefix = prefix + ("    " if index == len(entries) - 1 else "|   ")
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                _render_children(entry, child_prefix, level + 1)
            else:
                file_size = _format_size(entry.stat().st_size)
                lines.append(f"{prefix}{connector}{entry.name} ({file_size})")

    def _render_children(current: Path, prefix: str, level: int) -> None:
        if level > depth:
            return
        entries = sorted(current.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower()))
        for index, entry in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                next_prefix = prefix + ("    " if index == len(entries) - 1 else "|   ")
                _render_children(entry, next_prefix, level + 1)
            else:
                file_size = _format_size(entry.stat().st_size)
                lines.append(f"{prefix}{connector}{entry.name} ({file_size})")

    _render(root)
    output = "\n".join(lines)
    print(output)
    return output


def load(path: str | Path) -> Any:
    """Load common file types with format auto-detection.

    Args:
        path: File path to load.

    Returns:
        A loaded Python object, usually a DataFrame, dict, string, or ndarray.

    Example:
        >>> from dsbro.io import load
        >>> data = load("data.csv")
        >>> hasattr(data, "shape")
        True
    """
    file_path = _ensure_path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".feather":
        return pd.read_feather(file_path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(file_path)
    if suffix == ".json":
        try:
            return pd.read_json(file_path)
        except ValueError:
            with file_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    if suffix in {".yaml", ".yml"}:
        yaml = _safe_import("yaml", "pip install pyyaml")
        with file_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    if suffix in _IMAGE_SUFFIXES:
        return plt.imread(file_path)
    if suffix in _TEXT_SUFFIXES or suffix == "":
        encoding = _detect_text_encoding(file_path) or "utf-8"
        return file_path.read_text(encoding=encoding)

    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save(data: Any, path: str | Path) -> Path:
    """Save a Python object with format auto-detection from the file extension.

    Args:
        data: Object to save.
        path: Output file path.

    Returns:
        The written file path.

    Example:
        >>> import pandas as pd
        >>> from dsbro.io import save
        >>> target = save(pd.DataFrame({"x": [1]}), "sample.csv")
        >>> target.suffix
        '.csv'
    """
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower()

    if isinstance(data, pd.DataFrame):
        if suffix == ".csv":
            data.to_csv(file_path, index=False)
            return file_path
        if suffix == ".tsv":
            data.to_csv(file_path, index=False, sep="\t")
            return file_path
        if suffix == ".json":
            data.to_json(file_path, orient="records")
            return file_path
        if suffix in {".xlsx", ".xls"}:
            data.to_excel(file_path, index=False)
            return file_path
        if suffix == ".parquet":
            data.to_parquet(file_path, index=False)
            return file_path
        if suffix == ".feather":
            data.to_feather(file_path)
            return file_path
        if suffix in {".pkl", ".pickle"}:
            data.to_pickle(file_path)
            return file_path

    if isinstance(data, (dict, list, tuple)):
        if suffix == ".json":
            with file_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            return file_path
        if suffix in {".yaml", ".yml"}:
            yaml = _safe_import("yaml", "pip install pyyaml")
            with file_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, sort_keys=False)
            return file_path

    if isinstance(data, str):
        file_path.write_text(data, encoding="utf-8")
        return file_path

    if isinstance(data, bytes):
        file_path.write_bytes(data)
        return file_path

    if suffix in {".pkl", ".pickle"}:
        import pickle

        with file_path.open("wb") as handle:
            pickle.dump(data, handle)
        return file_path

    raise ValueError(f"Unsupported save combination for path: {file_path}")


def peek(path: str | Path, n: int = 5) -> Any:
    """Preview the first rows or lines of a file.

    Args:
        path: File path to inspect.
        n: Number of rows or lines to preview.

    Returns:
        A DataFrame head, string preview, or summary dictionary depending on file type.

    Example:
        >>> from dsbro.io import peek
        >>> preview = peek("data.csv", n=2)
        >>> hasattr(preview, "shape")
        True
    """
    if not isinstance(n, int):
        raise TypeError(f"Expected int for n, got {type(n).__name__}")
    if n <= 0:
        raise ValueError("n must be greater than 0")

    file_path = _ensure_path(path)
    suffix = file_path.suffix.lower()

    if suffix in _TABULAR_SUFFIXES or suffix in {".csv", ".tsv"}:
        data = load(file_path)
        return data.head(n)
    if suffix == ".json":
        data = load(file_path)
        if isinstance(data, pd.DataFrame):
            return data.head(n)
        if isinstance(data, list):
            return data[:n]
        if isinstance(data, dict):
            return dict(list(data.items())[:n])
    if suffix in _IMAGE_SUFFIXES:
        image = load(file_path)
        return {"shape": tuple(image.shape), "dtype": str(image.dtype)}

    encoding = _detect_text_encoding(file_path) or "utf-8"
    with file_path.open("r", encoding=encoding) as handle:
        lines = [line.rstrip("\n") for _, line in zip(range(n), handle)]
    return "\n".join(lines)


def find(path: str | Path, pattern: str) -> list[Path]:
    """Find files recursively using a glob or regular expression pattern.

    Args:
        path: Directory to search within.
        pattern: Glob pattern such as ``*.csv`` or a regular expression.

    Returns:
        A sorted list of matching file paths.

    Example:
        >>> from dsbro.io import find
        >>> matches = find(".", "*.py")
        >>> isinstance(matches, list)
        True
    """
    if not isinstance(pattern, str):
        raise TypeError(f"Expected str for pattern, got {type(pattern).__name__}")

    root = _ensure_path(path)
    if root.is_file():
        raise ValueError("find expects a directory path")

    if any(token in pattern for token in ("*", "?")):
        matches = [candidate for candidate in root.rglob(pattern) if candidate.is_file()]
    else:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {pattern}") from exc
        matches = [
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and regex.search(str(candidate.relative_to(root)))
        ]

    return sorted(matches)


def fileinfo(path: str | Path) -> dict[str, Any]:
    """Return basic metadata about a file or directory.

    Args:
        path: Target path.

    Returns:
        A dictionary with file metadata such as size, timestamps, encoding, and shape.

    Example:
        >>> from dsbro.io import fileinfo
        >>> info = fileinfo("data.csv")
        >>> "size" in info
        True
    """
    target = _ensure_path(path)
    stats = target.stat()

    info: dict[str, Any] = {
        "path": str(target),
        "name": target.name,
        "suffix": target.suffix.lower(),
        "is_dir": target.is_dir(),
        "size_bytes": stats.st_size,
        "size": _format_size(stats.st_size),
        "created": pd.Timestamp(stats.st_ctime, unit="s").isoformat(),
        "modified": pd.Timestamp(stats.st_mtime, unit="s").isoformat(),
    }

    if target.is_dir():
        info["file_count"] = sum(1 for candidate in target.rglob("*") if candidate.is_file())
        return info

    encoding = _detect_text_encoding(target)
    if encoding is not None:
        info["encoding"] = encoding
        info["line_count"] = _count_lines(target, encoding)

    if target.suffix.lower() in _TABULAR_SUFFIXES or target.suffix.lower() == ".json":
        try:
            data = load(target)
            if isinstance(data, pd.DataFrame):
                info["shape"] = tuple(data.shape)
                info["columns"] = list(data.columns)
        except (ImportError, ValueError):
            pass

    return info


def merge_csvs(folder: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Merge matching CSV files from a folder into one DataFrame.

    Args:
        folder: Folder containing CSV files.
        pattern: Glob pattern used to select CSV files.

    Returns:
        A concatenated DataFrame.

    Example:
        >>> from dsbro.io import merge_csvs
        >>> merged = merge_csvs("data")
        >>> hasattr(merged, "shape")
        True
    """
    csv_files = find(folder, pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched pattern '{pattern}' in {folder}")
    return pd.concat((pd.read_csv(file_path) for file_path in csv_files), ignore_index=True)


def copy_structure(src: str | Path, dst: str | Path) -> Path:
    """Copy a directory tree without copying any files.

    Args:
        src: Source directory.
        dst: Destination directory.

    Returns:
        The destination path.

    Example:
        >>> from dsbro.io import copy_structure
        >>> target = copy_structure("input", "output")
        >>> target.exists()
        True
    """
    source = _ensure_path(src)
    if source.is_file():
        raise ValueError("copy_structure expects a directory as src")

    destination = Path(dst).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    for directory in sorted(candidate for candidate in source.rglob("*") if candidate.is_dir()):
        relative = directory.relative_to(source)
        (destination / relative).mkdir(parents=True, exist_ok=True)

    return destination


def sample_files(folder: str | Path, n: int = 5) -> list[Path]:
    """Return a random sample of files from a directory tree.

    Args:
        folder: Folder to sample from.
        n: Number of files to sample.

    Returns:
        A list of sampled file paths.

    Example:
        >>> from dsbro.io import sample_files
        >>> files = sample_files(".", n=1)
        >>> len(files) == 1
        True
    """
    if not isinstance(n, int):
        raise TypeError(f"Expected int for n, got {type(n).__name__}")
    if n <= 0:
        raise ValueError("n must be greater than 0")

    root = _ensure_path(folder)
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        return []
    sample_size = min(n, len(files))
    return random.sample(files, sample_size)


def read_all(folder: str | Path, pattern: str = "*.csv") -> dict[str, Any]:
    """Load every matching file in a directory tree into a dictionary.

    Args:
        folder: Directory to read from.
        pattern: Glob pattern used to select files.

    Returns:
        A mapping of relative file paths to loaded objects.

    Example:
        >>> from dsbro.io import read_all
        >>> mapping = read_all("data")
        >>> isinstance(mapping, dict)
        True
    """
    root = _ensure_path(folder)
    matches = find(root, pattern)
    return {str(file_path.relative_to(root)): load(file_path) for file_path in matches}


def to_kaggle_submission(df: pd.DataFrame, filename: str | Path = "submission.csv") -> Path:
    """Save a DataFrame as a Kaggle-ready CSV submission file.

    Args:
        df: Submission DataFrame.
        filename: Output file path. Must have a ``.csv`` extension.

    Returns:
        The saved CSV file path.

    Example:
        >>> import pandas as pd
        >>> from dsbro.io import to_kaggle_submission
        >>> path = to_kaggle_submission(pd.DataFrame({"id": [1], "target": [0.1]}))
        >>> path.suffix
        '.csv'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Submission DataFrame must not be empty")

    output_path = Path(filename).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".csv")
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Kaggle submissions must be saved as a .csv file")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


__all__ = [
    "copy_structure",
    "fileinfo",
    "find",
    "load",
    "merge_csvs",
    "peek",
    "read_all",
    "sample_files",
    "save",
    "to_kaggle_submission",
    "tree",
]
