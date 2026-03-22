"""General utility functions for dsbro."""

from __future__ import annotations

import os
import platform
import random
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from dsbro._helpers import _format_size, _get_package_version, _get_total_memory_bytes, _safe_import
from dsbro._themes import apply_matplotlib_theme

T = TypeVar("T")
U = TypeVar("U")


def setup(
    dark: bool = True,
    max_columns: int | None = None,
    max_rows: int = 100,
    precision: int = 4,
    style: str | None = None,
) -> dict[str, Any]:
    """Configure a notebook-friendly dsbro environment.

    Args:
        dark: Whether to apply the dark plotting theme by default.
        max_columns: Maximum columns displayed by pandas. ``None`` shows all columns.
        max_rows: Maximum rows displayed by pandas.
        precision: Floating-point display precision for pandas.
        style: Explicit theme name. Overrides ``dark`` when provided.

    Returns:
        A dictionary summarizing the applied configuration.

    Example:
        >>> import dsbro
        >>> dsbro.setup()
        {'theme': 'dark', 'max_columns': None, 'max_rows': 100, ...}
    """
    selected_style = style or ("dark" if dark else "light")
    suppress_warnings()

    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.precision", precision)
    pd.set_option("display.width", 120)

    apply_matplotlib_theme(selected_style)
    sns = _safe_import("seaborn")
    sns.set_theme(style="darkgrid" if selected_style != "light" else "whitegrid")

    autoreload_enabled = False
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("load_ext", "autoreload")
            shell.run_line_magic("autoreload", "2")
            autoreload_enabled = True
    except Exception:
        autoreload_enabled = False

    return {
        "theme": selected_style,
        "max_columns": max_columns,
        "max_rows": max_rows,
        "precision": precision,
        "warnings_suppressed": True,
        "autoreload_enabled": autoreload_enabled,
    }


def notebook_setup(dark: bool = True) -> dict[str, Any]:
    """Alias for :func:`setup` with notebook-oriented defaults.

    Args:
        dark: Whether to use the dark theme.

    Returns:
        The configuration returned by :func:`setup`.

    Example:
        >>> from dsbro.utils import notebook_setup
        >>> notebook_setup(dark=False)["theme"]
        'light'
    """
    return setup(dark=dark)


def seed(n: int = 42) -> int:
    """Seed common random number generators.

    Args:
        n: Seed value.

    Returns:
        The seed that was applied.

    Example:
        >>> from dsbro.utils import seed
        >>> seed(123)
        123
    """
    if not isinstance(n, int):
        raise TypeError(f"Expected int for n, got {type(n).__name__}")

    os.environ["PYTHONHASHSEED"] = str(n)
    random.seed(n)
    np.random.seed(n)

    try:
        torch = _safe_import("torch")
        torch.manual_seed(n)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(n)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        tf = _safe_import("tensorflow")
        tf.random.set_seed(n)
    except ImportError:
        pass

    return n


@contextmanager
def timer(label: str = "Elapsed time") -> Iterator[dict[str, float]]:
    """Measure elapsed wall-clock time for a code block.

    Args:
        label: Label shown when timing output is printed.

    Returns:
        An iterator yielding a mutable result dictionary with an ``elapsed`` key after exit.

    Example:
        >>> from dsbro.utils import timer
        >>> with timer("training") as result:
        ...     _ = sum(range(1000))
        training: 0.000s
        >>> result["elapsed"] >= 0
        True
    """
    result: dict[str, float] = {}
    started = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - started
        print(f"{label}: {result['elapsed']:.3f}s")


def gpu_info() -> dict[str, Any]:
    """Return basic information about the available GPU, if any.

    Args:
        None.

    Returns:
        A dictionary describing GPU availability and memory details.

    Example:
        >>> from dsbro.utils import gpu_info
        >>> info = gpu_info()
        >>> "available" in info
        True
    """
    info: dict[str, Any] = {"available": False, "backend": None}

    try:
        torch = _safe_import("torch")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            memory_allocated = torch.cuda.memory_allocated(0) // (1024**2)
            total_memory = properties.total_memory // (1024**2)
            info.update(
                {
                    "available": True,
                    "backend": "cuda",
                    "name": properties.name,
                    "total_memory_mb": int(total_memory),
                    "used_memory_mb": int(memory_allocated),
                    "free_memory_mb": int(total_memory - memory_allocated),
                    "device_count": int(torch.cuda.device_count()),
                }
            )
            return info
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info.update({"available": True, "backend": "mps", "name": "Apple Metal"})
            return info
    except ImportError:
        pass

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            line = result.stdout.strip().splitlines()[0]
            name, total, used, free = [part.strip() for part in line.split(",")]
            info.update(
                {
                    "available": True,
                    "backend": "cuda",
                    "name": name,
                    "total_memory_mb": int(total),
                    "used_memory_mb": int(used),
                    "free_memory_mb": int(free),
                    "device_count": len(result.stdout.strip().splitlines()),
                }
            )
        except (IndexError, OSError, subprocess.SubprocessError, ValueError):
            pass

    return info


def system_info() -> dict[str, Any]:
    """Collect system and package metadata useful in notebook environments.

    Args:
        None.

    Returns:
        A dictionary containing Python, OS, disk, memory, and package version details.

    Example:
        >>> from dsbro.utils import system_info
        >>> info = system_info()
        >>> "python_version" in info
        True
    """
    cwd_anchor = Path.cwd().anchor or str(Path.cwd())
    disk_total, _, disk_free = shutil.disk_usage(cwd_anchor)
    total_memory = _get_total_memory_bytes()
    packages = {
        "numpy": _get_package_version("numpy"),
        "pandas": _get_package_version("pandas"),
        "matplotlib": _get_package_version("matplotlib"),
        "seaborn": _get_package_version("seaborn"),
        "sklearn": _get_package_version("sklearn"),
    }

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory,
        "total_memory": _format_size(total_memory) if total_memory is not None else None,
        "disk_total_bytes": disk_total,
        "disk_total": _format_size(disk_total),
        "disk_free_bytes": disk_free,
        "disk_free": _format_size(disk_free),
        "packages": packages,
    }


def show_versions() -> dict[str, str | None]:
    """Print and return installed versions of common data-science packages.

    Args:
        None.

    Returns:
        A mapping of package names to version strings.

    Example:
        >>> from dsbro.utils import show_versions
        >>> versions = show_versions()
        >>> "numpy" in versions
        True
    """
    versions = {
        "python": platform.python_version(),
        "numpy": _get_package_version("numpy"),
        "pandas": _get_package_version("pandas"),
        "matplotlib": _get_package_version("matplotlib"),
        "seaborn": _get_package_version("seaborn"),
        "sklearn": _get_package_version("sklearn"),
    }
    for name, version_value in versions.items():
        print(f"{name}: {version_value}")
    return versions


def suppress_warnings() -> None:
    """Suppress Python warnings globally.

    Args:
        None.

    Returns:
        None.

    Example:
        >>> from dsbro.utils import suppress_warnings
        >>> suppress_warnings()
    """
    warnings.filterwarnings("ignore")


def flatten(nested_list: Iterable[Any]) -> list[Any]:
    """Flatten arbitrarily nested iterables into a single list.

    Args:
        nested_list: An iterable that may contain nested iterables.

    Returns:
        A flat list of values.

    Example:
        >>> from dsbro.utils import flatten
        >>> flatten([1, [2, (3, 4)]])
        [1, 2, 3, 4]
    """
    if isinstance(nested_list, (str, bytes)):
        return [nested_list]

    flattened: list[Any] = []
    for item in nested_list:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes, dict, Path)):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened


def chunk(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield fixed-size chunks from an iterable.

    Args:
        iterable: Values to split into chunks.
        size: Number of items per chunk.

    Returns:
        An iterator of lists, each with up to ``size`` elements.

    Example:
        >>> from dsbro.utils import chunk
        >>> list(chunk([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    if not isinstance(size, int):
        raise TypeError(f"Expected int for size, got {type(size).__name__}")
    if size <= 0:
        raise ValueError("size must be greater than 0")

    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


def parallelize(func: Callable[[T], U], iterable: Iterable[T], n_jobs: int = -1) -> list[U]:
    """Apply a function to an iterable in parallel using joblib.

    Args:
        func: Single-argument callable applied to each item.
        iterable: Values passed to ``func``.
        n_jobs: Number of worker processes. ``-1`` uses all available workers.

    Returns:
        A list of results produced by ``func``.

    Example:
        >>> from dsbro.utils import parallelize
        >>> parallelize(lambda x: x * 2, [1, 2, 3], n_jobs=1)
        [2, 4, 6]
    """
    if not callable(func):
        raise TypeError("func must be callable")
    if not isinstance(n_jobs, int):
        raise TypeError(f"Expected int for n_jobs, got {type(n_jobs).__name__}")

    joblib = _safe_import("joblib", "pip install dsbro[dev]")
    return joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(func)(item) for item in iterable)


def download(url: str, path: str | Path | None = None, chunk_size: int = 8192) -> Path:
    """Download a file to disk.

    Args:
        url: Source URL to download.
        path: Optional output path. When omitted, the file name is inferred from the URL.
        chunk_size: Number of bytes read per chunk.

    Returns:
        The downloaded file path.

    Example:
        >>> from dsbro.utils import download
        >>> target = download("file:///tmp/example.txt", "copy.txt")
        >>> target.name
        'copy.txt'
    """
    if not isinstance(url, str):
        raise TypeError(f"Expected str for url, got {type(url).__name__}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    parsed = urllib.parse.urlparse(url)
    inferred_name = Path(parsed.path).name or "downloaded.file"
    target = Path(path) if path is not None else Path(inferred_name)
    target.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        while True:
            chunk_bytes = response.read(chunk_size)
            if not chunk_bytes:
                break
            handle.write(chunk_bytes)

    return target


__all__ = [
    "chunk",
    "download",
    "flatten",
    "gpu_info",
    "notebook_setup",
    "parallelize",
    "seed",
    "setup",
    "show_versions",
    "suppress_warnings",
    "system_info",
    "timer",
]
