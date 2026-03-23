"""Internal helpers used across the dsbro package."""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
from pathlib import Path
from typing import Any


def _ensure_path(path: str | Path, *, exists: bool = True) -> Path:
    """Return a normalized Path and optionally validate existence."""
    normalized = Path(path).expanduser()
    if exists and not normalized.exists():
        raise FileNotFoundError(f"Path does not exist: {normalized}")
    return normalized


def _format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable size string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def _safe_import(module_name: str, install_hint: str | None = None) -> Any:
    """Import a module and raise a helpful message when missing."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        message = f"Optional dependency '{module_name}' is required."
        if install_hint:
            message = f"{message} Install with: {install_hint}"
        raise ImportError(message) from exc


def _detect_text_encoding(path: Path) -> str | None:
    """Best-effort encoding detection for small text files."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            path.read_text(encoding=encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return None


def _count_lines(path: Path, encoding: str | None = None) -> int | None:
    """Count lines in a text file."""
    encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin-1"]
    for candidate in encodings:
        try:
            with path.open("r", encoding=candidate) as handle:
                return sum(1 for _ in handle)
        except UnicodeDecodeError:
            continue
    return None


def _get_package_version(name: str) -> str | None:
    """Return an installed package version or None."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _get_total_memory_bytes() -> int | None:
    """Return total system memory in bytes without requiring psutil."""
    if hasattr(os, "sysconf"):
        page_size = os.sysconf_names.get("SC_PAGE_SIZE")
        page_count = os.sysconf_names.get("SC_PHYS_PAGES")
        if page_size is not None and page_count is not None:
            try:
                return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            except (OSError, ValueError):
                pass

    if platform.system() == "Windows":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys)

    return None


def _print_header(title: str, width: int = 50) -> None:
    """Print a clean section header."""
    print(f"\n{'.' * width}")
    print(f"  {title}")
    print(f"{'.' * width}\n")


def _print_sub_header(title: str, width: int = 50) -> None:
    """Print a sub-section header."""
    del width
    print(f"\n  --- {title} ---\n")


def _print_divider(width: int = 50) -> None:
    """Print a simple divider line."""
    print(f"{'.' * width}")


def _print_kv(key: str, value: Any, key_width: int = 15) -> None:
    """Print a key-value pair aligned."""
    print(f"  {key:<{key_width}} {value}")


def _print_dataframe(df: Any) -> None:
    """Print a DataFrame-like object without wrapping columns."""
    try:
        import pandas as pd
    except ImportError:
        print(df)
        return

    if isinstance(df, pd.DataFrame):
        with pd.option_context(
            "display.width",
            200,
            "display.max_columns",
            None,
            "display.max_colwidth",
            20,
        ):
            print(df.to_string())
        return

    print(df)
