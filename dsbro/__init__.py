"""Top-level package exports for dsbro."""

from __future__ import annotations

import inspect
from typing import Any

from dsbro import eda, io, metrics, ml, prep, text, utils, viz
from dsbro._version import __version__
from dsbro.utils import setup

_MODULES = {
    "io": io,
    "eda": eda,
    "prep": prep,
    "viz": viz,
    "ml": ml,
    "metrics": metrics,
    "utils": utils,
    "text": text,
}

_MODULE_SUMMARIES = {
    "io": "File and directory utilities",
    "eda": "Exploratory data analysis helpers",
    "prep": "Preprocessing and feature engineering",
    "viz": "Visualization helpers with dsbro theming",
    "ml": "Model comparison, training, tuning, and ensembles",
    "metrics": "Classification and regression metrics",
    "utils": "Notebook and environment utilities",
    "text": "Text cleaning and NLP-style helpers",
}


def version() -> str:
    """Return the installed dsbro version string."""
    return __version__


def about() -> str:
    """Print and return basic project metadata."""
    message = (
        f"dsbro {__version__}\n"
        "Your Data Science Bro. One import away.\n"
        "Author: Muhammad Ibrahim Qasmi\n"
        "Homepage: https://ibrahimqasmi.com\n"
        "GitHub: https://github.com/muhammadibrahim313/dsbro"
    )
    print(message)
    return message


def help(topic: str | None = None) -> str:
    """Print a categorized help summary for dsbro modules or functions."""
    if topic is None:
        lines = ["dsbro available modules:"]
        for module_name, module in _MODULES.items():
            lines.append(f"- {module_name}: {_MODULE_SUMMARIES.get(module_name, '')}")
            for function_name in _public_functions(module):
                function = getattr(module, function_name)
                signature = inspect.signature(function)
                summary = _first_line(inspect.getdoc(function))
                lines.append(f"  {function_name}{signature}: {summary}")
        message = "\n".join(lines)
        print(message)
        return message

    normalized = topic.strip().lower()
    if normalized in _MODULES:
        module = _MODULES[normalized]
        lines = [f"dsbro.{normalized} - {_MODULE_SUMMARIES.get(normalized, '')}"]
        for function_name in _public_functions(module):
            function = getattr(module, function_name)
            signature = inspect.signature(function)
            summary = _first_line(inspect.getdoc(function))
            lines.append(f"- {function_name}{signature}: {summary}")
        message = "\n".join(lines)
        print(message)
        return message

    for module_name, module in _MODULES.items():
        if hasattr(module, normalized):
            function = getattr(module, normalized)
            message = inspect.getdoc(function) or (
                f"No help text available for dsbro.{module_name}.{normalized}."
            )
            print(message)
            return message

    raise ValueError(f"Unknown help topic: {topic}")


def _public_functions(module: Any) -> list[str]:
    """Return public functions defined directly in a module."""
    names: list[str] = []
    for name, member in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if inspect.getmodule(member) is module:
            names.append(name)
    return sorted(names)


def _first_line(docstring: str | None) -> str:
    """Return the first line of a docstring."""
    if not docstring:
        return ""
    return docstring.strip().splitlines()[0]


__all__ = [
    "__version__",
    "about",
    "eda",
    "help",
    "io",
    "metrics",
    "ml",
    "prep",
    "setup",
    "text",
    "utils",
    "version",
    "viz",
]
