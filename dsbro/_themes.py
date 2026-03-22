"""Theme definitions shared across dsbro modules."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

DEFAULT_THEME = "dark"

THEMES: dict[str, dict[str, Any]] = {
    "dark": {
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#1a1a2e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "axes.titlecolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "grid.color": "#3b3b58",
        "axes.prop_cycle": plt.cycler(
            color=["#00d4ff", "#00ff88", "#ff6b6b", "#ffd93d", "#c084fc"]
        ),
    },
    "light": {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#1f2933",
        "axes.labelcolor": "#1f2933",
        "axes.titlecolor": "#1f2933",
        "xtick.color": "#1f2933",
        "ytick.color": "#1f2933",
        "text.color": "#1f2933",
        "grid.color": "#d9e2ec",
        "axes.prop_cycle": plt.cycler(
            color=["#2563eb", "#059669", "#dc2626", "#d97706", "#7c3aed"]
        ),
    },
    "paper": {
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#0f172a",
        "axes.labelcolor": "#0f172a",
        "axes.titlecolor": "#0f172a",
        "xtick.color": "#0f172a",
        "ytick.color": "#0f172a",
        "text.color": "#0f172a",
        "grid.color": "#cbd5e1",
        "axes.prop_cycle": plt.cycler(
            color=["#0f766e", "#2563eb", "#be123c", "#ea580c", "#4f46e5"]
        ),
    },
    "kaggle": {
        "figure.facecolor": "#111827",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#f9fafb",
        "axes.labelcolor": "#f9fafb",
        "axes.titlecolor": "#f9fafb",
        "xtick.color": "#f9fafb",
        "ytick.color": "#f9fafb",
        "text.color": "#f9fafb",
        "grid.color": "#374151",
        "axes.prop_cycle": plt.cycler(
            color=["#20beff", "#34d399", "#fb7185", "#fbbf24", "#a78bfa"]
        ),
    },
    "neon": {
        "figure.facecolor": "#050816",
        "axes.facecolor": "#050816",
        "axes.edgecolor": "#f8fafc",
        "axes.labelcolor": "#f8fafc",
        "axes.titlecolor": "#f8fafc",
        "xtick.color": "#f8fafc",
        "ytick.color": "#f8fafc",
        "text.color": "#f8fafc",
        "grid.color": "#1e293b",
        "axes.prop_cycle": plt.cycler(
            color=["#00f5d4", "#f15bb5", "#fee440", "#00bbf9", "#9b5de5"]
        ),
    },
}


def get_theme(style: str = DEFAULT_THEME) -> dict[str, Any]:
    """Return a named theme configuration."""
    try:
        return THEMES[style].copy()
    except KeyError as exc:
        available = ", ".join(sorted(THEMES))
        raise ValueError(f"Unknown theme '{style}'. Available themes: {available}") from exc


def apply_matplotlib_theme(style: str = DEFAULT_THEME) -> dict[str, Any]:
    """Apply a theme to matplotlib and return the applied rcParams."""
    theme = get_theme(style)
    base_params = {
        "axes.grid": True,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "figure.autolayout": False,
    }
    applied = {**base_params, **theme}
    plt.rcParams.update(applied)
    return applied

