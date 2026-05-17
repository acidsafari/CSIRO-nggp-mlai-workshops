"""Static plotting helpers for Workshop 1.

The plotting functions are intentionally explicit and readable. They accept already
prepared arrays and return Matplotlib figures so the notebook can remain lightweight.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nextgen2026-matplotlib")
)

import matplotlib.pyplot as plt

from . import data as data_mod


def _format_axes(ax: plt.Axes, y_min=-1.5, y_max=1.5) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_measurement_world(
    x_train: np.ndarray,
    y_train: np.ndarray,
    noise_std: float | None = None,
    true_x: np.ndarray | None = None,
) -> plt.Figure:
    """Plot noisy observations and the faint true function."""
    if true_x is None:
        true_x = data_mod.make_test_grid((0.0, 1.0), n_points=500)

    fig, ax_main = plt.subplots(figsize=(10.5, 5.8))

    ax_main.plot(
        true_x,
        data_mod.true_function(true_x),
        color="#7a7a7a",
        linewidth=2.0,
        alpha=0.45,
        label="true f(x)",
    )
    ax_main.scatter(
        x_train,
        y_train,
        s=35,
        color="#1f77b4",
        label="observed (x_i, y_i)",
        zorder=3,
    )
    if noise_std is not None:
        ax_main.text(
            0.03,
            0.95,
            f"σ = {noise_std:.3f}",
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
    ax_main.set_title("Noisy Measurement World")
    _format_axes(ax_main, y_min=-1.6, y_max=1.6)
    ax_main.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    return fig


def plot_residual_histogram(
    x_train: np.ndarray,
    y_train: np.ndarray,
    bins: int = 12,
) -> plt.Figure:
    """Plot observation residuals against the hidden signal as a standalone figure."""
    residuals = y_train - data_mod.true_function(x_train)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.hist(residuals, bins=bins, color="#4c78a8", alpha=0.9, edgecolor="#ffffff")
    ax.set_title("Observation residuals")
    ax.set_xlabel("y - f(x)")
    ax.set_ylabel("count")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    return fig


__all__ = [
    "plot_measurement_world",
    "plot_residual_histogram",
]
