"""Reusable helpers for the Data Space notebook."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SEED = 7
theta_grid = np.linspace(0, 90, 500)

COLORS = {
    "data": "#2F5D7C",
    "truth": "#222222",
    "fit": "#C7502A",
    "alt": "#7B5E9E",
    "support": "#D9D9D9",
    "context0": "#2F5D7C",
    "context1": "#D18F24",
}


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Return a deterministic NumPy random generator."""
    return np.random.default_rng(seed)


def f0(theta: np.ndarray | float) -> np.ndarray:
    """Latent tilt-power response with a broad peak and narrow feature."""
    theta = np.asarray(theta)
    broad = np.exp(-((theta - 40) ** 2) / (2 * 15**2))
    narrow = 0.15 * np.exp(-((theta - 65) ** 2) / (2 * 3**2))
    return broad + narrow


def context_scale(c: np.ndarray | int) -> np.ndarray:
    """Return the multiplicative response scale for each binary context."""
    return np.where(np.asarray(c) == 1, 1.25, 0.75)


def context_probability(theta: np.ndarray | float, mode: str = "independent") -> np.ndarray:
    """Return P(C=1 | theta) under a named context assignment mode."""
    theta = np.asarray(theta)
    if mode == "independent":
        return np.full_like(theta, 0.5, dtype=float)
    if mode == "increasing":
        return 1 / (1 + np.exp(-(theta - 45) / 8))
    if mode == "reversed":
        return 1 / (1 + np.exp((theta - 45) / 8))
    raise ValueError(f"unknown context mode: {mode}")


def sample_theta(
    n: int,
    sampling: str = "uniform",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample latent true tilt values under a named sampling design."""
    rng = set_seed() if rng is None else rng
    if sampling == "uniform":
        theta = rng.uniform(0, 90, n)
    elif sampling == "gap65":
        theta_values: list[float] = []
        while len(theta_values) < n:
            draw = rng.uniform(0, 90, n)
            draw = draw[(draw < 58) | (draw > 72)]
            theta_values.extend(draw.tolist())
        theta = np.asarray(theta_values[:n])
    elif sampling == "cluster40":
        mix = rng.uniform(size=n)
        theta = np.where(mix < 0.75, rng.normal(40, 9, n), rng.uniform(0, 90, n))
        theta = np.clip(theta, 0, 90)
    elif sampling == "restricted":
        theta = rng.uniform(20, 60, n)
    elif sampling == "deployment65":
        theta = np.clip(rng.normal(65, 7, n), 0, 90)
    elif sampling == "sparse_feature":
        theta_values = []
        while len(theta_values) < n:
            draw = rng.uniform(0, 90, n)
            keep_prob = np.where((draw >= 58) & (draw <= 72), 0.18, 1.0)
            theta_values.extend(draw[rng.uniform(size=n) < keep_prob].tolist())
        theta = np.asarray(theta_values[:n])
    elif sampling == "dense_feature":
        mix = rng.uniform(size=n)
        theta = np.where(mix < 0.45, rng.normal(65, 4, n), rng.uniform(0, 90, n))
        theta = np.clip(theta, 0, 90)
    else:
        raise ValueError(f"unknown sampling pattern: {sampling}")
    return np.asarray(theta)


def sample_tilt_power(
    n: int = 120,
    scenario: str = "clean",
    seed: int = 0,
    x_noise: float = 0.0,
    y_noise: float = 0.05,
    context_strength: float = 0.35,
    sampling: str = "uniform",
    context_mode: str = "independent",
) -> dict[str, np.ndarray | str]:
    """Generate observed tilt-power samples from one latent physical curve."""
    rng = set_seed(seed)
    theta = sample_theta(n, sampling=sampling, rng=rng)
    if scenario == "clean":
        x_noise = 0.0
        y_noise = 0.0
        context_mode = "none"
    elif scenario == "output_noise":
        x_noise = 0.0
    elif scenario == "input_noise":
        x_noise = max(x_noise, 4.0)
    elif scenario == "hidden_context":
        pass
    elif scenario == "confounded_context":
        context_mode = "increasing"
    elif scenario == "confounded_reversed":
        context_mode = "reversed"

    if context_mode == "none":
        c = np.zeros(n, dtype=int)
        scale = np.ones(n)
    else:
        p = context_probability(theta, context_mode)
        c = (rng.uniform(size=n) < p).astype(int)
        scale = 1 + context_strength * np.where(c == 1, 1.0, -1.0)

    x = np.clip(theta + rng.normal(0, x_noise, n), 0, 90)
    y = scale * f0(theta) + rng.normal(0, y_noise, n)
    return {
        "theta": theta,
        "x": x,
        "y": y,
        "c": c,
        "scale": scale,
        "scenario": scenario,
        "sampling": sampling,
    }


def make_axis(
    title: str,
    xlabel: str = "observed tilt x",
    ylabel: str = "observed power y",
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Create a consistently labelled Matplotlib axis."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def style_xy_axis(ax: plt.Axes) -> None:
    """Apply the shared tilt-power axis bounds and grid."""
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.15, 1.55)
    ax.grid(alpha=0.2)


def plot_reference(ax: plt.Axes, label: str = "latent f0(theta)") -> None:
    """Plot the shared latent reference curve."""
    ax.plot(theta_grid, f0(theta_grid), color=COLORS["truth"], lw=2, label=label)


def plot_observed(
    ax: plt.Axes,
    data: dict[str, Any],
    reveal_context: bool = False,
    s: float = 28,
    alpha: float = 0.78,
) -> None:
    """Plot observed samples, optionally revealing the hidden binary context."""
    if reveal_context:
        for c, color, label in [
            (0, COLORS["context0"], "C=0"),
            (1, COLORS["context1"], "C=1"),
        ]:
            mask = data["c"] == c
            ax.scatter(
                data["x"][mask],
                data["y"][mask],
                s=s,
                alpha=alpha,
                color=color,
                label=label,
                edgecolor="white",
                linewidth=0.3,
            )
    else:
        ax.scatter(
            data["x"],
            data["y"],
            s=s,
            alpha=alpha,
            color=COLORS["data"],
            label="observed (X,Y)",
            edgecolor="white",
            linewidth=0.3,
        )


def local_count(x_grid: np.ndarray, x_obs: np.ndarray, radius: float) -> np.ndarray:
    """Count observations within radius of each grid point."""
    return np.array([np.sum(np.abs(x_obs - x) <= radius) for x in x_grid])


def local_mean(
    x_grid: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Estimate local conditional means within a fixed radius."""
    out = np.full_like(x_grid, np.nan, dtype=float)
    for i, x in enumerate(x_grid):
        mask = np.abs(x_obs - x) <= radius
        if np.any(mask):
            out[i] = np.mean(y_obs[mask])
    return out


def local_std(
    x_grid: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Estimate local conditional standard deviations within a fixed radius."""
    out = np.full_like(x_grid, np.nan, dtype=float)
    for i, x in enumerate(x_grid):
        mask = np.abs(x_obs - x) <= radius
        if np.sum(mask) >= 2:
            out[i] = np.std(y_obs[mask], ddof=1)
    return out


def local_variation(mu_grid: np.ndarray, x_grid: np.ndarray, radius: float) -> np.ndarray:
    """Compute local variation of a reference curve over a radius window."""
    out = np.full_like(x_grid, np.nan, dtype=float)
    for i, x in enumerate(x_grid):
        mask = np.abs(x_grid - x) <= radius
        if np.any(mask):
            out[i] = np.nanmax(mu_grid[mask]) - np.nanmin(mu_grid[mask])
    return out


def gaussian_smoother(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    bandwidth: float = 7.0,
) -> np.ndarray:
    """Predict with a Gaussian kernel smoother."""
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_eval = np.asarray(x_eval)
    weights = np.exp(-0.5 * ((x_eval[:, None] - x_train[None, :]) / bandwidth) ** 2)
    denom = np.sum(weights, axis=1)
    return (weights @ y_train) / np.maximum(denom, 1e-12)


def poly_ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 7,
    lam: float = 1e-2,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a polynomial ridge regressor and return its prediction function."""
    z = (np.asarray(x) - 45) / 45
    design = np.vander(z, degree + 1, increasing=True)
    penalty = lam * np.eye(degree + 1)
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)

    def predict(x_new: np.ndarray) -> np.ndarray:
        z_new = (np.asarray(x_new) - 45) / 45
        design_new = np.vander(z_new, degree + 1, increasing=True)
        return design_new @ coef

    return predict


def configure_matplotlib() -> None:
    """Apply shared notebook plotting defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
