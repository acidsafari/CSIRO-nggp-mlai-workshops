"""Reusable helpers for the Hypothesis Space II notebook."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from .data_space import (
    COLORS,
    f0,
    local_count,
    plot_observed,
    plot_reference,
    sample_tilt_power,
    style_xy_axis,
    theta_grid,
)

RAW_POLYNOMIAL_MAX_DEGREE = 9
DEPTH_W1_BASE_RAW = np.array([[0.055556, -0.044444, 0.066667]])
DEPTH_B1_BASE_RAW = np.array([-1.35, 2.60, -3.85])


def relu(z: np.ndarray | float) -> np.ndarray:
    """Evaluate the ReLU activation."""
    return np.maximum(0.0, z)


def validate_raw_polynomial_degree(degree: int) -> int:
    """Validate the raw-x polynomial degree used in notebook demonstrations."""
    degree = int(degree)
    if degree < 0 or degree > RAW_POLYNOMIAL_MAX_DEGREE:
        raise ValueError(
            "raw-x polynomial demonstrations use degree between "
            f"0 and {RAW_POLYNOMIAL_MAX_DEGREE}; higher degrees usually need "
            "an explicit scaling basis."
        )
    return degree


def polynomial_design(x: np.ndarray | float, degree: int) -> np.ndarray:
    """Return a raw-input polynomial design matrix with columns 1, x, ..., x^degree."""
    degree = validate_raw_polynomial_degree(degree)
    x = np.asarray(x)
    return np.vstack([x**k for k in range(degree + 1)]).T


def periodic_design(
    x: np.ndarray | float,
    frequency_count: int,
    period: float,
) -> np.ndarray:
    """Return a sine/cosine basis with an explicit period in raw x units."""
    z = np.asarray(x) / float(period)
    columns = [np.ones_like(z)]
    for k in range(1, int(frequency_count) + 1):
        columns.extend([np.sin(2 * np.pi * k * z), np.cos(2 * np.pi * k * z)])
    return np.vstack(columns).T


def relu_basis_design(
    x: np.ndarray | float,
    kinks: np.ndarray,
    include_linear: bool = True,
) -> np.ndarray:
    """Return a fixed-knot one-dimensional ReLU basis in raw x units."""
    x = np.asarray(x)
    kinks = np.asarray(kinks)
    columns = [np.ones_like(x)]
    if include_linear:
        columns.append(x)
    columns.extend([relu(x - knot) for knot in kinks])
    return np.vstack(columns).T


def ridge_fit(
    Phi: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-6,
    penalize_intercept: bool = False,
) -> np.ndarray:
    """Fit ridge coefficients for a supplied design matrix."""
    p = Phi.shape[1]
    penalty = float(lam) * np.eye(p)
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    return np.linalg.solve(Phi.T @ Phi + penalty, Phi.T @ y)


def fit_basis_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    design_fn: Callable[[np.ndarray], np.ndarray],
    lam: float = 1e-6,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """Fit a ridge model for a supplied basis and return predictor, coefficients."""
    coef = ridge_fit(design_fn(x_train), y_train, lam=lam)

    def predict(x_new: np.ndarray) -> np.ndarray:
        return design_fn(x_new) @ coef

    return predict, coef


def eval_one_hidden_relu(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    c: float = 0.0,
) -> np.ndarray:
    """Evaluate c + sum_j v_j ReLU(w_j x + b_j)."""
    A = relu(np.outer(np.asarray(x), np.asarray(w)) + np.asarray(b))
    return float(c) + A @ np.asarray(v)


def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Return mean squared error as a plain float."""
    return float(np.mean((np.asarray(y_hat) - np.asarray(y)) ** 2))


def shade_gap(ax: plt.Axes, label: str = "unsupported gap") -> None:
    """Shade the reused data-gap region from the Data Space notebook."""
    ax.axvspan(58, 72, color=COLORS["support"], alpha=0.32, lw=0, label=label)


def coefficient_vector(
    degree: int,
    preset: str,
    coefficient_scale: float,
) -> np.ndarray:
    """Create deterministic polynomial coefficients for a named preset."""
    degree = validate_raw_polynomial_degree(degree)
    coefficient_scale = float(coefficient_scale)
    k = np.arange(degree + 1, dtype=float)
    if preset == "smooth":
        shape = 0.55 * np.exp(-0.35 * k) * np.cos(1.2 * k)
    elif preset == "oscillating":
        shape = 0.50 * np.exp(-0.18 * k) * ((-1.0) ** k)
    elif preset == "tilted":
        shape = 0.35 * np.exp(-0.28 * k) * np.sin(0.9 * (k + 1))
        if degree >= 1:
            shape[1] += 0.45
    else:
        raise ValueError(f"unknown coefficient preset: {preset}")
    coef = shape / (coefficient_scale**k)
    coef[0] += 0.45
    return coef


def fixed_vs_learned_action_rows() -> list[list[str]]:
    """Return action rows used in the fixed-feature versus learned-feature demo."""
    return [
        ["change polynomial coefficients", "selected h inside fixed H_phi"],
        ["change polynomial degree", "redefine H"],
        ["change ReLU hidden weights or biases", "move within learned-feature H_NN"],
        ["change ReLU width", "redefine H"],
    ]


def plot_fixed_vs_learned(
    degree: int = 3,
    coefficient_preset: str = "smooth",
    coefficient_scale: float = 45.0,
    num_relu_features: int = 3,
    kink_shift: float = 0.0,
    output_scale: float = 1.0,
) -> None:
    """Plot one fixed-feature function and one learned-feature ReLU function."""
    degree = validate_raw_polynomial_degree(degree)
    theta = coefficient_vector(degree, coefficient_preset, coefficient_scale)
    fixed_y = polynomial_design(theta_grid, degree) @ theta

    num_relu_features = max(1, int(num_relu_features))
    if num_relu_features == 1:
        base_kinks = np.array([45.0])
    elif num_relu_features == 3:
        base_kinks = np.array([24.0, 45.0, 67.0])
    else:
        base_kinks = np.linspace(18.0, 78.0, num_relu_features)
    kinks = np.clip(base_kinks + float(kink_shift), 0.0, 90.0)
    w = np.full(len(kinks), 1 / 18.0)
    b = -w * kinks
    if num_relu_features == 3:
        v = np.array([0.50, -0.85, 0.70])
    else:
        signs = np.where(np.arange(num_relu_features) % 2 == 0, 1.0, -1.0)
        envelope = 0.42 + 0.28 * np.cos(np.arange(num_relu_features) * 0.8) ** 2
        v = signs * envelope
    v = float(output_scale) * v
    learned_y = eval_one_hidden_relu(theta_grid, w, b, v, c=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    ax = axes[0]
    ax.plot(theta_grid, fixed_y, color=COLORS["data"], lw=2.8, label=f"degree {degree}")
    ax.set_title("Fixed features: choose coefficients or degree")
    ax.set_xlabel("x")
    ax.set_ylabel("h(x)")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.35, 1.45)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(theta_grid, learned_y, color=COLORS["fit"], lw=2.8, label="one hidden layer")
    for kink in kinks:
        ax.axvline(kink, color="0.55", lw=0.9, ls=":")
        ax.text(kink, 1.33, "kink", rotation=90, va="top", ha="right", fontsize=7)
    ax.set_title("Learned features: move hidden kinks")
    ax.set_xlabel("x")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.35, 1.45)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    print(f"Raw polynomial coefficients theta = {np.array2string(theta, precision=6)}")
    print(f"Coefficient initialisation scale = {float(coefficient_scale):.1f}; raw x is not scaled.")
    print(f"ReLU feature count = {num_relu_features}")
    print(f"Raw ReLU hidden weights w = {np.array2string(w, precision=6)}")
    print(f"Raw ReLU hidden biases b = {np.array2string(b, precision=6)}")
    print(f"ReLU kink locations = {np.array2string(kinks, precision=3)}")
    print(f"Raw ReLU output weights v = {np.array2string(v, precision=6)}, c = {0.25:.6f}")
    print("What changed: coefficients move within H; degree or ReLU feature count redefines H.")
    print("Assumption: useful features are either chosen in advance or learned as movable ReLU kinks.")


def plot_relu_neuron(
    w: float,
    b: float,
    show_preactivation: bool = True,
    show_activation: bool = True,
) -> None:
    """Plot one ReLU neuron using an explicit raw-x weight and bias."""
    pre = float(w) * theta_grid + float(b)
    act = relu(pre)
    kink = -float(b) / float(w) if abs(float(w)) > 1e-12 else np.nan

    panels = []
    if show_preactivation:
        panels.append("preactivation")
    if show_activation:
        panels.append("activation")
    if not panels:
        panels = ["diagnostic"]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        if panel == "preactivation":
            ax.plot(theta_grid, pre, color=COLORS["data"], lw=2.2, label="$wx+b$")
            ax.axhline(0, color="0.25", lw=1, ls="--")
            ax.set_ylabel("preactivation")
            ax.set_title("Linear preactivation")
        elif panel == "activation":
            ax.plot(theta_grid, act, color=COLORS["fit"], lw=2.6, label="ReLU($wx+b$)")
            ax.set_ylabel("activation")
            ax.set_title("Activated learned feature")
        else:
            ax.text(0.5, 0.5, "Enable a plot toggle", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("No curve selected")
        if np.isfinite(kink):
            ax.axvline(kink, color=COLORS["fit"], lw=1.5, ls=":", label=f"$x^*$={kink:.1f}")
        ax.set_xlabel("x")
        ax.set_xlim(0, 90)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    plt.show()

    if abs(float(w)) <= 1e-12:
        active_side = "all x if b > 0, otherwise no x"
    elif float(w) > 0:
        active_side = f"x > {kink:.2f}"
    else:
        active_side = f"x < {kink:.2f}"
    print(f"Raw ReLU weight w = {float(w):.6f}, bias b = {float(b):.6f}")
    print(f"kink x* = {kink:.2f}" if np.isfinite(kink) else "no finite kink when w=0")
    print(f"Active side: {active_side}")
    print("What changed: editing w or b moves the realised one-unit function within this H.")
    print("Assumption: one movable piecewise-linear change point is available.")


def _resize_vector(values: list[float] | np.ndarray, defaults: np.ndarray, name: str) -> np.ndarray:
    """Return a one-dimensional vector resized to match default length."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return defaults
    if arr.size > defaults.size:
        return arr[: defaults.size]
    if arr.size < defaults.size:
        out = defaults.copy()
        out[: arr.size] = arr
        print(f"{name} had {arr.size} values; filled remaining entries with deterministic defaults.")
        return out
    return arr


def plot_relu_layer(
    num_units: int = 3,
    w: list[float] | np.ndarray = (0.08, 0.08, 0.08),
    b: list[float] | np.ndarray = (-2.4, -3.6, -5.0),
    v: list[float] | np.ndarray = (0.5, -0.8, 0.6),
    c: float = 0.0,
    show_units: bool = True,
    show_weighted_units: bool = True,
    show_sum: bool = True,
) -> None:
    """Plot hidden activations, weighted contributions, and their one-layer sum."""
    num_units = max(1, int(num_units))
    default_w = np.full(num_units, 0.08)
    default_b = -default_w * np.linspace(30.0, 70.0, num_units)
    default_v = np.where(np.arange(num_units) % 2 == 0, 0.6, -0.75)
    w = _resize_vector(w, default_w, "w")
    b = _resize_vector(b, default_b, "b")
    v = _resize_vector(v, default_v, "v")

    activations = relu(np.outer(theta_grid, w) + b)
    contributions = activations * v
    summed = float(c) + contributions.sum(axis=1)
    kinks = np.divide(-b, w, out=np.full_like(b, np.nan), where=np.abs(w) > 1e-12)
    visible = kinks[(kinks >= theta_grid.min()) & (kinks <= theta_grid.max()) & np.isfinite(kinks)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    if show_units:
        for j in range(num_units):
            axes[0].plot(theta_grid, activations[:, j], lw=1.8, label=f"unit {j + 1}")
            if np.isfinite(kinks[j]):
                axes[0].axvline(kinks[j], color="0.75", lw=0.7, ls=":")
    else:
        axes[0].text(0.5, 0.5, "Hidden activations hidden", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_title("Hidden activations")
    axes[0].set_ylabel("$\\sigma(w_jx+b_j)$")

    if show_weighted_units:
        for j in range(num_units):
            axes[1].plot(theta_grid, contributions[:, j], lw=1.8, label=f"v{j + 1} unit {j + 1}")
            if np.isfinite(kinks[j]):
                axes[1].axvline(kinks[j], color="0.75", lw=0.7, ls=":")
    else:
        axes[1].text(0.5, 0.5, "Weighted units hidden", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].axhline(0, color="0.25", lw=0.8, ls="--")
    axes[1].set_title("Weighted contributions")
    axes[1].set_ylabel("$v_j\\sigma(w_jx+b_j)$")

    if show_sum:
        axes[2].plot(theta_grid, summed, color=COLORS["fit"], lw=2.8, label="$h(x)$")
        for kink in visible:
            axes[2].axvline(kink, color="0.75", lw=0.7, ls=":")
    else:
        axes[2].text(0.5, 0.5, "Final sum hidden", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("Final one-layer function")
    axes[2].set_ylabel("h(x)")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_xlim(0, 90)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    plt.show()

    print(f"Number of units = {num_units}, intercept c = {float(c):.6f}")
    print(f"Raw hidden weights w = {np.array2string(w, precision=6)}")
    print(f"Raw hidden biases b = {np.array2string(b, precision=6)}")
    print(f"Raw output weights v = {np.array2string(v, precision=6)}")
    print(f"Kink locations x_j* = {np.array2string(kinks, precision=3)}")
    print(f"Visible kinks in plotted domain = {len(visible)}")
    print("What changed: w, b, v, and c move within this fixed-width H; num_units redefines H.")
    print("Assumption: the function is built by adding learned piecewise-linear features.")


def width_parameters(
    width: int,
    kink_shift: float = 0.0,
    output_scale: float = 1.0,
    parameter_preset: str = "spread",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return deterministic one-hidden-layer ReLU parameters for a width."""
    width = int(width)
    rng = np.random.default_rng(int(seed))
    if parameter_preset == "spread":
        kinks = np.linspace(12.0, 84.0, width) + float(kink_shift)
    elif parameter_preset == "clustered":
        kinks = 45.0 + np.linspace(-9.0, 9.0, width) + float(kink_shift)
    elif parameter_preset == "random":
        kinks = np.sort(rng.uniform(8.0, 86.0, size=width)) + float(kink_shift)
    else:
        raise ValueError(f"unknown parameter preset: {parameter_preset}")
    kinks = np.clip(kinks, 0.0, 90.0)
    if parameter_preset == "random":
        w = rng.choice([-1.0, 1.0], size=width) * rng.uniform(1 / 22.0, 1 / 11.0, size=width)
    else:
        w = np.full(width, 1 / 14.0)
    b = -w * kinks
    signs = np.where(np.arange(width) % 2 == 0, 1.0, -1.0)
    if parameter_preset == "random":
        envelope = rng.uniform(0.25, 0.75, size=width)
    else:
        envelope = 0.35 + 0.35 * np.cos(np.arange(width) * 0.9) ** 2
    v = float(output_scale) * signs * envelope
    return kinks, w, b, v, 0.35


def depth_raw_parameters(
    first_layer_scale: float,
    recombination_shift: float,
    output_mix: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return raw-x network parameters for the depth demonstration."""
    W1 = float(first_layer_scale) * DEPTH_W1_BASE_RAW
    b1 = DEPTH_B1_BASE_RAW.copy()
    W2 = np.array(
        [
            [0.90, -0.55],
            [0.35, 0.90],
            [-0.80, 0.65],
        ]
    )
    b2 = np.array([-0.65, -0.35]) + float(recombination_shift)
    output_w = float(output_mix) * np.array([0.85, -0.70])
    return W1, b1, W2, b2, output_w, 0.20


def plot_width_demo(
    num_units: int | None = None,
    parameter_preset: str = "spread",
    output_scale: float = 1.0,
    seed: int = 0,
    width: int | None = None,
    focus_unit: int | None = None,
    kink_shift: float = 0.0,
) -> None:
    """Plot a width-controlled one-hidden-layer ReLU network."""
    if num_units is None:
        num_units = 5 if width is None else width
    num_units = max(1, int(num_units))
    kinks, w, b, v, c = width_parameters(
        num_units,
        kink_shift=kink_shift,
        output_scale=output_scale,
        parameter_preset=parameter_preset,
        seed=seed,
    )
    contributions = relu(np.outer(theta_grid, w) + b) * v
    summed = c + contributions.sum(axis=1)
    focus = None if focus_unit is None else min(max(int(focus_unit) - 1, 0), len(kinks) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    ax = axes[0]
    for j, kink in enumerate(kinks):
        is_focus = focus is None or j == focus
        ax.plot(
            theta_grid,
            contributions[:, j],
            lw=1.8 if focus is None else 2.8 if is_focus else 1.1,
            alpha=0.7 if focus is None else 0.95 if is_focus else 0.25,
            color=COLORS["fit"] if is_focus else "0.35",
            label=f"unit {j + 1}" if is_focus else None,
        )
        ax.axvline(kink, color=COLORS["fit"] if is_focus else "0.7", lw=1.0, ls=":")
    ax.axhline(0, color="0.25", lw=0.9, ls="--")
    ax.set_title("Individual learned-feature contributions")
    ax.set_xlabel("x")
    ax.set_ylabel("$v_j a_j(x)$")
    ax.set_xlim(0, 90)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(theta_grid, summed, color=COLORS["fit"], lw=3, label="summed output")
    for kink in kinks:
        ax.axvline(kink, color="0.75", lw=0.6, ls=":")
    ax.set_title("Width controls how many hinges are available")
    ax.set_xlabel("x")
    ax.set_ylabel("h(x)")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.55, 2.10)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    visible = kinks[(kinks >= theta_grid.min()) & (kinks <= theta_grid.max())]
    if focus is not None:
        print(f"Focused unit {focus + 1}: kink at x={kinks[focus]:.1f}.")
    print(f"Width m = {num_units}")
    print(f"Parameter preset = {parameter_preset}; visible kink locations = {len(visible)}")
    print(f"Raw hidden weights w = {np.array2string(w, precision=6)}")
    print(f"Raw hidden biases b = {np.array2string(b, precision=6)}")
    print(f"Kink locations = {np.array2string(kinks, precision=3)}")
    print(f"Raw output weights v = {np.array2string(v, precision=6)}, c = {c:.6f}")
    print("What changed: num_units changes H_m; preset, seed, and output_scale select a function inside that width demo.")
    print("Assumption: width makes more local piecewise-linear changes available.")


def depth_representations(
    first_layer_scale: float = 1.0,
    recombination_shift: float = 0.0,
    output_mix: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return shallow output plus two-layer intermediate representations."""
    W1, b1, W2, b2, output_w, output_b = depth_raw_parameters(
        first_layer_scale,
        recombination_shift,
        output_mix,
    )
    z1 = relu(theta_grid[:, None] @ W1 + b1)
    z2 = relu(z1 @ W2 + b2)
    deep_output = output_b + z2 @ output_w

    shallow_kinks = np.array([12.0, 28.0, 48.0, 67.0, 82.0])
    shallow_coef = np.array([0.55, -0.90, 0.75, -0.45, 0.30])
    shallow_w = np.full_like(shallow_kinks, 1 / 18.0)
    shallow_b = -shallow_w * shallow_kinks
    shallow_output = eval_one_hidden_relu(theta_grid, shallow_w, shallow_b, shallow_coef, c=0.28)
    return shallow_output, z1, z2, deep_output


def _activation(z: np.ndarray, activation: str) -> np.ndarray:
    """Evaluate a small set of activations for depth demonstrations."""
    if activation == "relu":
        return relu(z)
    if activation == "tanh":
        return np.tanh(z)
    if activation == "linear":
        return z
    raise ValueError(f"unknown activation: {activation}")


def random_depth_representations(
    depth: int = 2,
    width: int = 4,
    activation: str = "relu",
    seed: int = 0,
) -> tuple[list[np.ndarray], np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray, float]:
    """Return hidden representations and output for a small feed-forward network."""
    depth = max(1, int(depth))
    width = max(1, int(width))
    rng = np.random.default_rng(int(seed))
    z = ((theta_grid - 45.0) / 22.5)[:, None]
    layer_outputs: list[np.ndarray] = []
    parameters: list[tuple[np.ndarray, np.ndarray]] = []
    in_dim = 1
    for ell in range(depth):
        if activation == "relu" and ell == 0:
            kink_z = np.linspace(-1.65, 1.65, width) + rng.normal(0.0, 0.08, size=width)
            directions = np.where(np.arange(width) % 2 == 0, 1.0, -1.0)
            W = (directions * rng.uniform(0.75, 1.35, size=width))[None, :]
            b = -W.ravel() * kink_z
        else:
            W = rng.normal(0.0, 1.15 / np.sqrt(in_dim), size=(in_dim, width))
            pre_without_bias = z @ W
            quantiles = np.linspace(0.2, 0.8, width)
            b = -np.array(
                [
                    np.quantile(pre_without_bias[:, j], quantiles[j])
                    for j in range(width)
                ]
            )
            b += rng.normal(0.0, 0.08, size=width)
        z = _activation(z @ W + b, activation)
        layer_outputs.append(z)
        parameters.append((W, b))
        in_dim = width
    output_w = rng.normal(0.0, 0.8 / np.sqrt(width), size=width)
    output_b = 0.35
    final_output = output_b + z @ output_w
    return layer_outputs, final_output, parameters, output_w, output_b


def approximate_region_count(y: np.ndarray, tol: float = 1e-3) -> int:
    """Estimate visible piecewise-linear regions from slope changes on the grid."""
    slope = np.diff(np.asarray(y))
    if slope.size < 2:
        return 1
    slope_change = np.abs(np.diff(slope))
    threshold = max(float(tol), 0.05 * float(np.nanmax(slope_change)) if np.any(slope_change) else tol)
    return int(1 + np.sum(slope_change > threshold))


def plot_depth_demo(
    depth: int = 2,
    width: int = 4,
    activation: str = "relu",
    seed: int = 0,
    show_layer_outputs: bool = True,
    show_final_output: bool = True,
    first_layer_scale: float | None = None,
    recombination_shift: float | None = None,
    output_mix: float | None = None,
) -> None:
    """Plot a depth-controlled feed-forward network demonstration."""
    if first_layer_scale is not None or recombination_shift is not None or output_mix is not None:
        first_layer_scale = 1.0 if first_layer_scale is None else first_layer_scale
        recombination_shift = 0.0 if recombination_shift is None else recombination_shift
        output_mix = 1.0 if output_mix is None else output_mix
        _, z1, z2, final_output = depth_representations(first_layer_scale, recombination_shift, output_mix)
        layer_outputs = [z1, z2]
        parameters = []
        output_w = np.array([])
        output_b = np.nan
        depth = len(layer_outputs)
        width = z1.shape[1]
        activation = "relu"
    else:
        layer_outputs, final_output, parameters, output_w, output_b = random_depth_representations(
            depth=depth,
            width=width,
            activation=activation,
            seed=seed,
        )

    panel_count = int(show_layer_outputs) + int(show_final_output)
    if panel_count == 0:
        panel_count = 1
    fig, axes = plt.subplots(1, panel_count, figsize=(6 * panel_count, 4), sharex=True)
    axes = np.atleast_1d(axes)
    axis_idx = 0
    if show_layer_outputs:
        ax = axes[axis_idx]
        for ell, layer in enumerate(layer_outputs, start=1):
            for j in range(min(layer.shape[1], 6)):
                ax.plot(theta_grid, layer[:, j], lw=1.4, alpha=0.8, label=f"L{ell} u{j + 1}")
        ax.set_title("Layer outputs")
        ax.set_ylabel("activation")
        axis_idx += 1
    if show_final_output:
        ax = axes[axis_idx]
        ax.plot(theta_grid, final_output, color=COLORS["fit"], lw=2.8, label="$h(x)$")
        ax.set_title("Final composed function")
        ax.set_ylabel("h(x)")
        axis_idx += 1
    if not show_layer_outputs and not show_final_output:
        axes[0].text(0.5, 0.5, "Enable a plot toggle", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("No curve selected")
    for ax in axes:
        ax.set_xlim(0, 90)
        ax.set_xlabel("x")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    plt.show()

    print(f"Depth = {int(depth)}, width = {int(width)}, activation = {activation}, seed = {int(seed)}")
    print(f"Approximate visible linear regions = {approximate_region_count(final_output)}")
    if parameters:
        W1, b1 = parameters[0]
        print(f"First-layer W shape = {W1.shape}, b shape = {b1.shape}")
        print(f"Output weights = {np.array2string(output_w, precision=5)}, output bias = {output_b:.5f}")
    print("What changed: depth, width, and activation redefine H; seed selects one parameter point inside it.")
    print("Assumption: useful functions can be built by composing and reusing intermediate features.")


def mlp_input_pattern(input_pattern: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a base input vector and a transformed comparison vector."""
    rng = np.random.default_rng(int(seed))
    base = np.zeros(16)
    base[[2, 3, 8, 9, 10, 13]] = [0.8, 1.1, 0.45, 1.0, 0.55, 0.75]
    if input_pattern == "base":
        comparison = base.copy()
    elif input_pattern == "shifted":
        comparison = np.roll(base, 3)
    elif input_pattern == "permuted":
        comparison = base[rng.permutation(len(base))]
    elif input_pattern == "noisy":
        comparison = np.clip(base + rng.normal(0.0, 0.08, size=base.size), 0.0, None)
    else:
        raise ValueError(f"unknown input pattern: {input_pattern}")
    return base, comparison


def mlp_forward_summary(
    x: np.ndarray,
    width: int,
    depth: int,
    activation: str,
    weight_scale: float,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
    """Evaluate a small dense MLP and return hidden states plus first-layer weights."""
    rng = np.random.default_rng(int(seed))
    width = max(1, int(width))
    depth = max(1, int(depth))
    z = np.asarray(x, dtype=float)
    hidden_states: list[np.ndarray] = []
    first_layer_W = np.empty((len(z), width))
    in_dim = len(z)
    for layer_idx in range(depth):
        W = rng.normal(0.0, float(weight_scale) / np.sqrt(in_dim), size=(in_dim, width))
        b = rng.normal(0.0, 0.12 * float(weight_scale), size=width)
        if layer_idx == 0:
            first_layer_W = W
        z = _activation(z @ W + b, activation)
        hidden_states.append(z)
        in_dim = width
    output_w = rng.normal(0.0, float(weight_scale) / np.sqrt(width), size=width)
    output_b = rng.normal(0.0, 0.12 * float(weight_scale))
    output = float(z @ output_w + output_b)
    return hidden_states, first_layer_W, output_w, output


def plot_architecture_demo(
    width: int = 8,
    depth: int = 2,
    activation: str = "relu",
    input_pattern: str = "shifted",
    weight_scale: float = 0.45,
    seed: int = 0,
    architecture: str = "mlp",
) -> None:
    """Plot an MLP-only inductive-bias diagnostic."""
    if architecture.lower() != "mlp":
        raise ValueError("this notebook section now supports only architecture='mlp'")

    width = max(1, int(width))
    depth = max(1, int(depth))
    base, comparison = mlp_input_pattern(input_pattern, seed=seed)
    base_hidden, W1, output_w, base_output = mlp_forward_summary(
        base,
        width=width,
        depth=depth,
        activation=activation,
        weight_scale=weight_scale,
        seed=seed,
    )
    comparison_hidden, _, _, comparison_output = mlp_forward_summary(
        comparison,
        width=width,
        depth=depth,
        activation=activation,
        weight_scale=weight_scale,
        seed=seed,
    )
    first_hidden_difference = np.linalg.norm(base_hidden[0] - comparison_hidden[0])
    output_difference = abs(base_output - comparison_output)
    weight_norm = float(np.linalg.norm(W1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(base, marker="o", label="base input")
    axes[0].plot(comparison, marker="o", alpha=0.75, label=input_pattern)
    axes[0].set_title("Coordinate-coded input")
    axes[0].set_xlabel("coordinate index")
    axes[0].set_ylabel("value")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    im = axes[1].imshow(W1.T, aspect="auto", cmap="coolwarm")
    axes[1].set_title("Dense first-layer weights")
    axes[1].set_xlabel("input coordinate")
    axes[1].set_ylabel("hidden unit")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    x_pos = np.arange(width)
    axes[2].bar(x_pos - 0.18, base_hidden[0], width=0.36, label="hidden(base)")
    axes[2].bar(x_pos + 0.18, comparison_hidden[0], width=0.36, label=f"hidden({input_pattern})")
    axes[2].set_title("First hidden representation")
    axes[2].set_xlabel("hidden unit")
    axes[2].set_ylabel("activation")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    plt.show()

    print("Architecture = mlp")
    print(f"Depth = {depth}, width = {width}, activation = {activation}, weight_scale = {float(weight_scale):.3f}")
    print(f"Seed = {int(seed)}, input_pattern = {input_pattern}")
    print(f"First-layer implementation matrix shape (input_dim, width) = {W1.shape}, Frobenius norm = {weight_norm:.3f}")
    print(f"Output weights = {np.array2string(output_w, precision=5)}")
    print(f"Output(base) = {base_output:.4f}; output({input_pattern}) = {comparison_output:.4f}")
    print(f"First-hidden L2 difference = {first_hidden_difference:.4f}; output difference = {output_difference:.4f}")
    print("What changed: width, depth, and activation redefine H_MLP; weight_scale and seed change the sampled parameter point.")
    print("Assumption: dense coordinate mixing is available, but no invariance is built in.")
    print("Solution bias: smaller weights and regularisation tend to prefer lower-norm or less rapidly varying realised functions.")


def initialise_mlp_inductive_bias_params(
    width: int = 16,
    depth: int = 2,
    activation: str = "relu",
    seed: int = 0,
    weight_scale: float = 1.0,
) -> dict[str, list[np.ndarray] | str]:
    """Initialise a small one-dimensional MLP for the inductive-bias demo."""
    if activation not in {"relu", "tanh"}:
        raise ValueError("activation must be 'relu' or 'tanh'")
    width = max(1, int(width))
    depth = max(1, int(depth))
    weight_scale = float(weight_scale)

    rng = np.random.default_rng(int(seed))
    dims = [1] + [width] * depth + [1]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for layer_idx, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        if layer_idx < depth:
            base_scale = np.sqrt(2.0 / fan_in) if activation == "relu" else np.sqrt(1.0 / fan_in)
            bias_scale = 0.12 * weight_scale
        else:
            base_scale = np.sqrt(1.0 / fan_in)
            bias_scale = 0.0
        weights.append(rng.normal(0.0, weight_scale * base_scale, size=(fan_in, fan_out)))
        biases.append(rng.normal(0.0, bias_scale, size=(1, fan_out)))
    return {"weights": weights, "biases": biases, "activation": activation}


def _activation_derivative(z: np.ndarray, activation: str) -> np.ndarray:
    """Return the derivative used by the small MLP training demo."""
    if activation == "relu":
        return (z > 0.0).astype(float)
    if activation == "tanh":
        a = np.tanh(z)
        return 1.0 - a**2
    raise ValueError("activation must be 'relu' or 'tanh'")


def _forward_mlp_inductive_bias(
    params: dict[str, list[np.ndarray] | str],
    x: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Evaluate the small one-dimensional MLP and keep caches for gradients."""
    weights = params["weights"]
    biases = params["biases"]
    activation = str(params["activation"])
    assert isinstance(weights, list)
    assert isinstance(biases, list)

    activations = [x]
    preactivations: list[np.ndarray] = []
    a = x
    for weight, bias in zip(weights[:-1], biases[:-1]):
        z = a @ weight + bias
        preactivations.append(z)
        a = _activation(z, activation)
        activations.append(a)
    y_hat = a @ weights[-1] + biases[-1]
    return y_hat, activations, preactivations


def _mlp_weight_norm(params: dict[str, list[np.ndarray] | str]) -> float:
    """Return the squared Frobenius norm of all MLP weight matrices."""
    weights = params["weights"]
    assert isinstance(weights, list)
    return float(sum(np.sum(weight**2) for weight in weights))


def _mlp_gradient_norm(grads: dict[str, list[np.ndarray]]) -> float:
    """Return the Euclidean norm of all gradient arrays."""
    return float(
        np.sqrt(
            sum(np.sum(grad**2) for grad in grads["weights"])
            + sum(np.sum(grad**2) for grad in grads["biases"])
        )
    )


def _backward_mlp_inductive_bias(
    params: dict[str, list[np.ndarray] | str],
    activations: list[np.ndarray],
    preactivations: list[np.ndarray],
    y_hat: np.ndarray,
    y: np.ndarray,
    weight_decay: float,
) -> dict[str, list[np.ndarray]]:
    """Backpropagate normalized MSE with L2 weight decay."""
    weights = params["weights"]
    activation = str(params["activation"])
    assert isinstance(weights, list)

    n = max(1, len(y))
    d_a = 2.0 * (y_hat - y) / n
    grad_w: list[np.ndarray] = [np.zeros_like(weight) for weight in weights]
    grad_b: list[np.ndarray] = [np.zeros((1, weight.shape[1])) for weight in weights]

    grad_w[-1] = activations[-1].T @ d_a + 2.0 * float(weight_decay) * weights[-1]
    grad_b[-1] = np.sum(d_a, axis=0, keepdims=True)
    d_a = d_a @ weights[-1].T

    for layer_idx in reversed(range(len(weights) - 1)):
        d_z = d_a * _activation_derivative(preactivations[layer_idx], activation)
        grad_w[layer_idx] = activations[layer_idx].T @ d_z + 2.0 * float(weight_decay) * weights[layer_idx]
        grad_b[layer_idx] = np.sum(d_z, axis=0, keepdims=True)
        if layer_idx > 0:
            d_a = d_z @ weights[layer_idx].T

    return {"weights": grad_w, "biases": grad_b}


def _copy_mlp_params(params: dict[str, list[np.ndarray] | str]) -> dict[str, list[np.ndarray] | str]:
    """Copy the small MLP parameter dictionary."""
    weights = params["weights"]
    biases = params["biases"]
    assert isinstance(weights, list)
    assert isinstance(biases, list)
    return {
        "weights": [weight.copy() for weight in weights],
        "biases": [bias.copy() for bias in biases],
        "activation": params["activation"],
    }


def _train_mlp_inductive_bias(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    depth: int,
    activation: str,
    seed: int,
    weight_scale: float,
    weight_decay: float,
    epochs: int,
    learning_rate: float = 0.015,
) -> tuple[dict[str, list[np.ndarray] | str], dict[str, list[float]]]:
    """Train the small MLP with full-batch Adam on normalized data."""
    params = initialise_mlp_inductive_bias_params(
        width=width,
        depth=depth,
        activation=activation,
        seed=seed,
        weight_scale=weight_scale,
    )
    weights = params["weights"]
    biases = params["biases"]
    assert isinstance(weights, list)
    assert isinstance(biases, list)

    state = {
        "mw": [np.zeros_like(weight) for weight in weights],
        "vw": [np.zeros_like(weight) for weight in weights],
        "mb": [np.zeros_like(bias) for bias in biases],
        "vb": [np.zeros_like(bias) for bias in biases],
    }
    history: dict[str, list[float]] = {"epoch": [], "loss": [], "objective": [], "grad_norm": []}
    epochs = max(0, int(epochs))
    log_every = max(1, epochs // 50) if epochs else 1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for epoch in range(epochs + 1):
        y_hat, activations, preactivations = _forward_mlp_inductive_bias(params, x)
        data_loss = float(np.mean((y_hat - y) ** 2))
        objective = data_loss + float(weight_decay) * _mlp_weight_norm(params)
        grads = _backward_mlp_inductive_bias(
            params,
            activations,
            preactivations,
            y_hat,
            y,
            weight_decay=weight_decay,
        )

        if epoch % log_every == 0 or epoch == epochs:
            history["epoch"].append(float(epoch))
            history["loss"].append(data_loss)
            history["objective"].append(objective)
            history["grad_norm"].append(_mlp_gradient_norm(grads))

        if epoch == epochs:
            break

        t = epoch + 1
        current_weights = params["weights"]
        current_biases = params["biases"]
        assert isinstance(current_weights, list)
        assert isinstance(current_biases, list)
        for idx in range(len(current_weights)):
            state["mw"][idx] = beta1 * state["mw"][idx] + (1.0 - beta1) * grads["weights"][idx]
            state["vw"][idx] = beta2 * state["vw"][idx] + (1.0 - beta2) * grads["weights"][idx] ** 2
            state["mb"][idx] = beta1 * state["mb"][idx] + (1.0 - beta1) * grads["biases"][idx]
            state["vb"][idx] = beta2 * state["vb"][idx] + (1.0 - beta2) * grads["biases"][idx] ** 2

            mw_hat = state["mw"][idx] / (1.0 - beta1**t)
            vw_hat = state["vw"][idx] / (1.0 - beta2**t)
            mb_hat = state["mb"][idx] / (1.0 - beta1**t)
            vb_hat = state["vb"][idx] / (1.0 - beta2**t)
            current_weights[idx] -= learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
            current_biases[idx] -= learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)

        if not all(np.all(np.isfinite(value)) for value in current_weights + current_biases):
            break

    return params, history


def visible_slope_change_locations(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    max_locations: int = 12,
    relative_threshold: float = 0.20,
) -> np.ndarray:
    """Estimate visible locations where a plotted one-dimensional curve changes slope."""
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    if len(x_grid) < 4 or len(y_grid) != len(x_grid):
        return np.array([])

    slopes = np.diff(y_grid) / np.maximum(np.diff(x_grid), 1e-12)
    slope_changes = np.abs(np.diff(slopes))
    if slope_changes.size == 0 or not np.any(np.isfinite(slope_changes)):
        return np.array([])
    max_change = float(np.nanmax(slope_changes))
    if max_change <= 1e-12:
        return np.array([])

    threshold = max(1e-6, float(relative_threshold) * max_change)
    candidates = np.flatnonzero(slope_changes >= threshold) + 1
    if candidates.size == 0:
        return np.array([])

    groups = np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1)
    peaks: list[tuple[float, float]] = []
    for group in groups:
        local_scores = slope_changes[group - 1]
        peak_idx = int(group[int(np.argmax(local_scores))])
        peaks.append((float(x_grid[peak_idx]), float(np.max(local_scores))))

    if len(peaks) > max_locations:
        peaks = sorted(peaks, key=lambda item: item[1], reverse=True)[: int(max_locations)]
    return np.asarray([location for location, _ in sorted(peaks)], dtype=float)


def exact_first_layer_thresholds(
    params: dict[str, list[np.ndarray] | str],
    x_mean: float,
    x_std: float,
) -> np.ndarray:
    """Return raw-x first-layer ReLU thresholds for a normalized scalar-input MLP."""
    weights = params["weights"]
    biases = params["biases"]
    assert isinstance(weights, list)
    assert isinstance(biases, list)
    w = weights[0][0, :]
    b = biases[0][0, :]
    nonzero = np.abs(w) > 1e-12
    thresholds = float(x_mean) - float(x_std) * b[nonzero] / w[nonzero]
    thresholds = thresholds[(thresholds >= float(np.min(theta_grid))) & (thresholds <= float(np.max(theta_grid)))]
    return np.sort(thresholds)


def fit_mlp_inductive_bias_demo(
    width: int = 16,
    depth: int = 2,
    activation: str = "relu",
    seed: int = 0,
    weight_scale: float = 1.0,
    weight_decay: float = 1e-4,
    epochs: int = 300,
) -> dict[str, object]:
    """Fit the one-dimensional MLP used in the section 6 inductive-bias demo."""
    data = sample_tilt_power(
        n=80,
        scenario="output_noise",
        seed=60 + int(seed),
        y_noise=0.045,
        sampling="sparse_feature",
    )
    x = np.asarray(data["x"], dtype=float).reshape(-1, 1)
    y = np.asarray(data["y"], dtype=float).reshape(-1, 1)
    x_mean = float(np.mean(x))
    x_std = float(max(np.std(x), 1e-6))
    y_mean = float(np.mean(y))
    y_std = float(max(np.std(y), 1e-6))
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    params, history = _train_mlp_inductive_bias(
        x_norm,
        y_norm,
        width=width,
        depth=depth,
        activation=activation,
        seed=seed,
        weight_scale=weight_scale,
        weight_decay=weight_decay,
        epochs=epochs,
    )

    x_grid = theta_grid.reshape(-1, 1)
    x_grid_norm = (x_grid - x_mean) / x_std
    y_grid_norm, _, _ = _forward_mlp_inductive_bias(params, x_grid_norm)
    y_grid = y_mean + y_std * y_grid_norm[:, 0]
    y_train_norm, _, _ = _forward_mlp_inductive_bias(params, x_norm)
    y_train = y_mean + y_std * y_train_norm[:, 0]
    slopes = np.gradient(y_grid, theta_grid)
    visible_changes = visible_slope_change_locations(theta_grid, y_grid)
    thresholds = (
        exact_first_layer_thresholds(params, x_mean=x_mean, x_std=x_std)
        if activation == "relu"
        else np.array([])
    )

    return {
        "data": data,
        "params": _copy_mlp_params(params),
        "history": history,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "y_grid": y_grid,
        "y_train": y_train,
        "slopes": slopes,
        "visible_changes": visible_changes,
        "first_layer_thresholds": thresholds,
        "train_mse": mse(y_train, y[:, 0]),
        "oracle_mse": mse(y_grid, f0(theta_grid)),
    }


def plot_mlp_inductive_bias_demo(
    width: int = 16,
    depth: int = 2,
    activation: str = "relu",
    seed: int = 0,
    weight_scale: float = 1.0,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    show_kinks: bool = True,
) -> None:
    """Plot the one-dimensional MLP inductive-bias diagnostic."""
    width = max(1, int(width))
    depth = max(1, int(depth))
    result = fit_mlp_inductive_bias_demo(
        width=width,
        depth=depth,
        activation=activation,
        seed=seed,
        weight_scale=weight_scale,
        weight_decay=weight_decay,
        epochs=epochs,
    )
    data = result["data"]
    history = result["history"]
    y_grid = np.asarray(result["y_grid"])
    slopes = np.asarray(result["slopes"])
    visible_changes = np.asarray(result["visible_changes"])
    first_layer_thresholds = np.asarray(result["first_layer_thresholds"])
    assert isinstance(data, dict)
    assert isinstance(history, dict)

    if activation == "relu" and show_kinks:
        if depth == 1:
            kink_locations = first_layer_thresholds
            kink_label = "first-layer threshold"
        else:
            kink_locations = visible_changes
            kink_label = "visible slope change"
    else:
        kink_locations = np.array([])
        kink_label = ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    ax.axvspan(60, 70, color=COLORS["support"], alpha=0.24, lw=0, label="narrow-feature region")
    plot_reference(ax)
    plot_observed(ax, data, alpha=0.78)
    ax.plot(theta_grid, y_grid, color=COLORS["fit"], lw=2.8, label="trained MLP")
    if show_kinks:
        for idx, location in enumerate(kink_locations):
            ax.axvline(
                location,
                color=COLORS["alt"],
                lw=1.0,
                ls=":",
                alpha=0.55,
                label=kink_label if idx == 0 else None,
            )
    style_xy_axis(ax)
    ax.set_title("One-dimensional MLP fit")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(theta_grid, slopes, color=COLORS["alt"], lw=2.2)
    if show_kinks:
        for location in kink_locations:
            ax.axvline(location, color=COLORS["alt"], lw=0.9, ls=":", alpha=0.5)
    ax.set_xlim(0, 90)
    ax.set_title("Local slope of $h_\\theta(x)$")
    ax.set_xlabel("observed tilt x")
    ax.set_ylabel("slope")
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(history["epoch"], history["loss"], color=COLORS["fit"], lw=2.2, label="normalized MSE")
    ax.plot(history["epoch"], history["objective"], color=COLORS["data"], lw=1.8, ls="--", label="objective")
    ax.set_title("Training trace")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()

    print("MLP inductive-bias demo")
    print(
        f"Depth = {depth}, width = {width}, activation = {activation}, "
        f"weight_scale = {float(weight_scale):.3f}, weight_decay = {float(weight_decay):.6g}"
    )
    print(f"Seed = {int(seed)}, epochs = {int(epochs)}")
    print(f"Training MSE = {float(result['train_mse']):.5f}; oracle grid MSE = {float(result['oracle_mse']):.5f}")
    print(f"Approximate visible linear regions = {approximate_region_count(y_grid)}")
    if activation == "relu" and show_kinks:
        if depth == 1:
            print(f"Exact first-layer thresholds in raw x = {np.array2string(kink_locations, precision=3)}")
        else:
            print(f"Visible slope-change locations in raw x = {np.array2string(kink_locations, precision=3)}")
    weights = result["params"]["weights"]
    assert isinstance(weights, list)
    print(f"First-layer implementation matrix shape (input_dim, width) = {weights[0].shape}")
    print(
        "What changed: width and depth redefine H_MLP; "
        "seed and weight_scale choose an initial parameter point; weight_decay and epochs change O."
    )
    print("Assumption: the target curve is learnable as a continuous piecewise-linear function of observed tilt.")
    print("Solution bias: weight decay and optimisation tend to prefer lower-norm, less rapidly varying realised functions.")


@lru_cache(maxsize=None)
def support_dataset(seed: int = 0) -> dict[str, np.ndarray | str]:
    """Return the fixed data-gap dataset for hypothesis-space comparisons."""
    return sample_tilt_power(
        n=70,
        scenario="output_noise",
        seed=70 + int(seed),
        y_noise=0.035,
        sampling="gap65",
    )


def _normalise_model_kind(model_kind: str) -> str:
    """Normalise public model-kind aliases used in the notebook."""
    aliases = {
        "poly": "polynomial",
        "polynomial": "polynomial",
        "relu": "ReLU basis",
        "ReLU basis": "ReLU basis",
        "periodic": "periodic basis",
        "periodic basis": "periodic basis",
    }
    if model_kind not in aliases:
        raise ValueError(f"unknown model kind: {model_kind}")
    return aliases[model_kind]


def fit_support_model(
    model_kind: str,
    complexity: int,
    lam: float,
    period: float = 90.0,
    seed: int = 0,
) -> tuple[Callable[[np.ndarray], np.ndarray], str, str, np.ndarray]:
    """Fit one selected basis model to the fixed support dataset."""
    model_kind = _normalise_model_kind(model_kind)
    data = support_dataset(seed)
    complexity = int(complexity)
    if model_kind == "polynomial":
        degree = validate_raw_polynomial_degree(complexity)
        predict, coef = fit_basis_model(
            data["x"],
            data["y"],
            lambda x: polynomial_design(x, degree),
            lam=lam,
        )
        return predict, f"degree {degree} polynomial", "global polynomial continuation", coef

    if model_kind == "ReLU basis":
        width = max(1, min(complexity, 18))
        kinks = np.linspace(8, 86, width)
        predict, coef = fit_basis_model(
            data["x"],
            data["y"],
            lambda x: relu_basis_design(x, kinks),
            lam=lam,
        )
        return predict, f"width {width} ReLU basis", "piecewise-linear continuation", coef

    if model_kind == "periodic basis":
        frequencies = max(1, min(complexity, 8))
        predict, coef = fit_basis_model(
            data["x"],
            data["y"],
            lambda x: periodic_design(x, frequencies, period=period),
            lam=lam,
        )
        return predict, f"{frequencies} periodic frequencies, period {period:.1f}", "periodicity imposed by basis", coef

    raise ValueError(f"unknown model kind: {model_kind}")


def support_diagnostics(
    predict: Callable[[np.ndarray], np.ndarray],
    radius: float = 4.0,
    seed: int = 0,
) -> dict[str, float]:
    """Compute training, observed-region, and gap oracle errors."""
    data = support_dataset(seed)
    counts = local_count(theta_grid, data["x"], radius=radius)
    observed_mask = counts >= 3
    gap_mask = (theta_grid >= 58) & (theta_grid <= 72)
    pred_grid = predict(theta_grid)
    return {
        "training": mse(predict(data["x"]), data["y"]),
        "observed_oracle": mse(pred_grid[observed_mask], f0(theta_grid)[observed_mask]),
        "gap_oracle": mse(pred_grid[gap_mask], f0(theta_grid)[gap_mask]),
    }


def plot_support_model(
    model_kind: str = "relu",
    complexity: int = 5,
    lam: float = 1e-3,
    period: float = 90.0,
    seed: int = 0,
) -> None:
    """Plot one selected hypothesis class on the fixed data-gap dataset."""
    normalised_model_kind = _normalise_model_kind(model_kind)
    data = support_dataset(seed)
    predict, label, assumption, coef = fit_support_model(
        normalised_model_kind,
        complexity,
        lam,
        period=period,
        seed=seed,
    )
    counts = local_count(theta_grid, data["x"], radius=4.0)
    observed_mask = counts >= 3
    diagnostics = support_diagnostics(predict, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=False)
    ax = axes[0]
    plot_reference(ax)
    shade_gap(ax)
    plot_observed(ax, data, alpha=0.80)
    ax.plot(theta_grid, predict(theta_grid), color=COLORS["fit"], lw=2.8, label=label)
    style_xy_axis(ax)
    ax.set_title("Fixed D, selected H")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(theta_grid, counts, color=COLORS["data"], lw=2.2)
    ax.fill_between(theta_grid, 0, counts, where=~observed_mask, color=COLORS["support"], alpha=0.55, label="weak support")
    ax.axvspan(58, 72, color=COLORS["support"], alpha=0.32, lw=0, label="deliberate gap")
    ax.axhline(3, color=COLORS["fit"], lw=1.3, ls="--", label="support threshold")
    ax.set_title("Where D provides local evidence")
    ax.set_xlabel("x")
    ax.set_ylabel("local count")
    ax.set_xlim(0, 90)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    print(f"Hypothesis class: {label}")
    print(f"Assumption outside support: {assumption}")
    print(f"Fitted raw-basis coefficients = {np.array2string(coef, precision=6)}")
    if normalised_model_kind == "periodic basis":
        print(f"Explicit period in raw x units = {float(period):.6f}")
    print(
        "Errors: "
        f"train={diagnostics['training']:.4f}, "
        f"observed-oracle={diagnostics['observed_oracle']:.4f}, "
        f"gap-oracle={diagnostics['gap_oracle']:.4f}"
    )
    print("What changed: model_kind and complexity redefine H; lam changes O; seed changes D.")
    print("Assumption: unsupported gap behaviour is supplied by the selected basis and penalty.")


def what_changed_rows() -> list[list[str]]:
    """Return the what-changed rows for the fixed-data support demo."""
    return [
        ["D", "yes", "no", "same observations and same data gap"],
        ["H", "no", "yes", "the selected feature family and continuation changed"],
        ["O", "mostly yes", "basis-specific penalty geometry", "same squared-loss ridge rule, but basis changes what the penalty prefers"],
        ["Estimand", "yes", "no", "same latent response used for oracle diagnostics"],
        ["Deployment distribution", "yes", "no", "same gap region is used as the stress test"],
    ]


@lru_cache(maxsize=None)
def capacity_data(
    noise_level: float = 0.1,
    train_size: int = 40,
    seed: int = 0,
    label_mode: str = "true",
) -> tuple[dict[str, np.ndarray | str], dict[str, np.ndarray | str]]:
    """Return deterministic train and validation data for capacity diagnostics."""
    train = sample_tilt_power(
        n=int(train_size),
        scenario="output_noise",
        seed=90 + int(seed),
        y_noise=float(noise_level),
        sampling="gap65",
    )
    validation = sample_tilt_power(
        n=300,
        scenario="output_noise",
        seed=910 + int(seed),
        y_noise=float(noise_level),
        sampling="uniform",
    )
    if label_mode == "random":
        random_y = np.random.default_rng(930 + int(seed)).normal(
            float(np.mean(train["y"])),
            float(np.std(train["y"]) + 1e-6),
            size=len(train["x"]),
        )
        train = {**train, "y": random_y, "scenario": "random_labels"}
    elif label_mode != "true":
        raise ValueError(f"unknown label mode: {label_mode}")
    return train, validation


def capacity_errors(
    max_degree: int = RAW_POLYNOMIAL_MAX_DEGREE,
    noise_level: float = 0.1,
    train_size: int = 40,
    seed: int = 0,
    label_mode: str = "true",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return degree, training, validation, and oracle errors."""
    train, validation = capacity_data(noise_level, train_size, seed, label_mode)
    max_degree = validate_raw_polynomial_degree(max_degree)
    degrees = np.arange(0, max_degree + 1)
    train_errors = []
    validation_errors = []
    oracle_errors = []
    for degree in degrees:
        predict, _ = fit_basis_model(
            train["x"],
            train["y"],
            lambda x, degree=degree: polynomial_design(x, degree),
            lam=0.004 + 0.001 * degree,
        )
        train_errors.append(mse(predict(train["x"]), train["y"]))
        validation_errors.append(mse(predict(validation["x"]), validation["y"]))
        oracle_errors.append(mse(predict(theta_grid), f0(theta_grid)))
    return degrees, np.asarray(train_errors), np.asarray(validation_errors), np.asarray(oracle_errors)


def plot_capacity_demo(
    capacity: int | None = None,
    noise_level: float = 0.1,
    label_mode: str = "true",
    train_size: int = 40,
    seed: int = 0,
    degree: int | None = None,
    show_random_labels: bool | None = None,
) -> None:
    """Plot a capacity diagnostic for a selected degree."""
    if capacity is None:
        capacity = 5 if degree is None else degree
    if show_random_labels is not None:
        label_mode = "random" if show_random_labels else "true"
    degree = capacity
    degree = validate_raw_polynomial_degree(degree)
    train, validation = capacity_data(noise_level, train_size, seed, label_mode)
    degrees, train_errors, validation_errors, oracle_errors = capacity_errors(
        noise_level=noise_level,
        train_size=train_size,
        seed=seed,
        label_mode=label_mode,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    if label_mode == "random":
        random_data = train
        random_y = train["y"]
        random_predict, _ = fit_basis_model(
            random_data["x"],
            random_y,
            lambda x: relu_basis_design(x, np.sort(random_data["x"])),
            lam=1e-7,
        )
        ax.scatter(random_data["x"], random_y, color=COLORS["data"], s=34, alpha=0.8, label="random labels")
        ax.plot(theta_grid, random_predict(theta_grid), color=COLORS["fit"], lw=2.4, label="raw ReLU-basis fit")
        ax.plot(theta_grid, f0(theta_grid), color=COLORS["truth"], lw=1.5, ls="--", label="latent f0, not used")
        ax.set_title("Random labels can still be fit")
        selected_train_error = mse(random_predict(random_data["x"]), random_y)
    else:
        predict, _ = fit_basis_model(
            train["x"],
            train["y"],
            lambda x: polynomial_design(x, degree),
            lam=0.004 + 0.001 * degree,
        )
        plot_reference(ax)
        shade_gap(ax)
        plot_observed(ax, train, alpha=0.75)
        ax.plot(theta_grid, predict(theta_grid), color=COLORS["fit"], lw=2.6, label=f"degree {degree}")
        ax.set_title("Selected capacity on the same training data")
        selected_train_error = mse(predict(train["x"]), train["y"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.25, 1.45)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(degrees, train_errors, marker="o", color=COLORS["data"], label="training error")
    ax.plot(degrees, validation_errors, marker="o", color=COLORS["fit"], label="validation error")
    ax.plot(degrees, oracle_errors, marker="o", color=COLORS["alt"], label="oracle grid error")
    ax.axvline(degree, color="0.25", lw=1.2, ls="--", label="selected degree")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("mean squared error")
    ax.set_title("Capacity curve")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    print(f"Selected training MSE = {selected_train_error:.4f}")
    print(f"Capacity = {degree}, label_mode = {label_mode}, noise_level = {float(noise_level):.3f}")
    print(f"Train size = {int(train_size)}, seed = {int(seed)}")
    print("What changed: capacity changes H; label_mode, noise_level, train_size, and seed change D.")
    print("Assumption: expressivity alone does not identify, optimise, or validate the selected function.")


def parameter_equivalence(
    transform: str = "permute",
    alpha: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original and transformed functions for parameter-equivalence demo."""
    transform_aliases = {
        "permute": "permutation",
        "permutation": "permutation",
        "positive_rescale": "positive rescaling",
        "positive rescaling": "positive rescaling",
    }
    transform = transform_aliases.get(transform, transform)
    rng = np.random.default_rng(int(seed))
    original_kinks = np.array([22.0, 48.0, 74.0])
    w = np.array([0.075, -0.060, 0.085]) + rng.normal(0.0, 0.0, size=3)
    b = -w * original_kinks
    v = np.array([0.85, -0.70, 0.55])
    c = 0.30
    original = eval_one_hidden_relu(theta_grid, w, b, v, c)

    if transform == "permutation":
        permutation = np.array([2, 0, 1])
        transformed = eval_one_hidden_relu(theta_grid, w[permutation], b[permutation], v[permutation], c)
    elif transform == "positive rescaling":
        scale = np.array([float(alpha), 1.0 / float(alpha), 1.5])
        transformed = eval_one_hidden_relu(theta_grid, scale * w, scale * b, v / scale, c)
    else:
        raise ValueError(f"unknown transform: {transform}")
    return original, transformed


def parameter_equivalence_tables(
    transform: str = "permute",
    alpha: float = 2.0,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray | float], str]:
    """Return original and transformed parameters for printing."""
    transform_aliases = {
        "permute": "permutation",
        "permutation": "permutation",
        "positive_rescale": "positive rescaling",
        "positive rescaling": "positive rescaling",
    }
    normalised = transform_aliases.get(transform, transform)
    rng = np.random.default_rng(int(seed))
    original_kinks = np.array([22.0, 48.0, 74.0])
    w = np.array([0.075, -0.060, 0.085]) + rng.normal(0.0, 0.0, size=3)
    b = -w * original_kinks
    v = np.array([0.85, -0.70, 0.55])
    c = 0.30
    if normalised == "permutation":
        permutation = np.array([2, 0, 1])
        transformed = {"w": w[permutation], "b": b[permutation], "v": v[permutation], "c": c}
    elif normalised == "positive rescaling":
        scale = np.array([float(alpha), 1.0 / float(alpha), 1.5])
        transformed = {"w": scale * w, "b": scale * b, "v": v / scale, "c": c}
    else:
        raise ValueError(f"unknown transform: {transform}")
    original = {"w": w, "b": b, "v": v, "c": c}
    return original, transformed, normalised


def plot_parameter_equivalence(
    transform: str = "permute",
    alpha: float = 2.0,
    seed: int = 0,
) -> None:
    """Plot two parameterisations that realise the same function."""
    original, transformed = parameter_equivalence(transform, alpha, seed)
    original_params, transformed_params, normalised = parameter_equivalence_tables(transform, alpha, seed)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(theta_grid, original, color=COLORS["fit"], lw=3, label="original parameters")
    ax.plot(theta_grid, transformed, color=COLORS["data"], lw=2, ls="--", label=normalised)
    ax.set_title("Different parameter vectors can realise the same function")
    ax.set_xlabel("x")
    ax.set_ylabel("h(x)")
    ax.set_xlim(0, 90)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    plt.show()
    print(f"Transform = {normalised}, alpha = {float(alpha):.6f}, seed = {int(seed)}")
    print(f"Original w = {np.array2string(np.asarray(original_params['w']), precision=6)}")
    print(f"Original b = {np.array2string(np.asarray(original_params['b']), precision=6)}")
    print(f"Original v = {np.array2string(np.asarray(original_params['v']), precision=6)}, c = {original_params['c']:.6f}")
    print(f"Transformed w = {np.array2string(np.asarray(transformed_params['w']), precision=6)}")
    print(f"Transformed b = {np.array2string(np.asarray(transformed_params['b']), precision=6)}")
    print(f"Transformed v = {np.array2string(np.asarray(transformed_params['v']), precision=6)}, c = {transformed_params['c']:.6f}")
    print(f"Max |h_original(x) - h_transformed(x)| = {np.max(np.abs(original - transformed)):.3e}")
    print("What changed: parameters changed, but the realised function stayed in the same point of H.")
    print("Assumption: scientific claims should target function behaviour, not arbitrary coordinates.")


def summary_rows() -> list[list[str]]:
    """Return the summary table rows for the notebook wrap-up."""
    return [
        ["H", "feature maps, ReLU units, width, depth, MLP architecture", "What functions did we make possible or natural?"],
        ["D", "fixed gap65 dataset and local support counts", "Where did observations provide evidence?"],
        ["O", "ridge penalties and selected coefficients", "Which compatible function was selected?"],
        ["s", "the plotted fitted function", "What claim does this realised function support?"],
    ]
