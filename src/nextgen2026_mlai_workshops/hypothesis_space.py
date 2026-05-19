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


def relu(z: np.ndarray | float) -> np.ndarray:
    """Evaluate the ReLU activation."""
    return np.maximum(0.0, z)


def scaled_x(x: np.ndarray | float) -> np.ndarray:
    """Scale the tilt-power x-axis to a numerically stable interval."""
    return (np.asarray(x) - 45.0) / 45.0


def polynomial_design(x: np.ndarray | float, degree: int) -> np.ndarray:
    """Return a polynomial design matrix with columns 1, z, ..., z^degree."""
    z = scaled_x(x)
    return np.vstack([z**k for k in range(int(degree) + 1)]).T


def periodic_design(x: np.ndarray | float, frequency_count: int) -> np.ndarray:
    """Return a sine/cosine basis on the 0-90 degree input range."""
    z = np.asarray(x) / 90.0
    columns = [np.ones_like(z)]
    for k in range(1, int(frequency_count) + 1):
        columns.extend([np.sin(2 * np.pi * k * z), np.cos(2 * np.pi * k * z)])
    return np.vstack(columns).T


def relu_basis_design(
    x: np.ndarray | float,
    kinks: np.ndarray,
    include_linear: bool = True,
) -> np.ndarray:
    """Return a fixed-knot one-dimensional ReLU basis design matrix."""
    z = scaled_x(x)
    kink_z = scaled_x(np.asarray(kinks))
    columns = [np.ones_like(z)]
    if include_linear:
        columns.append(z)
    columns.extend([relu(z - knot) for knot in kink_z])
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


def coefficient_vector(degree: int, preset: str) -> np.ndarray:
    """Create deterministic polynomial coefficients for an interactive preset."""
    degree = int(degree)
    k = np.arange(degree + 1, dtype=float)
    if preset == "smooth":
        coef = 0.55 * np.exp(-0.35 * k) * np.cos(1.2 * k)
    elif preset == "oscillating":
        coef = 0.50 * np.exp(-0.18 * k) * ((-1.0) ** k)
    elif preset == "tilted":
        coef = 0.35 * np.exp(-0.28 * k) * np.sin(0.9 * (k + 1))
        if degree >= 1:
            coef[1] += 0.45
    else:
        raise ValueError(f"unknown coefficient preset: {preset}")
    coef[0] += 0.45
    return coef


def fixed_vs_learned_action_rows() -> list[list[str]]:
    """Return action rows used in the fixed-feature versus learned-feature demo."""
    return [
        ["change polynomial coefficients", "selected h inside fixed H_phi"],
        ["change polynomial degree", "redefine H"],
        ["change ReLU hidden weights or biases", "move inside learned-feature H_NN"],
        ["change ReLU width", "redefine H"],
    ]


def plot_fixed_vs_learned(
    degree: int = 3,
    coefficient_preset: str = "smooth",
    kink_shift: float = 0.0,
    output_scale: float = 1.0,
) -> None:
    """Plot one fixed-feature function and one learned-feature ReLU function."""
    degree = int(degree)
    theta = coefficient_vector(degree, coefficient_preset)
    fixed_y = polynomial_design(theta_grid, degree) @ theta

    base_kinks = np.array([24.0, 45.0, 67.0])
    kinks = np.clip(base_kinks + float(kink_shift), 0.0, 90.0)
    w = np.full(len(kinks), 1 / 18.0)
    b = -w * kinks
    v = float(output_scale) * np.array([0.50, -0.85, 0.70])
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

    print(
        "Look for whether your change moves within a fixed H "
        "or redefines which functions are available."
    )


def plot_relu_neuron(w: float = 0.08, b: float = -3.6) -> None:
    """Plot preactivation and activation for one ReLU neuron."""
    pre = float(w) * theta_grid + float(b)
    act = relu(pre)
    kink = -float(b) / float(w) if abs(float(w)) > 1e-12 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    ax = axes[0]
    ax.plot(theta_grid, pre, color=COLORS["data"], lw=2.2, label="$wx+b$")
    ax.axhline(0, color="0.25", lw=1, ls="--")
    if np.isfinite(kink):
        ax.axvline(kink, color=COLORS["fit"], lw=1.5, ls=":", label=f"kink {kink:.1f}")
    ax.set_ylabel("preactivation")
    ax.set_title("Linear preactivation")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(theta_grid, act, color=COLORS["fit"], lw=2.6, label="ReLU($wx+b$)")
    if np.isfinite(kink):
        ax.axvline(kink, color=COLORS["fit"], lw=1.5, ls=":", label=f"$x^*$={kink:.1f}")
    ax.set_ylabel("activation")
    ax.set_title("Activated learned feature")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_xlim(0, 90)
    fig.tight_layout()
    plt.show()
    print(f"kink x* = {kink:.2f}" if np.isfinite(kink) else "no finite kink when w=0")


def width_parameters(
    width: int,
    kink_shift: float = 0.0,
    output_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return deterministic one-hidden-layer ReLU parameters for a width."""
    width = int(width)
    kinks = np.linspace(12.0, 84.0, width) + float(kink_shift)
    kinks = np.clip(kinks, 0.0, 90.0)
    w = np.full(width, 1 / 14.0)
    b = -w * kinks
    signs = np.where(np.arange(width) % 2 == 0, 1.0, -1.0)
    envelope = 0.35 + 0.35 * np.cos(np.arange(width) * 0.9) ** 2
    v = float(output_scale) * signs * envelope
    return kinks, w, b, v, 0.35


def plot_width_demo(
    width: int = 4,
    focus_unit: int = 1,
    kink_shift: float = 0.0,
    output_scale: float = 1.0,
) -> None:
    """Plot a width-controlled one-hidden-layer ReLU network."""
    kinks, w, b, v, c = width_parameters(width, kink_shift, output_scale)
    contributions = relu(np.outer(theta_grid, w) + b) * v
    summed = c + contributions.sum(axis=1)
    focus = min(max(int(focus_unit) - 1, 0), len(kinks) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    ax = axes[0]
    for j, kink in enumerate(kinks):
        is_focus = j == focus
        ax.plot(
            theta_grid,
            contributions[:, j],
            lw=2.8 if is_focus else 1.1,
            alpha=0.95 if is_focus else 0.25,
            color=COLORS["fit"] if is_focus else "0.35",
            label=f"focused unit {j + 1}" if is_focus else None,
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
    print(
        f"Focused unit {focus + 1}: kink at x={kinks[focus]:.1f}. "
        "Change width to redefine H; change focus to inspect one feature."
    )


def depth_representations(
    first_layer_scale: float = 1.0,
    recombination_shift: float = 0.0,
    output_mix: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return shallow output plus two-layer intermediate representations."""
    u = scaled_x(theta_grid)
    W1 = float(first_layer_scale) * np.array([[2.5, -2.0, 3.0]])
    b1 = np.array([1.15, 0.60, -0.85])
    z1 = relu(u[:, None] @ W1 + b1)
    W2 = np.array(
        [
            [0.90, -0.55],
            [0.35, 0.90],
            [-0.80, 0.65],
        ]
    )
    b2 = np.array([-0.65, -0.35]) + float(recombination_shift)
    z2 = relu(z1 @ W2 + b2)
    deep_output = 0.20 + z2 @ (float(output_mix) * np.array([0.85, -0.70]))

    shallow_kinks = np.array([12.0, 28.0, 48.0, 67.0, 82.0])
    shallow_coef = np.array([0.55, -0.90, 0.75, -0.45, 0.30])
    shallow_w = np.full_like(shallow_kinks, 1 / 18.0)
    shallow_b = -shallow_w * shallow_kinks
    shallow_output = eval_one_hidden_relu(theta_grid, shallow_w, shallow_b, shallow_coef, c=0.28)
    return shallow_output, z1, z2, deep_output


def plot_depth_demo(
    first_layer_scale: float = 1.0,
    recombination_shift: float = 0.0,
    output_mix: float = 1.0,
) -> None:
    """Plot a small manual depth/composition demonstration."""
    shallow_output, z1, z2, deep_output = depth_representations(
        first_layer_scale,
        recombination_shift,
        output_mix,
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for j in range(z1.shape[1]):
        axes[0, 0].plot(theta_grid, z1[:, j], lw=2, label=f"$z_1,{j + 1}$")
    axes[0, 0].set_title("First-layer features")

    for j in range(z2.shape[1]):
        axes[0, 1].plot(theta_grid, z2[:, j], lw=2, label=f"$z_2,{j + 1}$")
    axes[0, 1].set_title("Second layer recombines features")

    axes[1, 0].plot(theta_grid, shallow_output, color=COLORS["data"], lw=2.6, label="shallow reference")
    axes[1, 0].set_title("Shallow sum of hinges")

    axes[1, 1].plot(theta_grid, deep_output, color=COLORS["fit"], lw=2.8, label="composed output")
    axes[1, 1].set_title("Composed representation")

    for ax in axes.ravel():
        ax.set_xlim(0, 90)
        ax.set_xlabel("x")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    print("Look for which bends come from first-layer features and which come from recombination.")


def circular_filter(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a circular one-dimensional shared local filter."""
    offsets = np.arange(-(len(kernel) // 2), len(kernel) // 2 + 1)
    out = np.zeros_like(signal, dtype=float)
    for weight, offset in zip(kernel, offsets):
        out += weight * np.roll(signal, offset)
    return out


def attention_matrix(tokens: np.ndarray) -> np.ndarray:
    """Return a deterministic small attention matrix for token features."""
    Q = tokens @ np.array([[1.0, 0.2], [0.1, 0.9], [0.4, 0.3]])
    K = tokens @ np.array([[0.8, 0.1], [0.2, 1.0], [0.5, 0.4]])
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    attention = np.exp(scores - scores.max(axis=1, keepdims=True))
    return attention / attention.sum(axis=1, keepdims=True)


def gnn_update(
    node_features: np.ndarray,
    edges: list[tuple[int, int]],
    message_weight: float,
) -> np.ndarray:
    """Apply one simple mean-neighbour message-passing update."""
    node_features = np.asarray(node_features, dtype=float)
    adjacency = np.zeros((len(node_features), len(node_features)))
    for i, j in edges:
        adjacency[i, j] = adjacency[j, i] = 1
    degree = adjacency.sum(axis=1, keepdims=True)
    neighbour_mean = adjacency @ node_features[:, None] / np.maximum(degree, 1)
    return ((1 - float(message_weight)) * node_features[:, None] + float(message_weight) * neighbour_mean).ravel()


def plot_architecture_demo(
    architecture: str = "CNN local filter",
    shift: int = 6,
    query_token: int = 0,
    message_weight: float = 0.45,
) -> None:
    """Plot one lightweight architecture-bias diagnostic."""
    if architecture == "CNN local filter":
        signal = np.zeros(32)
        signal[[5, 6, 15, 16, 17, 25]] = [0.8, 1.1, 0.5, 1.0, 0.6, 0.9]
        kernel = np.array([-0.5, 1.0, -0.5])
        filtered = circular_filter(signal, kernel)
        shifted_filtered = circular_filter(np.roll(signal, int(shift)), kernel)
        error = np.max(np.abs(shifted_filtered - np.roll(filtered, int(shift))))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(signal, marker="o", label="input")
        axes[0].plot(np.roll(signal, int(shift)), marker="o", alpha=0.75, label="shifted input")
        axes[0].set_title("Shared local filter")
        axes[1].plot(filtered, marker="o", label="filter(input)")
        axes[1].plot(shifted_filtered, marker="o", alpha=0.75, label="filter(shifted input)")
        axes[1].plot(np.roll(filtered, int(shift)), ls="--", color="0.25", label="shifted filter(input)")
        axes[1].set_title(f"Translation diagnostic, max diff={error:.1e}")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()
        print("Vary shift. The output should shift with the input because the same local filter is reused.")
        return

    if architecture == "Transformer attention":
        tokens = np.array(
            [
                [1.0, 0.1, 0.0],
                [0.9, 0.2, 0.1],
                [0.0, 1.0, 0.2],
                [0.2, 0.8, 0.4],
                [0.1, 0.1, 1.0],
            ]
        )
        attention = attention_matrix(tokens)
        mixed = attention @ tokens
        query_token = min(max(int(query_token), 0), attention.shape[0] - 1)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        im = axes[0].imshow(attention, vmin=0, vmax=1, cmap="viridis")
        axes[0].axhline(query_token - 0.5, color="white", lw=1.5)
        axes[0].axhline(query_token + 0.5, color="white", lw=1.5)
        axes[0].set_title("Attention weights")
        axes[0].set_xlabel("key token")
        axes[0].set_ylabel("query token")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].bar(np.arange(tokens.shape[0]), attention[query_token], color=COLORS["fit"])
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"Query token {query_token} mixes values")
        axes[1].set_xlabel("key token")
        axes[1].set_ylabel("attention weight")
        axes[1].grid(alpha=0.2)
        print(f"Mixed value for query token {query_token}: {np.round(mixed[query_token], 3)}")
        fig.tight_layout()
        plt.show()
        return

    if architecture == "GNN message passing":
        node_features = np.array([0.2, 1.0, 0.4, 0.8, 0.1])
        positions = np.array([[0, 1], [1, 1.4], [2, 1], [1.6, 0], [0.4, 0]])
        edges = [(0, 1), (1, 2), (1, 3), (0, 4), (3, 4)]
        updated = gnn_update(node_features, edges, message_weight)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, values, title in [
            (axes[0], node_features, "input node features"),
            (axes[1], updated, "after neighbour update"),
        ]:
            for i, j in edges:
                ax.plot([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]], color="0.65", lw=1.5)
            sc = ax.scatter(positions[:, 0], positions[:, 1], c=values, s=430, cmap="plasma", vmin=0, vmax=1)
            for idx, (px, py) in enumerate(positions):
                ax.text(px, py, str(idx), ha="center", va="center", color="white", weight="bold")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.3, 2.3)
            ax.set_ylim(-0.3, 1.7)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.show()
        print("Vary message weight. Larger values make node features more neighbour-dependent.")
        return

    raise ValueError(f"unknown architecture demo: {architecture}")


@lru_cache(maxsize=1)
def support_dataset() -> dict[str, np.ndarray | str]:
    """Return the fixed data-gap dataset for hypothesis-space comparisons."""
    return sample_tilt_power(
        n=70,
        scenario="output_noise",
        seed=70,
        y_noise=0.035,
        sampling="gap65",
    )


def fit_support_model(
    model_kind: str,
    complexity: int,
    lam: float,
) -> tuple[Callable[[np.ndarray], np.ndarray], str, str]:
    """Fit one selected basis model to the fixed support dataset."""
    data = support_dataset()
    complexity = int(complexity)
    if model_kind == "polynomial":
        degree = max(0, min(complexity, 18))
        predict, _ = fit_basis_model(
            data["x"],
            data["y"],
            lambda x: polynomial_design(x, degree),
            lam=lam,
        )
        return predict, f"degree {degree} polynomial", "global polynomial continuation"

    if model_kind == "ReLU basis":
        width = max(1, min(complexity, 18))
        kinks = np.linspace(8, 86, width)
        predict, _ = fit_basis_model(
            data["x"],
            data["y"],
            lambda x: relu_basis_design(x, kinks),
            lam=lam,
        )
        return predict, f"width {width} ReLU basis", "piecewise-linear continuation"

    if model_kind == "periodic basis":
        frequencies = max(1, min(complexity, 8))
        predict, _ = fit_basis_model(
            data["x"],
            data["y"],
            lambda x: periodic_design(x, frequencies),
            lam=lam,
        )
        return predict, f"{frequencies} periodic frequencies", "periodicity imposed by basis"

    raise ValueError(f"unknown model kind: {model_kind}")


def support_diagnostics(
    predict: Callable[[np.ndarray], np.ndarray],
    radius: float = 4.0,
) -> dict[str, float]:
    """Compute training, observed-region, and gap oracle errors."""
    data = support_dataset()
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
    model_kind: str = "polynomial",
    complexity: int = 5,
    lam: float = 0.02,
) -> None:
    """Plot one selected hypothesis class on the fixed data-gap dataset."""
    data = support_dataset()
    predict, label, assumption = fit_support_model(model_kind, complexity, lam)
    counts = local_count(theta_grid, data["x"], radius=4.0)
    observed_mask = counts >= 3
    diagnostics = support_diagnostics(predict)

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
    print(
        "Errors: "
        f"train={diagnostics['training']:.4f}, "
        f"observed-oracle={diagnostics['observed_oracle']:.4f}, "
        f"gap-oracle={diagnostics['gap_oracle']:.4f}"
    )


def what_changed_rows() -> list[list[str]]:
    """Return the what-changed rows for the fixed-data support demo."""
    return [
        ["D", "yes", "no", "same observations and same data gap"],
        ["H", "no", "yes", "the selected feature family and continuation changed"],
        ["O", "mostly yes", "basis-specific penalty geometry", "same squared-loss ridge rule, but basis changes what the penalty prefers"],
        ["Estimand", "yes", "no", "same latent response used for oracle diagnostics"],
        ["Deployment distribution", "yes", "no", "same gap region is used as the stress test"],
    ]


@lru_cache(maxsize=1)
def capacity_data() -> tuple[dict[str, np.ndarray | str], dict[str, np.ndarray | str]]:
    """Return deterministic train and validation data for capacity diagnostics."""
    train = sample_tilt_power(n=80, scenario="output_noise", seed=90, y_noise=0.06, sampling="gap65")
    validation = sample_tilt_power(n=300, scenario="output_noise", seed=91, y_noise=0.06, sampling="uniform")
    return train, validation


def capacity_errors(max_degree: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return degree, training, validation, and oracle errors."""
    train, validation = capacity_data()
    degrees = np.arange(0, int(max_degree) + 1)
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


def plot_capacity_demo(degree: int = 5, show_random_labels: bool = False) -> None:
    """Plot an interactive capacity diagnostic."""
    degree = int(degree)
    train, validation = capacity_data()
    degrees, train_errors, validation_errors, oracle_errors = capacity_errors()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    if show_random_labels:
        random_data = sample_tilt_power(n=35, scenario="output_noise", seed=93, y_noise=0.0, sampling="uniform")
        random_y = np.random.default_rng(93).normal(0.55, 0.24, size=len(random_data["x"]))
        random_predict, _ = fit_basis_model(
            random_data["x"],
            random_y,
            lambda x: polynomial_design(x, 30),
            lam=1e-7,
        )
        ax.scatter(random_data["x"], random_y, color=COLORS["data"], s=34, alpha=0.8, label="random labels")
        ax.plot(theta_grid, random_predict(theta_grid), color=COLORS["fit"], lw=2.4, label="degree 30 fit")
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
    print("Look for whether lower training error also improves the validation or oracle diagnostic.")


def parameter_equivalence(
    transform: str = "permutation",
    alpha: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original and transformed functions for parameter-equivalence demo."""
    original_kinks = np.array([22.0, 48.0, 74.0])
    w = np.array([0.075, -0.060, 0.085])
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


def plot_parameter_equivalence(transform: str = "permutation", alpha: float = 2.0) -> None:
    """Plot two parameterisations that realise the same function."""
    original, transformed = parameter_equivalence(transform, alpha)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(theta_grid, original, color=COLORS["fit"], lw=3, label="original parameters")
    ax.plot(theta_grid, transformed, color=COLORS["data"], lw=2, ls="--", label=transform)
    ax.set_title("Different parameter vectors can realise the same function")
    ax.set_xlabel("x")
    ax.set_ylabel("h(x)")
    ax.set_xlim(0, 90)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    plt.show()
    print(f"Max |h_original(x) - h_transformed(x)| = {np.max(np.abs(original - transformed)):.3e}")


def summary_rows() -> list[list[str]]:
    """Return the summary table rows for the notebook wrap-up."""
    return [
        ["H", "feature maps, ReLU units, width, depth, architecture", "What functions did we make possible or natural?"],
        ["D", "fixed gap65 dataset and local support counts", "Where did observations provide evidence?"],
        ["O", "ridge penalties and selected coefficients", "Which compatible function was selected?"],
        ["s", "the plotted fitted function", "What claim does this realised function support?"],
    ]
