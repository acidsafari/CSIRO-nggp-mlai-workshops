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


def gap_bump(
    x: np.ndarray | float,
    centre: float = 65.0,
    width: float = 4.0,
) -> np.ndarray:
    """Return the local bump used to vary behaviour inside the data gap."""
    return np.exp(-((np.asarray(x) - centre) ** 2) / (2 * width**2))


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


def plot_observability_example(
    n: int = 170,
    seed: int = 21,
    x_noise: float = 4.0,
    y_noise: float = 0.06,
    context_strength: float = 0.35,
    sampling: str = "uniform",
) -> dict[str, Any]:
    """Show learner-visible data beside the hidden-context teaching view."""
    data = sample_tilt_power(
        n=n,
        scenario="hidden_context",
        seed=seed,
        x_noise=x_noise,
        y_noise=y_noise,
        context_strength=context_strength,
        sampling=sampling,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True, sharey=True)

    plot_observed(axes[0], data, reveal_context=False, alpha=0.72)
    style_xy_axis(axes[0])
    axes[0].set_title("What the learner receives")
    axes[0].legend(loc="upper right", fontsize=8)

    plot_observed(axes[1], data, reveal_context=True, alpha=0.72)
    plot_reference(axes[1], label="latent physical curve, not observed")
    style_xy_axis(axes[1])
    axes[1].set_title("Teaching view: hidden context revealed")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()
    return {"data": data}


def plot_claim_targets_example(
    n: int = 180,
    seed: int = 31,
    context_strength: float = 0.35,
    train_context_mode: str = "increasing",
    deployment_context_mode: str = "reversed",
    y_noise: float = 0.04,
    bandwidth: float = 7.0,
) -> dict[str, Any]:
    """Compare observed, physical, and deployment target relationships."""
    data = sample_tilt_power(
        n=n,
        scenario="hidden_context",
        seed=seed,
        y_noise=y_noise,
        context_strength=context_strength,
        sampling="uniform",
        context_mode=train_context_mode,
    )
    deployment_data = sample_tilt_power(
        n=800,
        scenario="hidden_context",
        seed=seed + 1,
        y_noise=0.0,
        context_strength=context_strength,
        sampling="uniform",
        context_mode=deployment_context_mode,
    )

    observed_target = gaussian_smoother(data["x"], data["y"], theta_grid, bandwidth=bandwidth)
    deployment_target = gaussian_smoother(
        deployment_data["x"],
        deployment_data["y"],
        theta_grid,
        bandwidth=bandwidth,
    )
    physical_target = f0(theta_grid)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot_observed(ax, data, reveal_context=False, alpha=0.62)
    ax.plot(theta_grid, observed_target, color=COLORS["fit"], lw=2.8, label="observed prediction target")
    ax.plot(theta_grid, physical_target, color=COLORS["truth"], lw=2.4, ls="--", label="latent physical response")
    ax.plot(theta_grid, deployment_target, color=COLORS["alt"], lw=2.4, ls=":", label="deployment target")
    style_xy_axis(ax)
    ax.set_title("Different claims point to different target relationships")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.show()

    observed_physical_gap = float(np.max(np.abs(observed_target - physical_target)))
    observed_deployment_gap = float(np.max(np.abs(observed_target - deployment_target)))
    print(f"Train context mode: {train_context_mode}; deployment context mode: {deployment_context_mode}")
    print(f"Max |observed target - physical target|: {observed_physical_gap:.3f}")
    print(f"Max |observed target - deployment target|: {observed_deployment_gap:.3f}")
    return {
        "data": data,
        "deployment_data": deployment_data,
        "observed_physical_gap": observed_physical_gap,
        "observed_deployment_gap": observed_deployment_gap,
    }


def plot_local_support_example(
    n: int = 105,
    seed: int = 45,
    y_noise: float = 0.055,
    sampling: str = "gap65",
    radius: float = 4.0,
    min_count: int = 6,
    bandwidth: float = 6.0,
) -> dict[str, Any]:
    """Plot a smoother with the local evidence count underneath."""
    data = sample_tilt_power(
        n=n,
        scenario="output_noise",
        seed=seed,
        y_noise=y_noise,
        sampling=sampling,
    )

    counts = local_count(theta_grid, data["x"], radius)
    weak_support = counts < min_count
    smooth = gaussian_smoother(data["x"], data["y"], theta_grid, bandwidth=bandwidth)

    fig, (ax, ax_count) = plt.subplots(
        2,
        1,
        figsize=(9, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for start, end, weak in zip(theta_grid[:-1], theta_grid[1:], weak_support[:-1]):
        if weak:
            ax.axvspan(start, end, color=COLORS["support"], alpha=0.35, lw=0)
            ax_count.axvspan(start, end, color=COLORS["support"], alpha=0.35, lw=0)

    plot_observed(ax, data, alpha=0.78)
    ax.plot(theta_grid, smooth, color=COLORS["fit"], lw=2.8, label="fitted smoother")
    ax.text(24, 1.25, "evidence-driven", color=COLORS["data"], fontsize=10, weight="bold")
    ax.text(58.5, 1.25, "assumption-driven", color=COLORS["fit"], fontsize=10, weight="bold")
    style_xy_axis(ax)
    ax.set_title("A fitted curve extends beyond local evidence")
    ax.legend(loc="upper right", fontsize=8)

    ax_count.plot(theta_grid, counts, color=COLORS["data"], lw=2.2, label="local count")
    ax_count.axhline(min_count, color=COLORS["fit"], lw=1.5, ls="--", label="weak-support threshold")
    ax_count.fill_between(theta_grid, 0, counts, where=weak_support, color=COLORS["support"], alpha=0.55)
    ax_count.set_ylabel("n_r(x)")
    ax_count.set_xlabel("observed tilt x")
    ax_count.set_ylim(0, max(18, counts.max() + 2))
    ax_count.grid(alpha=0.2)
    ax_count.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()
    return {"data": data, "counts": counts, "weak_support": weak_support}


def plot_resolution_diagnostic_example(
    n: int = 120,
    seed: int = 61,
    y_noise: float = 0.08,
    sampling: str = "sparse_feature",
    radius: float = 5.0,
    min_count: int = 5,
    min_signal_to_uncertainty: float = 2.0,
    claim_region: tuple[float, float] = (58, 72),
    bandwidth: float = 6.0,
) -> dict[str, Any]:
    """Show whether local evidence resolves the task-relevant feature."""
    data = sample_tilt_power(
        n=n,
        scenario="output_noise",
        seed=seed,
        y_noise=y_noise,
        sampling=sampling,
    )

    smooth = gaussian_smoother(data["x"], data["y"], theta_grid, bandwidth=bandwidth)
    counts = local_count(theta_grid, data["x"], radius)
    local_sd_values = local_std(theta_grid, data["x"], data["y"], radius)
    local_change = local_variation(f0(theta_grid), theta_grid, radius)
    standard_error = local_sd_values / np.sqrt(np.maximum(counts, 1))
    rho = np.divide(
        local_change,
        standard_error,
        out=np.full_like(theta_grid, np.nan),
        where=np.isfinite(standard_error) & (standard_error > 0),
    )
    weak_resolution = (counts < min_count) | (~np.isfinite(rho)) | (rho < min_signal_to_uncertainty)
    claim_mask = (theta_grid >= claim_region[0]) & (theta_grid <= claim_region[1])

    fig, (ax, ax_score) = plt.subplots(
        2,
        1,
        figsize=(9, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.axvspan(*claim_region, color=COLORS["support"], alpha=0.35, label="claim region")
    plot_observed(ax, data, alpha=0.72)
    plot_reference(ax, label="latent reference")
    ax.plot(theta_grid, smooth, color=COLORS["fit"], lw=2.6, label="observed smoother")
    style_xy_axis(ax)
    ax.set_title("Coverage is not the same as resolving the task-relevant feature")
    ax.legend(loc="upper right", fontsize=8)

    ax_score.axvspan(*claim_region, color=COLORS["support"], alpha=0.35)
    ax_score.plot(theta_grid, rho, color=COLORS["data"], lw=2.2, label="rho_r(x)")
    ax_score.axhline(
        min_signal_to_uncertainty,
        color=COLORS["fit"],
        lw=1.5,
        ls="--",
        label="resolution threshold",
    )
    ax_score.fill_between(
        theta_grid,
        0,
        np.nan_to_num(rho, nan=0.0),
        where=weak_resolution,
        color=COLORS["support"],
        alpha=0.55,
    )
    ax_score.set_ylabel("rho_r(x)")
    ax_score.set_xlabel("observed tilt x")
    ax_score.set_ylim(0, max(6, np.nanpercentile(rho, 95) + 0.5))
    ax_score.grid(alpha=0.2)
    ax_score.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()

    median_count = float(np.median(counts[claim_mask]))
    median_rho = float(np.nanmedian(rho[claim_mask]))
    print(f"Median local count in claim region: {median_count:.1f}")
    print(f"Median resolution score in claim region: {median_rho:.2f}")
    return {
        "data": data,
        "counts": counts,
        "rho": rho,
        "median_count": median_count,
        "median_rho": median_rho,
    }


def plot_ambiguity_example(
    n: int = 78,
    seed: int = 70,
    y_noise: float = 0.035,
    gap_region: tuple[float, float] = (58, 72),
    bump_strengths: list[float] | tuple[float, ...] = (-0.13, 0.0, 0.15),
) -> dict[str, Any]:
    """Show candidate functions that agree on samples but disagree in a gap."""
    data = sample_tilt_power(
        n=n,
        scenario="output_noise",
        seed=seed,
        y_noise=y_noise,
        sampling="gap65",
    )

    bump_centre = sum(gap_region) / 2
    bump_width = (gap_region[1] - gap_region[0]) / 3
    colors = [COLORS["fit"], COLORS["truth"], COLORS["alt"]]
    linestyles = ["-", "--", "-"]
    claim_mask = (theta_grid >= gap_region[0]) & (theta_grid <= gap_region[1])
    curve_values = []

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axvspan(*gap_region, color=COLORS["support"], alpha=0.35, label="claim region with little data")
    plot_observed(ax, data, alpha=0.78)

    for strength, color, linestyle in zip(bump_strengths, colors, linestyles):
        y_grid = f0(theta_grid) + strength * gap_bump(theta_grid, centre=bump_centre, width=bump_width)
        y_train = f0(data["x"]) + strength * gap_bump(data["x"], centre=bump_centre, width=bump_width)
        train_mse = np.mean((y_train - data["y"]) ** 2)
        curve_values.append(y_grid)
        ax.plot(
            theta_grid,
            y_grid,
            color=color,
            lw=2.4,
            ls=linestyle,
            label=f"bump strength {strength:+.2f}: train MSE {train_mse:.4f}",
        )

    curve_values = np.asarray(curve_values)
    disagreement = curve_values[:, claim_mask].max(axis=0) - curve_values[:, claim_mask].min(axis=0)
    max_disagreement = float(disagreement.max())

    style_xy_axis(ax)
    ax.set_title("Similar training error can leave the claim region ambiguous")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.show()

    print(f"Maximum candidate disagreement inside {gap_region}: {max_disagreement:.3f}")
    return {"data": data, "max_disagreement": max_disagreement}


def plot_context_shift_example(
    n_train: int = 180,
    n_test: int = 140,
    seed: int = 91,
    context_strength: float = 0.35,
    train_context_mode: str = "increasing",
    balanced_context_mode: str = "independent",
    shifted_context_mode: str = "reversed",
    y_noise: float = 0.05,
    bandwidth: float = 7.0,
) -> dict[str, Any]:
    """Evaluate one smoother under matched, balanced, and shifted contexts."""
    train_data = sample_tilt_power(
        n=n_train,
        scenario="hidden_context",
        seed=seed,
        y_noise=y_noise,
        context_strength=context_strength,
        sampling="uniform",
        context_mode=train_context_mode,
    )

    test_sets = [
        (
            "random test: same mixture",
            sample_tilt_power(
                n=n_test,
                scenario="hidden_context",
                seed=seed + 1,
                y_noise=y_noise,
                context_strength=context_strength,
                sampling="uniform",
                context_mode=train_context_mode,
            ),
        ),
        (
            "context-balanced test",
            sample_tilt_power(
                n=n_test,
                scenario="hidden_context",
                seed=seed + 2,
                y_noise=y_noise,
                context_strength=context_strength,
                sampling="uniform",
                context_mode=balanced_context_mode,
            ),
        ),
        (
            "shifted test: reversed mixture",
            sample_tilt_power(
                n=n_test,
                scenario="hidden_context",
                seed=seed + 3,
                y_noise=y_noise,
                context_strength=context_strength,
                sampling="uniform",
                context_mode=shifted_context_mode,
            ),
        ),
    ]

    train_curve = gaussian_smoother(train_data["x"], train_data["y"], theta_grid, bandwidth=bandwidth)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    metrics = []

    for ax, (name, data) in zip(axes, test_sets):
        y_pred = gaussian_smoother(train_data["x"], train_data["y"], data["x"], bandwidth=bandwidth)
        rmse = float(np.sqrt(np.mean((y_pred - data["y"]) ** 2)))
        frac_context_one = float(np.mean(data["c"] == 1))
        test_target = gaussian_smoother(data["x"], data["y"], theta_grid, bandwidth=bandwidth)
        metrics.append({"name": name, "rmse": rmse, "frac_context_one": frac_context_one})

        plot_observed(ax, data, reveal_context=False, alpha=0.50, s=22)
        ax.plot(theta_grid, train_curve, color=COLORS["fit"], lw=2.5, label="trained on original data")
        ax.plot(theta_grid, test_target, color=COLORS["alt"], lw=2.2, ls="--", label="test-set relationship")
        style_xy_axis(ax)
        ax.set_title(f"{name}\nRMSE {rmse:.3f}; frac C=1 {frac_context_one:.2f}")
        ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    plt.show()

    for row in metrics:
        print(f"{row['name']}: RMSE={row['rmse']:.3f}, fraction C=1={row['frac_context_one']:.2f}")
    return {"train_data": train_data, "metrics": metrics}


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
