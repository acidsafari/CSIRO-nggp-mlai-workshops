"""Reusable helpers for the Optimization Space notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .data_space import f0, sample_tilt_power, theta_grid

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - only used outside notebooks
    Markdown = None
    display = print

SEED = 7

COLORS = {
    "data": "#2F5D7C",
    "truth": "#222222",
    "selected": "#C7502A",
    "alt": "#7B5E9E",
    "regularised": "#2F7D4A",
    "sgd": "#D18F24",
    "support": "#D9D9D9",
    "validation": "#5B8CC0",
}

LINE_GRID = np.linspace(-3.2, 3.2, 400)
RESIDUAL_GRID = np.linspace(-4.0, 4.0, 400)
VIRIDIS_CMAP = "viridis"


def configure_optimization_matplotlib() -> None:
    """Apply the Matplotlib defaults used by the optimisation notebook."""
    plt.rcParams.update({
        "figure.figsize": (7, 4),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "legend.frameon": False,
    })


def set_seed(seed=SEED):
    return np.random.default_rng(seed)


def display_markdown(text):
    if Markdown is None:
        print(text)
    else:
        display(Markdown(text))


def format_cell(value):
    if isinstance(value, np.ndarray):
        return "[" + ", ".join(f"{v:.3f}" for v in value) + "]"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    text = str(value)
    return text.replace("|", "\\|")


def markdown_table(headers, rows):
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(format_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def display_table(headers, rows):
    display_markdown(markdown_table(headers, rows))


def make_axis(title, xlabel="x", ylabel="y", figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def true_linear_function(x):
    return 0.35 + 0.8 * np.asarray(x)


def make_linear_dataset(n=28, seed=SEED, outlier=True):
    rng = set_seed(seed)
    x = np.sort(rng.uniform(-2.6, 2.6, n))
    y = true_linear_function(x) + rng.normal(0.0, 0.22, n)
    if outlier:
        x = np.concatenate([x, [2.35]])
        y = np.concatenate([y, [-2.15]])
    return x, y


def linear_design(x):
    x = np.asarray(x)
    return np.column_stack([np.ones_like(x), x])


def linear_predictions(X, theta):
    return X @ theta


def squared_loss(pred, y):
    return (pred - y) ** 2


def absolute_loss(pred, y):
    return np.abs(pred - y)


def huber_loss(residual, delta=1.0):
    residual = np.asarray(residual)
    abs_r = np.abs(residual)
    return np.where(abs_r <= delta, 0.5 * residual**2, delta * (abs_r - 0.5 * delta))


def mse_objective(X, y, theta, lam=0.0):
    residual = X @ theta - y
    return np.mean(residual**2) + lam * np.sum(theta**2)


def mae_objective(X, y, theta):
    return np.mean(np.abs(X @ theta - y))


def mse_gradient(X, y, theta, lam=0.0):
    n = X.shape[0]
    return (2 / n) * X.T @ (X @ theta - y) + 2 * lam * theta


def ridge_solution(X, y, lam=0.0):
    n, p = X.shape
    gram = X.T @ X + n * lam * np.eye(p)
    rhs = X.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def fit_linear_absolute_grid(x, y, intercept_range=(-1.8, 1.8), slope_range=(-0.3, 1.8), points=241):
    intercepts = np.linspace(*intercept_range, points)
    slopes = np.linspace(*slope_range, points)
    best_loss = np.inf
    best_theta = None
    for intercept in intercepts:
        pred = intercept + slopes[:, None] * x[None, :]
        losses = np.mean(np.abs(pred - y[None, :]), axis=1)
        idx = int(np.argmin(losses))
        if losses[idx] < best_loss:
            best_loss = float(losses[idx])
            best_theta = np.array([intercept, slopes[idx]])
    return best_theta, best_loss


def gradient_descent(theta0, grad_fn, obj_fn, eta, steps):
    theta = np.array(theta0, dtype=float)
    path = [theta.copy()]
    values = [float(obj_fn(theta))]
    for _ in range(steps):
        theta = theta - eta * grad_fn(theta)
        path.append(theta.copy())
        values.append(float(obj_fn(theta)))
    return np.asarray(path), np.asarray(values)


def sgd_path(X, y, theta0, eta, steps, batch_size, seed=SEED, lam=0.0):
    rng = set_seed(seed)
    theta = np.array(theta0, dtype=float)
    path = [theta.copy()]
    values = [mse_objective(X, y, theta, lam=lam)]
    n = X.shape[0]
    replace = batch_size > n
    for _ in range(steps):
        idx = rng.choice(n, size=batch_size, replace=replace)
        theta = theta - eta * mse_gradient(X[idx], y[idx], theta, lam=lam)
        path.append(theta.copy())
        values.append(mse_objective(X, y, theta, lam=lam))
    return np.asarray(path), np.asarray(values)


def plot_observed(ax, x, y, label="observed data", color=None, alpha=0.88):
    ax.scatter(x, y, s=35, color=color or COLORS["data"], edgecolor="white", linewidth=0.4, alpha=alpha, label=label)


def plot_linear_fit(ax, theta, x_grid=LINE_GRID, label=None, color=None, lw=2.5, ls="-"):
    X_grid = linear_design(x_grid)
    ax.plot(x_grid, X_grid @ theta, color=color or COLORS["selected"], lw=lw, ls=ls, label=label)


def objective_grid_linear(x, y, intercepts, slopes, loss="mse", lam=0.0):
    I, S = np.meshgrid(intercepts, slopes, indexing="ij")
    residual = I[..., None] + S[..., None] * x[None, None, :] - y[None, None, :]
    if loss == "mae":
        Z = np.mean(np.abs(residual), axis=2)
    else:
        Z = np.mean(residual**2, axis=2)
    if lam:
        Z = Z + lam * (I**2 + S**2)
    return I, S, Z


def status_from_eta(eta, a):
    factor = abs(1 - eta * a)
    if eta <= 0:
        return "invalid"
    if factor < 0.25:
        return "fast stable"
    if factor < 1 and (1 - eta * a) < 0:
        return "oscillating stable"
    if factor < 1:
        return "slow stable"
    return "divergent"


def describe_lambda(lam):
    if lam == 0:
        return "least constrained fit"
    if lam < 1e-3:
        return "weak norm preference"
    if lam < 0.1:
        return "smoother compatible fit"
    return "strong norm preference"


def make_loss_mesh(theta0_range, theta1_range, points=120):
    theta0_values = np.linspace(theta0_range[0], theta0_range[1], points)
    theta1_values = np.linspace(theta1_range[0], theta1_range[1], points)
    return np.meshgrid(theta0_values, theta1_values, indexing="ij")


def evaluate_objective_mesh(T0, T1, objective_fn):
    flat = np.column_stack([T0.ravel(), T1.ravel()])
    values = np.array([objective_fn(theta) for theta in flat])
    return values.reshape(T0.shape)


def plot_3d_landscape(ax, T0, T1, Z, title, path=None, objective_fn=None, selected=None):
    ax.plot_surface(T0, T1, Z, cmap=VIRIDIS_CMAP, alpha=0.88, linewidth=0, antialiased=True)
    if path is not None and objective_fn is not None:
        path = np.asarray(path)
        z_path = np.array([objective_fn(theta) for theta in path])
        ax.plot(path[:, 0], path[:, 1], z_path, color=COLORS["selected"], lw=2.4, marker="o", ms=3.0, label="optimisation path")
    if selected is not None and objective_fn is not None:
        selected = np.asarray(selected)
        ax.scatter(selected[0], selected[1], objective_fn(selected), color=COLORS["selected"], s=55, label=r"$\theta_T$")
    ax.set_title(title)
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_zlabel(r"$J(\theta)$")
    ax.view_init(elev=28, azim=-132)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=7)


def format_theta(theta):
    theta = np.asarray(theta, dtype=float)
    return f"({theta[0]:.3f}, {theta[1]:.3f})"


def loss_values(residual, loss_name="squared", huber_delta=1.0):
    residual = np.asarray(residual, dtype=float)
    if loss_name == "squared":
        return residual**2
    if loss_name == "absolute":
        return np.abs(residual)
    if loss_name == "huber":
        return huber_loss(residual, delta=huber_delta)
    raise ValueError("loss_name must be 'squared', 'absolute', or 'huber'")


def linear_empirical_loss(X, y, theta, loss_name="squared", huber_delta=1.0):
    residual = X @ np.asarray(theta, dtype=float) - y
    return float(np.mean(loss_values(residual, loss_name=loss_name, huber_delta=huber_delta)))


def linear_objective(X, y, theta, loss_name="squared", lambda_ridge=0.0, huber_delta=1.0):
    theta = np.asarray(theta, dtype=float)
    risk = linear_empirical_loss(X, y, theta, loss_name=loss_name, huber_delta=huber_delta)
    penalty = float(lambda_ridge * np.sum(theta**2))
    return risk + penalty


def objective_grid_for_loss(
    x,
    y,
    theta0_range,
    theta1_range,
    loss_name="squared",
    lambda_ridge=0.0,
    huber_delta=1.0,
    grid_size=120,
):
    T0, T1 = make_loss_mesh(theta0_range, theta1_range, points=grid_size)
    residual = T0[..., None] + T1[..., None] * np.asarray(x)[None, None, :] - np.asarray(y)[None, None, :]
    Z = np.mean(loss_values(residual, loss_name=loss_name, huber_delta=huber_delta), axis=2)
    if lambda_ridge:
        Z = Z + lambda_ridge * (T0**2 + T1**2)
    return T0, T1, Z


def fit_linear_loss(
    x,
    y,
    loss_name="squared",
    lambda_ridge=0.0,
    huber_delta=1.0,
    theta0_range=(-2.5, 2.5),
    theta1_range=(-1.0, 2.0),
    grid_size=281,
):
    X = linear_design(x)
    if loss_name == "squared":
        theta = ridge_solution(X, y, lam=lambda_ridge)
        return (
            theta,
            linear_empirical_loss(X, y, theta, loss_name, huber_delta),
            linear_objective(X, y, theta, loss_name, lambda_ridge, huber_delta),
        )

    T0, T1, Z = objective_grid_for_loss(
        x,
        y,
        theta0_range,
        theta1_range,
        loss_name=loss_name,
        lambda_ridge=lambda_ridge,
        huber_delta=huber_delta,
        grid_size=grid_size,
    )
    idx = np.unravel_index(int(np.argmin(Z)), Z.shape)
    theta = np.array([T0[idx], T1[idx]])
    return theta, linear_empirical_loss(X, y, theta, loss_name, huber_delta), float(Z[idx])


def plot_2d_landscape(
    ax,
    T0,
    T1,
    Z,
    title,
    levels=None,
    path=None,
    selected=None,
    path_label="optimisation path",
    selected_label=r"$\theta_T$",
    path_color=None,
    selected_color=None,
    alpha=1.0,
    legend_fontsize=8,
):
    if levels is None:
        levels = np.linspace(float(np.min(Z)), float(np.quantile(Z, 0.92)), 22)
    ax.contourf(T0, T1, Z, levels=levels, cmap=VIRIDIS_CMAP, alpha=alpha)
    ax.contour(T0, T1, Z, levels=levels, colors="white", alpha=0.35, linewidths=0.6)
    if path is not None:
        path = np.asarray(path)
        ax.plot(
            path[:, 0],
            path[:, 1],
            marker="o",
            ms=3.0,
            lw=1.6,
            color=path_color or COLORS["selected"],
            label=path_label,
        )
    if selected is not None:
        selected = np.asarray(selected)
        ax.scatter(
            selected[0],
            selected[1],
            color=selected_color or COLORS["selected"],
            s=55,
            label=selected_label,
            zorder=5,
        )
    ax.set_title(title)
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=legend_fontsize)
    return levels


def plot_fit_with_residuals(ax, x, y, theta, title, label="selected fit", color=None, x_grid=None):
    if x_grid is None:
        x_grid = LINE_GRID
    theta = np.asarray(theta, dtype=float)
    y_hat = linear_design(x) @ theta
    plot_observed(ax, x, y, label="observed data")
    for xi, yi, pred in zip(x, y, y_hat):
        ax.plot([xi, xi], [yi, pred], color=color or COLORS["selected"], alpha=0.35, lw=1.1)
    plot_linear_fit(ax, theta, x_grid=x_grid, label=label, color=color or COLORS["selected"])
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)


def minibatch_training_path(X, y, theta0, learning_rate, num_updates, batch_size, seed=SEED, shuffle_each_epoch=True, lam=0.0):
    rng = set_seed(seed)
    theta = np.asarray(theta0, dtype=float).copy()
    path = [theta.copy()]
    values = [mse_objective(X, y, theta, lam=lam)]
    n = len(y)
    batch_size = int(max(1, min(batch_size, n)))
    order = rng.permutation(n)
    cursor = 0
    for _ in range(num_updates):
        if shuffle_each_epoch:
            if cursor + batch_size > n:
                order = rng.permutation(n)
                cursor = 0
            idx = order[cursor : cursor + batch_size]
            cursor += batch_size
        else:
            idx = rng.choice(n, size=batch_size, replace=False)
        theta = theta - learning_rate * mse_gradient(X[idx], y[idx], theta, lam=lam)
        path.append(theta.copy())
        values.append(mse_objective(X, y, theta, lam=lam))
    return np.asarray(path), np.asarray(values)


def finite_path_inside(path, theta0_range, theta1_range):
    path = np.asarray(path, dtype=float)
    mask = np.isfinite(path).all(axis=1)
    mask &= (path[:, 0] >= theta0_range[0]) & (path[:, 0] <= theta0_range[1])
    mask &= (path[:, 1] >= theta1_range[0]) & (path[:, 1] <= theta1_range[1])
    return path[mask]


def fit_squared_and_absolute(x, y):
    X = linear_design(x)
    theta_sq = ridge_solution(X, y, lam=0.0)
    theta_abs, _ = fit_linear_absolute_grid(x, y)
    return theta_sq, theta_abs


def quadratic_path(A, theta0, theta_star, eta, steps):
    def obj(theta):
        d = theta - theta_star
        return 0.5 * d @ A @ d
    def grad(theta):
        return A @ (theta - theta_star)
    return gradient_descent(theta0, grad, obj, eta, steps), obj


def nonconvex_objective(theta):
    theta = np.asarray(theta)
    theta0 = theta[..., 0]
    theta1 = theta[..., 1]
    return 0.08 * theta0**2 + 0.06 * theta1**2 + np.sin(3 * theta0) ** 2 + 0.7 * np.sin(3 * theta1) ** 2 + 0.15 * np.sin(2 * theta0 + theta1)


def nonconvex_gradient(theta):
    theta0, theta1 = theta
    d0 = 0.16 * theta0 + 3.0 * np.sin(6 * theta0) + 0.30 * np.cos(2 * theta0 + theta1)
    d1 = 0.12 * theta1 + 2.1 * np.sin(6 * theta1) + 0.15 * np.cos(2 * theta0 + theta1)
    return np.array([d0, d1])


def make_mlp_regression_data(n=40, noise=0.08, seed=0):
    """Small one-dimensional regression dataset for MLP optimisation examples."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, size=(n, 1)), axis=0)
    y_clean = np.sin(3.0 * np.pi * x) + 0.3 * x
    y = y_clean + noise * rng.normal(size=y_clean.shape)
    return x, y, y_clean


def make_mlp_grid(n=400):
    x_grid = np.linspace(-1.1, 1.1, n).reshape(-1, 1)
    y_true = np.sin(3.0 * np.pi * x_grid) + 0.3 * x_grid
    return x_grid, y_true


def activation_forward(z, activation="relu"):
    if activation == "relu":
        return np.maximum(0.0, z)
    if activation == "tanh":
        return np.tanh(z)
    raise ValueError("activation must be 'relu' or 'tanh'")


def activation_backward(z, activation="relu"):
    if activation == "relu":
        return (z > 0.0).astype(float)
    if activation == "tanh":
        a = np.tanh(z)
        return 1.0 - a**2
    raise ValueError("activation must be 'relu' or 'tanh'")


def initialise_mlp(width=20, init_mode="he", init_scale=1.0, seed=0, bias_shift=0.0):
    """Initialise a one-hidden-layer MLP with scalar input and scalar output."""
    rng = np.random.default_rng(seed)

    if init_mode == "small":
        s1 = init_scale * 0.05
        s2 = init_scale * 0.05
    elif init_mode == "xavier":
        s1 = init_scale * np.sqrt(1.0 / 1.0)
        s2 = init_scale * np.sqrt(1.0 / width)
    elif init_mode == "he":
        s1 = init_scale * np.sqrt(2.0 / 1.0)
        s2 = init_scale * np.sqrt(2.0 / width)
    elif init_mode == "large":
        s1 = init_scale * 2.0
        s2 = init_scale * 2.0 / np.sqrt(width)
    elif init_mode == "zero":
        s1 = 0.0
        s2 = 0.0
    else:
        raise ValueError("init_mode must be 'small', 'xavier', 'he', 'large', or 'zero'")

    return {
        "W1": rng.normal(0.0, s1, size=(1, width)),
        "b1": np.full((1, width), bias_shift, dtype=float),
        "W2": rng.normal(0.0, s2, size=(width, 1)),
        "b2": np.zeros((1, 1), dtype=float),
    }


def forward_mlp(params, x, activation="relu"):
    z1 = x @ params["W1"] + params["b1"]
    a1 = activation_forward(z1, activation=activation)
    yhat = a1 @ params["W2"] + params["b2"]
    cache = {"x": x, "z1": z1, "a1": a1, "yhat": yhat}
    return yhat, cache


def mse_loss(yhat, y):
    return float(np.mean((yhat - y) ** 2))


def backward_mlp(params, cache, y, activation="relu", weight_decay=0.0):
    x = cache["x"]
    z1 = cache["z1"]
    a1 = cache["a1"]
    yhat = cache["yhat"]
    n = x.shape[0]

    dyhat = 2.0 * (yhat - y) / n

    grads = {}
    grads["W2"] = a1.T @ dyhat + 2.0 * weight_decay * params["W2"]
    grads["b2"] = np.sum(dyhat, axis=0, keepdims=True)

    da1 = dyhat @ params["W2"].T
    dz1 = da1 * activation_backward(z1, activation=activation)

    grads["W1"] = x.T @ dz1 + 2.0 * weight_decay * params["W1"]
    grads["b1"] = np.sum(dz1, axis=0, keepdims=True)

    return grads


def gradient_norm(grads):
    return float(np.sqrt(sum(np.sum(g**2) for g in grads.values())))


def hidden_gradient_norm(grads):
    """Return the gradient norm for hidden-layer learning, excluding output bias."""
    return float(np.sqrt(np.sum(grads["W1"] ** 2) + np.sum(grads["b1"] ** 2) + np.sum(grads["W2"] ** 2)))


def parameter_norm(params):
    return float(np.sqrt(sum(np.sum(p**2) for p in params.values())))


def active_fraction(cache, activation="relu"):
    if activation == "relu":
        return float(np.mean(cache["z1"] > 0.0))
    if activation == "tanh":
        return float(np.mean(np.abs(cache["z1"]) < 2.0))
    return np.nan


def copy_params(params):
    return {key: value.copy() for key, value in params.items()}


def train_mlp(
    x,
    y,
    width=20,
    activation="relu",
    init_mode="he",
    init_scale=1.0,
    bias_shift=0.0,
    seed=0,
    learning_rate=0.03,
    epochs=1500,
    weight_decay=0.0,
    batch_size=None,
    log_every=10,
):
    params = initialise_mlp(
        width=width,
        init_mode=init_mode,
        init_scale=init_scale,
        seed=seed,
        bias_shift=bias_shift,
    )

    rng = np.random.default_rng(seed + 10_000)
    history = {
        "epoch": [],
        "loss": [],
        "grad_norm": [],
        "param_norm": [],
        "active_fraction": [],
    }

    n = x.shape[0]

    for epoch in range(epochs + 1):
        yhat, cache = forward_mlp(params, x, activation=activation)
        loss = mse_loss(yhat, y)

        if epoch % log_every == 0 or epoch == epochs:
            grads_full = backward_mlp(
                params,
                cache,
                y,
                activation=activation,
                weight_decay=weight_decay,
            )
            history["epoch"].append(epoch)
            history["loss"].append(loss)
            history["grad_norm"].append(gradient_norm(grads_full))
            history["param_norm"].append(parameter_norm(params))
            history["active_fraction"].append(active_fraction(cache, activation=activation))

        if epoch == epochs:
            break

        if batch_size is None or batch_size >= n:
            xb, yb = x, y
        else:
            idx = rng.choice(n, size=batch_size, replace=False)
            xb, yb = x[idx], y[idx]

        yhat_b, cache_b = forward_mlp(params, xb, activation=activation)
        grads = backward_mlp(
            params,
            cache_b,
            yb,
            activation=activation,
            weight_decay=weight_decay,
        )

        for key in params:
            params[key] -= learning_rate * grads[key]

        if not all(np.all(np.isfinite(value)) for value in params.values()):
            break

    return params, history


def plot_mlp_fit(ax, params, x, y, x_grid, y_true=None, activation="relu", label="MLP", color=None):
    y_grid, _ = forward_mlp(params, x_grid, activation=activation)
    ax.scatter(x[:, 0], y[:, 0], s=24, color=COLORS["data"], edgecolor="white", linewidth=0.4, label="observed data")
    if y_true is not None:
        ax.plot(x_grid[:, 0], y_true[:, 0], color=COLORS["truth"], linestyle="--", label="true curve")
    ax.plot(x_grid[:, 0], y_grid[:, 0], color=color, label=label)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(fontsize=8)
    return y_grid


def make_mlp_tilt_power_data(
    n=80,
    seed=42,
    sampling="sparse_feature",
    scenario="output_noise",
    y_noise=0.045,
    x_noise=0.0,
):
    """Return the shared tilt-power sample in raw and standardized MLP input coordinates."""
    data = sample_tilt_power(
        n=n,
        scenario=scenario,
        seed=seed,
        x_noise=x_noise,
        y_noise=y_noise,
        sampling=sampling,
    )
    order = np.argsort(np.asarray(data["x"], dtype=float))
    x_raw = np.asarray(data["x"], dtype=float)[order].reshape(-1, 1)
    y = np.asarray(data["y"], dtype=float)[order].reshape(-1, 1)
    theta = np.asarray(data["theta"], dtype=float)[order].reshape(-1, 1)
    context = np.asarray(data["c"], dtype=int)[order].reshape(-1, 1)

    x_mean = float(np.mean(x_raw))
    x_std = float(max(np.std(x_raw), 1e-6))
    x_scaled = (x_raw - x_mean) / x_std

    grid_raw = theta_grid.reshape(-1, 1)
    grid_scaled = (grid_raw - x_mean) / x_std
    y_true = f0(theta_grid).reshape(-1, 1)

    return {
        "raw_data": data,
        "theta": theta,
        "context": context,
        "x_raw": x_raw,
        "x_scaled": x_scaled,
        "y": y,
        "grid_raw": grid_raw,
        "grid_scaled": grid_scaled,
        "y_true": y_true,
        "x_mean": x_mean,
        "x_std": x_std,
        "sampling": sampling,
        "scenario": scenario,
    }


def _tilt_power_inputs(sample, input_mode="scaled"):
    if input_mode == "scaled":
        return sample["x_scaled"], sample["grid_scaled"]
    if input_mode == "raw":
        return sample["x_raw"], sample["grid_raw"]
    raise ValueError("input_mode must be 'scaled' or 'raw'")


def mlp_diagnostics(params, x, y, x_grid, y_true, activation="relu", weight_decay=0.0):
    """Summarise fit, gradient, and function-space diagnostics for one MLP state."""
    yhat, cache = forward_mlp(params, x, activation=activation)
    y_grid, _ = forward_mlp(params, x_grid, activation=activation)
    grads = backward_mlp(params, cache, y, activation=activation, weight_decay=weight_decay)
    return {
        "train_mse": mse_loss(yhat, y),
        "grid_oracle_mse": mse_loss(y_grid, y_true),
        "output_std": float(np.std(y_grid)),
        "grad_norm": gradient_norm(grads),
        "hidden_grad_norm": hidden_gradient_norm(grads),
        "param_norm": parameter_norm(params),
        "activity": active_fraction(cache, activation=activation),
    }


def train_mlp_tilt_power_recipe(
    sample,
    input_mode="scaled",
    width=24,
    activation="relu",
    init_mode="he",
    init_scale=1.0,
    bias_shift=0.0,
    seed=0,
    learning_rate=0.03,
    epochs=800,
    weight_decay=0.0,
    batch_size=None,
    log_every=100,
):
    """Train one MLP recipe on the shared tilt-power sample and return diagnostics."""
    x, x_grid = _tilt_power_inputs(sample, input_mode=input_mode)
    y = sample["y"]
    y_true = sample["y_true"]

    initial_params = initialise_mlp(
        width=width,
        init_mode=init_mode,
        init_scale=init_scale,
        seed=seed,
        bias_shift=bias_shift,
    )
    initial_diagnostics = mlp_diagnostics(
        initial_params,
        x,
        y,
        x_grid,
        y_true,
        activation=activation,
        weight_decay=weight_decay,
    )

    params, history = train_mlp(
        x,
        y,
        width=width,
        activation=activation,
        init_mode=init_mode,
        init_scale=init_scale,
        bias_shift=bias_shift,
        seed=seed,
        learning_rate=learning_rate,
        epochs=epochs,
        weight_decay=weight_decay,
        batch_size=batch_size,
        log_every=log_every,
    )
    final_diagnostics = mlp_diagnostics(
        params,
        x,
        y,
        x_grid,
        y_true,
        activation=activation,
        weight_decay=weight_decay,
    )

    return {
        "params": params,
        "history": history,
        "initial_diagnostics": initial_diagnostics,
        "final_diagnostics": final_diagnostics,
        "input_mode": input_mode,
        "activation": activation,
        "width": int(width),
        "init_mode": init_mode,
        "init_scale": float(init_scale),
        "bias_shift": float(bias_shift),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "weight_decay": float(weight_decay),
    }


def plot_mlp_tilt_power_fit(
    ax,
    sample,
    params,
    input_mode="scaled",
    activation="relu",
    label="MLP",
    color=None,
    show_data=True,
    show_truth=True,
    show_feature=True,
):
    """Plot an MLP trained in raw or scaled coordinates back on the raw tilt axis."""
    _, x_grid = _tilt_power_inputs(sample, input_mode=input_mode)
    y_grid, _ = forward_mlp(params, x_grid, activation=activation)

    if show_data:
        ax.scatter(
            sample["x_raw"][:, 0],
            sample["y"][:, 0],
            s=24,
            color=COLORS["data"],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.82,
            label="observed sample",
        )
    if show_truth:
        ax.plot(sample["grid_raw"][:, 0], sample["y_true"][:, 0], color=COLORS["truth"], lw=2.0, ls="--", label="latent f0")
    ax.plot(sample["grid_raw"][:, 0], y_grid[:, 0], color=color or COLORS["selected"], lw=2.3, label=label)
    if show_feature:
        ax.axvspan(60, 70, color=COLORS["support"], alpha=0.25, lw=0, label="narrow feature")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.2, 1.45)
    ax.set_xlabel("observed tilt X")
    ax.set_ylabel("observed power Y")
    ax.legend(fontsize=8)
    return y_grid


def print_mlp_summary(name, params, history, activation="relu"):
    final_loss = history["loss"][-1]
    final_grad = history["grad_norm"][-1]
    final_norm = history["param_norm"][-1]
    final_active = history["active_fraction"][-1]
    print(f"{name}")
    print(f"  final loss       = {final_loss:.5f}")
    print(f"  final grad norm  = {final_grad:.5f}")
    print(f"  final param norm = {final_norm:.5f}")
    if activation == "relu":
        print(f"  active fraction  = {final_active:.3f}")
    else:
        print(f"  unsaturated frac = {final_active:.3f}")


def _landscape_pair(T0, T1, Z, objective_fn, title_3d, title_2d, path=None, selected=None, selected_label=r"$\theta_T$"):
    fig = plt.figure(figsize=(12.4, 4.8))
    ax_surface = fig.add_subplot(1, 2, 1, projection="3d")
    ax_contour = fig.add_subplot(1, 2, 2)
    plot_3d_landscape(ax_surface, T0, T1, Z, title_3d, path=path, objective_fn=objective_fn, selected=selected)
    levels = np.linspace(float(np.min(Z)), float(np.quantile(Z, 0.92)), 24)
    plot_2d_landscape(
        ax_contour,
        T0,
        T1,
        Z,
        title_2d,
        levels=levels,
        path=path,
        selected=selected,
        selected_label=selected_label,
    )
    return fig, (ax_surface, ax_contour)


def plot_candidate_selection_example(
    seed=11,
    outlier=True,
    candidate_thetas=None,
    active_candidate=0,
):
    if candidate_thetas is None:
        candidate_thetas = [(0.2, 0.4), (0.8, 0.2), (1.4, 0.05)]

    x, y = make_linear_dataset(seed=seed, outlier=outlier)
    X = linear_design(x)
    candidate_thetas = [np.asarray(theta, dtype=float) for theta in candidate_thetas]
    active_candidate = int(np.clip(active_candidate, 0, len(candidate_thetas) - 1))
    risks = [mse_objective(X, y, theta) for theta in candidate_thetas]
    theta = candidate_thetas[active_candidate]
    risk = risks[active_candidate]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    plot_observed(ax, x, y)
    for i, candidate in enumerate(candidate_thetas):
        active = i == active_candidate
        color = COLORS["selected"] if active else COLORS["alt"]
        lw = 3.1 if active else 1.8
        alpha = 1.0 if active else 0.62
        plot_linear_fit(
            ax,
            candidate,
            label=f"{i}: theta={format_theta(candidate)}, risk={risks[i]:.3f}",
            color=color,
            lw=lw,
            ls="-" if active else "--",
        )
        ax.lines[-1].set_alpha(alpha)
    ax.set_title("Candidate functions inside the same hypothesis space")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

    print(f"Active theta = {theta}")
    print(f"Empirical risk = {risk:.3f}")
    print("Changed ingredient: selected candidate inside fixed H")
    return {"x": x, "y": y, "theta": theta, "risk": risk, "candidate_risks": risks}


def plot_fit_evaluation_example(
    seed=11,
    outlier=True,
    theta_selected=(0.8, 0.2),
    loss_name="squared",
    huber_delta=1.0,
):
    x, y = make_linear_dataset(seed=seed, outlier=outlier)
    X = linear_design(x)
    theta_selected = np.asarray(theta_selected, dtype=float)
    predictions = X @ theta_selected
    residual = predictions - y
    pointwise_penalty = loss_values(residual, loss_name=loss_name, huber_delta=huber_delta)
    empirical_risk = float(np.mean(pointwise_penalty))
    largest_residual = float(np.max(np.abs(residual)))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    plot_fit_with_residuals(
        axes[0],
        x,
        y,
        theta_selected,
        "Selected function and residuals",
        label="selected function",
    )
    for name, color, style in [
        ("squared", COLORS["selected"], "-"),
        ("absolute", COLORS["alt"], "--"),
        ("huber", COLORS["regularised"], ":"),
    ]:
        axes[1].plot(
            RESIDUAL_GRID,
            loss_values(RESIDUAL_GRID, loss_name=name, huber_delta=huber_delta),
            color=color,
            lw=2.0,
            ls=style,
            label=name,
        )
    axes[1].scatter(residual, pointwise_penalty, s=32, color=COLORS["data"], alpha=0.75, label="observed residuals")
    axes[1].set_title("Residuals become penalties")
    axes[1].set_xlabel(r"residual $h_\theta(x_i)-y_i$")
    axes[1].set_ylabel(r"$\ell(r_i)$")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"Evaluation loss = {loss_name}")
    print(f"Selected theta = {theta_selected}")
    print(f"Empirical risk = {empirical_risk:.3f}")
    print(f"Largest absolute residual = {largest_residual:.3f}")
    print("Changed ingredient: evaluation rule for a selected function")
    return {
        "x": x,
        "y": y,
        "theta": theta_selected,
        "residual": residual,
        "empirical_risk": empirical_risk,
    }


def plot_objective_example(
    seed=11,
    outlier=True,
    theta0_manual=0.4,
    theta1_manual=0.05,
    loss_name="squared",
    lambda_ridge=0.0,
    huber_delta=1.0,
):
    x, y = make_linear_dataset(seed=seed, outlier=outlier)
    X = linear_design(x)
    theta_manual = np.array([theta0_manual, theta1_manual], dtype=float)
    risk = linear_empirical_loss(X, y, theta_manual, loss_name=loss_name, huber_delta=huber_delta)
    penalty = float(lambda_ridge * np.sum(theta_manual**2))
    objective = risk + penalty
    theta_hat, _, _ = fit_linear_loss(
        x,
        y,
        loss_name=loss_name,
        lambda_ridge=lambda_ridge,
        huber_delta=huber_delta,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    plot_fit_with_residuals(
        axes[0],
        x,
        y,
        theta_manual,
        "Manual parameter value evaluated by the objective",
        label="manual theta",
        color=COLORS["alt"],
    )
    plot_linear_fit(axes[0], theta_hat, label="selected minimiser", color=COLORS["selected"], lw=2.2, ls="--")

    theta0_range = (min(theta0_manual, theta_hat[0]) - 1.2, max(theta0_manual, theta_hat[0]) + 1.2)
    theta1_range = (min(theta1_manual, theta_hat[1]) - 1.2, max(theta1_manual, theta_hat[1]) + 1.2)
    T0, T1, Z = objective_grid_for_loss(
        x,
        y,
        theta0_range,
        theta1_range,
        loss_name=loss_name,
        lambda_ridge=lambda_ridge,
        huber_delta=huber_delta,
    )
    levels = plot_2d_landscape(
        axes[1],
        T0,
        T1,
        Z,
        "Objective values over parameter space",
        selected=theta_manual,
        selected_label="manual theta",
    )
    plot_2d_landscape(
        axes[1],
        T0,
        T1,
        Z,
        "Objective values over parameter space",
        levels=levels,
        selected=theta_hat,
        selected_label="selected minimiser",
        selected_color=COLORS["truth"],
        alpha=0.0,
    )
    plt.tight_layout()
    plt.show()

    print(f"Manual theta = ({theta0_manual:.3f}, {theta1_manual:.3f})")
    print(f"Empirical risk = {risk:.3f}")
    print(f"Penalty = {penalty:.3f}")
    print(f"Objective J(theta) = {objective:.3f}")
    return {"theta_manual": theta_manual, "theta_hat": theta_hat, "risk": risk, "penalty": penalty, "objective": objective}


def plot_least_squares_vs_iterative_example(
    seed=11,
    outlier=True,
    learning_rate=0.05,
    num_steps=35,
    theta_init=(0.0, 0.0),
    scale_feature=True,
):
    x_raw, y = make_linear_dataset(seed=seed, outlier=outlier)
    x_model = (x_raw - x_raw.mean()) / x_raw.std() if scale_feature else x_raw
    X = linear_design(x_model)
    theta_least_squares = ridge_solution(X, y, lam=0.0)
    objective_fn = lambda theta: mse_objective(X, y, theta, lam=0.0)
    grad_fn = lambda theta: mse_gradient(X, y, theta, lam=0.0)
    theta_path, objective_values = gradient_descent(theta_init, grad_fn, objective_fn, learning_rate, num_steps)
    theta_iterative = theta_path[-1]
    distance_to_closed_form = float(np.linalg.norm(theta_iterative - theta_least_squares))

    x_grid_raw = np.linspace(x_raw.min(), x_raw.max(), 300)
    x_grid_model = (x_grid_raw - x_raw.mean()) / x_raw.std() if scale_feature else x_grid_raw
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].scatter(x_raw, y, s=35, color=COLORS["data"], edgecolor="white", linewidth=0.4, label="observed data")
    axes[0].plot(
        x_grid_raw,
        linear_design(x_grid_model) @ theta_least_squares,
        color=COLORS["truth"],
        lw=2.4,
        label="closed-form least squares",
    )
    axes[0].plot(
        x_grid_raw,
        linear_design(x_grid_model) @ theta_iterative,
        color=COLORS["selected"],
        lw=2.4,
        ls="--",
        label=f"finite-step GD, T={num_steps}",
    )
    axes[0].set_title("Direct solution versus finite-step optimisation")
    axes[0].set_xlabel("raw x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8)
    axes[1].plot(objective_values, color=COLORS["selected"], lw=2.2, label="gradient descent path")
    axes[1].axhline(objective_fn(theta_least_squares), color=COLORS["truth"], lw=1.8, ls="--", label="least-squares minimum")
    axes[1].set_title("Finite iterations approach the objective minimum")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$J(\theta_t)$")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"Closed-form least-squares theta = {theta_least_squares}")
    print(f"Final iterative theta_T = {theta_iterative}")
    print(f"Distance to least-squares solution = {distance_to_closed_form:.3f}")
    print(f"Final objective = {objective_values[-1]:.3f}")
    print("Changed ingredient: optimisation method, from direct solve to iterative path")
    return {
        "theta_least_squares": theta_least_squares,
        "theta_path": theta_path,
        "objective_values": objective_values,
        "distance_to_closed_form": distance_to_closed_form,
    }


def plot_gradient_descent_example(
    seed=13,
    learning_rate=0.05,
    num_steps=40,
    theta_init=(0.0, 0.0),
    scale_feature=True,
):
    x_raw, y = make_linear_dataset(seed=seed, outlier=True)
    x_model = (x_raw - x_raw.mean()) / x_raw.std() if scale_feature else x_raw
    X = linear_design(x_model)
    theta_star = ridge_solution(X, y)
    objective_fn = lambda theta: mse_objective(X, y, theta)
    grad_fn = lambda theta: mse_gradient(X, y, theta)
    theta_path, objective_values = gradient_descent(theta_init, grad_fn, objective_fn, learning_rate, num_steps)

    all_points = np.vstack([theta_path[np.isfinite(theta_path).all(axis=1)], theta_star[None, :]])
    theta0_range = (float(np.min(all_points[:, 0]) - 0.55), float(np.max(all_points[:, 0]) + 0.55))
    theta1_range = (float(np.min(all_points[:, 1]) - 0.55), float(np.max(all_points[:, 1]) + 0.55))
    visible_path = finite_path_inside(theta_path, theta0_range, theta1_range)
    T0, T1 = make_loss_mesh(theta0_range, theta1_range, points=130)
    Z = evaluate_objective_mesh(T0, T1, objective_fn)
    _landscape_pair(
        T0,
        T1,
        Z,
        objective_fn,
        "3D training landscape",
        "2D contours with GD path",
        path=visible_path,
        selected=theta_path[-1],
    )
    plt.tight_layout()
    plt.show()

    x_grid_raw = np.linspace(x_raw.min(), x_raw.max(), 300)
    x_grid_model = (x_grid_raw - x_raw.mean()) / x_raw.std() if scale_feature else x_grid_raw
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].plot(objective_values, color=COLORS["selected"], lw=2.2)
    axes[0].set_title("Objective value along the path")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(r"$J(\theta_t)$")
    axes[0].set_yscale("log")
    axes[1].scatter(x_raw, y, s=35, color=COLORS["data"], edgecolor="white", linewidth=0.4, label="observed data")
    for label, theta, color, style in [
        ("initial", theta_path[0], COLORS["alt"], ":"),
        ("final", theta_path[-1], COLORS["selected"], "-"),
        ("least-squares optimum", theta_star, COLORS["truth"], "--"),
    ]:
        axes[1].plot(x_grid_raw, linear_design(x_grid_model) @ theta, color=color, lw=2.2, ls=style, label=label)
    axes[1].set_title("Functions selected at different path points")
    axes[1].set_xlabel("raw x")
    axes[1].set_ylabel("y")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"Learning rate = {learning_rate}")
    print(f"Number of steps = {num_steps}")
    print(f"Final theta_T = {theta_path[-1]}")
    print(f"Final objective = {objective_values[-1]:.3f}")
    print("Changed ingredient: O, via update size")
    return {"theta_path": theta_path, "objective_values": objective_values, "theta_star": theta_star}


def plot_landscape_geometry_example(
    seed=19,
    scale_feature=False,
    theta0_range=(-2.0, 2.5),
    theta1_range=(-0.1, 0.1),
    grid_size=120,
    manual_theta=None,
):
    rng = set_seed(seed)
    x_raw = np.linspace(0, 90, 45)
    y = 1.2 + 0.045 * x_raw + rng.normal(0, 0.55, len(x_raw))
    if scale_feature:
        x_model = (x_raw - x_raw.mean()) / x_raw.std()
        theta1_plot_range = (-1.5, 1.8) if theta1_range == (-0.1, 0.1) else theta1_range
    else:
        x_model = x_raw
        theta1_plot_range = theta1_range
    X = linear_design(x_model)
    theta_hat = ridge_solution(X, y, lam=0.0)
    condition_number = np.linalg.cond(X.T @ X)

    x_grid_raw = np.linspace(x_raw.min(), x_raw.max(), 300)
    x_grid_model = (x_grid_raw - x_raw.mean()) / x_raw.std() if scale_feature else x_grid_raw
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(x_raw, y, s=35, color=COLORS["data"], edgecolor="white", linewidth=0.4, label="observed data")
    ax.plot(x_grid_raw, linear_design(x_grid_model) @ theta_hat, color=COLORS["selected"], lw=2.6, label="selected fit")
    ax.set_title("Same prediction task, different coordinates")
    ax.set_xlabel("raw x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    T0, T1, Z = objective_grid_for_loss(x_model, y, theta0_range, theta1_plot_range, grid_size=grid_size)
    objective_fn = lambda theta: mse_objective(X, y, theta)
    fig, (_, ax_contour) = _landscape_pair(
        T0,
        T1,
        Z,
        objective_fn,
        "3D objective landscape",
        "2D objective contours",
        selected=theta_hat,
        selected_label="selected minimiser",
    )
    if manual_theta is not None:
        manual_theta = np.asarray(manual_theta, dtype=float)
        ax_contour.scatter(manual_theta[0], manual_theta[1], color=COLORS["alt"], s=55, label="manual theta")
        ax_contour.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"Feature scaled = {scale_feature}")
    print(f"Condition number of X^T X = {condition_number:.2f}")
    print("Changed ingredient: O's coordinate geometry")
    return {"theta_hat": theta_hat, "condition_number": condition_number}


def plot_conditioning_example(
    lambda_min=1.0,
    lambda_max=5.0,
    learning_rate=0.12,
    num_steps=35,
    theta_init=(2.8, 2.4),
    theta_star=(0.4, -0.8),
):
    theta_init = np.asarray(theta_init, dtype=float)
    theta_star = np.asarray(theta_star, dtype=float)
    A = np.diag([lambda_min, lambda_max])
    (theta_path, objective_values), objective_fn = quadratic_path(A, theta_init, theta_star, learning_rate, num_steps)
    distances = np.linalg.norm(theta_path - theta_star, axis=1)
    eigen_errors = theta_path - theta_star
    kappa = lambda_max / lambda_min

    theta0_range = (min(theta_init[0], theta_star[0]) - 0.7, max(theta_init[0], theta_star[0]) + 0.7)
    theta1_range = (min(theta_init[1], theta_star[1]) - 0.7, max(theta_init[1], theta_star[1]) + 0.7)
    T0, T1 = make_loss_mesh(theta0_range, theta1_range, points=130)
    Z = evaluate_objective_mesh(T0, T1, objective_fn)
    visible_path = finite_path_inside(theta_path, theta0_range, theta1_range)
    _landscape_pair(
        T0,
        T1,
        Z,
        objective_fn,
        f"3D quadratic, kappa={kappa:.1f}",
        "2D contours with curvature path",
        path=visible_path,
        selected=theta_path[-1],
    )
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(distances, color=COLORS["selected"], lw=2.2, label=r"distance to $\theta^\star$")
    ax.plot(np.abs(eigen_errors[:, 0]), color=COLORS["alt"], lw=1.8, ls="--", label="direction 1 error")
    ax.plot(np.abs(eigen_errors[:, 1]), color=COLORS["regularised"], lw=1.8, ls=":", label="direction 2 error")
    ax.set_title("Progress can differ by curvature direction")
    ax.set_xlabel("iteration")
    ax.set_ylabel("absolute error")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"Condition number kappa = {kappa:.2f}")
    print(f"Learning rate = {learning_rate}")
    print(f"Final distance to theta_star = {distances[-1]:.3f}")
    return {"theta_path": theta_path, "objective_values": objective_values, "distances": distances, "kappa": kappa}


def plot_stochastic_updates_example(
    seed=41,
    batch_size=8,
    learning_rate=0.05,
    num_updates=80,
    num_repeats=5,
    shuffle_each_epoch=True,
):
    x, y = make_linear_dataset(n=80, seed=seed, outlier=False)
    X = linear_design(x)
    theta_init = np.array([-1.2, 1.7])
    objective_fn = lambda theta: mse_objective(X, y, theta)
    grad_fn = lambda theta: mse_gradient(X, y, theta)
    full_path, full_values = gradient_descent(theta_init, grad_fn, objective_fn, learning_rate, num_updates)

    sgd_paths = []
    sgd_values = []
    for repeat in range(num_repeats):
        path_repeat, values_repeat = minibatch_training_path(
            X,
            y,
            theta_init,
            learning_rate,
            num_updates,
            batch_size,
            seed=seed + 100 * repeat,
            shuffle_each_epoch=shuffle_each_epoch,
        )
        sgd_paths.append(path_repeat)
        sgd_values.append(values_repeat)
    sgd_values = np.asarray(sgd_values)
    final_objectives = sgd_values[:, -1]
    mean_final_objective = float(np.mean(final_objectives))
    std_final_objective = float(np.std(final_objectives))

    theta_star = ridge_solution(X, y)
    all_points = np.vstack([full_path, *sgd_paths, theta_star[None, :]])
    theta0_range = (all_points[:, 0].min() - 0.35, all_points[:, 0].max() + 0.35)
    theta1_range = (all_points[:, 1].min() - 0.35, all_points[:, 1].max() + 0.35)
    T0, T1 = make_loss_mesh(theta0_range, theta1_range, points=135)
    Z = evaluate_objective_mesh(T0, T1, objective_fn)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    levels = np.linspace(float(np.min(Z)), float(np.quantile(Z, 0.92)), 24)
    plot_2d_landscape(axes[0], T0, T1, Z, "Full-data objective with stochastic paths", levels=levels)
    axes[0].plot(full_path[:, 0], full_path[:, 1], color=COLORS["truth"], lw=2.5, marker="o", markevery=12, ms=3.2, label="full-batch path")
    for repeat, path_repeat in enumerate(sgd_paths):
        axes[0].plot(path_repeat[:, 0], path_repeat[:, 1], lw=1.3, alpha=0.78, marker="o", markevery=14, ms=2.2, label="SGD path" if repeat == 0 else None)
        axes[0].scatter(path_repeat[-1, 0], path_repeat[-1, 1], color=COLORS["sgd"], s=28, alpha=0.75)
    axes[0].scatter(theta_star[0], theta_star[1], color=COLORS["truth"], s=75, marker="*", label="full-data minimiser")
    axes[0].legend(fontsize=7)
    axes[1].plot(full_values, color=COLORS["truth"], lw=2.4, label="full-batch")
    for repeat, values_repeat in enumerate(sgd_values):
        axes[1].plot(values_repeat, color=COLORS["sgd"], lw=1.2, alpha=0.55, label="minibatch repeats" if repeat == 0 else None)
    axes[1].plot(np.mean(sgd_values, axis=0), color=COLORS["selected"], lw=2.2, label="mean minibatch objective")
    axes[1].set_title("Full-data objective evaluated over updates")
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("full-data MSE")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.show()

    rows = [(repeat, path_repeat[-1], sgd_values[repeat, -1]) for repeat, path_repeat in enumerate(sgd_paths)]
    display_table(["Repeat", "Final theta", "Final full-data objective"], rows)
    print(f"Batch size = {batch_size}")
    print(f"Number of repeats = {num_repeats}")
    print(f"Mean final objective = {mean_final_objective:.3f}")
    print(f"Std final objective = {std_final_objective:.3f}")
    print("Changed ingredient: O, via stochastic update path")
    return {"full_path": full_path, "sgd_paths": sgd_paths, "sgd_values": sgd_values}


def plot_regularisation_example(
    seed=31,
    outlier=True,
    lambda_ridge=0.0,
    validation_seed=32,
    lambda_values=None,
):
    if lambda_values is None:
        lambda_values = [0.0, 0.01, 0.1, 1.0]
    x_train, y_train = make_linear_dataset(n=26, seed=seed, outlier=outlier)
    rng_val = set_seed(validation_seed)
    x_val = np.sort(rng_val.uniform(-2.7, 2.7, 80))
    y_val = true_linear_function(x_val) + rng_val.normal(0.0, 0.18, len(x_val))
    X_train = linear_design(x_train)
    X_val = linear_design(x_val)
    if lambda_ridge not in lambda_values:
        lambda_values = sorted([*lambda_values, lambda_ridge])

    rows = []
    thetas = {}
    train_errors = []
    val_errors = []
    norms = []
    for lam in lambda_values:
        theta_lam = ridge_solution(X_train, y_train, lam=lam)
        thetas[lam] = theta_lam
        train_error_lam = mse_objective(X_train, y_train, theta_lam, lam=0.0)
        val_error_lam = mse_objective(X_val, y_val, theta_lam, lam=0.0)
        theta_norm_lam = float(np.linalg.norm(theta_lam))
        train_errors.append(train_error_lam)
        val_errors.append(val_error_lam)
        norms.append(theta_norm_lam)
        rows.append((lam, theta_lam, train_error_lam, val_error_lam, theta_norm_lam, describe_lambda(lam)))

    theta_hat = thetas[lambda_ridge]
    train_error = mse_objective(X_train, y_train, theta_hat, lam=0.0)
    val_error = mse_objective(X_val, y_val, theta_hat, lam=0.0)
    theta_norm = float(np.linalg.norm(theta_hat))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    plot_observed(ax, x_train, y_train, label="training data")
    for lam in lambda_values:
        active = lam == lambda_ridge
        color = COLORS["selected"] if active else COLORS["regularised"]
        lw = 3.0 if active else 1.7
        alpha = 1.0 if active else 0.58
        plot_linear_fit(ax, thetas[lam], label=f"lambda={lam:g}", color=color, lw=lw, ls="-" if active else "--")
        ax.lines[-1].set_alpha(alpha)
    ax.set_title("Ridge penalty selects among compatible lines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    objective_fn = lambda theta: mse_objective(X_train, y_train, theta, lam=lambda_ridge)
    theta0_range = (theta_hat[0] - 1.2, theta_hat[0] + 1.2)
    theta1_range = (theta_hat[1] - 1.2, theta_hat[1] + 1.2)
    T0, T1 = make_loss_mesh(theta0_range, theta1_range, points=130)
    Z = evaluate_objective_mesh(T0, T1, objective_fn)
    _landscape_pair(
        T0,
        T1,
        Z,
        objective_fn,
        f"3D ridge landscape, lambda={lambda_ridge:g}",
        "2D ridge contours",
        selected=theta_hat,
        selected_label="ridge solution",
    )
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    axes[0].plot(lambda_values, train_errors, marker="o", color=COLORS["selected"], lw=2.0, label="training error")
    axes[0].plot(lambda_values, val_errors, marker="o", color=COLORS["validation"], lw=2.0, label="validation error")
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_title("Error versus ridge strength")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("MSE")
    axes[0].legend(fontsize=8)
    axes[1].plot(lambda_values, norms, marker="o", color=COLORS["regularised"], lw=2.0)
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_title("Parameter norm versus ridge strength")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$\|\theta\|_2$")
    plt.tight_layout()
    plt.show()

    display_table(["lambda", "theta", "Training error", "Validation error", "Parameter norm", "Preference"], rows)
    print(f"lambda = {lambda_ridge}")
    print(f"Selected theta = {theta_hat}")
    print(f"Training error = {train_error:.3f}")
    print(f"Validation error = {val_error:.3f}")
    print(f"Parameter norm = {theta_norm:.3f}")
    return {"theta_hat": theta_hat, "train_error": train_error, "val_error": val_error, "theta_norm": theta_norm}


def plot_initialisation_example(
    seed=7,
    learning_rate=0.045,
    num_steps=120,
    start_points=None,
):
    if start_points is None:
        start_points = [(-2.8, -2.4), (-1.1, -0.4), (0.65, -2.2), (2.65, -0.1)]
    starts = np.asarray(start_points, dtype=float)
    objective_fn = nonconvex_objective
    grad_fn = nonconvex_gradient
    paths = []
    values = []
    for start in starts:
        path_start, values_start = gradient_descent(start, grad_fn, objective_fn, learning_rate, num_steps)
        paths.append(path_start)
        values.append(values_start)

    T0, T1 = make_loss_mesh((-3.2, 3.2), (-3.0, 3.0), points=170)
    Z = evaluate_objective_mesh(T0, T1, objective_fn)
    levels = np.linspace(float(np.min(Z)), float(np.quantile(Z, 0.96)), 28)

    fig = plt.figure(figsize=(12.4, 4.8))
    ax_surface = fig.add_subplot(1, 2, 1, projection="3d")
    ax_contour = fig.add_subplot(1, 2, 2)
    ax_surface.plot_surface(T0, T1, Z, cmap=VIRIDIS_CMAP, alpha=0.88, linewidth=0, antialiased=True)
    for i, path_start in enumerate(paths):
        z_path = np.array([objective_fn(theta) for theta in path_start])
        ax_surface.plot(path_start[:, 0], path_start[:, 1], z_path, lw=1.6, marker="o", markevery=18, ms=2.3, label=f"start {i}")
    ax_surface.set_title("3D non-convex objective")
    ax_surface.set_xlabel(r"$\theta_0$")
    ax_surface.set_ylabel(r"$\theta_1$")
    ax_surface.set_zlabel(r"$J(\theta)$")
    ax_surface.view_init(elev=30, azim=-132)
    ax_surface.legend(fontsize=7)
    plot_2d_landscape(ax_contour, T0, T1, Z, "2D basins and paths", levels=levels)
    for i, path_start in enumerate(paths):
        ax_contour.plot(path_start[:, 0], path_start[:, 1], lw=1.5, marker="o", markevery=18, ms=2.3, label=f"start {i}")
        ax_contour.scatter(path_start[0, 0], path_start[0, 1], color="white", edgecolor=COLORS["truth"], s=34)
        ax_contour.scatter(path_start[-1, 0], path_start[-1, 1], color=COLORS["selected"], s=36)
    ax_contour.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

    x_grid = np.linspace(-3.0, 3.0, 300)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, path_start in enumerate(paths):
        theta_final = path_start[-1]
        y_function = np.sin(theta_final[0] * x_grid) + 0.25 * theta_final[1] * x_grid
        ax.plot(x_grid, y_function, lw=1.8, label=f"final {i}: theta={format_theta(theta_final)}")
    ax.set_title("Toy functions represented by final parameters")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$h_\theta(x)$")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

    rows = [(i, start, path_start[-1], values_start[-1]) for i, (start, path_start, values_start) in enumerate(zip(starts, paths, values))]
    final_table = markdown_table(["Start", "theta_0", "theta_T", "Final objective"], rows)
    display_markdown(final_table)
    print(f"Number of starts = {len(start_points)}")
    print("Final objective values:")
    print(final_table)
    print("Changed ingredient: O, via initialisation")
    return {"paths": paths, "values": values, "final_table": final_table}


def plot_stopping_rule_example(
    seed=42,
    learning_rate=0.05,
    max_steps=300,
    validation_fraction=0.3,
    selected_step="validation",
):
    rng = set_seed(seed)
    x_all = np.sort(rng.uniform(-2.6, 2.6, 46))
    y_all = true_linear_function(x_all) + rng.normal(0.0, 0.18, len(x_all))
    order = rng.permutation(len(x_all))
    n_val = int(round(validation_fraction * len(x_all)))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]
    x_train = np.concatenate([x_train, [2.45]])
    y_train = np.concatenate([y_train, [-5.0]])
    X_train = linear_design(x_train)
    X_val = linear_design(x_val)

    objective_train = lambda theta: mse_objective(X_train, y_train, theta)
    grad_train = lambda theta: mse_gradient(X_train, y_train, theta)
    theta_path, train_error = gradient_descent(np.array([0.0, 0.0]), grad_train, objective_train, learning_rate, max_steps)
    val_error = np.array([mse_objective(X_val, y_val, theta) for theta in theta_path])
    validation_t = int(np.argmin(val_error))
    early_t = min(10, max_steps)
    late_t = max_steps
    if isinstance(selected_step, str):
        selected_key = selected_step.lower()
        if selected_key == "early":
            T_star = early_t
        elif selected_key == "validation":
            T_star = validation_t
        elif selected_key == "late":
            T_star = late_t
        else:
            raise ValueError("selected_step must be 'early', 'validation', 'late', or an integer")
    else:
        T_star = int(np.clip(selected_step, 0, max_steps))

    checkpoints = {"early": early_t, "validation-selected": validation_t, "late": late_t}
    theta_hat = ridge_solution(X_train, y_train)
    theta0_range = (min(theta_path[:, 0].min(), theta_hat[0]) - 0.45, max(theta_path[:, 0].max(), theta_hat[0]) + 0.45)
    theta1_range = (min(theta_path[:, 1].min(), theta_hat[1]) - 0.45, max(theta_path[:, 1].max(), theta_hat[1]) + 0.45)
    T0, T1 = make_loss_mesh(theta0_range, theta1_range, points=130)
    Z = evaluate_objective_mesh(T0, T1, objective_train)

    fig, (_, ax_contour) = _landscape_pair(
        T0,
        T1,
        Z,
        objective_train,
        "3D training loss surface",
        "2D path with stopping choices",
        path=theta_path,
        selected=theta_path[T_star],
        selected_label=r"selected $T^\star$",
    )
    for label, t in checkpoints.items():
        ax_contour.scatter(theta_path[t, 0], theta_path[t, 1], s=70, label=label)
    ax_contour.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].plot(train_error, color=COLORS["selected"], lw=2.1, label="training MSE")
    axes[0].plot(val_error, color=COLORS["validation"], lw=2.1, label="validation MSE")
    for label, t in checkpoints.items():
        axes[0].axvline(t, ls="--" if t != T_star else "-", lw=1.2, alpha=0.72, label=f"{label} t={t}")
    axes[0].axvline(T_star, color=COLORS["truth"], lw=2.0, alpha=0.9, label=f"reported t={T_star}")
    axes[0].set_title("Stopping rule selects an iterate")
    axes[0].set_xlabel("update t")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)

    x_grid = np.linspace(min(x_all.min(), x_train.min()), max(x_all.max(), x_train.max()), 300)
    axes[1].plot(x_grid, true_linear_function(x_grid), color=COLORS["truth"], lw=2.2, label="latent reference")
    plot_observed(axes[1], x_train, y_train, label="training data")
    axes[1].scatter(x_val, y_val, s=35, color=COLORS["validation"], edgecolor="white", linewidth=0.4, label="validation data")
    for label, t in checkpoints.items():
        color = COLORS["regularised"] if label == "early" else COLORS["selected"] if label == "validation-selected" else COLORS["alt"]
        lw = 3.0 if t == T_star else 1.9
        plot_linear_fit(axes[1], theta_path[t], x_grid=x_grid, label=f"{label} h_theta_t", color=color, lw=lw, ls="-" if t == T_star else "--")
    if T_star not in checkpoints.values():
        plot_linear_fit(axes[1], theta_path[T_star], x_grid=x_grid, label="reported iterate", color=COLORS["truth"], lw=2.5)
    axes[1].set_title("Different stopping times report different functions")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.show()

    rows = [(label, t, train_error[t], val_error[t], theta_path[t]) for label, t in checkpoints.items()]
    rows.append(("reported", T_star, train_error[T_star], val_error[T_star], theta_path[T_star]))
    display_table(["Stopping rule", "t", "Training MSE", "Validation MSE", "theta_t"], rows)
    print(f"Selected stopping rule = {selected_step}")
    print(f"T_star = {T_star}")
    print(f"Training error at T_star = {train_error[T_star]:.3f}")
    print(f"Validation error at T_star = {val_error[T_star]:.3f}")
    print("Changed ingredient: O, via stopping rule")
    return {"theta_path": theta_path, "train_error": train_error, "val_error": val_error, "T_star": T_star}


def display_optimisation_audit(scenario="high_train_error"):
    audit_rows = [
        ("High training error", r"$\mathcal{H}$ or $\mathcal{O}$", "capacity, optimisation path"),
        (r"Same $\mathcal{D}$ and $\mathcal{H}$, different solutions", r"$\mathcal{O}$", "loss, seed, batch size, stopping"),
        ("Good random-test error, bad shifted-context error", r"$\mathcal{D}$ or deployment mismatch", "validation design"),
        (r"Strong sensitivity to $\lambda$", r"$\mathcal{O}$", "regularisation preference"),
        ("Training loss keeps falling, validation worsens", r"$\mathcal{O}$", "stopping rule"),
        ("Finite-step optimiser differs from least-squares benchmark", r"$\mathcal{O}$", "learning rate, steps, scaling, convergence"),
    ]
    scenario_focus = {
        "high_train_error": ("High training error", r"$\mathcal{H}$ or $\mathcal{O}$", "capacity, optimisation path"),
        "good_train_bad_shift": ("Good random-test error, bad shifted-context error", r"$\mathcal{D}$ or deployment mismatch", "validation design"),
        "seed_sensitive": (r"Same $\mathcal{D}$ and $\mathcal{H}$, different solutions", r"$\mathcal{O}$", "initialisation, batch order, stochastic path"),
        "regularisation_sensitive": (r"Strong sensitivity to $\lambda$", r"$\mathcal{O}$", "regularisation preference"),
        "early_stopping_sensitive": ("Training loss keeps falling, validation worsens", r"$\mathcal{O}$", "stopping rule"),
        "loss_sensitive": ("Same data, different fit after changing loss", r"$\mathcal{O}$", "loss and estimand"),
        "iterative_not_converged": ("Finite-step optimiser differs from least-squares benchmark", r"$\mathcal{O}$", "learning rate, steps, scaling, convergence"),
    }
    focus = scenario_focus.get(scenario, scenario_focus["high_train_error"])
    display_table(["Observation", "Likely source", "What to check"], audit_rows)
    display_markdown(
        "\n".join(
            [
                f"**Selected scenario:** `{scenario}`",
                "",
                markdown_table(["Observation", "Likely source", "What to check"], [focus]),
            ]
        )
    )
    return {"scenario": scenario, "focus": focus, "audit_rows": audit_rows}


__all__ = [
    "configure_optimization_matplotlib",
    "SEED",
    "COLORS",
    "LINE_GRID",
    "RESIDUAL_GRID",
    "VIRIDIS_CMAP",
    "set_seed",
    "display_markdown",
    "format_cell",
    "markdown_table",
    "display_table",
    "make_axis",
    "true_linear_function",
    "make_linear_dataset",
    "linear_design",
    "linear_predictions",
    "squared_loss",
    "absolute_loss",
    "huber_loss",
    "mse_objective",
    "mae_objective",
    "mse_gradient",
    "ridge_solution",
    "fit_linear_absolute_grid",
    "gradient_descent",
    "sgd_path",
    "plot_observed",
    "plot_linear_fit",
    "objective_grid_linear",
    "status_from_eta",
    "describe_lambda",
    "make_loss_mesh",
    "evaluate_objective_mesh",
    "plot_3d_landscape",
    "format_theta",
    "loss_values",
    "linear_empirical_loss",
    "linear_objective",
    "objective_grid_for_loss",
    "fit_linear_loss",
    "plot_2d_landscape",
    "plot_fit_with_residuals",
    "minibatch_training_path",
    "finite_path_inside",
    "fit_squared_and_absolute",
    "quadratic_path",
    "nonconvex_objective",
    "nonconvex_gradient",
    "make_mlp_regression_data",
    "make_mlp_grid",
    "activation_forward",
    "activation_backward",
    "initialise_mlp",
    "forward_mlp",
    "mse_loss",
    "backward_mlp",
    "gradient_norm",
    "hidden_gradient_norm",
    "parameter_norm",
    "active_fraction",
    "copy_params",
    "train_mlp",
    "plot_mlp_fit",
    "make_mlp_tilt_power_data",
    "mlp_diagnostics",
    "train_mlp_tilt_power_recipe",
    "plot_mlp_tilt_power_fit",
    "print_mlp_summary",
    "plot_candidate_selection_example",
    "plot_fit_evaluation_example",
    "plot_objective_example",
    "plot_least_squares_vs_iterative_example",
    "plot_gradient_descent_example",
    "plot_landscape_geometry_example",
    "plot_conditioning_example",
    "plot_stochastic_updates_example",
    "plot_regularisation_example",
    "plot_initialisation_example",
    "plot_stopping_rule_example",
    "display_optimisation_audit",
]
