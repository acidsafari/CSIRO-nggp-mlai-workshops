"""Reusable helpers for the Solution Space notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - only used outside notebooks
    Markdown = None
    display = print

from .data_space import (
    COLORS as DATA_COLORS,
    configure_matplotlib,
    f0,
    local_count,
    sample_tilt_power,
    theta_grid,
)

SEED = 7
GRID = theta_grid

EVAL_COLORS = {
    "train": "#2F5D7C",
    "val": "#D18F24",
    "test": "#2F7D4A",
    "diagnostic": "#7B5E9E",
    "truth": "#222222",
    "fit": "#C7502A",
    "residual": "#6A737D",
    "support": "#D9D9D9",
    "context0": DATA_COLORS["context0"],
    "context1": DATA_COLORS["context1"],
}

SLICE_NOTES = {
    "X < 20": "boundary / low-angle region",
    "20 <= X < 40": "rising region",
    "40 <= X < 60": "broad peak",
    "60 <= X < 70": "narrow feature",
    "X >= 70": "tail / extrapolation",
    "low-density": "coverage weakness",
    "high-density": "well-supported region",
    "C = 0": "context",
    "C = 1": "context",
}


def configure_solution_matplotlib() -> None:
    """Apply the Matplotlib defaults used by the solution-space notebook."""
    configure_matplotlib()
    plt.rcParams.update(
        {
            "figure.figsize": (7, 4),
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
        }
    )


def set_seed(seed=SEED):
    return np.random.default_rng(seed)


def display_markdown(text):
    if Markdown is None:
        print(text)
    else:
        display(Markdown(text))


def format_cell(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(item) for item in row) + " |")
    return "\n".join(lines)


def display_table(headers, rows):
    display_markdown(markdown_table(headers, rows))


def clone_data(data, idx=None, name=None):
    keys = ["theta", "x", "y", "c", "id", "group", "time"]
    out = {}
    for key in keys:
        if key in data:
            out[key] = np.asarray(data[key] if idx is None else data[key][idx]).copy()
    out["name"] = name if name is not None else data.get("name", "data")
    out["metadata"] = dict(data.get("metadata", {}))
    return out


def data_matrix(data, input_keys=("x",)):
    columns = []
    for key in input_keys:
        values = np.asarray(data[key], dtype=float)
        columns.append(values.reshape(-1, 1))
    return np.hstack(columns)


def make_grid_data(context=0):
    return {
        "theta": GRID.copy(),
        "x": GRID.copy(),
        "y": f0(GRID),
        "c": np.full_like(GRID, context, dtype=int),
        "id": np.arange(len(GRID)),
        "name": "grid",
        "metadata": {},
    }


def make_tilt_dataset(
    n=420,
    scenario="hidden_context",
    seed=SEED,
    context_mode="increasing",
    sampling="sparse_feature",
    x_noise=1.5,
    y_noise=0.055,
    context_strength=0.35,
    name="tilt-power",
):
    raw = sample_tilt_power(
        n=n,
        scenario=scenario,
        seed=seed,
        x_noise=x_noise,
        y_noise=y_noise,
        context_strength=context_strength,
        sampling=sampling,
        context_mode=context_mode,
    )
    order = np.argsort(raw["x"])
    data = {
        "theta": np.asarray(raw["theta"])[order],
        "x": np.asarray(raw["x"])[order],
        "y": np.asarray(raw["y"])[order],
        "c": np.asarray(raw["c"])[order].astype(int),
        "id": np.arange(n)[order],
        "group": (np.asarray(raw["theta"])[order] // 10).astype(int),
        "time": np.arange(n)[order],
        "name": name,
        "metadata": {
            "scenario": scenario,
            "sampling": sampling,
            "context_mode": context_mode,
            "x_noise": x_noise,
            "y_noise": y_noise,
        },
    }
    return data


def _allocate_random(indices, rng, val_size, test_size):
    indices = np.asarray(indices)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(round(test_size * n)))
    n_val = max(1, int(round(val_size * n)))
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def _stratified_indices(labels, rng, val_size, test_size):
    train_parts, val_parts, test_parts = [], [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 5:
            train_parts.append(idx)
            continue
        train, val, test = _allocate_random(idx, rng, val_size, test_size)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
    return (
        np.concatenate(train_parts),
        np.concatenate(val_parts),
        np.concatenate(test_parts),
    )


def make_split_suite(data, split_type="stratified_by_x_bin", seed=SEED, val_size=0.2, test_size=0.2):
    rng = set_seed(seed)
    n = len(data["x"])
    all_idx = np.arange(n)

    if split_type == "iid_random":
        train_idx, val_idx, test_idx = _allocate_random(all_idx, rng, val_size, test_size)
    elif split_type == "stratified_by_x_bin":
        bins = np.digitize(data["x"], np.linspace(0, 90, 8), right=False)
        train_idx, val_idx, test_idx = _stratified_indices(bins, rng, val_size, test_size)
    elif split_type == "range_holdout":
        val_idx = np.flatnonzero((data["x"] >= 60) & (data["x"] < 70))
        test_idx = np.flatnonzero(data["x"] >= 70)
        train_idx = np.setdiff1d(all_idx, np.concatenate([val_idx, test_idx]), assume_unique=False)
        if min(len(train_idx), len(val_idx), len(test_idx)) < 12:
            train_idx, val_idx, test_idx = _allocate_random(all_idx, rng, val_size, test_size)
    elif split_type == "context_balanced":
        labels = 10 * np.digitize(data["x"], np.linspace(0, 90, 7), right=False) + data["c"]
        train_idx, val_idx, test_idx = _stratified_indices(labels, rng, val_size, test_size)
    elif split_type == "context_shifted":
        score = data["c"] + 0.25 * (data["x"] > 45) + 0.05 * rng.normal(size=n)
        ordered = np.argsort(score)
        n_test = max(20, int(round(test_size * n)))
        n_val = max(20, int(round(val_size * n)))
        test_idx = ordered[-n_test:]
        val_idx = ordered[-n_test - n_val : -n_test]
        train_idx = ordered[: -n_test - n_val]
    else:
        raise ValueError(f"unknown split_type: {split_type}")

    suite = {
        "split_type": split_type,
        "train": clone_data(data, np.sort(train_idx), "train"),
        "val": clone_data(data, np.sort(val_idx), "validation"),
        "test": clone_data(data, np.sort(test_idx), "final test"),
        "diagnostics": {},
    }
    suite["diagnostics"]["iid_validation"] = make_tilt_dataset(
        n=240,
        scenario="hidden_context",
        seed=seed + 101,
        context_mode=data["metadata"]["context_mode"],
        sampling=data["metadata"]["sampling"],
        name="iid diagnostic",
    )
    suite["diagnostics"]["balanced_context"] = make_tilt_dataset(
        n=240,
        scenario="hidden_context",
        seed=seed + 202,
        context_mode="independent",
        sampling="uniform",
        name="balanced-context diagnostic",
    )
    suite["diagnostics"]["shifted_context"] = make_tilt_dataset(
        n=240,
        scenario="hidden_context",
        seed=seed + 303,
        context_mode="reversed",
        sampling="deployment65",
        name="shifted-context diagnostic",
    )
    suite["diagnostics"]["range_stress"] = make_tilt_dataset(
        n=240,
        scenario="hidden_context",
        seed=seed + 404,
        context_mode="independent",
        sampling="deployment65",
        name="range stress diagnostic",
    )
    return suite


def describe_split(split_suite):
    rows = []
    for name in ["train", "val", "test"]:
        split = split_suite[name]
        rows.append(
            [
                split["name"],
                len(split["x"]),
                np.min(split["x"]),
                np.max(split["x"]),
                np.mean(split["y"]),
                np.mean(split["c"]),
                np.sum((split["x"] >= 60) & (split["x"] < 70)),
                np.median(local_count(split["x"], split_suite["train"]["x"], radius=5.0)),
            ]
        )
    return rows


def plot_split_distributions(split_suite):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(0, 90, 19)
    for key in ["train", "val", "test"]:
        axes[0].hist(
            split_suite[key]["x"],
            bins=bins,
            alpha=0.45,
            label=split_suite[key]["name"],
            color=EVAL_COLORS[key],
        )
    axes[0].axvspan(60, 70, color=EVAL_COLORS["support"], alpha=0.35, label="narrow feature slice")
    axes[0].set_title(f"X distribution: {split_suite['split_type']}")
    axes[0].set_xlabel("observed tilt X")
    axes[0].set_ylabel("count")
    axes[0].legend()

    labels = ["train", "validation", "test"]
    values = [np.mean(split_suite[key]["c"]) for key in ["train", "val", "test"]]
    axes[1].bar(labels, values, color=[EVAL_COLORS["train"], EVAL_COLORS["val"], EVAL_COLORS["test"]])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction with C=1")
    axes[1].set_title("Hidden context composition")
    plt.tight_layout()


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    mse = np.mean(residual**2)
    mae = np.mean(np.abs(residual))
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if denom <= 1e-12 else 1 - np.sum(residual**2) / denom
    return {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": mae, "R2": r2, "Bias": np.mean(residual)}


def make_slices(data, reference_train=None, radius=5.0):
    x = data["x"]
    slices = {
        "X < 20": x < 20,
        "20 <= X < 40": (x >= 20) & (x < 40),
        "40 <= X < 60": (x >= 40) & (x < 60),
        "60 <= X < 70": (x >= 60) & (x < 70),
        "X >= 70": x >= 70,
        "C = 0": data["c"] == 0,
        "C = 1": data["c"] == 1,
    }
    if reference_train is None:
        counts = local_count(x, x, radius)
    else:
        counts = local_count(x, reference_train["x"], radius)
    low_cut = np.quantile(counts, 0.30)
    high_cut = np.quantile(counts, 0.70)
    slices["low-density"] = counts <= low_cut
    slices["high-density"] = counts >= high_cut
    return slices


def slice_metrics(data, y_pred, slices):
    rows = []
    for name, mask in slices.items():
        count = int(np.sum(mask))
        if count == 0:
            rows.append([name, 0, np.nan, np.nan, np.nan, SLICE_NOTES.get(name, "")])
            continue
        metrics = regression_metrics(data["y"][mask], np.asarray(y_pred)[mask])
        rows.append([name, count, metrics["RMSE"], metrics["MAE"], metrics["Bias"], SLICE_NOTES.get(name, "")])
    return rows


def task_weighted_metrics(data, y_pred, weight_fn):
    weights = np.asarray(weight_fn(data["x"]), dtype=float)
    residual = np.asarray(y_pred) - data["y"]
    weighted_mse = np.sum(weights * residual**2) / np.maximum(np.sum(weights), 1e-12)
    return {"weighted_MSE": weighted_mse, "weighted_RMSE": np.sqrt(weighted_mse)}


def narrow_feature_weight(x):
    return np.where((x >= 60) & (x < 70), 4.0, 1.0)


def worst_slice_rmse(data, y_pred, reference_train):
    rows = slice_metrics(data, y_pred, make_slices(data, reference_train))
    values = [row[2] for row in rows if row[1] >= 10 and np.isfinite(row[2])]
    return np.nan if not values else float(np.max(values))


def plot_predictions(model, split_suite, title=None, show_context=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(GRID, f0(GRID), color=EVAL_COLORS["truth"], lw=2, label="latent f0(theta)")
    if "c" in model.get("input_keys", ("x",)) and show_context:
        for context, color in [(0, EVAL_COLORS["context0"]), (1, EVAL_COLORS["context1"])]:
            grid_data = make_grid_data(context=context)
            ax.plot(GRID, predict_model(model, grid_data), lw=2.3, color=color, label=f"{model['name']} prediction, C={context}")
    else:
        ax.plot(GRID, predict_model(model, make_grid_data()), color=EVAL_COLORS["fit"], lw=2.5, label=f"{model['name']} prediction")
    for key, marker in [("train", "o"), ("val", "^"), ("test", "s")]:
        split = split_suite[key]
        ax.scatter(
            split["x"],
            split["y"],
            s=24,
            alpha=0.58,
            marker=marker,
            color=EVAL_COLORS[key],
            label=split["name"],
            edgecolor="white",
            linewidth=0.25,
        )
    ax.axvspan(60, 70, color=EVAL_COLORS["support"], alpha=0.22, label="narrow feature slice")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.15, 1.55)
    ax.set_xlabel("observed tilt X")
    ax.set_ylabel("observed power Y")
    ax.set_title(title or f"Prediction curve: {model['name']}")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()


def plot_residuals(model, data, title=None):
    y_pred = predict_model(model, data)
    residual = y_pred - data["y"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].axhline(0, color="#555555", lw=1)
    axes[0].scatter(data["x"], residual, c=data["c"], cmap="coolwarm", s=28, alpha=0.75, edgecolor="white", linewidth=0.25)
    axes[0].axvspan(60, 70, color=EVAL_COLORS["support"], alpha=0.25)
    axes[0].set_xlabel("observed tilt X")
    axes[0].set_ylabel("prediction residual")
    axes[0].set_title(title or f"Residuals by X: {data['name']}")
    axes[1].axhline(0, color="#555555", lw=1)
    axes[1].scatter(y_pred, residual, c=data["c"], cmap="coolwarm", s=28, alpha=0.75, edgecolor="white", linewidth=0.25)
    axes[1].set_xlabel("predicted Y")
    axes[1].set_ylabel("prediction residual")
    axes[1].set_title("Residuals by fitted value")
    plt.tight_layout()


def error_by_bin_rows(data, y_pred, bins=np.array([0, 20, 40, 60, 70, 90])):
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (data["x"] >= left) & (data["x"] < right if right < bins[-1] else data["x"] <= right)
        label = f"[{left:.0f}, {right:.0f})" if right < bins[-1] else f"[{left:.0f}, {right:.0f}]"
        if np.sum(mask) == 0:
            rows.append([label, 0, np.nan, np.nan])
        else:
            metrics = regression_metrics(data["y"][mask], np.asarray(y_pred)[mask])
            rows.append([label, np.sum(mask), metrics["RMSE"], metrics["Bias"]])
    return rows


def fit_mean_baseline(train):
    return {"kind": "mean", "name": "Mean baseline", "value": float(np.mean(train["y"])), "input_keys": ("x",)}


def fit_linear_baseline(train):
    z = ((train["x"] - 45.0) / 45.0).reshape(-1, 1)
    X = np.column_stack([np.ones(len(z)), z])
    coef, *_ = np.linalg.lstsq(X, train["y"], rcond=None)
    return {"kind": "poly", "name": "Linear baseline", "coef": coef, "degree": 1, "input_keys": ("x",)}


def fit_polynomial_baseline(train, degree=7, ridge_lambda=1e-3):
    z = (train["x"] - 45.0) / 45.0
    X = np.vander(z, degree + 1, increasing=True)
    penalty = ridge_lambda * np.eye(degree + 1)
    penalty[0, 0] = 0
    coef = np.linalg.solve(X.T @ X + penalty, X.T @ train["y"])
    return {
        "kind": "poly",
        "name": f"Polynomial ridge degree {degree}",
        "coef": coef,
        "degree": degree,
        "ridge_lambda": ridge_lambda,
        "input_keys": ("x",),
    }


def predict_model(model, data):
    if model["kind"] == "mean":
        return np.full(len(data["x"]), model["value"], dtype=float)
    if model["kind"] == "poly":
        z = (data["x"] - 45.0) / 45.0
        X = np.vander(z, model["degree"] + 1, increasing=True)
        return X @ model["coef"]
    if model["kind"] == "mlp":
        return predict_numpy_mlp(model, data)
    raise ValueError(f"unknown model kind: {model['kind']}")


def evaluate_model(model, split_suite, split_names=("train", "val", "test")):
    rows = []
    for split_name in split_names:
        data = split_suite[split_name]
        pred = predict_model(model, data)
        metrics = regression_metrics(data["y"], pred)
        weighted = task_weighted_metrics(data, pred, narrow_feature_weight)
        rows.append(
            [
                model["name"],
                data["name"],
                len(data["x"]),
                metrics["RMSE"],
                metrics["MAE"],
                metrics["R2"],
                weighted["weighted_RMSE"],
                worst_slice_rmse(data, pred, split_suite["train"]),
            ]
        )
    return rows


def _activation(z, kind):
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "tanh":
        return np.tanh(z)
    if kind == "gelu":
        return 0.5 * z * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
    raise ValueError(f"unknown activation: {kind}")


def _activation_grad(z, kind):
    if kind == "relu":
        return (z > 0).astype(float)
    if kind == "tanh":
        a = np.tanh(z)
        return 1.0 - a**2
    if kind == "gelu":
        # Smooth enough for this workshop demonstration; exactness is not the teaching point.
        tanh_arg = np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)
        t = np.tanh(tanh_arg)
        return 0.5 * (1 + t) + 0.5 * z * (1 - t**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * z**2)
    raise ValueError(f"unknown activation: {kind}")


def initialise_mlp(input_dim=1, width=64, depth=2, activation="relu", seed=SEED):
    rng = set_seed(seed)
    dims = [input_dim] + [width] * depth + [1]
    weights, biases = [], []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        scale = np.sqrt(2.0 / fan_in) if activation == "relu" else np.sqrt(1.0 / fan_in)
        weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
        biases.append(np.zeros((1, fan_out)))
    return {"weights": weights, "biases": biases, "activation": activation}


def copy_params(params):
    return {
        "weights": [w.copy() for w in params["weights"]],
        "biases": [b.copy() for b in params["biases"]],
        "activation": params["activation"],
    }


def forward_mlp(params, X):
    activations = [X]
    preactivations = []
    A = X
    for W, b in zip(params["weights"][:-1], params["biases"][:-1]):
        Z = A @ W + b
        preactivations.append(Z)
        A = _activation(Z, params["activation"])
        activations.append(A)
    y_hat = A @ params["weights"][-1] + params["biases"][-1]
    return y_hat, activations, preactivations


def backward_mlp(params, activations, preactivations, y_hat, y_true, weight_decay=0.0):
    n = max(1, len(y_true))
    dA = 2.0 * (y_hat - y_true) / n
    grad_w = [None] * len(params["weights"])
    grad_b = [None] * len(params["biases"])

    grad_w[-1] = activations[-1].T @ dA + weight_decay * params["weights"][-1]
    grad_b[-1] = np.sum(dA, axis=0, keepdims=True)
    dA = dA @ params["weights"][-1].T

    for layer in reversed(range(len(params["weights"]) - 1)):
        dZ = dA * _activation_grad(preactivations[layer], params["activation"])
        grad_w[layer] = activations[layer].T @ dZ + weight_decay * params["weights"][layer]
        grad_b[layer] = np.sum(dZ, axis=0, keepdims=True)
        if layer > 0:
            dA = dZ @ params["weights"][layer].T
    return {"weights": grad_w, "biases": grad_b}


def _init_optimizer_state(params):
    zeros_w = [np.zeros_like(w) for w in params["weights"]]
    zeros_b = [np.zeros_like(b) for b in params["biases"]]
    return {
        "mw": [z.copy() for z in zeros_w],
        "vw": [z.copy() for z in zeros_w],
        "mb": [z.copy() for z in zeros_b],
        "vb": [z.copy() for z in zeros_b],
        "velocity_w": [z.copy() for z in zeros_w],
        "velocity_b": [z.copy() for z in zeros_b],
        "t": 0,
    }


def _apply_update(params, grads, state, lr, optimizer="adam", beta1=0.9, beta2=0.999, momentum=0.9):
    eps = 1e-8
    if optimizer == "adam":
        state["t"] += 1
        for i in range(len(params["weights"])):
            state["mw"][i] = beta1 * state["mw"][i] + (1 - beta1) * grads["weights"][i]
            state["vw"][i] = beta2 * state["vw"][i] + (1 - beta2) * (grads["weights"][i] ** 2)
            state["mb"][i] = beta1 * state["mb"][i] + (1 - beta1) * grads["biases"][i]
            state["vb"][i] = beta2 * state["vb"][i] + (1 - beta2) * (grads["biases"][i] ** 2)
            mw_hat = state["mw"][i] / (1 - beta1 ** state["t"])
            vw_hat = state["vw"][i] / (1 - beta2 ** state["t"])
            mb_hat = state["mb"][i] / (1 - beta1 ** state["t"])
            vb_hat = state["vb"][i] / (1 - beta2 ** state["t"])
            params["weights"][i] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
            params["biases"][i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
    elif optimizer == "sgd":
        for i in range(len(params["weights"])):
            params["weights"][i] -= lr * grads["weights"][i]
            params["biases"][i] -= lr * grads["biases"][i]
    elif optimizer == "momentum":
        for i in range(len(params["weights"])):
            state["velocity_w"][i] = momentum * state["velocity_w"][i] - lr * grads["weights"][i]
            state["velocity_b"][i] = momentum * state["velocity_b"][i] - lr * grads["biases"][i]
            params["weights"][i] += state["velocity_w"][i]
            params["biases"][i] += state["velocity_b"][i]
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")


def predict_numpy_mlp(model, data):
    X = data_matrix(data, model["input_keys"])
    Xn = (X - model["x_mean"]) / model["x_std"]
    yn, _, _ = forward_mlp(model["params"], Xn)
    return yn.ravel() * model["y_std"] + model["y_mean"]


def _predict_with_params(params, X, x_mean, x_std, y_mean, y_std):
    Xn = (X - x_mean) / x_std
    yn, _, _ = forward_mlp(params, Xn)
    return yn.ravel() * y_std + y_mean


def train_model(train, val, config):
    input_keys = tuple(config.get("input_keys", ("x",)))
    X_train = data_matrix(train, input_keys)
    X_val = data_matrix(val, input_keys)
    y_train = train["y"].reshape(-1, 1)

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = np.maximum(X_train.std(axis=0, keepdims=True), 1e-6)
    y_mean = float(y_train.mean())
    y_std = float(max(y_train.std(), 1e-6))

    Xn = (X_train - x_mean) / x_std
    yn = (y_train - y_mean) / y_std

    params = initialise_mlp(
        input_dim=Xn.shape[1],
        width=config.get("width", 64),
        depth=config.get("depth", 2),
        activation=config.get("activation", "relu"),
        seed=config.get("seed", SEED),
    )
    state = _init_optimizer_state(params)
    rng = set_seed(config.get("seed", SEED) + 1000)
    epochs = config.get("epochs", 320)
    batch_size = min(config.get("batch_size", 64), len(Xn))
    lr = config.get("lr", 0.01)
    weight_decay = config.get("weight_decay", 1e-4)
    optimizer = config.get("optimizer", "adam")
    name = config.get("name", "Reference MLP")

    history = []
    best_params = copy_params(params)
    final_params = None
    early_params = None
    best_val_rmse = np.inf
    best_epoch = 0

    def make_model(current_params):
        return {
            "kind": "mlp",
            "name": name,
            "params": current_params,
            "input_keys": input_keys,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "config": dict(config),
        }

    for epoch in range(1, epochs + 1):
        order = np.arange(len(Xn))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            y_hat, activations, preactivations = forward_mlp(params, Xn[batch])
            grads = backward_mlp(params, activations, preactivations, y_hat, yn[batch], weight_decay=weight_decay)
            _apply_update(params, grads, state, lr=lr, optimizer=optimizer)

        model_now = make_model(params)
        train_pred = predict_model(model_now, train)
        val_pred = predict_model(model_now, val)
        train_metrics = regression_metrics(train["y"], train_pred)
        val_metrics = regression_metrics(val["y"], val_pred)
        val_slices = make_slices(val, reference_train=train)
        feature_mask = val_slices["60 <= X < 70"]
        low_density_mask = val_slices["low-density"]
        feature_rmse = np.nan
        low_density_rmse = np.nan
        if np.sum(feature_mask) > 0:
            feature_rmse = regression_metrics(val["y"][feature_mask], val_pred[feature_mask])["RMSE"]
        if np.sum(low_density_mask) > 0:
            low_density_rmse = regression_metrics(val["y"][low_density_mask], val_pred[low_density_mask])["RMSE"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["MSE"],
                "val_loss": val_metrics["MSE"],
                "train_RMSE": train_metrics["RMSE"],
                "val_RMSE": val_metrics["RMSE"],
                "feature_RMSE": feature_rmse,
                "low_density_RMSE": low_density_rmse,
            }
        )

        if epoch == max(1, epochs // 10):
            early_params = copy_params(params)
        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            best_params = copy_params(params)

    final_params = copy_params(params)
    model = make_model(best_params)
    model["best_epoch"] = best_epoch
    model["history"] = history
    model["early_params"] = early_params
    model["final_params"] = final_params
    return model


def plot_training_history(history, title="Reference MLP training history"):
    epochs = np.array([row["epoch"] for row in history])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train MSE", color=EVAL_COLORS["train"])
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation MSE", color=EVAL_COLORS["val"])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Aggregate loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["val_RMSE"] for row in history], label="validation RMSE", color=EVAL_COLORS["val"])
    axes[1].plot(epochs, [row["feature_RMSE"] for row in history], label="[60,70] RMSE", color=EVAL_COLORS["diagnostic"])
    axes[1].plot(epochs, [row["low_density_RMSE"] for row in history], label="low-density RMSE", color=EVAL_COLORS["residual"])
    axes[1].axvline(np.argmin([row["val_RMSE"] for row in history]) + 1, color="#555555", ls="--", lw=1, label="best val epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Validation diagnostics")
    axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout()


def run_seed_sweep(split_suite, config, seeds=(17, 23, 31), epochs=180):
    models, rows = [], []
    for seed in seeds:
        cfg = dict(config, seed=seed, epochs=epochs, name=f"MLP seed {seed}")
        model = train_model(split_suite["train"], split_suite["val"], cfg)
        pred = predict_model(model, split_suite["test"])
        metrics = regression_metrics(split_suite["test"]["y"], pred)
        feature = make_slices(split_suite["test"], split_suite["train"])["60 <= X < 70"]
        feature_rmse = np.nan
        if np.sum(feature) > 0:
            feature_rmse = regression_metrics(split_suite["test"]["y"][feature], pred[feature])["RMSE"]
        rows.append([seed, model["best_epoch"], metrics["RMSE"], feature_rmse])
        models.append(model)
    return models, rows


def plot_prediction_diagnostics(model, data, reference_train):
    y_pred = predict_model(model, data)
    residual = y_pred - data["y"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].scatter(data["y"], y_pred, c=data["c"], cmap="coolwarm", s=28, alpha=0.75, edgecolor="white", linewidth=0.25)
    lo = min(np.min(data["y"]), np.min(y_pred))
    hi = max(np.max(data["y"]), np.max(y_pred))
    axes[0].plot([lo, hi], [lo, hi], color="#555555", lw=1, ls="--")
    axes[0].set_xlabel("observed Y")
    axes[0].set_ylabel("predicted Y")
    axes[0].set_title("Observed vs predicted")

    axes[1].axhline(0, color="#555555", lw=1)
    axes[1].scatter(data["x"], residual, c=data["c"], cmap="coolwarm", s=28, alpha=0.75, edgecolor="white", linewidth=0.25)
    axes[1].axvspan(60, 70, color=EVAL_COLORS["support"], alpha=0.25)
    axes[1].set_xlabel("observed tilt X")
    axes[1].set_ylabel("residual")
    axes[1].set_title("Residuals by X")

    bin_rows = error_by_bin_rows(data, y_pred)
    labels = [row[0] for row in bin_rows]
    rmse = [row[2] for row in bin_rows]
    axes[2].bar(labels, rmse, color=EVAL_COLORS["diagnostic"])
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("Error by tilt bin")
    axes[2].tick_params(axis="x", rotation=35)
    plt.tight_layout()

    display_table(["Tilt bin", "Count", "RMSE", "Bias"], bin_rows)
    display_table(["Slice", "Count", "RMSE", "MAE", "Bias", "Notes"], slice_metrics(data, y_pred, make_slices(data, reference_train)))


def evaluate_on_named_sets(model, split_suite):
    rows = []
    named_sets = {
        "validation": split_suite["val"],
        "final test": split_suite["test"],
        **split_suite["diagnostics"],
    }
    for name, data in named_sets.items():
        pred = predict_model(model, data)
        metrics = regression_metrics(data["y"], pred)
        weighted = task_weighted_metrics(data, pred, narrow_feature_weight)
        rows.append(
            [
                name,
                len(data["x"]),
                metrics["RMSE"],
                weighted["weighted_RMSE"],
                worst_slice_rmse(data, pred, split_suite["train"]),
                np.mean(data["c"]),
                np.sum((data["x"] >= 60) & (data["x"] < 70)),
            ]
        )
    return rows


def get_evidence_set(split_suite, evidence_set):
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "validation": "val",
        "test": "test",
        "final test": "test",
    }
    key = aliases.get(evidence_set, evidence_set)
    if key in {"train", "val", "test"}:
        return split_suite[key]
    if key in split_suite["diagnostics"]:
        return split_suite["diagnostics"][key]
    raise KeyError(f"unknown evidence_set: {evidence_set}")


def feature_mask(data, region=(60, 70)):
    left, right = region
    return (data["x"] >= left) & (data["x"] < right)


def weight_for_region(region=(60, 70), feature_weight=5.0):
    left, right = region

    def _weight(x):
        return np.where((x >= left) & (x < right), feature_weight, 1.0)

    return _weight


def safe_region_metrics(data, pred, mask):
    count = int(np.sum(mask))
    if count == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "R2": np.nan}
    metrics = regression_metrics(data["y"][mask], np.asarray(pred)[mask])
    metrics["n"] = count
    return metrics


def density_masks(data, reference_train, radius=5.0, low_threshold=5, high_threshold=20):
    counts = local_count(data["x"], reference_train["x"], radius=radius)
    low = counts <= low_threshold
    high = counts >= high_threshold
    low_rule = f"N_r(x) <= {low_threshold}"
    high_rule = f"N_r(x) >= {high_threshold}"
    if not np.any(low):
        low_cut = np.quantile(counts, 0.30)
        low = counts <= low_cut
        low_rule = f"lowest 30% support, cutoff {low_cut:.1f}"
    if not np.any(high):
        high_cut = np.quantile(counts, 0.70)
        high = counts >= high_cut
        high_rule = f"highest 30% support, cutoff {high_cut:.1f}"
    return counts, low, high, low_rule, high_rule


def metric_dashboard(
    model,
    suite,
    feature_region=(60, 70),
    density_radius=5.0,
    low_density_threshold=5,
    high_density_threshold=20,
    seed_std=None,
):
    rows = []
    for label, split_key in [("train RMSE", "train"), ("validation RMSE", "val"), ("final test RMSE", "test")]:
        data = suite[split_key]
        pred = predict_model(model, data)
        rows.append([label, regression_metrics(data["y"], pred)["RMSE"], data["name"]])

    test = suite["test"]
    test_pred = predict_model(model, test)
    feature = feature_mask(test, feature_region)
    rows.append([f"{feature_region} RMSE", safe_region_metrics(test, test_pred, feature)["RMSE"], "final test slice"])

    _, low, _, low_rule, _ = density_masks(
        test,
        suite["train"],
        radius=density_radius,
        low_threshold=low_density_threshold,
        high_threshold=high_density_threshold,
    )
    rows.append(["low-density RMSE", safe_region_metrics(test, test_pred, low)["RMSE"], low_rule])

    shifted = suite["diagnostics"]["shifted_context"]
    shifted_pred = predict_model(model, shifted)
    rows.append(["shifted-context RMSE", regression_metrics(shifted["y"], shifted_pred)["RMSE"], "diagnostic set"])
    rows.append(["worst-slice RMSE", worst_slice_rmse(test, test_pred, suite["train"]), "final test slices with n >= 10"])
    rows.append(["seed standard deviation", seed_std, "computed in the stability section" if seed_std is None else "test RMSE across seeds"])
    return rows


def plot_observed_splits(suite, feature_region=(60, 70), title="Observed data and evidence roles"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, marker in [("train", "o"), ("val", "^"), ("test", "s")]:
        data = suite[key]
        ax.scatter(
            data["x"],
            data["y"],
            s=28,
            alpha=0.65,
            marker=marker,
            color=EVAL_COLORS[key],
            edgecolor="white",
            linewidth=0.25,
            label=data["name"],
        )
    ax.axvspan(feature_region[0], feature_region[1], color=EVAL_COLORS["support"], alpha=0.3, label=f"local feature {feature_region}")
    ax.plot(GRID, f0(GRID), color=EVAL_COLORS["truth"], lw=2, label="latent teaching curve")
    ax.set_xlabel("observed tilt X")
    ax.set_ylabel("observed power Y")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    return fig, ax


def candidate_summary(candidate_name, model, suite, notes="", shifted_data=None, seed_std=None):
    train_pred = predict_model(model, suite["train"])
    val_pred = predict_model(model, suite["val"])
    test_pred = predict_model(model, suite["test"])
    train_rmse = regression_metrics(suite["train"]["y"], train_pred)["RMSE"]
    val_rmse = regression_metrics(suite["val"]["y"], val_pred)["RMSE"]
    test_rmse = regression_metrics(suite["test"]["y"], test_pred)["RMSE"]
    feature_mask = make_slices(suite["test"], suite["train"])["60 <= X < 70"]
    feature_rmse = np.nan
    if np.sum(feature_mask) > 0:
        feature_rmse = regression_metrics(suite["test"]["y"][feature_mask], test_pred[feature_mask])["RMSE"]
    low_mask = make_slices(suite["test"], suite["train"])["low-density"]
    low_rmse = np.nan
    if np.sum(low_mask) > 0:
        low_rmse = regression_metrics(suite["test"]["y"][low_mask], test_pred[low_mask])["RMSE"]
    if shifted_data is None:
        shifted_data = suite["diagnostics"]["shifted_context"]
    shifted_pred = predict_model(model, shifted_data)
    shifted_rmse = regression_metrics(shifted_data["y"], shifted_pred)["RMSE"]
    return [
        candidate_name,
        train_rmse,
        val_rmse,
        test_rmse,
        feature_rmse,
        low_rmse,
        shifted_rmse,
        seed_std,
        notes,
    ]


def compact_train_config(reference_config, **overrides):
    cfg = dict(reference_config)
    cfg.update({"epochs": 180, "width": 64, "depth": 2, "seed": 41, "name": "Compact MLP"})
    cfg.update(overrides)
    return cfg


def fit_poly_on_indices(train, indices, degree=7, ridge_lambda=1e-3):
    subset = clone_data(train, indices, name="train subset")
    return fit_polynomial_baseline(subset, degree=degree, ridge_lambda=ridge_lambda)


def polynomial_learning_curve(train, val, sizes=(40, 80, 140, 220)):
    rng = set_seed(91)
    order = np.arange(len(train["x"]))
    rng.shuffle(order)
    rows = []
    for size in sizes:
        idx = np.sort(order[: min(size, len(order))])
        model = fit_poly_on_indices(train, idx)
        train_subset = clone_data(train, idx, name="train subset")
        train_pred = predict_model(model, train_subset)
        val_pred = predict_model(model, val)
        rows.append(
            [
                len(idx),
                regression_metrics(train_subset["y"], train_pred)["RMSE"],
                regression_metrics(val["y"], val_pred)["RMSE"],
                regression_metrics(val["y"], val_pred)["Bias"],
            ]
        )
    return rows


def polynomial_validation_curve(train, val, degrees=(1, 3, 5, 7, 11, 15)):
    rows = []
    for degree in degrees:
        model = fit_polynomial_baseline(train, degree=degree, ridge_lambda=1e-3)
        train_pred = predict_model(model, train)
        val_pred = predict_model(model, val)
        rows.append(
            [
                degree,
                regression_metrics(train["y"], train_pred)["RMSE"],
                regression_metrics(val["y"], val_pred)["RMSE"],
                worst_slice_rmse(val, val_pred, train),
            ]
        )
    return rows


__all__ = [
    "configure_solution_matplotlib",
    "SEED",
    "GRID",
    "EVAL_COLORS",
    "SLICE_NOTES",
    "set_seed",
    "display_markdown",
    "format_cell",
    "markdown_table",
    "display_table",
    "clone_data",
    "data_matrix",
    "make_grid_data",
    "make_tilt_dataset",
    "_allocate_random",
    "_stratified_indices",
    "make_split_suite",
    "describe_split",
    "plot_split_distributions",
    "regression_metrics",
    "make_slices",
    "slice_metrics",
    "task_weighted_metrics",
    "narrow_feature_weight",
    "worst_slice_rmse",
    "plot_predictions",
    "plot_residuals",
    "error_by_bin_rows",
    "fit_mean_baseline",
    "fit_linear_baseline",
    "fit_polynomial_baseline",
    "predict_model",
    "evaluate_model",
    "_activation",
    "_activation_grad",
    "initialise_mlp",
    "copy_params",
    "forward_mlp",
    "backward_mlp",
    "_init_optimizer_state",
    "_apply_update",
    "predict_numpy_mlp",
    "_predict_with_params",
    "train_model",
    "plot_training_history",
    "run_seed_sweep",
    "plot_prediction_diagnostics",
    "evaluate_on_named_sets",
    "get_evidence_set",
    "feature_mask",
    "weight_for_region",
    "safe_region_metrics",
    "density_masks",
    "metric_dashboard",
    "plot_observed_splits",
    "candidate_summary",
    "compact_train_config",
    "fit_poly_on_indices",
    "polynomial_learning_curve",
    "polynomial_validation_curve",
]
