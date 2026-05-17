"""Data-generation helpers for Workshop 1.

The module keeps the data-generation surface small and deterministic so it is easy
to reuse for both static assets and optional interactive experimentation.
"""

from __future__ import annotations

import numpy as np


def true_function(x: np.ndarray | float) -> np.ndarray:
    """Evaluate the fixed teaching target function.

    The workshop uses a periodic signal so it is easy to see bias/variance
    effects across hypothesis spaces and regularisation strengths.
    """
    x_array = np.asarray(x, dtype=float)
    return np.sin(2.0 * np.pi * x_array)


def _to_rng(seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _sample_uniform(
    n: int, rng: np.random.Generator, x_min: float, x_max: float
) -> np.ndarray:
    return np.sort(rng.uniform(low=x_min, high=x_max, size=n))


def _sample_clustered(
    n: int, rng: np.random.Generator, x_min: float, x_max: float
) -> np.ndarray:
    # Two compact clusters in [0, 1] with mild overlap.
    centers = np.array([0.24, 0.74])
    centers = x_min + (x_max - x_min) * centers
    spread = (x_max - x_min) * 0.07

    n_left = max(1, n // 2)
    n_right = max(1, n - n_left)
    x_left = rng.normal(loc=centers[0], scale=spread, size=n_left)
    x_right = rng.normal(loc=centers[1], scale=spread, size=n_right)
    x = np.concatenate([x_left, x_right])
    return np.sort(np.clip(x, x_min, x_max))


def _sample_partial_domain(
    n: int, rng: np.random.Generator, x_min: float, x_max: float
) -> np.ndarray:
    # Keep all x in a strict sub-range of the plotting domain.
    partial_min = x_min
    partial_max = x_min + 0.7 * (x_max - x_min)
    return _sample_uniform(n=n, rng=rng, x_min=partial_min, x_max=partial_max)


def make_regression_data(
    n: int = 20,
    noise_std: float = 0.2,
    seed: int | None = None,
    x_range: tuple[float, float] = (0.0, 1.0),
    sampling: str = "uniform",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy target observations from :func:`true_function`.

    Parameters
    ----------
    n:
        Number of observations.
    noise_std:
        Standard deviation of Gaussian noise.
    seed:
        Optional RNG seed. Ignored when *rng* is provided.
    x_range:
        Inclusive range for x values.
    sampling:
        One of ``"uniform"``, ``"clustered"``, or ``"partial_domain"``.
    rng:
        Optional explicit RNG to support full caller control.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    x_min, x_max = x_range
    if not x_min < x_max:
        raise ValueError("x_range must be an increasing pair")

    sampler = {
        "uniform": _sample_uniform,
        "clustered": _sample_clustered,
        "partial_domain": _sample_partial_domain,
    }
    sampling_key = sampling.lower()
    if sampling_key not in sampler:
        raise ValueError(
            "sampling must be 'uniform', 'clustered', or 'partial_domain'"
        )

    local_rng = _to_rng(seed, rng)
    x = sampler[sampling_key](n=n, rng=local_rng, x_min=x_min, x_max=x_max)
    noise = local_rng.normal(loc=0.0, scale=noise_std, size=n)
    y = true_function(x) + noise
    return x, y


def make_test_grid(
    x_range: tuple[float, float] = (0.0, 1.0), n_points: int = 400
) -> np.ndarray:
    """Return a stable x-grid for plotting/evaluation."""
    x_min, x_max = x_range
    if not x_min < x_max:
        raise ValueError("x_range must be an increasing pair")
    return np.linspace(x_min, x_max, n_points)


def train_test_split_for_workshop(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    seed: int | None = 42,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple deterministic train/test split used by Workshop 1 assets."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")

    n = len(x)
    n_test = int(round(n * test_size))
    n_test = min(max(1, n_test), n - 1)

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return (
        x[train_idx],
        y[train_idx],
        x[test_idx],
        y[test_idx],
    )


__all__ = [
    "true_function",
    "make_regression_data",
    "make_test_grid",
    "train_test_split_for_workshop",
]
