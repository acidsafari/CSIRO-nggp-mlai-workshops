"""Diagnostic metric helpers for the probabilistic regression workshop."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared error."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean((y_true_array - y_pred_array) ** 2))


def gaussian_negative_log_likelihood(
    y_true: np.ndarray, y_pred: np.ndarray, sigma: float = 1.0
) -> float:
    """Gaussian negative log-likelihood under variance ``sigma**2``.

    This is returned as a sum across observations so values are directly
    comparable when sample sizes are fixed.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    residual = y_true_array - y_pred_array
    return float(
        0.5
        * np.sum((residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
    )


def coefficient_norm(coeffs: np.ndarray) -> float:
    """Return the L2 norm of fitted coefficients."""
    return float(np.linalg.norm(np.asarray(coeffs, dtype=float)))


__all__ = [
    "mse",
    "gaussian_negative_log_likelihood",
    "coefficient_norm",
]
