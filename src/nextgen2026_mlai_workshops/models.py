"""Model-fitting helpers for the probabilistic regression workshop."""

from __future__ import annotations

import numpy as np


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Build a Vandermonde design matrix with powers x^0 .. x^degree."""
    if degree < 0:
        raise ValueError("degree must be >= 0")

    x_array = np.asarray(x, dtype=float)
    if x_array.ndim != 1:
        raise ValueError("x must be one-dimensional")

    return np.vander(x_array, N=degree + 1, increasing=True)


def fit_polynomial_mle(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Least-squares polynomial fit (maximum-likelihood under Gaussian noise)."""
    x_matrix = polynomial_features(x=x, degree=degree)
    coeffs, *_ = np.linalg.lstsq(x_matrix, y, rcond=None)
    return coeffs


def fit_polynomial_ridge_map(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    lambda_reg: float = 0.0,
    penalize_bias: bool = False,
) -> np.ndarray:
    """Closed-form ridge/MAP coefficients with optional intercept-exemption."""
    if lambda_reg < 0.0:
        raise ValueError("lambda_reg must be >= 0")

    x_matrix = polynomial_features(x=x, degree=degree)
    gram = x_matrix.T @ x_matrix
    rhs = x_matrix.T @ y
    penalty = np.eye(degree + 1)
    if not penalize_bias:
        penalty[0, 0] = 0.0
    regularized = gram + lambda_reg * penalty
    try:
        return np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        # Numerical fallback for edge cases such as duplicate x locations.
        if lambda_reg == 0.0:
            return np.linalg.lstsq(x_matrix, y, rcond=None)[0]

        augmented_left = np.vstack(
            [x_matrix, np.sqrt(lambda_reg) * np.sqrt(penalty)]
        )
        augmented_right = np.concatenate([y, np.zeros(degree + 1)])
        return np.linalg.lstsq(augmented_left, augmented_right, rcond=None)[0]


def predict_polynomial(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial at x with coefficients from degree 0..d."""
    x_matrix = polynomial_features(x=x, degree=len(coeffs) - 1)
    return x_matrix @ coeffs


__all__ = [
    "polynomial_features",
    "fit_polynomial_mle",
    "fit_polynomial_ridge_map",
    "predict_polynomial",
]
