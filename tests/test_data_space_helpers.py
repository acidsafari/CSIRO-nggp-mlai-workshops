"""Tests for reusable Data Space notebook helpers."""

from __future__ import annotations

import unittest

import numpy as np

from nextgen2026_mlai_workshops import data_space


class DataSpaceHelperTests(unittest.TestCase):
    def test_sample_tilt_power_is_deterministic(self) -> None:
        first = data_space.sample_tilt_power(n=20, scenario="hidden_context", seed=123)
        second = data_space.sample_tilt_power(n=20, scenario="hidden_context", seed=123)

        np.testing.assert_allclose(first["x"], second["x"])
        np.testing.assert_allclose(first["y"], second["y"])
        np.testing.assert_array_equal(first["c"], second["c"])

    def test_sampling_gap_avoids_narrow_feature_region(self) -> None:
        theta = data_space.sample_theta(200, sampling="gap65", rng=data_space.set_seed(5))

        self.assertTrue(np.all((theta < 58) | (theta > 72)))

    def test_local_count_and_smoother_shapes(self) -> None:
        x_obs = np.array([0.0, 1.0, 3.0])
        y_obs = np.array([0.0, 2.0, 4.0])
        grid = np.array([0.0, 2.0, 4.0])

        counts = data_space.local_count(grid, x_obs, radius=1.0)
        smoothed = data_space.gaussian_smoother(x_obs, y_obs, grid, bandwidth=1.5)

        np.testing.assert_array_equal(counts, np.array([2, 2, 1]))
        self.assertEqual(smoothed.shape, grid.shape)
        self.assertTrue(np.all(np.isfinite(smoothed)))

    def test_poly_ridge_fit_returns_vectorised_predictor(self) -> None:
        x = np.linspace(0, 90, 20)
        y = data_space.f0(x)
        predict = data_space.poly_ridge_fit(x, y, degree=3, lam=0.1)

        y_hat = predict(np.array([0.0, 45.0, 90.0]))

        self.assertEqual(y_hat.shape, (3,))
        self.assertTrue(np.all(np.isfinite(y_hat)))


if __name__ == "__main__":
    unittest.main()
