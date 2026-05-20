"""Tests for reusable Optimization Space notebook helpers."""

from __future__ import annotations

import unittest

import numpy as np

from nextgen2026_mlai_workshops import optimization_space


class OptimizationSpaceHelperTests(unittest.TestCase):
    def test_mlp_data_and_initialisation_are_deterministic(self) -> None:
        x_first, y_first, clean_first = optimization_space.make_mlp_regression_data(n=16, seed=11)
        x_second, y_second, clean_second = optimization_space.make_mlp_regression_data(n=16, seed=11)

        np.testing.assert_allclose(x_first, x_second)
        np.testing.assert_allclose(y_first, y_second)
        np.testing.assert_allclose(clean_first, clean_second)

        params_first = optimization_space.initialise_mlp(width=6, seed=5)
        params_second = optimization_space.initialise_mlp(width=6, seed=5)

        for key in params_first:
            np.testing.assert_allclose(params_first[key], params_second[key])

    def test_mlp_forward_and_backward_shapes(self) -> None:
        x, y, _ = optimization_space.make_mlp_regression_data(n=10, seed=2)
        params = optimization_space.initialise_mlp(width=8, seed=3)

        yhat, cache = optimization_space.forward_mlp(params, x)
        grads = optimization_space.backward_mlp(params, cache, y)

        self.assertEqual(yhat.shape, y.shape)
        self.assertEqual(cache["z1"].shape, (10, 8))
        self.assertEqual(set(grads), set(params))
        for key in params:
            self.assertEqual(grads[key].shape, params[key].shape)
        self.assertTrue(np.isfinite(optimization_space.gradient_norm(grads)))

    def test_mlp_training_reduces_loss(self) -> None:
        x, y, _ = optimization_space.make_mlp_regression_data(n=30, noise=0.05, seed=4)

        _, history = optimization_space.train_mlp(
            x,
            y,
            width=12,
            seed=1,
            learning_rate=0.03,
            epochs=300,
            log_every=50,
        )

        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertEqual(history["epoch"][-1], 300)
        self.assertTrue(all(np.isfinite(value) for value in history["loss"]))

    def test_tilt_power_mlp_data_has_scaled_and_raw_views(self) -> None:
        sample = optimization_space.make_mlp_tilt_power_data(n=24, seed=9)

        self.assertEqual(sample["x_raw"].shape, (24, 1))
        self.assertEqual(sample["x_scaled"].shape, (24, 1))
        self.assertEqual(sample["y"].shape, (24, 1))
        self.assertEqual(sample["grid_raw"].shape, sample["grid_scaled"].shape)
        self.assertEqual(sample["grid_raw"].shape, sample["y_true"].shape)
        self.assertAlmostEqual(float(np.mean(sample["x_scaled"])), 0.0, places=12)
        self.assertAlmostEqual(float(np.std(sample["x_scaled"])), 1.0, places=12)

    def test_tilt_power_recipe_returns_finite_diagnostics(self) -> None:
        sample = optimization_space.make_mlp_tilt_power_data(n=36, seed=12)

        run = optimization_space.train_mlp_tilt_power_recipe(
            sample,
            input_mode="scaled",
            width=12,
            init_mode="he",
            seed=2,
            learning_rate=0.03,
            epochs=120,
            log_every=40,
        )

        self.assertEqual(run["history"]["epoch"][-1], 120)
        self.assertLess(
            run["final_diagnostics"]["train_mse"],
            run["initial_diagnostics"]["train_mse"],
        )
        for diagnostics in [run["initial_diagnostics"], run["final_diagnostics"]]:
            self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))

    def test_relu_parameter_equivalence_transforms_match_original(self) -> None:
        width = 8
        x_grid, _ = optimization_space.make_mlp_grid(n=80)
        params = optimization_space.initialise_mlp(width=width, seed=4)

        permuted = optimization_space.copy_params(params)
        perm = np.random.default_rng(127).permutation(width)
        permuted["W1"] = params["W1"][:, perm]
        permuted["b1"] = params["b1"][:, perm]
        permuted["W2"] = params["W2"][perm, :]

        alpha = 2.5
        rescaled = optimization_space.copy_params(params)
        rescaled["W1"] = alpha * params["W1"]
        rescaled["b1"] = alpha * params["b1"]
        rescaled["W2"] = params["W2"] / alpha

        original_y, _ = optimization_space.forward_mlp(params, x_grid, activation="relu")
        permuted_y, _ = optimization_space.forward_mlp(permuted, x_grid, activation="relu")
        rescaled_y, _ = optimization_space.forward_mlp(rescaled, x_grid, activation="relu")

        np.testing.assert_allclose(original_y, permuted_y, atol=1e-12)
        np.testing.assert_allclose(original_y, rescaled_y, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
