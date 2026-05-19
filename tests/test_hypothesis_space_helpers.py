"""Tests for reusable Hypothesis Space notebook helpers."""

from __future__ import annotations

import unittest

import numpy as np

from nextgen2026_mlai_workshops import hypothesis_space


class HypothesisSpaceHelperTests(unittest.TestCase):
    def test_design_matrices_have_expected_shapes(self) -> None:
        x = np.array([0.0, 45.0, 90.0])

        self.assertEqual(hypothesis_space.polynomial_design(x, degree=4).shape, (3, 5))
        self.assertEqual(hypothesis_space.periodic_design(x, frequency_count=2).shape, (3, 5))
        self.assertEqual(hypothesis_space.relu_basis_design(x, np.array([30.0, 60.0])).shape, (3, 4))

    def test_ridge_fit_returns_vectorised_predictor(self) -> None:
        x = np.linspace(0, 90, 12)
        y = hypothesis_space.f0(x)
        predict, coef = hypothesis_space.fit_basis_model(
            x,
            y,
            lambda values: hypothesis_space.polynomial_design(values, degree=3),
            lam=0.01,
        )

        self.assertEqual(coef.shape, (4,))
        self.assertEqual(predict(np.array([0.0, 45.0])).shape, (2,))
        self.assertTrue(np.all(np.isfinite(predict(x))))

    def test_parameter_equivalence_transforms_match_original(self) -> None:
        original, permuted = hypothesis_space.parameter_equivalence("permutation")
        original_again, rescaled = hypothesis_space.parameter_equivalence("positive rescaling", alpha=2.5)

        np.testing.assert_allclose(original, original_again)
        np.testing.assert_allclose(original, permuted, atol=1e-12)
        np.testing.assert_allclose(original, rescaled, atol=1e-12)

    def test_support_diagnostics_are_finite(self) -> None:
        predict, label, assumption = hypothesis_space.fit_support_model("ReLU basis", complexity=5, lam=0.02)
        diagnostics = hypothesis_space.support_diagnostics(predict)

        self.assertIn("ReLU", label)
        self.assertIn("continuation", assumption)
        self.assertEqual(set(diagnostics), {"training", "observed_oracle", "gap_oracle"})
        self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))


if __name__ == "__main__":
    unittest.main()
