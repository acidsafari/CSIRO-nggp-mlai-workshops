"""Tests for reusable Solution Space notebook helpers."""

from __future__ import annotations

import unittest

import numpy as np

from nextgen2026_mlai_workshops import solution_space


class SolutionSpaceHelperTests(unittest.TestCase):
    def test_evidence_set_lookup_and_feature_mask(self) -> None:
        data = solution_space.make_tilt_dataset(n=90, seed=5, sampling="uniform")
        suite = solution_space.make_split_suite(data, split_type="stratified_by_x_bin", seed=5)

        self.assertIs(solution_space.get_evidence_set(suite, "validation"), suite["val"])
        self.assertIs(solution_space.get_evidence_set(suite, "final test"), suite["test"])
        self.assertIs(solution_space.get_evidence_set(suite, "shifted_context"), suite["diagnostics"]["shifted_context"])

        mask = solution_space.feature_mask(suite["test"], region=(60, 70))
        self.assertEqual(mask.shape, suite["test"]["x"].shape)
        self.assertEqual(mask.dtype, np.bool_)

    def test_safe_region_and_density_metrics_are_finite(self) -> None:
        data = solution_space.make_tilt_dataset(n=120, seed=7, sampling="sparse_feature")
        suite = solution_space.make_split_suite(data, split_type="stratified_by_x_bin", seed=7)
        model = solution_space.fit_mean_baseline(suite["train"])
        pred = solution_space.predict_model(model, suite["test"])

        counts, low, high, low_rule, high_rule = solution_space.density_masks(suite["test"], suite["train"])
        self.assertEqual(counts.shape, suite["test"]["x"].shape)
        self.assertEqual(low.shape, suite["test"]["x"].shape)
        self.assertEqual(high.shape, suite["test"]["x"].shape)
        self.assertTrue(low_rule.startswith("N_r") or "support" in low_rule)
        self.assertTrue(high_rule.startswith("N_r") or "support" in high_rule)

        metrics = solution_space.safe_region_metrics(suite["test"], pred, low)
        self.assertGreaterEqual(metrics["n"], 0)
        if metrics["n"]:
            self.assertTrue(np.isfinite(metrics["RMSE"]))

    def test_metric_dashboard_returns_expected_rows(self) -> None:
        data = solution_space.make_tilt_dataset(n=100, seed=13, sampling="uniform")
        suite = solution_space.make_split_suite(data, split_type="iid_random", seed=13)
        model = solution_space.fit_mean_baseline(suite["train"])

        rows = solution_space.metric_dashboard(model, suite, feature_region=(60, 70), seed_std=0.02)

        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[-1][0], "seed standard deviation")
        self.assertEqual(rows[-1][1], 0.02)
        self.assertTrue(all(len(row) == 3 for row in rows))


if __name__ == "__main__":
    unittest.main()
