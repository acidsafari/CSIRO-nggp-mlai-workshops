"""Tests for Workshop 3 solar PV helpers."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from nextgen2026_mlai_workshops import solar_pv


class SolarPVSimulatorTests(unittest.TestCase):
    def test_observations_are_deterministic_and_in_range(self) -> None:
        first = solar_pv.make_pv_observations(40, "deployment_conditions", seed=123)
        second = solar_pv.make_pv_observations(40, "deployment_conditions", seed=123)
        np.testing.assert_allclose(first["normalized_power"], second["normalized_power"])
        self.assertTrue(np.all((first["irradiance"] >= 0) & (first["irradiance"] <= 1000)))
        self.assertTrue(np.all((first["ambient_temperature"] >= 5) & (first["ambient_temperature"] <= 45)))
        self.assertTrue(np.all((first["tilt_angle"] >= 0) & (first["tilt_angle"] <= 60)))
        self.assertTrue(np.all((first["normalized_power"] >= 0) & (first["normalized_power"] <= 1)))

    def test_bundle_contract(self) -> None:
        bundle = solar_pv.make_workshop3_bundle("baseline", seed=7)
        self.assertEqual(len(bundle["train"]["normalized_power"]), 600)
        self.assertEqual(len(bundle["validation"]["normalized_power"]), 200)
        self.assertEqual(len(bundle["test"]["normalized_power"]), 200)
        self.assertEqual(bundle["visible_inputs"], ["irradiance", "ambient_temperature", "tilt_angle"])
        self.assertIn("cloud_cover", bundle["revealed_inputs"])
        self.assertIn("panel_temperature", bundle["revealed_inputs"])

    def test_baseline_config_uses_fixed_architecture(self) -> None:
        config = solar_pv.baseline_config()
        self.assertEqual(config["hidden_width"], 16)
        self.assertEqual(config["depth"], 2)
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["optimizer"], "sgd")
        self.assertEqual(config["momentum"], 0.0)

    def test_challenge_bundle_uses_stricter_criterion(self) -> None:
        bundle = solar_pv.make_challenge_bundle(seed=7)
        self.assertEqual(bundle["metadata"]["scenario"], "experiment_lab_challenge")
        self.assertEqual(bundle["metadata"]["data_policy"], "none")
        self.assertEqual(len(bundle["train"]["normalized_power"]), 600)
        self.assertEqual(bundle["criterion"].overall_rmse, 0.038)
        self.assertEqual(bundle["criterion"].key_range_rmse, 0.045)

    def test_challenge_bundle_can_add_policy_rows(self) -> None:
        bundle = solar_pv.make_challenge_bundle(
            seed=7,
            data_policy="deployment_matched_mix",
            added_n=25,
        )
        self.assertEqual(bundle["metadata"]["data_policy"], "deployment_matched_mix")
        self.assertEqual(bundle["metadata"]["added_n"], 25)
        self.assertEqual(len(bundle["train"]["normalized_power"]), 625)
        self.assertEqual(len(bundle["validation"]["normalized_power"]), 200)
        self.assertEqual(len(bundle["test"]["normalized_power"]), 200)

    def test_challenge_bundle_rejects_added_rows_without_policy(self) -> None:
        with self.assertRaises(ValueError):
            solar_pv.make_challenge_bundle(seed=7, added_n=25)

    def test_challenge_bundle_can_target_selected_points(self) -> None:
        bundle = solar_pv.make_challenge_bundle(
            seed=7,
            data_policy="target_selected_points",
            added_n=30,
            target_point_indices=[0, 5, 12],
        )
        self.assertEqual(bundle["metadata"]["data_policy"], "target_selected_points")
        self.assertEqual(bundle["metadata"]["target_split"], "validation")
        self.assertEqual(bundle["metadata"]["target_point_indices"], [0, 5, 12])
        self.assertEqual(len(bundle["train"]["normalized_power"]), 630)

    def test_challenge_bundle_rejects_invalid_selected_points(self) -> None:
        with self.assertRaises(ValueError):
            solar_pv.make_challenge_bundle(
                seed=7,
                data_policy="target_selected_points",
                added_n=30,
                target_point_indices=[999],
            )

    def test_challenge_option_rows_include_multiple_spaces(self) -> None:
        rows = solar_pv.challenge_option_rows()
        categories = {row[0] for row in rows}
        self.assertIn("Data policy", categories)
        self.assertIn("Selected points", categories)
        self.assertIn("Inputs", categories)
        self.assertIn("Optimizer", categories)
        self.assertIn("Objective", categories)

    def test_challenge_candidate_point_rows_have_validation_indices(self) -> None:
        bundle = solar_pv.make_challenge_bundle(seed=7)
        rows = solar_pv.challenge_candidate_point_rows(bundle, top_k=4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(isinstance(row[0], int) for row in rows))
        self.assertTrue(all(0 <= row[0] < len(bundle["validation"]["normalized_power"]) for row in rows))

    def test_visible_only_bundle_hides_nonvisible_fields_and_sampling_metadata(self) -> None:
        bundle = solar_pv.visible_only_bundle(solar_pv.make_workshop3_bundle("baseline", seed=7))
        self.assertEqual(bundle["revealed_inputs"], [])
        self.assertEqual(bundle["metadata"], {"view": "visible_only"})
        for split in ("train", "validation", "test"):
            self.assertNotIn("cloud_cover", bundle[split])
            self.assertNotIn("panel_temperature", bundle[split])
            self.assertNotIn("sampling_policy", bundle[split]["metadata"])
            self.assertEqual(bundle[split]["metadata"], {"view": "visible_only"})

    def test_missing_combination_scenario_has_training_gap(self) -> None:
        bundle = solar_pv.make_workshop3_bundle("data_missing_combinations", seed=7)
        self.assertEqual(int(np.sum(solar_pv.joint_gap_mask(bundle["train"]))), 0)
        self.assertGreater(int(np.sum(solar_pv.joint_gap_mask(bundle["validation"]))), 0)

    def test_print_table_pads_and_aligns_numeric_columns(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            solar_pv.print_table(
                ["Label", "Width", "Validation RMSE"],
                [
                    ["tiny", 2, 0.123456],
                    ["wide regularized", 128, 0.045],
                ],
            )

        lines = buffer.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("---:", lines[1])
        self.assertIn("0.1235", lines[2])
        self.assertIn("0.0450", lines[3])
        self.assertEqual([len(line) for line in lines], [len(lines[0])] * len(lines))

    def test_custom_input_observations_preserve_requested_visible_values(self) -> None:
        values = [
            {"irradiance": 820.0, "ambient_temperature": 36.0, "tilt_angle": 12.0},
            {"irradiance": 880.0, "ambient_temperature": 38.0, "tilt_angle": 50.0},
        ]
        data = solar_pv.make_pv_observations_at_inputs(values, seed=7)
        np.testing.assert_allclose(data["irradiance"], [820.0, 880.0])
        np.testing.assert_allclose(data["ambient_temperature"], [36.0, 38.0])
        np.testing.assert_allclose(data["tilt_angle"], [12.0, 50.0])
        self.assertTrue(np.all((data["normalized_power"] >= 0) & (data["normalized_power"] <= 1)))


@unittest.skipIf(solar_pv.torch is None, "PyTorch is not installed")
class SolarPVTrainingTests(unittest.TestCase):
    def test_short_training_run_produces_report(self) -> None:
        bundle = solar_pv.make_workshop3_bundle("baseline", seed=7)
        run = solar_pv.train_with_config(bundle, solar_pv.baseline_config(epochs=30), name="test run")
        report = solar_pv.evaluate_model_report(run, bundle, include_test=False)
        self.assertEqual(len(run["history"]), 30)
        self.assertIn("metric_rows", report)
        self.assertFalse(report["include_test"])
        self.assertTrue(np.isfinite(report["criterion"]["overall_validation_rmse"]))

    def test_short_training_supports_added_optimizer_choices(self) -> None:
        bundle = solar_pv.make_workshop3_bundle("baseline", seed=7)
        for optimizer, momentum, learning_rate in [
            ("sgd", 0.0, 0.035),
            ("sgd_momentum", 0.8, 0.02),
            ("adam", 0.0, 0.005),
        ]:
            with self.subTest(optimizer=optimizer):
                run = solar_pv.train_with_config(
                    bundle,
                    solar_pv.baseline_config(
                        epochs=3,
                        optimizer=optimizer,
                        momentum=momentum,
                        learning_rate=learning_rate,
                    ),
                    name=optimizer,
                )
                self.assertEqual(run["config"]["optimizer"], optimizer)
                self.assertEqual(len(run["history"]), 3)

    def test_plain_sgd_rejects_nonzero_momentum(self) -> None:
        bundle = solar_pv.make_workshop3_bundle("baseline", seed=7)
        with self.assertRaises(ValueError):
            solar_pv.train_with_config(
                bundle,
                solar_pv.baseline_config(epochs=1, optimizer="sgd", momentum=0.2),
            )

    def test_challenge_runner_hides_test_by_default(self) -> None:
        bundle = solar_pv.make_challenge_bundle(seed=7)
        result = solar_pv.run_challenge_experiment(
            bundle,
            solar_pv.challenge_config(epochs=3),
            include_test=False,
            name="hidden test run",
        )
        self.assertFalse(result["report"]["include_test"])
        self.assertEqual(len(result["row"]), 13)
        self.assertNotIn("test", [row[0] for row in result["report"]["metric_rows"]])

    def test_challenge_runner_includes_test_when_requested(self) -> None:
        bundle = solar_pv.make_challenge_bundle(seed=7)
        result = solar_pv.run_challenge_experiment(
            bundle,
            solar_pv.challenge_config(epochs=3),
            include_test=True,
            name="final check run",
        )
        self.assertTrue(result["report"]["include_test"])
        self.assertEqual(len(result["row"]), 14)
        self.assertIn("test", [row[0] for row in result["report"]["metric_rows"]])

    def test_useful_measurement_beats_candidate_sensor_c_in_calibration(self) -> None:
        bundle = solar_pv.make_workshop3_bundle("hypothesis_new_measurements", seed=7)
        result = solar_pv.run_input_set_options(
            bundle,
            input_sets=[
                ("baseline", solar_pv.VISIBLE_INPUTS),
                ("add cloud", [*solar_pv.VISIBLE_INPUTS, "cloud_cover"]),
                ("add candidate sensor C", [*solar_pv.VISIBLE_INPUTS, solar_pv.CANDIDATE_SENSOR_C]),
            ],
            seed=7,
        )
        rows = {row[0]: row for row in result["rows"]}
        self.assertLess(rows["add cloud"][2], rows["baseline"][2])
        self.assertLess(rows["add cloud"][2], rows["add candidate sensor C"][2])

    def test_data_collection_options_accept_custom_input_values(self) -> None:
        bundle = solar_pv.make_workshop3_bundle("data_missing_combinations", seed=7)
        result = solar_pv.run_data_collection_options(
            bundle,
            options=[
                {
                    "label": "custom edge rows",
                    "input_values": [
                        {"irradiance": 860.0, "ambient_temperature": 37.0, "tilt_angle": 8.0},
                        {"irradiance": 860.0, "ambient_temperature": 37.0, "tilt_angle": 52.0},
                    ],
                }
            ],
            added_n=12,
            seed=7,
        )
        self.assertEqual(result["rows"][0][0], "custom edge rows")
        self.assertEqual(result["rows"][0][1], 12)
        self.assertEqual(len(solar_pv.target_vector(result["bundles"]["custom edge rows"]["train"])), 612)


if __name__ == "__main__":
    unittest.main()
