"""Workshop 3 solar PV simulator, training, and investigation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:  # Keep simulator helpers importable when torch has not been installed yet.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - exercised only in minimal environments
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


SEED = 7
VISIBLE_INPUTS = ["irradiance", "ambient_temperature", "tilt_angle"]
REVEALED_INPUTS = ["cloud_cover", "panel_temperature"]
CANDIDATE_SENSOR_C = "candidate_sensor_c"
ALL_INPUTS = [*VISIBLE_INPUTS, *REVEALED_INPUTS, CANDIDATE_SENSOR_C]
TARGET = "normalized_power"

DATA_COLLECTION_OPTIONS = (
    "uniform_sampling",
    "target_low_support",
    "target_high_error",
    "stratified_regime",
    "more_normal_conditions",
)
COLLECTION_MIX_OPTIONS = (
    "normal_condition_mix",
    "deployment_matched_mix",
    "input_regime_stratified_mix",
    "rare_regime_oversampled",
    "temperature_irradiance_balanced_mix",
)
REGULARITY_OPTIONS = ("none", "weak", "medium", "strong")
ACTIVATION_OPTIONS = ("relu", "tanh", "silu", "gelu")
SCALING_OPTIONS = ("raw", "minmax", "standard")
INITIALIZATION_OPTIONS = ("small", "xavier", "he", "large")
LOSS_OPTIONS = ("mse", "mae", "huber")
SELECTION_METRIC_OPTIONS = ("rmse", "mae")

COLORS = {
    "train": "#2F5D7C",
    "validation": "#D18F24",
    "test": "#2F7D4A",
    "fit": "#C7502A",
    "diagnostic": "#7B5E9E",
    "residual": "#6A737D",
    "support": "#D9D9D9",
}


@dataclass(frozen=True)
class SuccessCriterion:
    """Visible-input success criterion used across Workshop 3."""

    overall_rmse: float = 0.085
    key_range_rmse: float = 0.115
    key_irradiance_min: float = 700.0
    key_ambient_min: float = 30.0
    key_tilt_min: float = 15.0
    key_tilt_max: float = 45.0


def challenge_success_criterion() -> SuccessCriterion:
    """Return the stricter criterion used by the final experiment lab."""
    return SuccessCriterion(overall_rmse=0.038, key_range_rmse=0.045)


def require_torch() -> Any:
    """Return the torch module or raise a helpful dependency error."""
    if torch is None:
        raise ImportError(
            "Workshop 3 training helpers require PyTorch. Install project "
            "dependencies before running model training cells."
        )
    return torch


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Return a deterministic NumPy random generator."""
    return np.random.default_rng(int(seed))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic helper for simulator components."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def clone_data(data: dict[str, Any], idx: np.ndarray | Sequence[int] | None = None, name: str | None = None) -> dict[str, Any]:
    """Copy a PV observation dictionary, optionally selecting rows."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key == "metadata":
            out[key] = dict(value)
        elif isinstance(value, np.ndarray):
            out[key] = value.copy() if idx is None else value[np.asarray(idx)].copy()
        else:
            out[key] = value
    if name is not None:
        out["name"] = name
    return out


def concat_data(parts: Sequence[dict[str, Any]], name: str = "combined") -> dict[str, Any]:
    """Concatenate same-schema PV observation dictionaries."""
    keys = [key for key, value in parts[0].items() if isinstance(value, np.ndarray)]
    out = {key: np.concatenate([np.asarray(part[key]) for part in parts]) for key in keys}
    out["name"] = name
    out["metadata"] = {"parts": [part.get("name", "data") for part in parts]}
    return out


def visible_only_data(data: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """Return a copy containing only visible inputs and the target."""
    keep = [*VISIBLE_INPUTS, TARGET]
    out = {key: np.asarray(data[key]).copy() for key in keep}
    out["name"] = name or data.get("name", "visible data")
    out["metadata"] = {"view": "visible_only"}
    return out


def visible_only_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return a bundle view with hidden simulator fields removed."""
    out = dict(bundle)
    for split in ("train", "validation", "test"):
        out[split] = visible_only_data(bundle[split], bundle[split].get("name", split))
    out["revealed_inputs"] = []
    out["metadata"] = {"view": "visible_only"}
    return out


def key_range_mask(data: dict[str, Any], criterion: SuccessCriterion | None = None) -> np.ndarray:
    """Return the visible-input key operating range mask."""
    criterion = criterion or SuccessCriterion()
    return (
        (data["irradiance"] >= criterion.key_irradiance_min)
        & (data["ambient_temperature"] >= criterion.key_ambient_min)
        & (data["tilt_angle"] >= criterion.key_tilt_min)
        & (data["tilt_angle"] <= criterion.key_tilt_max)
    )


def joint_gap_mask(data: dict[str, Any]) -> np.ndarray:
    """Return the visible-input joint region used for data-support diagnostics."""
    central_tilt = (data["tilt_angle"] >= 22.0) & (data["tilt_angle"] <= 38.0)
    return (data["irradiance"] >= 760.0) & (data["ambient_temperature"] >= 32.0) & ~central_tilt


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def _sample_visible(n: int, sampling_policy: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample visible variables plus cloud cover under a named policy."""
    n = int(n)
    if sampling_policy == "normal_conditions":
        irradiance = _clip(rng.normal(560, 180, n), 0, 1000)
        ambient = _clip(rng.normal(23, 6, n), 5, 45)
        tilt = _clip(rng.normal(30, 9, n), 0, 60)
        cloud = rng.beta(1.4, 4.8, n)
    elif sampling_policy == "deployment_conditions":
        mix = rng.uniform(size=n)
        irradiance = np.where(mix < 0.68, rng.normal(800, 130, n), rng.uniform(150, 1000, n))
        ambient = np.where(mix < 0.68, rng.normal(34, 5, n), rng.uniform(10, 45, n))
        tilt = _clip(rng.normal(30, 11, n), 0, 60)
        cloud = np.where(mix < 0.55, rng.beta(1.7, 2.4, n), rng.beta(1.1, 5.0, n))
        irradiance = _clip(irradiance, 0, 1000)
        ambient = _clip(ambient, 5, 45)
    elif sampling_policy in {"broad_operating_conditions", "uniform_visible"}:
        irradiance = rng.uniform(0, 1000, n)
        ambient = rng.uniform(5, 45, n)
        tilt = rng.uniform(0, 60, n)
        cloud = rng.beta(1.3, 2.2, n)
    elif sampling_policy == "visible_regime_stratified":
        irradiance_bins = rng.choice([125, 375, 625, 875], n)
        ambient_bins = rng.choice([10, 20, 30, 40], n)
        tilt_bins = rng.choice([7.5, 22.5, 37.5, 52.5], n)
        irradiance = _clip(rng.normal(irradiance_bins, 70), 0, 1000)
        ambient = _clip(rng.normal(ambient_bins, 3.5), 5, 45)
        tilt = _clip(rng.normal(tilt_bins, 4.5), 0, 60)
        cloud = rng.beta(1.4, 2.4, n)
    elif sampling_policy == "rare_regime_oversampled":
        mix = rng.uniform(size=n)
        irradiance = np.where(mix < 0.55, rng.normal(850, 90, n), rng.uniform(0, 1000, n))
        ambient = np.where(mix < 0.55, rng.normal(36, 4, n), rng.uniform(5, 45, n))
        tilt = np.where(mix < 0.55, rng.choice([rng.normal(10, 4, n), rng.normal(50, 4, n)]), rng.uniform(0, 60, n))
        cloud = rng.beta(1.4, 2.0, n)
        irradiance = _clip(irradiance, 0, 1000)
        ambient = _clip(ambient, 5, 45)
        tilt = _clip(tilt, 0, 60)
    elif sampling_policy == "temperature_irradiance_balanced":
        irr_levels = rng.choice([150, 350, 550, 750, 925], n)
        temp_levels = rng.choice([10, 18, 26, 34, 42], n)
        irradiance = _clip(rng.normal(irr_levels, 55), 0, 1000)
        ambient = _clip(rng.normal(temp_levels, 2.8), 5, 45)
        tilt = _clip(rng.normal(30, 12, n), 0, 60)
        cloud = rng.beta(1.5, 2.4, n)
    elif sampling_policy == "joint_gap_region":
        irradiance = _clip(rng.normal(860, 75, n), 720, 1000)
        ambient = _clip(rng.normal(36, 3.8, n), 28, 45)
        low_side = rng.uniform(size=n) < 0.5
        tilt = np.where(low_side, rng.normal(10, 4, n), rng.normal(50, 4, n))
        tilt = _clip(tilt, 0, 60)
        cloud = rng.beta(1.5, 2.6, n)
    elif sampling_policy == "more_normal_conditions":
        irradiance, ambient, tilt, cloud = _sample_visible(n, "normal_conditions", rng)
    elif sampling_policy == "missing_joint_region":
        values: list[tuple[float, float, float, float]] = []
        while len(values) < n:
            batch_n = max(256, n)
            draws = make_pv_observations(batch_n, "broad_operating_conditions", seed=int(rng.integers(0, 1_000_000)), noise=0.0)
            keep = ~joint_gap_mask(draws)
            for row in zip(
                draws["irradiance"][keep],
                draws["ambient_temperature"][keep],
                draws["tilt_angle"][keep],
                draws["cloud_cover"][keep],
            ):
                values.append(tuple(float(v) for v in row))
                if len(values) >= n:
                    break
        arr = np.asarray(values)
        irradiance, ambient, tilt, cloud = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    else:
        raise ValueError(f"unknown sampling_policy: {sampling_policy}")
    return irradiance, ambient, tilt, _clip(cloud, 0, 1)


def pv_latent_power(
    irradiance: np.ndarray,
    ambient_temperature: np.ndarray,
    tilt_angle: np.ndarray,
    cloud_cover: np.ndarray,
    panel_temperature: np.ndarray,
) -> np.ndarray:
    """Return the deterministic normalized PV power before measurement noise."""
    irr_norm = np.asarray(irradiance, dtype=float) / 1000.0
    irradiance_response = (1.0 - np.exp(-3.2 * irr_norm)) / (1.0 - np.exp(-3.2))
    tilt_factor = 0.70 + 0.30 * np.exp(-((np.asarray(tilt_angle) - 30.0) ** 2) / (2 * 17.0**2))
    cloud_factor = 1.0 - 0.58 * np.asarray(cloud_cover) ** 1.25
    temperature_efficiency = 1.0 - 0.0048 * (np.asarray(panel_temperature) - 25.0)
    temperature_efficiency = np.clip(temperature_efficiency, 0.70, 1.08)
    derating = 1.0 - 0.13 * sigmoid((np.asarray(panel_temperature) - 52.0) / 4.5) * sigmoid((np.asarray(irradiance) - 760.0) / 65.0)
    return np.clip(0.96 * irradiance_response * tilt_factor * cloud_factor * temperature_efficiency * derating, 0, 1)


def make_pv_observations(
    n: int,
    sampling_policy: str = "normal_conditions",
    seed: int = SEED,
    noise: float = 0.025,
    name: str | None = None,
) -> dict[str, Any]:
    """Generate synthetic PV observations for Workshop 3."""
    rng = set_seed(seed)
    irradiance, ambient, tilt, cloud = _sample_visible(n, sampling_policy, rng)
    panel_temp = ambient + 0.030 * irradiance * (1.0 - 0.45 * cloud) + rng.normal(0.0, 2.2, int(n))
    panel_temp = _clip(panel_temp, 10.0, 75.0)
    clean = pv_latent_power(irradiance, ambient, tilt, cloud, panel_temp)
    heteroskedastic = float(noise) * (0.75 + 0.8 * cloud)
    y = np.clip(clean + rng.normal(0.0, heteroskedastic, int(n)), 0, 1)
    return {
        "irradiance": irradiance.astype(float),
        "ambient_temperature": ambient.astype(float),
        "tilt_angle": tilt.astype(float),
        "cloud_cover": cloud.astype(float),
        "panel_temperature": panel_temp.astype(float),
        CANDIDATE_SENSOR_C: rng.normal(0.0, 1.0, int(n)).astype(float),
        "noisy_measurement": rng.normal(0.0, 1.0, int(n)).astype(float),
        TARGET: y.astype(float),
        "clean_power": clean.astype(float),
        "measurement_quality": np.zeros(int(n), dtype=int),
        "name": name or sampling_policy,
        "metadata": {"sampling_policy": sampling_policy, "seed": int(seed), "noise": float(noise)},
    }


def make_capacity_observations(n: int, seed: int = SEED, name: str | None = None) -> dict[str, Any]:
    """Generate a visible-input nonlinear scenario for capacity diagnostics."""
    data = make_pv_observations(n, "broad_operating_conditions", seed=seed, noise=0.012, name=name or "capacity data")
    rng = set_seed(seed + 17)
    data["cloud_cover"] = np.clip(rng.normal(0.18, 0.045, int(n)), 0, 1)
    data["panel_temperature"] = np.clip(
        data["ambient_temperature"] + 0.026 * data["irradiance"] * (1.0 - 0.35 * data["cloud_cover"]) + rng.normal(0, 1.2, int(n)),
        10,
        75,
    )
    base = pv_latent_power(
        data["irradiance"],
        data["ambient_temperature"],
        data["tilt_angle"],
        data["cloud_cover"],
        data["panel_temperature"],
    )
    visible_bump = 0.28 * np.exp(-((data["irradiance"] - 640.0) ** 2) / (2 * 42.0**2)) * np.exp(
        -((data["tilt_angle"] - 47.0) ** 2) / (2 * 3.0**2)
    )
    cool_morning_bump = 0.22 * np.exp(-((data["irradiance"] - 360.0) ** 2) / (2 * 38.0**2)) * np.exp(
        -((data["ambient_temperature"] - 16.0) ** 2) / (2 * 2.6**2)
    )
    visible_dip = 0.26 * np.exp(-((data["irradiance"] - 820.0) ** 2) / (2 * 42.0**2)) * np.exp(
        -((data["ambient_temperature"] - 36.0) ** 2) / (2 * 2.8**2)
    )
    smooth_ripple = 0.11 * np.sin((data["irradiance"] - 220.0) / 42.0) * np.exp(
        -((data["tilt_angle"] - 30.0) ** 2) / (2 * 12.0**2)
    )
    tilt_ripple = 0.08 * np.sin((data["tilt_angle"] - 8.0) / 4.5) * np.exp(
        -((data["irradiance"] - 700.0) ** 2) / (2 * 180.0**2)
    )
    clean = np.clip(base + visible_bump + cool_morning_bump - visible_dip + smooth_ripple + tilt_ripple, 0, 1)
    data["clean_power"] = clean
    data[TARGET] = np.clip(clean + rng.normal(0.0, 0.014, int(n)), 0, 1)
    data["metadata"] = {**data.get("metadata", {}), "scenario": "capacity_visible_nonlinearity"}
    return data


def make_pv_observations_near(
    centers: dict[str, Any],
    n: int,
    seed: int = SEED,
    name: str = "targeted new observations",
) -> dict[str, Any]:
    """Generate new observations near supplied visible-input center rows."""
    rng = set_seed(seed)
    center_count = len(target_vector(centers))
    chosen = rng.choice(center_count, size=int(n), replace=True)
    irradiance = np.clip(centers["irradiance"][chosen] + rng.normal(0, 45, int(n)), 0, 1000)
    ambient = np.clip(centers["ambient_temperature"][chosen] + rng.normal(0, 2.5, int(n)), 5, 45)
    tilt = np.clip(centers["tilt_angle"][chosen] + rng.normal(0, 3.5, int(n)), 0, 60)
    cloud = np.clip(centers.get("cloud_cover", np.full(center_count, 0.25))[chosen] + rng.normal(0, 0.10, int(n)), 0, 1)
    panel = np.clip(ambient + 0.030 * irradiance * (1.0 - 0.45 * cloud) + rng.normal(0, 2.0, int(n)), 10, 75)
    clean = pv_latent_power(irradiance, ambient, tilt, cloud, panel)
    y = np.clip(clean + rng.normal(0, 0.025 * (0.75 + 0.8 * cloud), int(n)), 0, 1)
    return {
        "irradiance": irradiance.astype(float),
        "ambient_temperature": ambient.astype(float),
        "tilt_angle": tilt.astype(float),
        "cloud_cover": cloud.astype(float),
        "panel_temperature": panel.astype(float),
        CANDIDATE_SENSOR_C: rng.normal(0.0, 1.0, int(n)).astype(float),
        "noisy_measurement": rng.normal(0.0, 1.0, int(n)).astype(float),
        TARGET: y.astype(float),
        "clean_power": clean.astype(float),
        "measurement_quality": np.zeros(int(n), dtype=int),
        "name": name,
        "metadata": {"sampling_policy": "near_centers", "seed": int(seed)},
    }


def _visible_input_center_data(input_values: Mapping[str, Sequence[float]] | Sequence[Mapping[str, float] | Sequence[float]]) -> dict[str, Any]:
    """Return visible-input center rows from user-specified values."""
    if isinstance(input_values, Mapping):
        missing = [col for col in VISIBLE_INPUTS if col not in input_values]
        if missing:
            raise ValueError(f"custom input values must include: {', '.join(missing)}")
        columns = {col: np.asarray(input_values[col], dtype=float) for col in VISIBLE_INPUTS}
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise ValueError("custom input value columns must have the same length")
    else:
        rows = list(input_values)
        if not rows:
            raise ValueError("custom input values must contain at least one row")
        if isinstance(rows[0], Mapping):
            columns = {
                col: np.asarray([row[col] for row in rows], dtype=float)
                for col in VISIBLE_INPUTS
            }
        else:
            arr = np.asarray(rows, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != len(VISIBLE_INPUTS):
                raise ValueError(
                    "custom input value rows must be mappings or "
                    "(irradiance, ambient_temperature, tilt_angle) triples"
                )
            columns = {col: arr[:, i] for i, col in enumerate(VISIBLE_INPUTS)}

    ranges = {
        "irradiance": (0.0, 1000.0),
        "ambient_temperature": (5.0, 45.0),
        "tilt_angle": (0.0, 60.0),
    }
    for col, values in columns.items():
        low, high = ranges[col]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"custom input values for {col} must be finite")
        if np.any((values < low) | (values > high)):
            raise ValueError(f"custom input values for {col} must be in the range {low:g}-{high:g}")

    n = len(columns[VISIBLE_INPUTS[0]])
    out: dict[str, Any] = {col: values.astype(float) for col, values in columns.items()}
    out[TARGET] = np.zeros(n, dtype=float)
    out["name"] = "custom input centers"
    out["metadata"] = {"sampling_policy": "custom_input_centers"}
    return out


def make_pv_observations_at_inputs(
    input_values: Mapping[str, Sequence[float]] | Sequence[Mapping[str, float] | Sequence[float]],
    n: int | None = None,
    seed: int = SEED,
    noise: float = 0.025,
    name: str = "custom input observations",
) -> dict[str, Any]:
    """Generate new observations at user-specified visible input values."""
    centers = _visible_input_center_data(input_values)
    rng = set_seed(seed)
    center_count = len(centers[TARGET])
    n = center_count if n is None else int(n)
    if n <= 0:
        raise ValueError("n must be positive for custom input observations")
    chosen = np.arange(center_count) if n == center_count else rng.choice(center_count, size=n, replace=True)
    irradiance = centers["irradiance"][chosen]
    ambient = centers["ambient_temperature"][chosen]
    tilt = centers["tilt_angle"][chosen]
    cloud = rng.beta(1.4, 2.4, n)
    panel_temp = _clip(ambient + 0.030 * irradiance * (1.0 - 0.45 * cloud) + rng.normal(0.0, 2.2, n), 10.0, 75.0)
    clean = pv_latent_power(irradiance, ambient, tilt, cloud, panel_temp)
    heteroskedastic = float(noise) * (0.75 + 0.8 * cloud)
    y = np.clip(clean + rng.normal(0.0, heteroskedastic, n), 0, 1)
    return {
        "irradiance": irradiance.astype(float),
        "ambient_temperature": ambient.astype(float),
        "tilt_angle": tilt.astype(float),
        "cloud_cover": cloud.astype(float),
        "panel_temperature": panel_temp.astype(float),
        CANDIDATE_SENSOR_C: rng.normal(0.0, 1.0, n).astype(float),
        "noisy_measurement": rng.normal(0.0, 1.0, n).astype(float),
        TARGET: y.astype(float),
        "clean_power": clean.astype(float),
        "measurement_quality": np.zeros(n, dtype=int),
        "name": name,
        "metadata": {"sampling_policy": "custom_input_values", "seed": int(seed), "noise": float(noise)},
    }


def add_noisy_labels(data: dict[str, Any], n_outliers: int = 10, seed: int = SEED, magnitude: float = 0.35) -> dict[str, Any]:
    """Return a copy with a few corrupted labels for robust-objective experiments."""
    rng = set_seed(seed)
    out = clone_data(data, name=f"{data.get('name', 'train')} with noisy labels")
    n = len(out[TARGET])
    idx = rng.choice(n, size=min(int(n_outliers), n), replace=False)
    direction = rng.choice([-1.0, 1.0], size=len(idx))
    out["original_power"] = out[TARGET].copy()
    out[TARGET][idx] = np.clip(out[TARGET][idx] + direction * float(magnitude), 0, 1)
    out["measurement_quality"] = np.zeros(n, dtype=int)
    out["measurement_quality"][idx] = 1
    out["metadata"] = {**out.get("metadata", {}), "noisy_label_count": int(len(idx))}
    return out


def make_workshop3_bundle(scenario: str = "baseline", seed: int = SEED) -> dict[str, Any]:
    """Create a deterministic data bundle for a Workshop 3 scenario."""
    criterion = SuccessCriterion()
    if scenario in {"baseline", "optimizer_scaling_initialization", "optimizer_schedule"}:
        train = make_pv_observations(600, "normal_conditions", seed, name="train")
        validation = make_pv_observations(200, "deployment_conditions", seed + 1, name="validation")
        test = make_pv_observations(200, "deployment_conditions", seed + 2, name="final test")
    elif scenario in {"hypothesis_capacity", "hypothesis_activation"}:
        train = make_capacity_observations(600, seed, name="train")
        validation = make_capacity_observations(200, seed + 1, name="validation")
        test = make_capacity_observations(200, seed + 2, name="final test")
    elif scenario == "data_missing_combinations":
        train = make_pv_observations(600, "missing_joint_region", seed, name="train")
        validation = make_pv_observations(200, "broad_operating_conditions", seed + 1, name="validation")
        test = make_pv_observations(200, "broad_operating_conditions", seed + 2, name="final test")
    elif scenario == "data_deployment_mismatch":
        train = make_pv_observations(600, "normal_conditions", seed, name="train")
        validation = make_pv_observations(200, "deployment_conditions", seed + 1, name="validation")
        test = make_pv_observations(200, "deployment_conditions", seed + 2, name="final test")
    elif scenario == "hypothesis_new_measurements":
        train = make_pv_observations(600, "deployment_conditions", seed, name="train")
        validation = make_pv_observations(200, "deployment_conditions", seed + 1, name="validation")
        test = make_pv_observations(200, "deployment_conditions", seed + 2, name="final test")
    elif scenario == "optimizer_noisy_labels":
        clean_train = make_pv_observations(600, "deployment_conditions", seed, name="train")
        train = add_noisy_labels(clean_train, n_outliers=12, seed=seed + 91, magnitude=0.42)
        validation = make_pv_observations(200, "deployment_conditions", seed + 1, name="validation")
        test = make_pv_observations(200, "deployment_conditions", seed + 2, name="final test")
    elif scenario == "data_split_probe":
        pool = make_pv_observations(1000, "broad_operating_conditions", seed, name="split pool")
        train, validation, test = make_split_from_pool(pool, strategy="random_split", seed=seed)
    else:
        raise ValueError(f"unknown Workshop 3 scenario: {scenario}")
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "visible_inputs": list(VISIBLE_INPUTS),
        "revealed_inputs": list(REVEALED_INPUTS),
        "target": TARGET,
        "criterion": criterion,
        "metadata": {"scenario": scenario, "seed": int(seed)},
    }


def make_challenge_bundle(
    seed: int = SEED,
    data_policy: str | None = None,
    added_n: int = 0,
    criterion: SuccessCriterion | None = None,
    target_point_indices: Sequence[int] | None = None,
    target_split: str = "validation",
) -> dict[str, Any]:
    """Create the fixed validation-first bundle for the final experiment lab."""
    base = make_workshop3_bundle("baseline", seed=seed)
    policy = "target_selected_points" if target_point_indices is not None and data_policy in {None, "", "none"} else str(data_policy or "none")
    added_n = int(added_n)
    if policy == "none":
        if added_n != 0:
            raise ValueError("added_n must be 0 when data_policy is 'none'")
        train = clone_data(base["train"], name="challenge train")
    elif policy == "target_selected_points":
        if added_n <= 0:
            raise ValueError("added_n must be positive when targeting selected points")
        if target_point_indices is None:
            raise ValueError("target_point_indices must be provided when data_policy is 'target_selected_points'")
        if target_split not in {"train", "validation", "test"}:
            raise ValueError("target_split must be one of: train, validation, test")
        point_indices = np.asarray(target_point_indices, dtype=int)
        if len(point_indices) == 0:
            raise ValueError("target_point_indices must contain at least one row index")
        split_data = base[target_split]
        n_rows = len(target_vector(split_data))
        if np.any(point_indices < 0) or np.any(point_indices >= n_rows):
            raise ValueError(f"target_point_indices must be between 0 and {n_rows - 1}")
        centers = clone_data(split_data, point_indices, name=f"selected {target_split} centers")
        extra = make_pv_observations_near(centers, added_n, seed=seed + 101, name="added selected-point data")
        train = concat_data([base["train"], extra], name="challenge train + selected points")
    elif policy in {"target_low_support", "target_high_error"}:
        if added_n <= 0:
            raise ValueError(f"added_n must be positive when data_policy is {policy!r}")
        extra = _targeted_extra_from_validation(base, policy, added_n, seed=seed + 101)
        train = concat_data([base["train"], extra], name=f"challenge train + {policy}")
    else:
        if added_n <= 0:
            raise ValueError("added_n must be positive when data_policy adds rows")
        extra = _extra_data_for_option(policy, added_n, seed + 101)
        train = concat_data([base["train"], extra], name=f"challenge train + {policy}")

    return {
        "train": train,
        "validation": clone_data(base["validation"], name="challenge validation"),
        "test": clone_data(base["test"], name="challenge final test"),
        "visible_inputs": list(VISIBLE_INPUTS),
        "revealed_inputs": list(REVEALED_INPUTS),
        "target": TARGET,
        "criterion": criterion or challenge_success_criterion(),
        "metadata": {
            "scenario": "experiment_lab_challenge",
            "seed": int(seed),
            "data_policy": policy,
            "added_n": added_n,
            "target_split": target_split if policy == "target_selected_points" else None,
            "target_point_indices": [int(idx) for idx in np.asarray(target_point_indices, dtype=int)] if target_point_indices is not None else [],
        },
    }


def feature_matrix(data: dict[str, Any], input_columns: Sequence[str]) -> np.ndarray:
    """Return a 2D feature matrix for the requested columns."""
    return np.column_stack([np.asarray(data[col], dtype=float) for col in input_columns])


def target_vector(data: dict[str, Any]) -> np.ndarray:
    """Return target values as a 1D array."""
    return np.asarray(data[TARGET], dtype=float)


def describe_visible_inputs(data: dict[str, Any]) -> list[list[Any]]:
    """Return schema-style rows for visible inputs."""
    units = {
        "irradiance": "W/m2",
        "ambient_temperature": "C",
        "tilt_angle": "degrees",
    }
    expected = {
        "irradiance": "0-1000",
        "ambient_temperature": "5-45",
        "tilt_angle": "0-60",
    }
    rows = []
    for col in VISIBLE_INPUTS:
        values = np.asarray(data[col], dtype=float)
        rows.append([col, units[col], expected[col], float(np.min(values)), float(np.median(values)), float(np.max(values))])
    return rows


def describe_inputs(data: dict[str, Any]) -> list[list[Any]]:
    """Return schema-style rows for the baseline inputs."""
    return describe_visible_inputs(data)


def representative_rows(data: dict[str, Any], k: int = 2) -> list[dict[str, float]]:
    """Return representative low, typical, and high target examples."""
    y = target_vector(data)
    order = np.argsort(y)
    mid = np.argsort(np.abs(y - np.median(y)))
    chosen = np.concatenate([order[:k], mid[:k], order[-k:]])
    rows = []
    for idx in chosen:
        rows.append({col: float(data[col][idx]) for col in [*VISIBLE_INPUTS, TARGET]})
    return rows


def binned_target_summary(data: dict[str, Any], column: str, bins: int = 6) -> list[list[Any]]:
    """Return target mean/spread by bins of one visible input."""
    x = np.asarray(data[column], dtype=float)
    y = target_vector(data)
    edges = np.linspace(float(np.min(x)), float(np.max(x)), int(bins) + 1)
    rows: list[list[Any]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x >= left) & (x < right if right < edges[-1] else x <= right)
        if np.any(mask):
            rows.append([f"{left:.1f}-{right:.1f}", int(mask.sum()), float(np.mean(y[mask])), float(np.std(y[mask]))])
        else:
            rows.append([f"{left:.1f}-{right:.1f}", 0, np.nan, np.nan])
    return rows


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)), "MAE": mae, "Bias": float(np.mean(residual))}


def visible_regime_labels(data: dict[str, Any]) -> np.ndarray:
    """Return compact visible regime labels for splitting and diagnostics."""
    irr = np.digitize(data["irradiance"], [300, 700])
    temp = np.digitize(data["ambient_temperature"], [20, 30])
    tilt = np.digitize(data["tilt_angle"], [15, 45])
    return irr * 9 + temp * 3 + tilt


def slice_masks(data: dict[str, Any], criterion: SuccessCriterion | None = None) -> dict[str, np.ndarray]:
    """Return simple visible-input slice masks."""
    return {
        "key operating range": key_range_mask(data, criterion),
        "high irradiance": data["irradiance"] >= 700,
        "hot ambient": data["ambient_temperature"] >= 30,
        "low tilt": data["tilt_angle"] < 15,
        "central tilt": (data["tilt_angle"] >= 15) & (data["tilt_angle"] <= 45),
        "high tilt": data["tilt_angle"] > 45,
    }


def slice_metric_rows(data: dict[str, Any], y_pred: np.ndarray, criterion: SuccessCriterion | None = None) -> list[list[Any]]:
    """Return visible-slice metric rows."""
    rows = []
    for name, mask in slice_masks(data, criterion).items():
        count = int(np.sum(mask))
        if count == 0:
            rows.append([name, 0, np.nan, np.nan, np.nan])
        else:
            metrics = regression_metrics(target_vector(data)[mask], np.asarray(y_pred)[mask])
            rows.append([name, count, metrics["RMSE"], metrics["MAE"], metrics["Bias"]])
    return rows


class PVDataset(Dataset):
    """Simple in-memory PyTorch dataset for PV regression."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        require_torch()
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        return self.X[idx], self.y[idx]


if nn is not None:

    class SolarPVMLP(nn.Module):
        """Configurable MLP used in Workshop 3."""

        def __init__(
            self,
            input_dim: int = 3,
            hidden_width: int = 16,
            depth: int = 2,
            activation: str = "relu",
            initialization: str = "he",
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_width = int(hidden_width)
            self.depth = int(depth)
            self.activation_name = activation
            activation_layer = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "silu": nn.SiLU,
                "gelu": nn.GELU,
            }.get(activation)
            if activation_layer is None:
                raise ValueError(f"unknown activation: {activation}")
            layers: list[nn.Module] = []
            last = self.input_dim
            for _ in range(self.depth):
                layers.append(nn.Linear(last, self.hidden_width))
                layers.append(activation_layer())
                last = self.hidden_width
            layers.append(nn.Linear(last, 1))
            self.network = nn.Sequential(*layers)
            self.reset_parameters(initialization)

        def reset_parameters(self, initialization: str = "he") -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    if initialization == "he":
                        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    elif initialization == "xavier":
                        nn.init.xavier_normal_(module.weight)
                    elif initialization == "small":
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    elif initialization == "large":
                        nn.init.normal_(module.weight, mean=0.0, std=1.2)
                    else:
                        raise ValueError(f"unknown initialization: {initialization}")
                    nn.init.zeros_(module.bias)

        def forward(self, x: Any) -> Any:
            return self.network(x)

else:

    class SolarPVMLP:  # type: ignore[no-redef]
        """Placeholder class that raises a dependency error if torch is missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def fit_scaler(X: np.ndarray, scaling: str) -> dict[str, np.ndarray | str]:
    """Fit a train-only input scaler."""
    if scaling == "standard":
        center = X.mean(axis=0, keepdims=True)
        scale = np.maximum(X.std(axis=0, keepdims=True), 1e-6)
    elif scaling == "minmax":
        center = X.min(axis=0, keepdims=True)
        scale = np.maximum(X.max(axis=0, keepdims=True) - center, 1e-6)
    elif scaling == "raw":
        center = np.zeros((1, X.shape[1]))
        scale = np.ones((1, X.shape[1]))
    else:
        raise ValueError(f"unknown scaling: {scaling}")
    return {"scaling": scaling, "center": center, "scale": scale}


def transform_with_scaler(X: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    """Transform features with a fitted scaler."""
    return (X - scaler["center"]) / scaler["scale"]


def loss_function(name: str, huber_delta: float = 0.08) -> Any:
    """Return a PyTorch loss module by name."""
    require_torch()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=float(huber_delta))
    raise ValueError(f"unknown loss: {name}")


def predict_run(run: dict[str, Any], data: dict[str, Any]) -> np.ndarray:
    """Predict with a trained run on a PV data dictionary."""
    require_torch()
    model = run["model"]
    model.eval()
    X = feature_matrix(data, run["config"]["input_columns"])
    Xn = transform_with_scaler(X, run["scaler"])
    with torch.no_grad():
        pred = model(torch.as_tensor(Xn, dtype=torch.float32)).cpu().numpy().ravel()
    return np.clip(pred, 0, 1)


def _epoch_metrics(run: dict[str, Any], bundle: dict[str, Any]) -> dict[str, float]:
    val_pred = predict_run(run, bundle["validation"])
    train_pred = predict_run(run, bundle["train"])
    return {
        "train_RMSE": regression_metrics(target_vector(bundle["train"]), train_pred)["RMSE"],
        "validation_RMSE": regression_metrics(target_vector(bundle["validation"]), val_pred)["RMSE"],
    }


def train_solar_mlp(
    bundle: dict[str, Any],
    input_columns: Sequence[str] | None = None,
    hidden_width: int = 16,
    depth: int = 2,
    activation: str = "relu",
    scaling: str = "standard",
    initialization: str = "he",
    regularity_strength: str = "none",
    weight_decay: float | None = None,
    loss: str = "mse",
    huber_delta: float = 0.08,
    learning_rate: float = 0.035,
    batch_size: int = 32,
    epochs: int = 220,
    optimizer: str = "sgd",
    momentum: float = 0.0,
    seed: int = SEED,
    selection_metric: str = "rmse",
    name: str | None = None,
) -> dict[str, Any]:
    """Train a SolarPVMLP using a small set of workshop optimizer choices."""
    require_torch()
    torch.manual_seed(int(seed))
    input_columns = list(input_columns or bundle["visible_inputs"])
    train = bundle["train"]
    validation = bundle["validation"]
    X_train = feature_matrix(train, input_columns)
    y_train = target_vector(train)
    X_val = feature_matrix(validation, input_columns)
    y_val = target_vector(validation)
    scaler = fit_scaler(X_train, scaling)
    Xn = transform_with_scaler(X_train, scaler)
    Xv = transform_with_scaler(X_val, scaler)
    model = SolarPVMLP(
        input_dim=len(input_columns),
        hidden_width=hidden_width,
        depth=depth,
        activation=activation,
        initialization=initialization,
    )
    criterion = loss_function(loss, huber_delta=huber_delta)
    regularity_map = {"none": 0.0, "weak": 1e-5, "medium": 1e-4, "strong": 1e-3}
    if weight_decay is None:
        if regularity_strength not in regularity_map:
            raise ValueError(f"unknown regularity_strength: {regularity_strength}")
        weight_decay = regularity_map[regularity_strength]
    optimizer_name = str(optimizer)
    momentum = float(momentum)
    if optimizer_name == "sgd":
        if momentum != 0.0:
            raise ValueError("optimizer='sgd' uses momentum=0.0; use optimizer='sgd_momentum' for momentum")
        torch_optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate), momentum=0.0, weight_decay=float(weight_decay))
    elif optimizer_name == "sgd_momentum":
        torch_optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate), momentum=momentum, weight_decay=float(weight_decay))
    elif optimizer_name == "adam":
        torch_optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name}")
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 123)
    loader = DataLoader(PVDataset(Xn, y_train), batch_size=min(int(batch_size), len(Xn)), shuffle=True, generator=generator)

    history: list[dict[str, float]] = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = np.inf
    best_epoch = 0
    final_epoch_metrics: dict[str, float] = {}
    diverged = False
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for xb, yb in loader:
            torch_optimizer.zero_grad()
            loss_value = criterion(model(xb), yb)
            if not torch.isfinite(loss_value):
                diverged = True
                break
            loss_value.backward()
            torch_optimizer.step()
        if diverged:
            break
        model.eval()
        with torch.no_grad():
            train_pred = model(torch.as_tensor(Xn, dtype=torch.float32)).cpu().numpy().ravel()
            val_pred = model(torch.as_tensor(Xv, dtype=torch.float32)).cpu().numpy().ravel()
        if not (np.all(np.isfinite(train_pred)) and np.all(np.isfinite(val_pred))):
            diverged = True
            break
        train_metrics = regression_metrics(y_train, np.clip(train_pred, 0, 1))
        val_metrics = regression_metrics(y_val, np.clip(val_pred, 0, 1))
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["MSE"],
            "validation_loss": val_metrics["MSE"],
            "train_RMSE": train_metrics["RMSE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_MAE": val_metrics["MAE"],
        }
        history.append(row)
        score = row["validation_MAE"] if selection_metric == "mae" else row["validation_RMSE"]
        if np.isfinite(score) and score < best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        final_epoch_metrics = row
    model.load_state_dict(best_state)
    config = {
        "input_columns": input_columns,
        "hidden_width": int(hidden_width),
        "depth": int(depth),
        "activation": activation,
        "scaling": scaling,
        "initialization": initialization,
        "regularity_strength": regularity_strength,
        "weight_decay": float(weight_decay),
        "loss": loss,
        "huber_delta": float(huber_delta),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "optimizer": optimizer_name,
        "momentum": momentum,
        "seed": int(seed),
        "selection_metric": selection_metric,
    }
    run = {
        "model": model,
        "history": history,
        "config": config,
        "scaler": scaler,
        "best_validation_epoch": int(best_epoch),
        "final_epoch_metrics": final_epoch_metrics,
        "diverged": bool(diverged or best_epoch == 0),
        "name": name or f"MLP width={hidden_width} depth={depth}",
    }
    run["best_metrics"] = _epoch_metrics(run, bundle)
    return run


def make_mean_baseline(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return a simple mean baseline run-like object."""
    return {
        "kind": "mean",
        "value": float(np.mean(target_vector(bundle["train"]))),
        "config": {"input_columns": []},
        "name": "Mean baseline",
    }


def predict_any(run: dict[str, Any], data: dict[str, Any]) -> np.ndarray:
    """Predict with either an MLP run or a mean baseline."""
    if run.get("kind") == "mean":
        return np.full(len(target_vector(data)), run["value"], dtype=float)
    return predict_run(run, data)


def evaluate_model_report(
    run: dict[str, Any],
    bundle: dict[str, Any],
    criterion: SuccessCriterion | None = None,
    include_test: bool = False,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Return a compact report for shared evaluation cells."""
    criterion = criterion or bundle["criterion"]
    split_names = ["train", "validation"] + (["test"] if include_test else [])
    metric_rows = []
    predictions = {}
    for split_name in split_names:
        data = bundle[split_name]
        pred = predict_any(run, data)
        predictions[split_name] = pred
        metrics = regression_metrics(target_vector(data), pred)
        key_mask = key_range_mask(data, criterion)
        key_metrics = regression_metrics(target_vector(data)[key_mask], pred[key_mask]) if np.any(key_mask) else {"RMSE": np.nan, "MAE": np.nan}
        metric_rows.append([split_name, len(target_vector(data)), metrics["RMSE"], metrics["MAE"], key_metrics["RMSE"]])
    validation_pred = predictions["validation"]
    validation = bundle["validation"]
    validation_metrics = regression_metrics(target_vector(validation), validation_pred)
    key_mask = key_range_mask(validation, criterion)
    key_rmse = np.nan
    if np.any(key_mask):
        key_rmse = regression_metrics(target_vector(validation)[key_mask], validation_pred[key_mask])["RMSE"]
    passes = bool(validation_metrics["RMSE"] <= criterion.overall_rmse and np.isfinite(key_rmse) and key_rmse <= criterion.key_range_rmse)
    report = {
        "run_name": run.get("name", run.get("kind", "model")),
        "metric_rows": metric_rows,
        "slice_rows": slice_metric_rows(validation, validation_pred, criterion),
        "criterion": {
            "overall_validation_rmse": validation_metrics["RMSE"],
            "overall_threshold": criterion.overall_rmse,
            "key_range_validation_rmse": key_rmse,
            "key_range_threshold": criterion.key_range_rmse,
            "passes": passes,
        },
        "predictions": predictions,
        "include_test": include_test,
    }
    if show_advanced:
        residual = validation_pred - target_vector(validation)
        report["advanced"] = {
            "residual_quantiles": np.quantile(residual, [0, 0.1, 0.5, 0.9, 1.0]).tolist(),
            "key_range_count": int(np.sum(key_mask)),
        }
    return report


def plot_training_curves(run: dict[str, Any]) -> tuple[Any, Any]:
    """Plot train and validation loss/RMSE curves."""
    history = run["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(epochs, [row["train_loss"] for row in history], color=COLORS["train"], label="train MSE")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], color=COLORS["validation"], label="validation MSE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_RMSE"] for row in history], color=COLORS["train"], label="train RMSE")
    axes[1].plot(epochs, [row["validation_RMSE"] for row in history], color=COLORS["validation"], label="validation RMSE")
    axes[1].axvline(run["best_validation_epoch"], color="#555555", ls="--", lw=1, label="best validation epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE")
    axes[1].legend()
    fig.tight_layout()
    return fig, axes


def plot_observed_vs_predicted(run: dict[str, Any], bundle: dict[str, Any], split: str = "validation") -> tuple[Any, Any]:
    """Plot observed vs predicted target values."""
    data = bundle[split]
    pred = predict_any(run, data)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(target_vector(data), pred, s=28, alpha=0.72, color=COLORS.get(split, COLORS["diagnostic"]), edgecolor="white", linewidth=0.3)
    ax.plot([0, 1], [0, 1], color="#333333", lw=1, ls="--")
    ax.set_xlabel("observed normalized power")
    ax.set_ylabel("predicted normalized power")
    ax.set_title(f"Observed vs predicted: {split}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def nearest_training_distance(bundle: dict[str, Any], split: str = "validation", input_columns: Sequence[str] | None = None) -> np.ndarray:
    """Compute nearest train distance in standardized visible-input space."""
    input_columns = input_columns or bundle["visible_inputs"]
    X_train = feature_matrix(bundle["train"], input_columns)
    X_eval = feature_matrix(bundle[split], input_columns)
    scaler = fit_scaler(X_train, "standard")
    train_z = transform_with_scaler(X_train, scaler)
    eval_z = transform_with_scaler(X_eval, scaler)
    distances = np.sqrt(np.sum((eval_z[:, None, :] - train_z[None, :, :]) ** 2, axis=2))
    return np.min(distances, axis=1)


def low_support_rows(bundle: dict[str, Any], run: dict[str, Any] | None = None, top_k: int = 8) -> list[list[Any]]:
    """Return validation examples with largest nearest-training distances."""
    validation = bundle["validation"]
    dist = nearest_training_distance(bundle)
    pred = predict_any(run, validation) if run is not None else np.full(len(dist), np.nan)
    residual = pred - target_vector(validation)
    order = np.argsort(dist)[-int(top_k) :][::-1]
    rows = []
    for idx in order:
        rows.append([
            int(idx),
            float(dist[idx]),
            float(validation["irradiance"][idx]),
            float(validation["ambient_temperature"][idx]),
            float(validation["tilt_angle"][idx]),
            float(residual[idx]) if np.isfinite(residual[idx]) else np.nan,
        ])
    return rows


def _allocate_random(indices: np.ndarray, rng: np.random.Generator, train_n: int = 600, val_n: int = 200, test_n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = np.asarray(indices).copy()
    rng.shuffle(shuffled)
    return shuffled[:train_n], shuffled[train_n : train_n + val_n], shuffled[train_n + val_n : train_n + val_n + test_n]


def make_split_from_pool(pool: dict[str, Any], strategy: str = "random_split", seed: int = SEED) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create train/validation/test splits from a pool using a named strategy."""
    rng = set_seed(seed)
    n = len(target_vector(pool))
    all_idx = np.arange(n)
    if strategy == "random_split":
        train_idx, val_idx, test_idx = _allocate_random(all_idx, rng)
    elif strategy in {"visible_regime_stratified_split", "input_regime_stratified_split"}:
        labels = visible_regime_labels(pool)
        train_parts, val_parts, test_parts = [], [], []
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            if len(idx) < 6:
                train_parts.append(idx)
                continue
            rng.shuffle(idx)
            n_label = len(idx)
            n_val = max(1, int(round(0.2 * n_label)))
            n_test = max(1, int(round(0.2 * n_label)))
            test_parts.append(idx[:n_test])
            val_parts.append(idx[n_test : n_test + n_val])
            train_parts.append(idx[n_test + n_val :])
        train_idx = np.concatenate(train_parts)
        val_idx = np.concatenate(val_parts)
        test_idx = np.concatenate(test_parts)
        if len(train_idx) > 600:
            train_idx = rng.choice(train_idx, 600, replace=False)
        if len(val_idx) > 200:
            val_idx = rng.choice(val_idx, 200, replace=False)
        if len(test_idx) > 200:
            test_idx = rng.choice(test_idx, 200, replace=False)
    elif strategy == "deployment_structured_split":
        score = (
            1.2 * (pool["irradiance"] / 1000.0)
            + 1.1 * ((pool["ambient_temperature"] - 5.0) / 40.0)
            + 0.35 * np.abs(pool["tilt_angle"] - 30.0) / 30.0
            + rng.normal(0, 0.02, n)
        )
        order = np.argsort(score)
        train_idx = order[:600]
        val_idx = order[600:800]
        test_idx = order[800:1000]
    elif strategy == "rare_regime_holdout_split":
        rare = np.flatnonzero(joint_gap_mask(pool) | key_range_mask(pool))
        common = np.setdiff1d(all_idx, rare)
        rng.shuffle(rare)
        rng.shuffle(common)
        val_idx = np.concatenate([rare[:100], common[:100]])
        test_idx = np.concatenate([rare[100:200], common[100:200]])
        used = np.concatenate([val_idx, test_idx])
        remaining = np.setdiff1d(all_idx, used)
        rng.shuffle(remaining)
        train_idx = remaining[:600]
    else:
        raise ValueError(f"unknown split strategy: {strategy}")
    return (
        clone_data(pool, np.sort(train_idx), "train"),
        clone_data(pool, np.sort(val_idx), "validation"),
        clone_data(pool, np.sort(test_idx), "final test"),
    )


def split_regime_summary(train: dict[str, Any], validation: dict[str, Any], test: dict[str, Any], criterion: SuccessCriterion | None = None) -> list[list[Any]]:
    """Return visible regime counts for split diagnostics."""
    criterion = criterion or SuccessCriterion()
    rows = []
    for split_name, data in [("train", train), ("validation", validation), ("test", test)]:
        rows.append([
            split_name,
            len(target_vector(data)),
            int(np.sum(key_range_mask(data, criterion))),
            float(np.mean(data["irradiance"])),
            float(np.mean(data["ambient_temperature"])),
            float(np.mean(data["tilt_angle"])),
        ])
    return rows


def baseline_config(**overrides: Any) -> dict[str, Any]:
    """Return the fixed baseline training config with optional overrides."""
    config = {
        "hidden_width": 16,
        "depth": 2,
        "activation": "relu",
        "scaling": "standard",
        "initialization": "he",
        "regularity_strength": "none",
        "loss": "mse",
        "learning_rate": 0.035,
        "batch_size": 32,
        "epochs": 220,
        "optimizer": "sgd",
        "momentum": 0.0,
        "seed": 7,
        "selection_metric": "rmse",
    }
    config.update(overrides)
    return config


def challenge_config(**overrides: Any) -> dict[str, Any]:
    """Return default experiment-lab config with optional overrides."""
    config = baseline_config()
    config.update(overrides)
    return config


def train_with_config(bundle: dict[str, Any], config: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """Train using a config dictionary."""
    config = dict(config)
    input_columns = config.pop("input_columns", bundle["visible_inputs"])
    return train_solar_mlp(bundle, input_columns=input_columns, name=name, **config)


def challenge_option_rows() -> list[list[str]]:
    """Return the option table shown in the experiment-lab notebook."""
    return [
        ["Data policy", "data_policy", "none; target_selected_points; target_low_support; target_high_error; uniform_sampling; stratified_regime; more_normal_conditions; normal_condition_mix; deployment_matched_mix; input_regime_stratified_mix; rare_regime_oversampled; temperature_irradiance_balanced_mix"],
        ["Selected points", "target_point_indices", "validation row IDs to collect near when data_policy='target_selected_points'"],
        ["Added rows", "added_n", "0; 100; 200; 300; 500"],
        ["Inputs", "input_columns", "visible inputs; add cloud_cover; add panel_temperature; add both; add candidate_sensor_c"],
        ["Width/depth", "hidden_width, depth", "8-128 width; 1-4 hidden layers"],
        ["Activation", "activation", "relu; tanh; silu; gelu"],
        ["Regularity", "regularity_strength", "none; weak; medium; strong"],
        ["Scaling", "scaling", "standard; minmax; raw"],
        ["Initialization", "initialization", "he; xavier; small; large"],
        ["Optimizer", "optimizer, momentum", "sgd; sgd_momentum; adam"],
        ["Schedule", "learning_rate, batch_size, epochs", "learning rate; minibatch size; training length"],
        ["Objective", "loss, huber_delta", "mse; mae; huber"],
        ["Selection", "selection_metric", "rmse; mae"],
    ]


def challenge_candidate_point_rows(bundle: dict[str, Any], run: dict[str, Any] | None = None, top_k: int = 10) -> list[list[Any]]:
    """Return validation rows that are useful anchors for targeted data collection."""
    validation = bundle["validation"]
    distances = nearest_training_distance(bundle)
    abs_residual = np.full(len(distances), np.nan)
    if run is not None:
        abs_residual = np.abs(predict_any(run, validation) - target_vector(validation))
        distance_scale = np.std(distances) if np.std(distances) > 1e-9 else 1.0
        residual_scale = np.std(abs_residual) if np.std(abs_residual) > 1e-9 else 1.0
        score = (distances - np.mean(distances)) / distance_scale + (abs_residual - np.mean(abs_residual)) / residual_scale
    else:
        score = distances
    order = np.argsort(score)[-int(top_k) :][::-1]
    rows = []
    for idx in order:
        rows.append([
            int(idx),
            float(distances[idx]),
            float(abs_residual[idx]) if np.isfinite(abs_residual[idx]) else np.nan,
            float(validation["irradiance"][idx]),
            float(validation["ambient_temperature"][idx]),
            float(validation["tilt_angle"][idx]),
            float(target_vector(validation)[idx]),
        ])
    return rows


def _challenge_summary_row(name: str, bundle: dict[str, Any], run: dict[str, Any], report: dict[str, Any]) -> list[Any]:
    config = run["config"]
    data_policy = bundle.get("metadata", {}).get("data_policy", "unknown")
    added_n = bundle.get("metadata", {}).get("added_n", 0)
    row = [
        name,
        data_policy,
        added_n,
        len(config["input_columns"]),
        config["hidden_width"],
        config["depth"],
        config["activation"],
        config["optimizer"],
        config["loss"],
        report["criterion"]["overall_validation_rmse"],
        report["criterion"]["key_range_validation_rmse"],
        report["criterion"]["passes"],
        run["best_validation_epoch"],
    ]
    if report.get("include_test"):
        test_rows = [metric_row for metric_row in report["metric_rows"] if metric_row[0] == "test"]
        row.append(test_rows[0][2] if test_rows else np.nan)
    return row


def run_challenge_experiment(
    bundle: dict[str, Any],
    config: dict[str, Any],
    include_test: bool = False,
    name: str | None = None,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Train one experiment-lab run and return validation-first outputs."""
    config = challenge_config(**dict(config))
    run_name = name or "challenge experiment"
    run = train_with_config(bundle, config, name=run_name)
    report = evaluate_model_report(run, bundle, include_test=include_test, show_advanced=show_advanced)
    return {
        "bundle": bundle,
        "run": run,
        "report": report,
        "row": _challenge_summary_row(run_name, bundle, run, report),
    }


def compare_split_strategies(
    split_options: Sequence[str] | None = None,
    seed: int = SEED,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Train/evaluate the baseline under several split designs."""
    split_options = list(split_options or ["random_split", "input_regime_stratified_split", "deployment_structured_split", "rare_regime_holdout_split"])
    pool = make_pv_observations(1000, "broad_operating_conditions", seed, name="split pool")
    rows = []
    reports = {}
    summaries = {}
    for option in split_options:
        train, validation, test = make_split_from_pool(pool, option, seed)
        bundle = {
            "train": train,
            "validation": validation,
            "test": test,
            "visible_inputs": list(VISIBLE_INPUTS),
            "revealed_inputs": list(REVEALED_INPUTS),
            "target": TARGET,
            "criterion": SuccessCriterion(),
            "metadata": {"scenario": "data_split_probe", "split_strategy": option},
        }
        run = train_with_config(bundle, baseline_config(seed=seed + 20, epochs=160), name=option)
        report = evaluate_model_report(run, bundle, include_test=False, show_advanced=show_advanced)
        rows.append([
            option,
            int(np.sum(key_range_mask(train))),
            int(np.sum(key_range_mask(validation))),
            report["criterion"]["overall_validation_rmse"],
            report["criterion"]["key_range_validation_rmse"],
        ])
        reports[option] = report
        summaries[option] = split_regime_summary(train, validation, test)
    return {"rows": rows, "reports": reports, "summaries": summaries}


def _extra_data_for_option(option: str, added_n: int, seed: int) -> dict[str, Any]:
    policy = {
        "uniform_sampling": "uniform_visible",
        "target_low_support": "joint_gap_region",
        "target_high_error": "joint_gap_region",
        "stratified_regime": "visible_regime_stratified",
        "more_normal_conditions": "more_normal_conditions",
        "normal_condition_mix": "normal_conditions",
        "deployment_matched_mix": "deployment_conditions",
        "visible_regime_stratified_mix": "visible_regime_stratified",
        "input_regime_stratified_mix": "visible_regime_stratified",
        "rare_regime_oversampled": "rare_regime_oversampled",
        "temperature_irradiance_balanced_mix": "temperature_irradiance_balanced",
    }.get(option)
    if policy is None:
        raise ValueError(f"unknown data option: {option}")
    return make_pv_observations(added_n, policy, seed=seed, name=f"added {option}")


def _custom_extra_from_option(option: Mapping[str, Any], added_n: int, seed: int) -> tuple[str, dict[str, Any]]:
    """Generate custom extra data from a user-specified option mapping."""
    label = str(option.get("label", "custom input values"))
    if "input_values" not in option:
        raise ValueError("custom data collection options must include an 'input_values' field")
    n = int(option.get("added_n", added_n))
    extra = make_pv_observations_at_inputs(
        option["input_values"],
        n=n,
        seed=seed,
        noise=float(option.get("noise", 0.025)),
        name=f"added {label}",
    )
    return label, extra


def _targeted_extra_from_validation(
    bundle: dict[str, Any],
    option: str,
    added_n: int,
    seed: int,
    baseline_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate new training observations near validation diagnostics."""
    validation = visible_only_data(bundle["validation"], "validation diagnostic centers")
    if option == "target_low_support":
        distances = nearest_training_distance(bundle)
        order = np.argsort(distances)[-max(8, min(24, len(distances))) :]
    elif option == "target_high_error":
        if baseline_run is None:
            baseline_run = train_with_config(bundle, baseline_config(seed=seed + 311, epochs=140), name="diagnostic baseline")
        residual = np.abs(predict_run(baseline_run, validation) - target_vector(validation))
        order = np.argsort(residual)[-max(8, min(24, len(residual))) :]
    else:
        raise ValueError(f"unsupported targeted option: {option}")
    centers = clone_data(validation, order, name=f"{option} centers")
    return make_pv_observations_near(centers, added_n, seed=seed, name=f"added {option}")


def _bundle_with_train(bundle: dict[str, Any], train: dict[str, Any], scenario_note: str) -> dict[str, Any]:
    out = dict(bundle)
    out["train"] = train
    out["metadata"] = {**bundle.get("metadata", {}), "scenario_note": scenario_note}
    return out


def run_data_collection_options(
    bundle: dict[str, Any],
    options: Sequence[str | Mapping[str, Any]] | None = None,
    added_n: int = 100,
    seed: int = SEED,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Compare targeted data-collection options for missing-combinations activity."""
    options = list(options or DATA_COLLECTION_OPTIONS)
    rows = []
    runs = {}
    bundles = {}
    diagnostic_run = train_with_config(bundle, baseline_config(seed=seed + 300, epochs=150), name="diagnostic baseline")
    baseline_dist = nearest_training_distance(bundle)
    low_support = baseline_dist >= np.quantile(baseline_dist, 0.75)
    for i, option in enumerate(options):
        if isinstance(option, str):
            label = option
            if option in {"target_low_support", "target_high_error"}:
                extra = _targeted_extra_from_validation(bundle, option, added_n, seed + i, baseline_run=diagnostic_run)
            else:
                extra = _extra_data_for_option(option, added_n, seed + i)
        elif isinstance(option, Mapping):
            label, extra = _custom_extra_from_option(option, added_n, seed + i)
        else:
            raise TypeError("data collection options must be strings or mappings")
        extra_n = len(target_vector(extra))
        train = concat_data([bundle["train"], extra], name=f"train + {label}")
        option_bundle = _bundle_with_train(bundle, train, label)
        run = train_with_config(option_bundle, baseline_config(seed=seed + 50, epochs=180), name=label)
        report = evaluate_model_report(run, option_bundle, include_test=False, show_advanced=show_advanced)
        pred = predict_run(run, option_bundle["validation"])
        low_rmse = regression_metrics(target_vector(option_bundle["validation"])[low_support], pred[low_support])["RMSE"]
        rows.append([label, extra_n, report["criterion"]["overall_validation_rmse"], report["criterion"]["key_range_validation_rmse"], low_rmse])
        runs[label] = run
        bundles[label] = option_bundle
    return {"rows": rows, "runs": runs, "bundles": bundles}


def run_collection_mix_options(
    bundle: dict[str, Any],
    options: Sequence[str] | None = None,
    added_n: int = 150,
    seed: int = SEED,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Compare collection-mix options for deployment-mismatch activity."""
    options = list(options or COLLECTION_MIX_OPTIONS)
    rows = []
    runs = {}
    bundles = {}
    for i, option in enumerate(options):
        extra = _extra_data_for_option(option, added_n, seed + i)
        train = concat_data([bundle["train"], extra], name=f"train + {option}")
        option_bundle = _bundle_with_train(bundle, train, option)
        run = train_with_config(option_bundle, baseline_config(seed=seed + 70, epochs=180), name=option)
        report = evaluate_model_report(run, option_bundle, include_test=False, show_advanced=show_advanced)
        rows.append([option, added_n, report["criterion"]["overall_validation_rmse"], report["criterion"]["key_range_validation_rmse"], report["criterion"]["passes"]])
        runs[option] = run
        bundles[option] = option_bundle
    return {"rows": rows, "runs": runs, "bundles": bundles}


def run_model_family_options(
    bundle: dict[str, Any],
    options: Sequence[dict[str, Any]] | None = None,
    seed: int = SEED,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Compare model-family options for capacity/regularity activity."""
    options = list(options or [
        {"label": "small", "hidden_width": 8, "depth": 1},
        {"label": "baseline", "hidden_width": 16, "depth": 2},
        {"label": "wide", "hidden_width": 128, "depth": 2},
        {"label": "wide regularized", "hidden_width": 128, "depth": 2, "regularity_strength": "medium"},
    ])
    rows = []
    runs = {}
    for i, option in enumerate(options):
        label = option.get("label", f"option {i}")
        regularity = option.get("regularity_strength", "none")
        config = baseline_config(
            hidden_width=option.get("hidden_width", 16),
            depth=option.get("depth", 2),
            regularity_strength=regularity,
            learning_rate=option.get("learning_rate", 0.035),
            epochs=option.get("epochs", 200),
            seed=seed,
        )
        run = train_with_config(bundle, config, name=label)
        report = evaluate_model_report(run, bundle, include_test=False, show_advanced=show_advanced)
        train_rmse = report["metric_rows"][0][2]
        val_rmse = report["metric_rows"][1][2]
        rows.append([label, config["hidden_width"], config["depth"], regularity, train_rmse, val_rmse, val_rmse - train_rmse])
        runs[label] = run
    return {"rows": rows, "runs": runs}


def run_input_set_options(
    bundle: dict[str, Any],
    input_sets: Sequence[tuple[str, Sequence[str]]] | None = None,
    seed: int = SEED,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Compare candidate input sets for additional-measurements activity."""
    input_sets = list(input_sets or [
        ("baseline", VISIBLE_INPUTS),
        ("add cloud", [*VISIBLE_INPUTS, "cloud_cover"]),
        ("add panel temperature", [*VISIBLE_INPUTS, "panel_temperature"]),
        ("add both", [*VISIBLE_INPUTS, "cloud_cover", "panel_temperature"]),
        ("add candidate sensor C", [*VISIBLE_INPUTS, CANDIDATE_SENSOR_C]),
    ])
    rows = []
    runs = {}
    for i, (label, columns) in enumerate(input_sets):
        config = baseline_config(input_columns=list(columns), seed=seed, epochs=200)
        run = train_with_config(bundle, config, name=label)
        report = evaluate_model_report(run, bundle, include_test=False, show_advanced=show_advanced)
        rows.append([label, len(columns), report["criterion"]["overall_validation_rmse"], report["criterion"]["key_range_validation_rmse"], report["criterion"]["passes"]])
        runs[label] = run
    return {"rows": rows, "runs": runs}


def residual_by_candidate_measurement(bundle: dict[str, Any], run: dict[str, Any], bins: int = 5) -> dict[str, list[list[Any]]]:
    """Summarise residuals by the newly revealed candidate variables."""
    validation = bundle["validation"]
    residual = predict_run(run, validation) - target_vector(validation)
    out: dict[str, list[list[Any]]] = {}
    for col in REVEALED_INPUTS:
        x = validation[col]
        edges = np.linspace(float(np.min(x)), float(np.max(x)), int(bins) + 1)
        rows = []
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (x >= left) & (x < right if right < edges[-1] else x <= right)
            rows.append([f"{left:.2f}-{right:.2f}", int(mask.sum()), float(np.mean(residual[mask])) if np.any(mask) else np.nan, float(np.mean(np.abs(residual[mask]))) if np.any(mask) else np.nan])
        out[col] = rows
    return out


def run_activation_options(
    bundle: dict[str, Any],
    activations: Sequence[str] | None = None,
    seed: int = SEED,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Compare activation options under fixed size and optimizer."""
    activations = list(activations or ACTIVATION_OPTIONS)
    rows = []
    runs = {}
    for i, activation in enumerate(activations):
        init = "he"
        config = baseline_config(activation=activation, initialization=init, seed=seed, epochs=200)
        run = train_with_config(bundle, config, name=activation)
        report = evaluate_model_report(run, bundle, include_test=False, show_advanced=show_advanced)
        rows.append([activation, init, report["criterion"]["overall_validation_rmse"], report["criterion"]["key_range_validation_rmse"], report["criterion"]["passes"]])
        runs[activation] = run
    return {"rows": rows, "runs": runs}


def run_scaling_initialization_options(
    bundle: dict[str, Any],
    options: Sequence[dict[str, Any]] | None = None,
    seed: int = SEED,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Compare scaling and initialization options."""
    options = list(options or [
        {"label": "raw + he", "scaling": "raw", "initialization": "he"},
        {"label": "standard + tiny", "scaling": "standard", "initialization": "small"},
        {"label": "standard + he", "scaling": "standard", "initialization": "he"},
        {"label": "standard + large", "scaling": "standard", "initialization": "large"},
    ])
    rows = []
    runs = {}
    for option in options:
        label = option["label"]
        config = baseline_config(
            scaling=option["scaling"],
            initialization=option["initialization"],
            seed=seed,
            epochs=120,
            learning_rate=option.get("learning_rate", 0.035),
        )
        run = train_with_config(bundle, config, name=label)
        report = evaluate_model_report(run, bundle, include_test=False, show_advanced=show_advanced)
        initial_range = initial_prediction_range(bundle, config)
        status = "diverged" if run.get("diverged") else "valid"
        rows.append([label, option["scaling"], option["initialization"], initial_range[0], initial_range[1], status, report["criterion"]["overall_validation_rmse"], run["best_validation_epoch"]])
        runs[label] = run
    return {"rows": rows, "runs": runs}


def initial_prediction_range(bundle: dict[str, Any], config: dict[str, Any]) -> tuple[float, float]:
    """Return initial prediction min/max before training for a config."""
    require_torch()
    torch.manual_seed(int(config.get("seed", SEED)))
    columns = config.get("input_columns", bundle["visible_inputs"])
    X_train = feature_matrix(bundle["train"], columns)
    scaler = fit_scaler(X_train, config.get("scaling", "standard"))
    X_val = transform_with_scaler(feature_matrix(bundle["validation"], columns), scaler)
    model = SolarPVMLP(
        input_dim=len(columns),
        hidden_width=config.get("hidden_width", 16),
        depth=config.get("depth", 2),
        activation=config.get("activation", "relu"),
        initialization=config.get("initialization", "he"),
    )
    with torch.no_grad():
        pred = model(torch.as_tensor(X_val, dtype=torch.float32)).numpy().ravel()
    return float(np.min(pred)), float(np.max(pred))


def run_sgd_schedule_options(
    bundle: dict[str, Any],
    options: Sequence[dict[str, Any]] | None = None,
    seed: int = SEED,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Compare SGD schedule options."""
    options = list(options or [
        {"label": "too slow", "learning_rate": 0.003, "batch_size": 32, "epochs": 120},
        {"label": "baseline", "learning_rate": 0.035, "batch_size": 32, "epochs": 220},
        {"label": "large batch", "learning_rate": 0.035, "batch_size": 256, "epochs": 220},
        {"label": "aggressive", "learning_rate": 0.12, "batch_size": 32, "epochs": 120},
    ])
    rows = []
    runs = {}
    for option in options:
        label = option["label"]
        config = baseline_config(
            learning_rate=option["learning_rate"],
            batch_size=option["batch_size"],
            epochs=option["epochs"],
            seed=seed,
        )
        run = train_with_config(bundle, config, name=label)
        report = evaluate_model_report(run, bundle, include_test=False, show_advanced=show_advanced)
        final_rmse = run.get("final_epoch_metrics", {}).get("validation_RMSE", np.nan)
        rows.append([
            label,
            option["learning_rate"],
            option["batch_size"],
            option["epochs"],
            report["criterion"]["overall_validation_rmse"],
            run["best_validation_epoch"],
            final_rmse,
        ])
        runs[label] = run
    return {"rows": rows, "runs": runs}


def run_objective_options(
    bundle: dict[str, Any],
    options: Sequence[dict[str, Any]] | None = None,
    seed: int = SEED,
    show_advanced: bool = False,
) -> dict[str, Any]:
    """Compare objective/loss options for noisy-label activity."""
    options = list(options or [
        {"label": "MSE", "loss": "mse", "selection_metric": "rmse"},
        {"label": "MAE", "loss": "mae", "selection_metric": "mae", "learning_rate": 0.02},
        {"label": "Huber small", "loss": "huber", "huber_delta": 0.04, "selection_metric": "rmse"},
        {"label": "Huber medium", "loss": "huber", "huber_delta": 0.10, "selection_metric": "rmse"},
    ])
    rows = []
    runs = {}
    for option in options:
        label = option["label"]
        config = baseline_config(
            loss=option["loss"],
            huber_delta=option.get("huber_delta", 0.08),
            learning_rate=option.get("learning_rate", 0.03),
            selection_metric=option.get("selection_metric", "rmse"),
            seed=seed,
            epochs=220,
        )
        run = train_with_config(bundle, config, name=label)
        report = evaluate_model_report(run, bundle, include_test=False, show_advanced=show_advanced)
        val_pred = predict_run(run, bundle["validation"])
        val_metrics = regression_metrics(target_vector(bundle["validation"]), val_pred)
        rows.append([label, config["loss"], config["huber_delta"], config["selection_metric"], val_metrics["RMSE"], val_metrics["MAE"], report["criterion"]["key_range_validation_rmse"]])
        runs[label] = run
    return {"rows": rows, "runs": runs}


def high_residual_rows(
    bundle: dict[str, Any],
    run: dict[str, Any],
    split: str = "train",
    top_k: int = 8,
    include_quality: bool = False,
) -> list[list[Any]]:
    """Return rows with largest residuals for noisy-label diagnostics."""
    data = bundle[split]
    pred = predict_run(run, data)
    residual = pred - target_vector(data)
    order = np.argsort(np.abs(residual))[-int(top_k) :][::-1]
    rows = []
    for idx in order:
        row = [
            int(idx),
            float(pred[idx]),
            float(target_vector(data)[idx]),
            float(residual[idx]),
        ]
        if include_quality:
            row.append(int(data.get("measurement_quality", np.zeros(len(residual), dtype=int))[idx]))
        rows.append(row)
    return rows


def final_check(run: dict[str, Any], bundle: dict[str, Any], show_advanced: bool = False) -> dict[str, Any]:
    """Run the shared report with test included after selection is fixed."""
    return evaluate_model_report(run, bundle, include_test=True, show_advanced=show_advanced)


def _is_numeric_table_cell(value: Any) -> bool:
    """Return whether a cell should be right-aligned in printed tables."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _format_table_cell(value: Any) -> str:
    """Format one value for compact markdown table output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}" if np.isfinite(value) else "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def _format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Return a padded markdown table with numeric columns right-aligned."""
    row_values = [list(row) for row in rows]
    column_count = max([len(headers), *(len(row) for row in row_values)] or [0])
    padded_headers = [str(headers[index]) if index < len(headers) else "" for index in range(column_count)]
    padded_rows = [
        [row[index] if index < len(row) else "" for index in range(column_count)]
        for row in row_values
    ]
    formatted_rows = [[_format_table_cell(item) for item in row] for row in padded_rows]
    numeric_columns = [
        bool(padded_rows) and all(_is_numeric_table_cell(row[index]) for row in padded_rows if row[index] != "")
        for index in range(column_count)
    ]
    widths = [
        max(
            3,
            len(padded_headers[index]),
            *(len(row[index]) for row in formatted_rows),
        )
        for index in range(column_count)
    ]

    def align_cell(text: str, index: int) -> str:
        return text.rjust(widths[index]) if numeric_columns[index] else text.ljust(widths[index])

    header_line = "| " + " | ".join(padded_headers[index].ljust(widths[index]) for index in range(column_count)) + " |"
    separators = [
        ("-" * max(3, widths[index] - 1) + ":").rjust(widths[index])
        if numeric_columns[index]
        else (":" + "-" * max(3, widths[index] - 1)).ljust(widths[index])
        for index in range(column_count)
    ]
    separator_line = "| " + " | ".join(separators) + " |"
    body_lines = [
        "| " + " | ".join(align_cell(row[index], index) for index in range(column_count)) + " |"
        for row in formatted_rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Print a padded markdown table in notebooks or terminals."""
    print(_format_markdown_table(headers, rows))


def print_report(report: dict[str, Any]) -> None:
    """Print the shared evaluation report as compact markdown tables."""
    print(f"Report: {report['run_name']}")
    print_table(["Split", "n", "RMSE", "MAE", "Key range RMSE"], report["metric_rows"])
    criterion = report["criterion"]
    print()
    print_table(
        ["Criterion", "Value", "Threshold", "Pass"],
        [
            [
                "Overall validation RMSE",
                criterion["overall_validation_rmse"],
                criterion["overall_threshold"],
                criterion["overall_validation_rmse"] <= criterion["overall_threshold"],
            ],
            [
                "Key range validation RMSE",
                criterion["key_range_validation_rmse"],
                criterion["key_range_threshold"],
                criterion["key_range_validation_rmse"] <= criterion["key_range_threshold"],
            ],
            ["Combined criterion", "", "", criterion["passes"]],
        ],
    )
    print()
    print_table(["Slice", "n", "RMSE", "MAE", "Bias"], report["slice_rows"])


__all__ = [
    "ALL_INPUTS",
    "ACTIVATION_OPTIONS",
    "CANDIDATE_SENSOR_C",
    "COLLECTION_MIX_OPTIONS",
    "DATA_COLLECTION_OPTIONS",
    "INITIALIZATION_OPTIONS",
    "LOSS_OPTIONS",
    "REGULARITY_OPTIONS",
    "REVEALED_INPUTS",
    "SCALING_OPTIONS",
    "SELECTION_METRIC_OPTIONS",
    "TARGET",
    "VISIBLE_INPUTS",
    "SuccessCriterion",
    "SolarPVMLP",
    "PVDataset",
    "add_noisy_labels",
    "baseline_config",
    "binned_target_summary",
    "challenge_candidate_point_rows",
    "challenge_config",
    "challenge_option_rows",
    "challenge_success_criterion",
    "clone_data",
    "compare_split_strategies",
    "concat_data",
    "describe_inputs",
    "describe_visible_inputs",
    "evaluate_model_report",
    "feature_matrix",
    "final_check",
    "joint_gap_mask",
    "key_range_mask",
    "low_support_rows",
    "make_mean_baseline",
    "make_capacity_observations",
    "make_challenge_bundle",
    "make_pv_observations_at_inputs",
    "make_pv_observations",
    "make_pv_observations_near",
    "make_split_from_pool",
    "make_workshop3_bundle",
    "nearest_training_distance",
    "plot_observed_vs_predicted",
    "plot_training_curves",
    "print_table",
    "print_report",
    "regression_metrics",
    "representative_rows",
    "residual_by_candidate_measurement",
    "run_activation_options",
    "run_collection_mix_options",
    "run_challenge_experiment",
    "run_data_collection_options",
    "run_input_set_options",
    "run_model_family_options",
    "run_objective_options",
    "run_scaling_initialization_options",
    "run_sgd_schedule_options",
    "slice_metric_rows",
    "split_regime_summary",
    "target_vector",
    "train_solar_mlp",
    "train_with_config",
    "visible_regime_labels",
]
