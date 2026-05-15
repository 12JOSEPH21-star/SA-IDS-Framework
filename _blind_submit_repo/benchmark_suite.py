from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from experiment import PreparedExperimentData, ResearchExperimentRunner, TensorBatch
from models import SensorMetadataBatch
from pipeline import SilenceAwareIDS, SilenceAwareIDSConfig
from reliability import ConformalPredictor
from task_cli.framework import load_framework_config

if TYPE_CHECKING:
    from experiment import ExperimentRunConfig, TabularDataConfig


@dataclass
class BenchmarkSuiteConfig:
    """Configuration for the fixed paper-style benchmark suite.

    Attributes:
        framework_config_path: Base framework config used as the canonical starting point.
        output_dir: Directory where benchmark artifacts are written.
        station_limit: Maximum number of sensor instances kept in the canonical slice.
        station_offset: Optional offset applied before station selection.
        temporal_stride_hours: Temporal thinning factor applied after splitting.
        max_train_rows: Cap for chronologically ordered training rows.
        max_calibration_rows: Cap for chronologically ordered calibration rows.
        max_evaluation_rows: Cap for chronologically ordered evaluation rows.
        state_epochs: State-model epochs used in benchmark runs.
        missingness_epochs: Missingness-model epochs used in benchmark runs.
        inducing_points: Sparse GP inducing-point count used in the benchmark.
        prediction_batch_size: Inference batch size.
        target_coverage: Target predictive coverage for calibration-error reporting.
        shift_fraction: Fraction of evaluation timestamps treated as regime-shift stress.
        policy_budget: Optional active-sensing budget. When omitted, one is inferred.
        max_selections: Maximum number of selected candidates.
        missing_intensities: Missingness rates used in the MNAR benchmark.
        predictive_mechanisms: Structural-missingness mechanisms benchmarked in Table 1.
        fault_scenarios: Synthetic fault scenarios benchmarked in Table 2.
        cache_prepared_data: Whether to cache the canonical slice after the first preparation pass.
        repeat_seeds: Seeds used for repeated benchmark evaluation.
        region_holdout_variants: Variants evaluated under coarse region-holdout generalization.
        bootstrap_samples: Number of bootstrap or randomization samples used for
            confidence intervals and paired significance tests.
        significance_alpha: Alpha level used when marking significant gains.
        seed: Global random seed.
    """

    framework_config_path: Path
    output_dir: Path = Path("outputs/benchmark_suite")
    station_limit: int = 12
    station_offset: int = 0
    temporal_stride_hours: int = 6
    max_train_rows: int = 4096
    max_calibration_rows: int = 1024
    max_evaluation_rows: int = 1024
    state_epochs: int = 1
    missingness_epochs: int = 1
    inducing_points: int = 16
    prediction_batch_size: int = 256
    target_coverage: float = 0.9
    shift_fraction: float = 0.35
    policy_budget: float | None = None
    max_selections: int = 5
    missing_intensities: tuple[float, ...] = (0.3, 0.5, 0.7)
    predictive_mechanisms: tuple[str, ...] = (
        "mar",
        "state_dependent_mnar",
        "value_dependent_mnar",
    )
    fault_scenarios: tuple[str, ...] = (
        "random_dropout",
        "block_missingness",
        "frozen_sensor",
        "drift",
        "spike_burst",
        "extreme_event_confound",
    )
    cache_prepared_data: bool = True
    repeat_seeds: tuple[int, ...] = (7,)
    region_holdout_variants: tuple[str, ...] = ()
    bootstrap_samples: int = 256
    significance_alpha: float = 0.05
    seed: int = 7

    def __post_init__(self) -> None:
        if self.station_limit <= 0:
            raise ValueError("station_limit must be positive.")
        if self.temporal_stride_hours <= 0:
            raise ValueError("temporal_stride_hours must be positive.")
        if self.max_train_rows <= 0 or self.max_calibration_rows <= 0 or self.max_evaluation_rows <= 0:
            raise ValueError("row caps must be positive.")
        if self.state_epochs <= 0 or self.missingness_epochs <= 0:
            raise ValueError("training epochs must be positive.")
        if self.inducing_points <= 0:
            raise ValueError("inducing_points must be positive.")
        if self.prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive.")
        if not 0.0 < self.target_coverage < 1.0:
            raise ValueError("target_coverage must lie in (0, 1).")
        if not 0.0 < self.shift_fraction < 1.0:
            raise ValueError("shift_fraction must lie in (0, 1).")
        if self.max_selections <= 0:
            raise ValueError("max_selections must be positive.")
        if not self.repeat_seeds:
            raise ValueError("repeat_seeds must contain at least one seed.")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive.")
        if not 0.0 < self.significance_alpha < 1.0:
            raise ValueError("significance_alpha must lie in (0, 1).")


@dataclass
class BenchmarkArtifacts:
    """Artifact paths emitted by the benchmark suite."""

    summary_path: Path
    predictive_path: Path
    fault_path: Path
    reliability_path: Path
    coverage_timeseries_path: Path
    region_holdout_path: Path
    ablation_path: Path
    policy_path: Path
    runtime_path: Path
    significance_path: Path
    paper_tables_path: Path
    report_path: Path


@dataclass
class BenchmarkResult:
    """Structured output for the benchmark suite."""

    canonical_setup: dict[str, Any]
    predictive_rows: list[dict[str, Any]]
    fault_rows: list[dict[str, Any]]
    reliability_rows: list[dict[str, Any]]
    coverage_timeseries_rows: list[dict[str, Any]]
    region_holdout_rows: list[dict[str, Any]]
    ablation_rows: list[dict[str, Any]]
    policy_rows: list[dict[str, Any]]
    runtime_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert results to a JSON-serializable dictionary."""
        return {
            "canonical_setup": self.canonical_setup,
            "predictive_rows": self.predictive_rows,
            "fault_rows": self.fault_rows,
            "reliability_rows": self.reliability_rows,
            "coverage_timeseries_rows": self.coverage_timeseries_rows,
            "region_holdout_rows": self.region_holdout_rows,
            "ablation_rows": self.ablation_rows,
            "policy_rows": self.policy_rows,
            "runtime_rows": self.runtime_rows,
        }


def _default_framework_config() -> Path:
    """Resolve the best available real-data framework config for benchmarking."""
    candidates = (
        Path("data/joint_weather_network_q1_2025/framework_joint_q1.json"),
        Path("data/noaa_isd_korea_q1_2025/framework_isd_q1.json"),
        Path("data/era5_korea_q1_2025/framework_era5_q1.json"),
        Path("framework_config.json"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("No default framework config was found for benchmark initialization.")


def write_benchmark_template(
    config_path: Path,
    *,
    framework_config_path: Path | None = None,
    preset: str = "small",
    force: bool = False,
) -> Path:
    """Write a benchmark-suite config JSON.

    Args:
        config_path: Output config path.
        framework_config_path: Optional base framework config. Defaults to the best
            available real-data config in the repository.
        force: Whether to overwrite an existing file.

    Returns:
        Path to the written config.
    """
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} already exists. Re-run with --force to overwrite.")
    base_framework = (framework_config_path or _default_framework_config()).resolve()
    preset_payloads = {
        "small": {
            "station_limit": 12,
            "temporal_stride_hours": 6,
            "max_train_rows": 4096,
            "max_calibration_rows": 1024,
            "max_evaluation_rows": 1024,
            "state_epochs": 1,
            "missingness_epochs": 1,
            "inducing_points": 16,
            "prediction_batch_size": 256,
            "repeat_seeds": [7, 11, 19],
            "region_holdout_variants": [],
        },
        "medium": {
            "station_limit": 16,
            "temporal_stride_hours": 3,
            "max_train_rows": 8192,
            "max_calibration_rows": 2048,
            "max_evaluation_rows": 2048,
            "state_epochs": 2,
            "missingness_epochs": 2,
            "inducing_points": 24,
            "prediction_batch_size": 384,
            "repeat_seeds": [7, 11, 19, 23, 29],
            "region_holdout_variants": [],
        },
        "large": {
            "station_limit": 96,
            "temporal_stride_hours": 1,
            "max_train_rows": 32768,
            "max_calibration_rows": 8192,
            "max_evaluation_rows": 8192,
            "state_epochs": 2,
            "missingness_epochs": 2,
            "inducing_points": 32,
            "prediction_batch_size": 512,
            "repeat_seeds": [7, 11, 19, 23, 29],
            "region_holdout_variants": [
                "base_gp_only",
                "gp_plus_joint_generative_jvi_training",
                "full_model",
            ],
        },
    }
    if preset not in preset_payloads:
        raise ValueError("preset must be one of 'small', 'medium', or 'large'.")
    payload = {
        "framework_config_path": str(base_framework),
        "output_dir": str((config_path.parent / "outputs" / "benchmark_suite").resolve()),
        "preset": preset,
        "station_limit": preset_payloads[preset]["station_limit"],
        "station_offset": 0,
        "temporal_stride_hours": preset_payloads[preset]["temporal_stride_hours"],
        "max_train_rows": preset_payloads[preset]["max_train_rows"],
        "max_calibration_rows": preset_payloads[preset]["max_calibration_rows"],
        "max_evaluation_rows": preset_payloads[preset]["max_evaluation_rows"],
        "state_epochs": preset_payloads[preset]["state_epochs"],
        "missingness_epochs": preset_payloads[preset]["missingness_epochs"],
        "inducing_points": preset_payloads[preset]["inducing_points"],
        "prediction_batch_size": preset_payloads[preset]["prediction_batch_size"],
        "target_coverage": 0.9,
        "shift_fraction": 0.35,
        "policy_budget": None,
        "max_selections": 5,
        "missing_intensities": [0.3, 0.5, 0.7],
        "predictive_mechanisms": ["mar", "state_dependent_mnar", "value_dependent_mnar"],
        "fault_scenarios": [
            "random_dropout",
            "block_missingness",
            "frozen_sensor",
            "drift",
            "spike_burst",
            "extreme_event_confound",
        ],
        "cache_prepared_data": True,
        "repeat_seeds": preset_payloads[preset]["repeat_seeds"],
        "region_holdout_variants": preset_payloads[preset]["region_holdout_variants"],
        "bootstrap_samples": 256,
        "significance_alpha": 0.05,
        "seed": 7,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def load_benchmark_config(config_path: Path) -> BenchmarkSuiteConfig:
    """Load a benchmark-suite config from JSON."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return BenchmarkSuiteConfig(
        framework_config_path=Path(payload["framework_config_path"]).resolve(),
        output_dir=Path(payload.get("output_dir", "outputs/benchmark_suite")).resolve(),
        station_limit=int(payload.get("station_limit", 12)),
        station_offset=int(payload.get("station_offset", 0)),
        temporal_stride_hours=int(payload.get("temporal_stride_hours", 6)),
        max_train_rows=int(payload.get("max_train_rows", 4096)),
        max_calibration_rows=int(payload.get("max_calibration_rows", 1024)),
        max_evaluation_rows=int(payload.get("max_evaluation_rows", 1024)),
        state_epochs=int(payload.get("state_epochs", 1)),
        missingness_epochs=int(payload.get("missingness_epochs", 1)),
        inducing_points=int(payload.get("inducing_points", 16)),
        prediction_batch_size=int(payload.get("prediction_batch_size", 256)),
        target_coverage=float(payload.get("target_coverage", 0.9)),
        shift_fraction=float(payload.get("shift_fraction", 0.35)),
        policy_budget=payload.get("policy_budget"),
        max_selections=int(payload.get("max_selections", 5)),
        missing_intensities=tuple(float(value) for value in payload.get("missing_intensities", [0.3, 0.5, 0.7])),
        predictive_mechanisms=tuple(
            str(value) for value in payload.get("predictive_mechanisms", ["mar", "state_dependent_mnar", "value_dependent_mnar"])
        ),
        fault_scenarios=tuple(
            str(value)
            for value in payload.get(
                "fault_scenarios",
                ["random_dropout", "block_missingness", "frozen_sensor", "drift", "spike_burst", "extreme_event_confound"],
            )
        ),
        cache_prepared_data=bool(payload.get("cache_prepared_data", True)),
        repeat_seeds=tuple(int(value) for value in payload.get("repeat_seeds", [payload.get("seed", 7)])),
        region_holdout_variants=tuple(str(value) for value in payload.get("region_holdout_variants", [])),
        bootstrap_samples=int(payload.get("bootstrap_samples", 256)),
        significance_alpha=float(payload.get("significance_alpha", 0.05)),
        seed=int(payload.get("seed", 7)),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write stable CSV artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: Any) -> str:
    """Format scalar outputs for compact CSV tables."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _parse_float(value: Any) -> float | None:
    """Best-effort float parsing for stringified benchmark tables."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bootstrap_confidence_interval(
    values: list[float],
    *,
    alpha: float,
    samples: int,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Estimate a bootstrap confidence interval for one scalar metric."""
    if not values:
        return None, None
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() == 1:
        scalar = float(tensor.item())
        return scalar, scalar
    generator = torch.Generator().manual_seed(seed)
    sample_index = torch.randint(
        low=0,
        high=int(tensor.numel()),
        size=(samples, int(tensor.numel())),
        generator=generator,
    )
    sample_means = tensor[sample_index].mean(dim=1)
    lower = float(torch.quantile(sample_means, alpha / 2.0).item())
    upper = float(torch.quantile(sample_means, 1.0 - alpha / 2.0).item())
    return lower, upper


def _paired_randomization_test(
    candidate_values: list[float],
    baseline_values: list[float],
    *,
    lower_is_better: bool,
    samples: int,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    """Run a paired sign-flip randomization test with bootstrap CI on deltas."""
    if len(candidate_values) != len(baseline_values) or not candidate_values:
        return None, None, None
    candidate = torch.tensor(candidate_values, dtype=torch.float32)
    baseline = torch.tensor(baseline_values, dtype=torch.float32)
    delta = baseline - candidate if lower_is_better else candidate - baseline
    effect_mean = float(delta.mean().item())
    if delta.numel() == 1:
        ci_low = effect_mean
        ci_high = effect_mean
        return effect_mean, ci_low, 1.0
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (samples, int(delta.numel())), generator=generator, dtype=torch.int64)
    signs = signs.mul(2).sub(1).to(dtype=torch.float32)
    null_distribution = (signs * delta.unsqueeze(0)).mean(dim=1)
    p_value = float((null_distribution.abs() >= abs(effect_mean)).float().mean().item())
    ci_low, ci_high = _bootstrap_confidence_interval(
        delta.tolist(),
        alpha=0.05,
        samples=samples,
        seed=seed + 17,
    )
    return effect_mean, ci_low, p_value if ci_high is not None else p_value


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    numeric_keys: tuple[str, ...],
    bootstrap_samples: int = 256,
    bootstrap_alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Aggregate repeated benchmark rows into mean/std summaries."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        output = {column: value for column, value in zip(group_keys, key)}
        output["repeat_count"] = len(group_rows)
        for numeric_key in numeric_keys:
            values = [_parse_float(row.get(numeric_key)) for row in group_rows]
            numeric_values = [value for value in values if value is not None and math.isfinite(value)]
            if not numeric_values:
                output[f"{numeric_key}_mean"] = ""
                output[f"{numeric_key}_std"] = ""
                output[f"{numeric_key}_ci_low"] = ""
                output[f"{numeric_key}_ci_high"] = ""
                continue
            tensor = torch.tensor(numeric_values, dtype=torch.float32)
            output[f"{numeric_key}_mean"] = _format_float(float(tensor.mean().item()))
            output[f"{numeric_key}_std"] = _format_float(
                float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
            )
            ci_low, ci_high = _bootstrap_confidence_interval(
                numeric_values,
                alpha=bootstrap_alpha,
                samples=bootstrap_samples,
                seed=len(group_rows) + len(numeric_key),
            )
            output[f"{numeric_key}_ci_low"] = _format_float(ci_low)
            output[f"{numeric_key}_ci_high"] = _format_float(ci_high)
        aggregated.append(output)
    return aggregated


def _significance_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    metric_key: str,
    baseline_variant: str,
    lower_is_better: bool,
    bootstrap_samples: int,
    significance_alpha: float,
) -> list[dict[str, Any]]:
    """Compute paired significance summaries over repeat seeds."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in group_keys)
        grouped.setdefault(key, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        by_variant: dict[str, dict[int, float]] = {}
        for row in group_rows:
            value = _parse_float(row.get(metric_key))
            seed = row.get("repeat_seed")
            variant_name = str(row.get("variant_name"))
            if value is None or seed is None:
                continue
            by_variant.setdefault(variant_name, {})[int(seed)] = float(value)
        baseline = by_variant.get(baseline_variant)
        if not baseline:
            continue
        for variant_name, values_by_seed in by_variant.items():
            if variant_name == baseline_variant:
                continue
            aligned_seeds = sorted(set(values_by_seed).intersection(baseline))
            if not aligned_seeds:
                continue
            candidate_values = [values_by_seed[seed] for seed in aligned_seeds]
            baseline_values = [baseline[seed] for seed in aligned_seeds]
            effect_mean, effect_ci_low, p_value = _paired_randomization_test(
                candidate_values,
                baseline_values,
                lower_is_better=lower_is_better,
                samples=bootstrap_samples,
                seed=sum(aligned_seeds) + len(metric_key),
            )
            effect_ci_high = None
            if effect_mean is not None:
                _, effect_ci_high = _bootstrap_confidence_interval(
                    [
                        baseline_value - candidate_value if lower_is_better else candidate_value - baseline_value
                        for candidate_value, baseline_value in zip(candidate_values, baseline_values)
                    ],
                    alpha=significance_alpha,
                    samples=bootstrap_samples,
                    seed=sum(aligned_seeds) + len(metric_key) + 23,
                )
            row_payload = {column: value for column, value in zip(group_keys, key)}
            row_payload.update(
                {
                    "baseline_variant": baseline_variant,
                    "variant_name": variant_name,
                    "metric": metric_key,
                    "repeat_count": len(aligned_seeds),
                    "effect_mean": _format_float(effect_mean),
                    "effect_ci_low": _format_float(effect_ci_low),
                    "effect_ci_high": _format_float(effect_ci_high),
                    "p_value": _format_float(p_value),
                    "significant": bool(p_value is not None and p_value <= significance_alpha),
                }
            )
            output_rows.append(row_payload)
    return output_rows


def _slice_metadata_batch(
    metadata: SensorMetadataBatch | None,
    indices: Tensor,
) -> SensorMetadataBatch:
    """Slice one metadata batch using one-dimensional row indices."""
    if metadata is None:
        return SensorMetadataBatch()
    return SensorMetadataBatch(
        sensor_instance=None if metadata.sensor_instance is None else metadata.sensor_instance[indices],
        sensor_type=None if metadata.sensor_type is None else metadata.sensor_type[indices],
        sensor_group=None if metadata.sensor_group is None else metadata.sensor_group[indices],
        sensor_modality=None if metadata.sensor_modality is None else metadata.sensor_modality[indices],
        installation_environment=(
            None if metadata.installation_environment is None else metadata.installation_environment[indices]
        ),
        maintenance_state=None if metadata.maintenance_state is None else metadata.maintenance_state[indices],
        continuous=None if metadata.continuous is None else metadata.continuous[indices],
    )


def _slice_batch(batch: TensorBatch, indices: Tensor) -> TensorBatch:
    """Slice a tensor batch without changing column semantics."""
    return TensorBatch(
        X=batch.X[indices],
        y=batch.y[indices],
        context=None if batch.context is None else batch.context[indices],
        M=None if batch.M is None else batch.M[indices],
        S=None if batch.S is None else batch.S[indices],
        missing_indicator=batch.missing_indicator[indices],
        cost=None if batch.cost is None else batch.cost[indices],
        sensor_metadata=_slice_metadata_batch(batch.sensor_metadata, indices),
        indices=batch.indices[indices],
    )


def _batch_to_cache_dict(batch: TensorBatch) -> dict[str, Any]:
    """Serialize a tensor batch for torch.save."""
    return {
        "X": batch.X.cpu(),
        "y": batch.y.cpu(),
        "context": None if batch.context is None else batch.context.cpu(),
        "M": None if batch.M is None else batch.M.cpu(),
        "S": None if batch.S is None else batch.S.cpu(),
        "missing_indicator": batch.missing_indicator.cpu(),
        "cost": None if batch.cost is None else batch.cost.cpu(),
        "indices": batch.indices.cpu(),
        "sensor_metadata": {
            "sensor_instance": None if batch.sensor_metadata.sensor_instance is None else batch.sensor_metadata.sensor_instance.cpu(),
            "sensor_type": None if batch.sensor_metadata.sensor_type is None else batch.sensor_metadata.sensor_type.cpu(),
            "sensor_group": None if batch.sensor_metadata.sensor_group is None else batch.sensor_metadata.sensor_group.cpu(),
            "sensor_modality": None if batch.sensor_metadata.sensor_modality is None else batch.sensor_metadata.sensor_modality.cpu(),
            "installation_environment": (
                None
                if batch.sensor_metadata.installation_environment is None
                else batch.sensor_metadata.installation_environment.cpu()
            ),
            "maintenance_state": (
                None if batch.sensor_metadata.maintenance_state is None else batch.sensor_metadata.maintenance_state.cpu()
            ),
            "continuous": None if batch.sensor_metadata.continuous is None else batch.sensor_metadata.continuous.cpu(),
        },
    }


def _batch_from_cache_dict(payload: dict[str, Any]) -> TensorBatch:
    """Restore a tensor batch from a cached dictionary."""
    metadata = payload["sensor_metadata"]
    return TensorBatch(
        X=payload["X"],
        y=payload["y"],
        context=payload["context"],
        M=payload["M"],
        S=payload["S"],
        missing_indicator=payload["missing_indicator"],
        cost=payload["cost"],
        sensor_metadata=SensorMetadataBatch(
            sensor_instance=metadata["sensor_instance"],
            sensor_type=metadata["sensor_type"],
            sensor_group=metadata["sensor_group"],
            sensor_modality=metadata["sensor_modality"],
            installation_environment=metadata["installation_environment"],
            maintenance_state=metadata["maintenance_state"],
            continuous=metadata["continuous"],
        ),
        indices=payload["indices"],
    )


def _standardize(vector: Tensor) -> Tensor:
    """Standardize a one-dimensional tensor while tolerating NaNs."""
    finite_mask = torch.isfinite(vector)
    if not finite_mask.any():
        return torch.zeros_like(vector)
    centered = vector.clone()
    mean = centered[finite_mask].mean()
    std = centered[finite_mask].std(unbiased=False)
    scale = torch.clamp(std, min=1e-6)
    centered[finite_mask] = (centered[finite_mask] - mean) / scale
    centered[~finite_mask] = 0.0
    return centered


def _temporal_station_difference(values: Tensor, sensor_instance: Tensor | None) -> Tensor:
    """Compute station-local first differences for chronological series."""
    diff = torch.zeros_like(values)
    if sensor_instance is None or values.numel() == 0:
        return diff
    for station_id in torch.unique(sensor_instance):
        station_mask = sensor_instance == station_id
        station_index = station_mask.nonzero(as_tuple=False).squeeze(-1)
        if station_index.numel() < 2:
            continue
        station_values = values[station_index]
        step_diff = torch.zeros_like(station_values)
        step_diff[1:] = torch.abs(station_values[1:] - station_values[:-1])
        diff[station_index] = step_diff
    diff[~torch.isfinite(diff)] = 0.0
    return diff


def _context_index(prepared: PreparedExperimentData, column_name: str) -> int | None:
    """Return one context-column index from the prepared feature schema."""
    context_columns = prepared.feature_schema.get("context_columns", [])
    if column_name not in context_columns:
        return None
    return int(context_columns.index(column_name))


def _region_label(latitude: float, longitude: float) -> str:
    """Assign one coarse region label from latitude and longitude."""
    north = latitude >= 36.5
    east = longitude >= 127.75
    if north and east:
        return "northeast"
    if north and not east:
        return "northwest"
    if not north and east:
        return "southeast"
    return "southwest"


def _region_labels_for_batch(batch: TensorBatch) -> list[str]:
    """Assign one coarse region label per row from the first two GP coordinates."""
    latitudes = batch.X[:, 0].detach().cpu().tolist()
    longitudes = batch.X[:, 1].detach().cpu().tolist()
    return [_region_label(float(lat), float(lon)) for lat, lon in zip(latitudes, longitudes)]


class BenchmarkSuiteRunner:
    """Fixed benchmark runner for the paper's core experimental claims."""

    def __init__(self, config: BenchmarkSuiteConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        (
            self.data_config,
            self.pipeline_config,
            self.run_config,
            _,
        ) = load_framework_config(config.framework_config_path)
        self.run_config = replace(
            self.run_config,
            prediction_batch_size=config.prediction_batch_size,
            max_selections=config.max_selections,
        )
        self.base_runner = ResearchExperimentRunner(self.data_config, self.pipeline_config, self.run_config)
        self._prepared: PreparedExperimentData | None = None
        self._canonical: PreparedExperimentData | None = None
        self._resolved_pipeline_config: SilenceAwareIDSConfig | None = None

    def _progress_path(self) -> Path:
        """Return the benchmark progress JSON path."""
        return self.config.output_dir / "benchmark_progress.json"

    def _heartbeat_path(self) -> Path:
        """Return the benchmark heartbeat JSONL path."""
        return self.config.output_dir / "benchmark_heartbeat.jsonl"

    def _write_progress(
        self,
        *,
        status: str,
        stage: str,
        repeat_seed: int | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist one benchmark progress snapshot."""
        progress = {
            "status": status,
            "stage": stage,
            "repeat_seed": repeat_seed,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "error": error,
            "payload": payload or {},
        }
        self._progress_path().write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_heartbeat(
        self,
        *,
        stage: str,
        step: str,
        repeat_seed: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append one benchmark heartbeat line."""
        heartbeat = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "step": step,
            "repeat_seed": repeat_seed,
            "payload": payload or {},
        }
        with self._heartbeat_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(heartbeat, ensure_ascii=False) + "\n")

    def _cache_path(self) -> Path:
        """Resolve the cache path for the canonical benchmark slice."""
        payload = {
            "framework_config_path": str(self.config.framework_config_path.resolve()),
            "data_path": str(self.data_config.data_path.resolve()),
            "data_mtime_ns": self.data_config.data_path.stat().st_mtime_ns,
            "framework_mtime_ns": self.config.framework_config_path.stat().st_mtime_ns,
            "station_limit": self.config.station_limit,
            "station_offset": self.config.station_offset,
            "temporal_stride_hours": self.config.temporal_stride_hours,
            "max_train_rows": self.config.max_train_rows,
            "max_calibration_rows": self.config.max_calibration_rows,
            "max_evaluation_rows": self.config.max_evaluation_rows,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        cache_dir = self.config.output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"canonical_{digest}.pt"

    def _try_load_cached_prepared(self) -> PreparedExperimentData | None:
        """Load a cached canonical slice when available."""
        if not self.config.cache_prepared_data:
            return None
        cache_path = self._cache_path()
        if not cache_path.exists():
            return None
        payload = torch.load(cache_path, map_location="cpu")
        prepared = PreparedExperimentData(
            full=_batch_from_cache_dict(payload["full"]),
            train=_batch_from_cache_dict(payload["train"]),
            calibration=_batch_from_cache_dict(payload["calibration"]),
            evaluation=_batch_from_cache_dict(payload["evaluation"]),
            metadata_cardinalities=payload["metadata_cardinalities"],
            feature_schema=payload["feature_schema"],
        )
        self._resolved_pipeline_config = self.base_runner._resolve_pipeline_config(prepared)
        return prepared

    def _save_cached_prepared(self, prepared: PreparedExperimentData) -> None:
        """Persist the canonical slice for reuse in later runs."""
        if not self.config.cache_prepared_data:
            return
        cache_path = self._cache_path()
        torch.save(
            {
                "full": _batch_to_cache_dict(prepared.full),
                "train": _batch_to_cache_dict(prepared.train),
                "calibration": _batch_to_cache_dict(prepared.calibration),
                "evaluation": _batch_to_cache_dict(prepared.evaluation),
                "metadata_cardinalities": prepared.metadata_cardinalities,
                "feature_schema": prepared.feature_schema,
            },
            cache_path,
        )

    def prepare_canonical_data(self) -> PreparedExperimentData:
        """Prepare the fixed benchmark slice used by all reported experiments."""
        if self._canonical is not None:
            return self._canonical
        self._write_progress(status="running", stage="preparing_canonical_data")
        self._append_heartbeat(stage="preparing_canonical_data", step="start")
        cached = self._try_load_cached_prepared()
        if cached is not None:
            self._canonical = cached
            self._append_heartbeat(
                stage="preparing_canonical_data",
                step="cache_hit",
                payload={
                    "train_rows": int(cached.train.X.shape[0]),
                    "calibration_rows": int(cached.calibration.X.shape[0]),
                    "evaluation_rows": int(cached.evaluation.X.shape[0]),
                },
            )
            self._write_progress(
                status="running",
                stage="prepared_canonical_data",
                payload={
                    "train_rows": int(cached.train.X.shape[0]),
                    "calibration_rows": int(cached.calibration.X.shape[0]),
                    "evaluation_rows": int(cached.evaluation.X.shape[0]),
                    "cache_hit": True,
                },
            )
            return cached
        prepared = self.base_runner.prepare_data()
        self._prepared = prepared
        self._resolved_pipeline_config = self.base_runner._resolve_pipeline_config(prepared)

        selected_stations = self._select_station_instances(prepared.full)
        full = self._filter_batch(prepared.full, selected_stations, max_rows=None)
        train = self._filter_batch(prepared.train, selected_stations, max_rows=self.config.max_train_rows)
        calibration = self._filter_batch(
            prepared.calibration,
            selected_stations,
            max_rows=self.config.max_calibration_rows,
        )
        evaluation = self._filter_batch(
            prepared.evaluation,
            selected_stations,
            max_rows=self.config.max_evaluation_rows,
        )
        self._canonical = PreparedExperimentData(
            full=full,
            train=train,
            calibration=calibration,
            evaluation=evaluation,
            metadata_cardinalities=self._metadata_cardinalities(full.sensor_metadata),
            feature_schema=prepared.feature_schema,
        )
        self._save_cached_prepared(self._canonical)
        self._append_heartbeat(
            stage="preparing_canonical_data",
            step="prepared",
            payload={
                "train_rows": int(train.X.shape[0]),
                "calibration_rows": int(calibration.X.shape[0]),
                "evaluation_rows": int(evaluation.X.shape[0]),
                "cache_hit": False,
            },
        )
        self._write_progress(
            status="running",
            stage="prepared_canonical_data",
            payload={
                "train_rows": int(train.X.shape[0]),
                "calibration_rows": int(calibration.X.shape[0]),
                "evaluation_rows": int(evaluation.X.shape[0]),
                "cache_hit": False,
            },
        )
        return self._canonical

    def _select_station_instances(self, batch: TensorBatch) -> Tensor:
        """Choose a heterogeneous subset of station instances for the benchmark."""
        sensor_instance = batch.sensor_metadata.sensor_instance
        if sensor_instance is None:
            raise ValueError("Benchmarking requires station identifiers encoded as sensor_instance.")
        unique_station_ids = torch.unique(sensor_instance, sorted=True)
        if unique_station_ids.numel() <= self.config.station_limit:
            return unique_station_ids
        start = min(self.config.station_offset, max(int(unique_station_ids.numel()) - self.config.station_limit, 0))
        if batch.sensor_metadata.sensor_type is None:
            return unique_station_ids[start : start + self.config.station_limit]

        selected: list[int] = []
        by_type: dict[int, list[int]] = {}
        for station_id in unique_station_ids.tolist():
            station_mask = sensor_instance == station_id
            first_index = int(station_mask.nonzero(as_tuple=False).squeeze(-1)[0].item())
            sensor_type = int(batch.sensor_metadata.sensor_type[first_index].item())
            by_type.setdefault(sensor_type, []).append(station_id)
        ordered_types = sorted(by_type)
        type_offset = {key: 0 for key in ordered_types}
        while len(selected) < self.config.station_limit:
            progress = False
            for sensor_type in ordered_types:
                candidates = by_type[sensor_type]
                pointer = type_offset[sensor_type]
                if pointer >= len(candidates):
                    continue
                station_id = candidates[pointer]
                type_offset[sensor_type] += 1
                if station_id < start:
                    continue
                selected.append(station_id)
                progress = True
                if len(selected) >= self.config.station_limit:
                    break
            if not progress:
                break
        if len(selected) < self.config.station_limit:
            fallback = unique_station_ids[start : start + self.config.station_limit].tolist()
            selected = fallback
        return torch.tensor(selected[: self.config.station_limit], device=batch.X.device, dtype=torch.long)

    def _filter_batch(
        self,
        batch: TensorBatch,
        selected_stations: Tensor,
        *,
        max_rows: int | None,
    ) -> TensorBatch:
        """Apply station filtering, temporal thinning, and row caps."""
        sensor_instance = batch.sensor_metadata.sensor_instance
        if sensor_instance is None:
            raise ValueError("Benchmarking requires sensor_instance metadata.")
        station_mask = torch.zeros_like(sensor_instance, dtype=torch.bool)
        for station_id in selected_stations:
            station_mask |= sensor_instance == station_id
        hour_key = torch.round(batch.X[:, 2]).to(dtype=torch.long)
        stride_mask = torch.remainder(hour_key, self.config.temporal_stride_hours) == 0
        keep_mask = station_mask & stride_mask
        indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            indices = station_mask.nonzero(as_tuple=False).squeeze(-1)
        if max_rows is not None and indices.numel() > max_rows:
            indices = indices[:max_rows]
        return _slice_batch(batch, indices)

    @staticmethod
    def _metadata_cardinalities(metadata: SensorMetadataBatch) -> dict[str, int]:
        """Recompute metadata cardinalities after filtering."""
        return {
            "sensor_type": 0 if metadata.sensor_type is None else int(torch.unique(metadata.sensor_type).numel()),
            "sensor_group": 0 if metadata.sensor_group is None else int(torch.unique(metadata.sensor_group).numel()),
            "sensor_modality": (
                0 if metadata.sensor_modality is None else int(torch.unique(metadata.sensor_modality).numel())
            ),
            "installation_environment": (
                0
                if metadata.installation_environment is None
                else int(torch.unique(metadata.installation_environment).numel())
            ),
            "maintenance_state": (
                0 if metadata.maintenance_state is None else int(torch.unique(metadata.maintenance_state).numel())
            ),
        }

    def _base_config(self) -> SilenceAwareIDSConfig:
        """Resolve the benchmark-adjusted base pipeline config."""
        prepared = self.prepare_canonical_data()
        assert self._resolved_pipeline_config is not None
        base = self._resolved_pipeline_config
        return replace(
            base,
            state=replace(base.state, inducing_points=min(base.state.inducing_points, self.config.inducing_points)),
            state_training=replace(
                base.state_training,
                epochs=self.config.state_epochs,
            ),
            missingness_training=replace(
                base.missingness_training,
                epochs=self.config.missingness_epochs,
            ),
            reliability=replace(
                base.reliability,
                epsilon=1.0 - self.config.target_coverage,
            ),
            missingness=replace(
                base.missingness,
                num_sensor_types=base.missingness.num_sensor_types or prepared.metadata_cardinalities["sensor_type"],
                num_sensor_groups=base.missingness.num_sensor_groups or prepared.metadata_cardinalities["sensor_group"],
                num_sensor_modalities=(
                    base.missingness.num_sensor_modalities or prepared.metadata_cardinalities["sensor_modality"]
                ),
                num_installation_environments=(
                    base.missingness.num_installation_environments
                    or prepared.metadata_cardinalities["installation_environment"]
                ),
                num_maintenance_states=(
                    base.missingness.num_maintenance_states or prepared.metadata_cardinalities["maintenance_state"]
                ),
            ),
            observation=replace(
                base.observation,
                num_sensor_types=base.observation.num_sensor_types or prepared.metadata_cardinalities["sensor_type"],
                num_sensor_groups=base.observation.num_sensor_groups or prepared.metadata_cardinalities["sensor_group"],
                num_sensor_modalities=(
                    base.observation.num_sensor_modalities or prepared.metadata_cardinalities["sensor_modality"]
                ),
                num_installation_environments=(
                    base.observation.num_installation_environments
                    or prepared.metadata_cardinalities["installation_environment"]
                ),
                num_maintenance_states=(
                    base.observation.num_maintenance_states or prepared.metadata_cardinalities["maintenance_state"]
                ),
            ),
        )

    def _variant_config(self, variant_name: str) -> SilenceAwareIDSConfig:
        """Construct one benchmark variant config from the base setup."""
        base = self._base_config()
        graph_reliability = replace(
            base.reliability,
            mode="graph_corel",
            graph_score_weight=min(base.reliability.graph_score_weight, 0.08),
            graph_covariance_weight=min(base.reliability.graph_covariance_weight, 0.2),
            graph_message_passing_steps=min(base.reliability.graph_message_passing_steps, 1),
            graph_min_quantile_factor=max(base.reliability.graph_min_quantile_factor, 0.95),
            adaptation_rate=max(base.reliability.adaptation_rate, 0.08),
        )
        relational_reliability = replace(
            base.reliability,
            mode="relational_adaptive",
            relational_neighbor_weight=min(base.reliability.relational_neighbor_weight, 0.1),
            adaptation_rate=max(base.reliability.adaptation_rate, 0.05),
        )
        ablation_variants = SilenceAwareIDS(base).build_ablation_configs()
        if variant_name in ablation_variants:
            variant = ablation_variants[variant_name]
            if variant_name in {"ppo_warmstart_baseline", "rollout_policy_baseline", "myopic_policy_baseline"}:
                variant = replace(variant, reliability=graph_reliability)
            if variant_name == "full_model":
                reliability_override = (
                    relational_reliability
                    if self.config.max_train_rows >= 8192
                    else graph_reliability
                )
                variant = replace(variant, reliability=reliability_override)
            if variant_name in {
                "variance_policy_baseline",
                "myopic_policy_baseline",
                "rollout_policy_baseline",
                "ppo_warmstart_baseline",
                "full_model",
            }:
                route_weight = 0.2
                selection_mode = variant.policy.selection_mode
                if variant_name == "ppo_warmstart_baseline":
                    route_weight = 0.6
                    selection_mode = "ratio"
                if variant_name == "full_model":
                    route_weight = 0.2
                    selection_mode = "budget"
                return replace(
                    variant,
                    policy=replace(
                        variant.policy,
                        selection_mode=selection_mode,
                        route_distance_weight=max(variant.policy.route_distance_weight, route_weight),
                    ),
                )
            return variant
        if variant_name == "no_conformal":
            return replace(base, use_m5=False)
        if variant_name == "split_conformal":
            return replace(base, use_m5=True, reliability=replace(base.reliability, mode="split"))
        if variant_name == "adaptive_conformal":
            return replace(base, use_m5=True, reliability=replace(base.reliability, mode="adaptive"))
        if variant_name == "relational_adaptive":
            return replace(base, use_m5=True, reliability=relational_reliability)
        if variant_name == "graph_corel":
            return replace(base, use_m5=True, reliability=graph_reliability)
        raise KeyError(f"Unknown benchmark variant: {variant_name}")

    def _fit_variant(
        self,
        variant_name: str,
        train_batch: TensorBatch,
        calibration_batch: TensorBatch,
    ) -> tuple[SilenceAwareIDS, float]:
        """Fit one pipeline variant and return the fitted model with elapsed seconds."""
        variant_seed = self.config.seed + int(hashlib.sha1(variant_name.encode("utf-8")).hexdigest()[:8], 16)
        torch.manual_seed(variant_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(variant_seed)
        variant = SilenceAwareIDS(self._variant_config(variant_name))
        fit_start = time.perf_counter()
        variant.fit(
            train_batch.X,
            train_batch.y,
            missing_indicator_train=train_batch.missing_indicator,
            context_train=train_batch.context,
            M_train=train_batch.M,
            S_train=train_batch.S,
            sensor_metadata_train=train_batch.sensor_metadata,
            X_cal=calibration_batch.X,
            y_cal=calibration_batch.y,
            context_cal=calibration_batch.context,
            M_cal=calibration_batch.M,
            S_cal=calibration_batch.S,
            sensor_metadata_cal=calibration_batch.sensor_metadata,
        )
        return variant, time.perf_counter() - fit_start

    def _predictive_state_summary(
        self,
        pipeline: SilenceAwareIDS,
        batch: TensorBatch,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Resolve predictive mean, variance, and optional missingness probabilities."""
        if pipeline.use_m3:
            return pipeline.missingness_aware_state_summary(
                batch.X,
                y=batch.y,
                context=batch.context,
                M=batch.M,
                S=batch.S,
                sensor_metadata=batch.sensor_metadata,
                batch_size=self.config.prediction_batch_size,
                include_observation_noise=True,
            )
        mu, var = pipeline.predict_state(
            batch.X,
            batch_size=self.config.prediction_batch_size,
            include_observation_noise=True,
        )
        return mu, var, None

    def _nominal_interval(self, mu: Tensor, var: Tensor) -> tuple[Tensor, Tensor]:
        """Construct Gaussian nominal intervals at the target coverage level."""
        alpha = (1.0 + self.config.target_coverage) / 2.0
        z_score = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(alpha, device=mu.device, dtype=mu.dtype))
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        return mu - z_score * std, mu + z_score * std

    def _coverage_metrics(self, y_true: Tensor, lower: Tensor, upper: Tensor) -> dict[str, float]:
        """Compute interval coverage and mean width on observed targets."""
        observed_mask = torch.isfinite(y_true)
        if not observed_mask.any():
            return {"coverage": float("nan"), "interval_width": float("nan")}
        width = upper[observed_mask] - lower[observed_mask]
        covered = (y_true[observed_mask] >= lower[observed_mask]) & (y_true[observed_mask] <= upper[observed_mask])
        return {
            "coverage": float(covered.float().mean().item()),
            "interval_width": float(width.mean().item()),
        }

    def _calibration_error(self, coverage: float) -> float:
        """Compute absolute target-coverage error."""
        return abs(coverage - self.config.target_coverage)

    def _shift_mask(self, evaluation: TensorBatch) -> Tensor:
        """Mark the latest evaluation timestamps as the regime-shift region."""
        unique_times = torch.unique(evaluation.X[:, 2], sorted=True)
        if unique_times.numel() == 0:
            return torch.zeros(evaluation.X.shape[0], dtype=torch.bool, device=evaluation.X.device)
        cutoff_index = max(0, int(math.floor(unique_times.numel() * (1.0 - self.config.shift_fraction))) - 1)
        cutoff = unique_times[cutoff_index]
        return evaluation.X[:, 2] >= cutoff

    def _policy_budget(self, candidate_cost: Tensor | None) -> float | None:
        """Infer a benchmark policy budget when one is not explicitly configured."""
        if candidate_cost is None:
            return None
        if self.config.policy_budget is not None:
            return float(self.config.policy_budget)
        if candidate_cost.numel() == 0:
            return None
        median_cost = float(torch.median(candidate_cost).item())
        return max(median_cost * float(self.config.max_selections), median_cost)

    def _selection_surrogate_gain(
        self,
        selection: dict[str, Any] | None,
        candidate_variance: Tensor,
    ) -> float:
        """Estimate uncertainty reduction using selected candidate variances."""
        if selection is None:
            return 0.0
        selected = selection.get("selected_indices", [])
        if not selected:
            return 0.0
        index_tensor = torch.tensor(selected, device=candidate_variance.device, dtype=torch.long)
        return float(candidate_variance[index_tensor].sum().item())

    def _route_distance(self, selected_x: Tensor) -> float:
        """Compute a simple routing length over selected physical coordinates."""
        if selected_x.shape[0] <= 1:
            return 0.0
        diffs = selected_x[1:, :2] - selected_x[:-1, :2]
        return float(torch.sqrt(torch.clamp(diffs.pow(2).sum(dim=-1), min=1e-6)).sum().item())

    def _random_selection(
        self,
        candidate_cost: Tensor,
        *,
        budget: float | None,
        max_selections: int,
    ) -> dict[str, Any]:
        """Randomly select candidates under a budget for the random policy baseline."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed + 17001)
            permutation = torch.randperm(candidate_cost.shape[0], device=candidate_cost.device)
        selected: list[int] = []
        total_cost = 0.0
        for index in permutation.tolist():
            cost = float(candidate_cost[index].item())
            if budget is not None and total_cost + cost > budget:
                continue
            selected.append(index)
            total_cost += cost
            if len(selected) >= max_selections:
                break
        return {
            "selected_indices": selected,
            "total_cost": total_cost,
        }

    def _missingness_probability(self, batch: TensorBatch, mechanism: str, intensity: float) -> Tensor:
        """Construct synthetic structural-missingness probabilities for Table 1."""
        observed_mask = torch.isfinite(batch.y)
        probability = torch.full_like(batch.y, fill_value=float(intensity))
        if mechanism == "mar":
            probability[~observed_mask] = 1.0
            return torch.clamp(probability, min=1e-3, max=0.999)

        y_filled = torch.nan_to_num(batch.y, nan=0.0)
        if mechanism == "value_dependent_mnar":
            signal = torch.abs(_standardize(y_filled))
        elif mechanism == "state_dependent_mnar":
            signal = _temporal_station_difference(y_filled, batch.sensor_metadata.sensor_instance)
            if batch.context is not None:
                signal = signal + 0.5 * _standardize(batch.context.norm(p=2, dim=-1))
            signal = _standardize(signal)
        else:
            raise ValueError(f"Unknown missingness mechanism: {mechanism}")
        probability = torch.sigmoid(signal)
        mean_probability = (
            float(torch.clamp(probability[observed_mask].mean(), min=1e-6).item())
            if observed_mask.any()
            else 1.0
        )
        probability = probability * (float(intensity) / mean_probability)
        probability[~observed_mask] = 1.0
        return torch.clamp(probability, min=1e-3, max=0.999)

    def _apply_missingness_mechanism(
        self,
        batch: TensorBatch,
        mechanism: str,
        intensity: float,
        *,
        seed_offset: int,
    ) -> tuple[TensorBatch, Tensor, Tensor]:
        """Sample synthetic missingness and return a masked training batch."""
        probability = self._missingness_probability(batch, mechanism, intensity)
        observed_mask = torch.isfinite(batch.y)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed + seed_offset)
            sampled_mask = torch.rand_like(probability) < probability
        missing_mask = observed_mask & sampled_mask
        masked_y = batch.y.clone()
        masked_y[missing_mask] = float("nan")
        missing_indicator = batch.missing_indicator.clone()
        missing_indicator[missing_mask] = 1.0
        masked_batch = TensorBatch(
            X=batch.X,
            y=masked_y,
            context=batch.context,
            M=missing_indicator,
            S=batch.S,
            missing_indicator=missing_indicator,
            cost=batch.cost,
            sensor_metadata=batch.sensor_metadata,
            indices=batch.indices,
        )
        return masked_batch, missing_mask, probability

    def _fault_batch(
        self,
        batch: TensorBatch,
        scenario: str,
        *,
        seed_offset: int,
        event_context_index: int | None = None,
        event_information_index: int | None = None,
    ) -> tuple[TensorBatch, Tensor, Tensor]:
        """Inject one synthetic sensor-fault scenario into an evaluation batch."""
        y = batch.y.clone()
        labels = torch.zeros_like(y, dtype=torch.bool)
        event_ids = torch.full_like(batch.indices, fill_value=-1, dtype=torch.long)
        observed = torch.isfinite(y)
        if not observed.any():
            return batch, labels, event_ids
        y_mean = y[observed].mean()
        y_std = torch.clamp(y[observed].std(unbiased=False), min=1.0)
        sensor_instance = batch.sensor_metadata.sensor_instance
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed + seed_offset)
            event_counter = 0
            if scenario == "random_dropout":
                candidate = observed.nonzero(as_tuple=False).squeeze(-1)
                count = max(1, int(candidate.numel() * 0.12))
                selected = candidate[torch.randperm(candidate.numel(), device=candidate.device)[:count]]
                y[selected] = y_mean - 6.0 * y_std
                labels[selected] = True
                event_ids[selected] = torch.arange(count, device=event_ids.device, dtype=torch.long)
            elif scenario in {"block_missingness", "frozen_sensor", "drift", "spike_burst"}:
                if sensor_instance is None:
                    sensor_groups = [observed.nonzero(as_tuple=False).squeeze(-1)]
                else:
                    sensor_groups = [
                        ((sensor_instance == station_id) & observed).nonzero(as_tuple=False).squeeze(-1)
                        for station_id in torch.unique(sensor_instance)
                    ]
                for station_rows in sensor_groups:
                    if station_rows.numel() < 3:
                        continue
                    block_length = 3 if scenario != "drift" else min(5, int(station_rows.numel()))
                    start_limit = max(1, int(station_rows.numel()) - block_length + 1)
                    start = int(torch.randint(0, start_limit, (1,), device=station_rows.device).item())
                    block = station_rows[start : start + block_length]
                    labels[block] = True
                    event_ids[block] = event_counter
                    event_counter += 1
                    if scenario == "block_missingness":
                        baseline = y[block[0] - 1].clone() if start > 0 else (y_mean - 5.0 * y_std)
                        y[block] = baseline
                    elif scenario == "frozen_sensor":
                        y[block] = y[block[0]].clone()
                    elif scenario == "drift":
                        ramp = torch.linspace(0.0, 3.0, block.shape[0], device=y.device, dtype=y.dtype)
                        y[block] = y[block].clone() + ramp * y_std
                    elif scenario == "spike_burst":
                        signs = torch.where(
                            torch.rand(block.shape[0], device=y.device) > 0.5,
                            1.0,
                            -1.0,
                        )
                        y[block] = y[block].clone() + signs * 4.0 * y_std
            elif scenario == "extreme_event_confound":
                event_mask = torch.zeros_like(observed)
                if batch.context is not None and event_context_index is not None:
                    event_mask = event_mask | ((batch.context[:, event_context_index] > 0.0) & observed)
                if batch.context is not None and event_information_index is not None:
                    event_mask = event_mask | ((batch.context[:, event_information_index] > 0.0) & observed)
                if not event_mask.any():
                    unique_times = torch.unique(batch.X[:, 2], sorted=True)
                    if unique_times.numel() > 0:
                        event_time = unique_times[max(0, int(unique_times.numel() * 0.8) - 1)]
                        event_mask = (batch.X[:, 2] == event_time) & observed
                y[event_mask] = y[event_mask].clone() + 2.5 * y_std
            else:
                raise ValueError(f"Unknown fault scenario: {scenario}")

        faulty_batch = TensorBatch(
            X=batch.X,
            y=y,
            context=batch.context,
            M=batch.M,
            S=batch.S,
            missing_indicator=batch.missing_indicator,
            cost=batch.cost,
            sensor_metadata=batch.sensor_metadata,
            indices=batch.indices,
        )
        return faulty_batch, labels, event_ids

    @staticmethod
    def _auroc(labels: Tensor, scores: Tensor) -> float | None:
        """Compute AUROC from binary labels and continuous scores without sklearn."""
        positives = labels.bool()
        negatives = ~positives
        n_pos = int(positives.sum().item())
        n_neg = int(negatives.sum().item())
        if n_pos == 0 or n_neg == 0:
            return None
        order = torch.argsort(scores, stable=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, order.numel() + 1, device=order.device, dtype=torch.float32)
        rank_sum = float(ranks[positives].sum().item())
        return (rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)

    @staticmethod
    def _detection_delay(event_ids: Tensor, flags: Tensor) -> float | None:
        """Compute average detection delay across fault events."""
        valid_events = torch.unique(event_ids[event_ids >= 0])
        if valid_events.numel() == 0:
            return None
        delays: list[float] = []
        for event_id in valid_events.tolist():
            event_index = (event_ids == event_id).nonzero(as_tuple=False).squeeze(-1)
            if event_index.numel() == 0:
                continue
            first_event = int(event_index[0].item())
            flagged = event_index[flags[event_index]]
            if flagged.numel() == 0:
                delays.append(float(event_index.numel()))
            else:
                delays.append(float(int(flagged[0].item()) - first_event))
        return sum(delays) / max(len(delays), 1)

    def _fault_variant_config(self, variant_name: str) -> SilenceAwareIDSConfig:
        """Construct observation-only diagnosis benchmark variants."""
        base = self._base_config()
        sequential = replace(base.state_training, training_strategy="sequential")
        stripped = replace(base, use_m2=True, use_m3=False, use_m5=False, state_training=sequential)
        advanced_mode = "temporal_nwp" if stripped.observation.context_dim > 0 else "temporal"
        anchor_index = (
            stripped.observation.nwp_context_index
            if stripped.observation.nwp_context_index is not None
            else (0 if stripped.observation.context_dim > 0 else None)
        )
        if variant_name == "pointwise_threshold_baseline":
            return replace(
                stripped,
                observation=replace(
                    stripped.observation,
                    diagnosis_mode="pointwise",
                    use_dbn=False,
                    use_pi_ssd=False,
                    use_latent_ode=False,
                    use_fault_head=False,
                ),
            )
        if variant_name == "temporal_residual_baseline":
            return replace(
                stripped,
                observation=replace(
                    stripped.observation,
                    diagnosis_mode="temporal",
                    use_dbn=False,
                    use_pi_ssd=False,
                    use_latent_ode=False,
                    use_fault_head=False,
                ),
            )
        if variant_name == "dbn_lite":
            return replace(
                stripped,
                observation=replace(
                    stripped.observation,
                    diagnosis_mode=advanced_mode,
                    use_dbn=True,
                    use_pi_ssd=False,
                    use_latent_ode=False,
                    use_fault_head=True,
                    nwp_context_index=anchor_index,
                    nwp_anchor_weight=0.75 if anchor_index is not None else stripped.observation.nwp_anchor_weight,
                    link_fit_steps=max(40, stripped.observation.link_fit_steps),
                    link_learning_rate=0.03,
                    fault_self_supervised_weight=0.4,
                    fault_corruption_probability=0.25,
                    fault_score_state_weight=0.2,
                    fault_score_temporal_weight=0.95,
                    fault_score_persistence_weight=0.45,
                    fault_score_probability_weight=0.0,
                    fault_target_false_alarm_rate=0.02,
                ),
            )
        if variant_name == "pi_ssd_only":
            return replace(
                stripped,
                observation=replace(
                    stripped.observation,
                    diagnosis_mode=advanced_mode,
                    use_dbn=False,
                    use_pi_ssd=True,
                    use_latent_ode=True,
                    use_fault_head=True,
                    nwp_context_index=anchor_index,
                    nwp_anchor_weight=0.75 if anchor_index is not None else stripped.observation.nwp_anchor_weight,
                    link_fit_steps=max(60, stripped.observation.link_fit_steps),
                    link_learning_rate=0.03,
                    fault_self_supervised_weight=0.5,
                    fault_corruption_probability=0.3,
                    fault_score_embedding_weight=0.02,
                    fault_score_temporal_weight=1.0,
                    fault_score_persistence_weight=0.45,
                    fault_score_probability_weight=0.0,
                    fault_target_false_alarm_rate=0.02,
                ),
            )
        if variant_name == "dbn_plus_pi_ssd":
            return replace(
                stripped,
                observation=replace(
                    stripped.observation,
                    diagnosis_mode=advanced_mode,
                    use_dbn=True,
                    use_pi_ssd=True,
                    use_latent_ode=True,
                    use_fault_head=True,
                    nwp_context_index=anchor_index,
                    nwp_anchor_weight=1.0 if anchor_index is not None else stripped.observation.nwp_anchor_weight,
                    link_fit_steps=max(80, stripped.observation.link_fit_steps),
                    link_learning_rate=0.025,
                    fault_self_supervised_weight=0.6,
                    fault_corruption_probability=0.3,
                    fault_score_state_weight=0.15,
                    fault_score_embedding_weight=0.03,
                    fault_score_temporal_weight=1.05,
                    fault_score_persistence_weight=0.5,
                    fault_score_probability_weight=0.0,
                    fault_target_false_alarm_rate=0.02,
                ),
            )
        raise KeyError(f"Unknown fault diagnosis variant: {variant_name}")

    def _fault_rows(self, prepared: PreparedExperimentData) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run the fault diagnosis benchmark used for Table 2."""
        rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        event_context_index = _context_index(prepared, "event_station_active")
        if event_context_index is None:
            event_context_index = _context_index(prepared, "event_warning_active")
        event_information_index = _context_index(prepared, "event_station_count")
        if event_information_index is None:
            event_information_index = _context_index(prepared, "event_information_count")
        variants = (
            "pointwise_threshold_baseline",
            "temporal_residual_baseline",
            "dbn_lite",
            "pi_ssd_only",
            "dbn_plus_pi_ssd",
        )
        fitted: dict[str, tuple[SilenceAwareIDS, float]] = {}
        for variant_name in variants:
            pipeline = SilenceAwareIDS(self._fault_variant_config(variant_name))
            fit_start = time.perf_counter()
            pipeline.fit(
                prepared.train.X,
                prepared.train.y,
                missing_indicator_train=prepared.train.missing_indicator,
                context_train=prepared.train.context,
                M_train=prepared.train.M,
                S_train=prepared.train.S,
                sensor_metadata_train=prepared.train.sensor_metadata,
                X_cal=prepared.calibration.X,
                y_cal=prepared.calibration.y,
                context_cal=prepared.calibration.context,
                M_cal=prepared.calibration.M,
                S_cal=prepared.calibration.S,
                sensor_metadata_cal=prepared.calibration.sensor_metadata,
            )
            clean_threshold = pipeline.calibrate_fault_detection(
                prepared.calibration.X,
                prepared.calibration.y,
                context=prepared.calibration.context,
                sensor_metadata=prepared.calibration.sensor_metadata,
                batch_size=self.config.prediction_batch_size,
            )
            if variant_name in {"dbn_lite", "pi_ssd_only", "dbn_plus_pi_ssd"}:
                calibration_scores: list[Tensor] = []
                calibration_labels: list[Tensor] = []
                for scenario_index, scenario in enumerate(self.config.fault_scenarios):
                    faulty_calibration, labels_calibration, _ = self._fault_batch(
                        prepared.calibration,
                        scenario,
                        seed_offset=4000 + scenario_index,
                        event_context_index=event_context_index,
                        event_information_index=event_information_index,
                    )
                    diagnosis = pipeline.detect_faults(
                        faulty_calibration.X,
                        faulty_calibration.y,
                        context=faulty_calibration.context,
                        sensor_metadata=faulty_calibration.sensor_metadata,
                        batch_size=self.config.prediction_batch_size,
                    )
                    available = diagnosis["available"].bool()
                    calibration_scores.append(diagnosis["fault_score"][available])
                    calibration_labels.append(labels_calibration[available].float())
                if calibration_scores:
                    tuned_threshold = pipeline.tune_fault_threshold(
                        torch.cat(calibration_scores, dim=0),
                        torch.cat(calibration_labels, dim=0),
                        target_false_alarm_rate=pipeline.config.observation.fault_target_false_alarm_rate,
                        far_penalty_weight=1.25,
                    )
                    pipeline._fault_detection_threshold = float(
                        torch.maximum(clean_threshold, tuned_threshold).item()
                    )
            fitted[variant_name] = (pipeline, time.perf_counter() - fit_start)

        for scenario_index, scenario in enumerate(self.config.fault_scenarios):
            faulty_batch, labels, event_ids = self._fault_batch(
                prepared.evaluation,
                scenario,
                seed_offset=2000 + scenario_index,
                event_context_index=event_context_index,
                event_information_index=event_information_index,
            )
            labels_float = labels.float()
            for variant_name, (pipeline, fit_seconds) in fitted.items():
                detect_start = time.perf_counter()
                diagnosis = pipeline.detect_faults(
                    faulty_batch.X,
                    faulty_batch.y,
                    context=faulty_batch.context,
                    sensor_metadata=faulty_batch.sensor_metadata,
                    batch_size=self.config.prediction_batch_size,
                )
                detect_seconds = time.perf_counter() - detect_start
                flags = diagnosis["fault_flag"].bool()
                scores = diagnosis["fault_score"].float()
                tp = float((flags & labels).float().sum().item())
                fp = float((flags & ~labels).float().sum().item())
                fn = float((~flags & labels).float().sum().item())
                precision = tp / max(tp + fp, 1.0)
                recall = tp / max(tp + fn, 1.0) if labels.any() else None
                f1 = (
                    2.0 * precision * recall / max(precision + recall, 1e-6)
                    if recall is not None
                    else None
                )
                false_alarm_rate = float((flags & ~labels).float().sum().item()) / max(
                    float((~labels).float().sum().item()),
                    1.0,
                )
                rows.append(
                    {
                        "variant_name": variant_name,
                        "fault_scenario": scenario,
                        "f1": _format_float(f1),
                        "precision": _format_float(precision),
                        "recall": _format_float(recall),
                        "auroc": _format_float(self._auroc(labels, scores)),
                        "detection_delay": _format_float(self._detection_delay(event_ids, flags)),
                        "false_alarm_rate": _format_float(false_alarm_rate),
                        "fault_fraction": _format_float(float(labels_float.mean().item())),
                    }
                )
                runtime_rows.append(
                    {
                        "section": "fault_diagnosis",
                        "variant_name": variant_name,
                        "scenario": scenario,
                        "fit_seconds": _format_float(fit_seconds),
                        "predict_seconds": _format_float(detect_seconds),
                        "selection_seconds": "",
                    }
                )
        return rows, runtime_rows

    def _predictive_rows(self, prepared: PreparedExperimentData) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run the core JVI-vs-plug-in benchmark under MAR and MNAR."""
        rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        predictive_variants = {
            "gp_only": "base_gp_only",
            "homogeneous_missingness": "gp_plus_homogeneous_missingness",
            "sensor_conditional_plugin_missingness": "gp_plus_sensor_conditional_missingness",
            "joint_variational_missingness": "gp_plus_joint_variational_missingness",
            "joint_generative_missingness": "gp_plus_joint_generative_missingness",
            "full_joint_jvi_training": "gp_plus_joint_generative_jvi_training",
        }
        for mechanism_index, mechanism in enumerate(self.config.predictive_mechanisms):
            for intensity_index, intensity in enumerate(self.config.missing_intensities):
                self._append_heartbeat(
                    stage="predictive_rows",
                    step="scenario_start",
                    repeat_seed=self.config.seed,
                    payload={"mechanism": mechanism, "missing_intensity": float(intensity)},
                )
                train_batch, _, _ = self._apply_missingness_mechanism(
                    prepared.train,
                    mechanism,
                    intensity,
                    seed_offset=100 * mechanism_index + intensity_index,
                )
                _, eval_gap_mask, eval_probability = self._apply_missingness_mechanism(
                    prepared.evaluation,
                    mechanism,
                    intensity,
                    seed_offset=1000 + 100 * mechanism_index + intensity_index,
                )
                gap_indices = eval_gap_mask.nonzero(as_tuple=False).squeeze(-1)
                gap_batch = _slice_batch(prepared.evaluation, gap_indices) if gap_indices.numel() > 0 else None
                for label, variant_name in predictive_variants.items():
                    self._append_heartbeat(
                        stage="predictive_rows",
                        step="variant_fit_start",
                        repeat_seed=self.config.seed,
                        payload={
                            "label": label,
                            "variant_name": variant_name,
                            "mechanism": mechanism,
                            "missing_intensity": float(intensity),
                        },
                    )
                    pipeline, fit_seconds = self._fit_variant(variant_name, train_batch, prepared.calibration)
                    self._append_heartbeat(
                        stage="predictive_rows",
                        step="variant_fit_complete",
                        repeat_seed=self.config.seed,
                        payload={
                            "label": label,
                            "variant_name": variant_name,
                            "mechanism": mechanism,
                            "missing_intensity": float(intensity),
                            "fit_seconds": float(fit_seconds),
                        },
                    )
                    metrics = pipeline.evaluate_predictions(
                        prepared.evaluation.X,
                        prepared.evaluation.y,
                        context=prepared.evaluation.context,
                        M=prepared.evaluation.M,
                        S=prepared.evaluation.S,
                        sensor_metadata=prepared.evaluation.sensor_metadata,
                        integrate_missingness=pipeline.use_m3,
                        batch_size=self.config.prediction_batch_size,
                    )
                    gap_metrics: dict[str, float] = {}
                    if gap_batch is not None and gap_batch.X.shape[0] > 0:
                        gap_metrics = pipeline.evaluate_predictions(
                            gap_batch.X,
                            gap_batch.y,
                            context=gap_batch.context,
                            M=gap_batch.M,
                            S=gap_batch.S,
                            sensor_metadata=gap_batch.sensor_metadata,
                            integrate_missingness=pipeline.use_m3,
                            batch_size=self.config.prediction_batch_size,
                        )
                    mu, var, p_miss = self._predictive_state_summary(pipeline, prepared.evaluation)
                    lower, upper = self._nominal_interval(mu, var)
                    coverage_metrics = self._coverage_metrics(prepared.evaluation.y, lower, upper)
                    rows.append(
                        {
                            "variant_name": label,
                            "pipeline_variant": variant_name,
                            "mechanism": mechanism,
                            "missing_intensity": _format_float(float(intensity)),
                            "rmse": _format_float(metrics.get("rmse")),
                            "mae": _format_float(metrics.get("mae")),
                            "crps": _format_float(metrics.get("crps")),
                            "log_score": _format_float(metrics.get("log_score")),
                            "gaussian_coverage": _format_float(coverage_metrics.get("coverage")),
                            "calibration_error": _format_float(
                                self._calibration_error(float(coverage_metrics["coverage"]))
                            ),
                            "gap_rmse": _format_float(gap_metrics.get("rmse")),
                            "gap_crps": _format_float(gap_metrics.get("crps")),
                            "gap_fraction": _format_float(float(eval_gap_mask.float().mean().item())),
                            "mean_synthetic_missingness_probability": _format_float(
                                float(eval_probability[torch.isfinite(prepared.evaluation.y)].mean().item())
                            ),
                            "mean_predicted_missingness_probability": _format_float(
                                float(p_miss.mean().item()) if p_miss is not None else 0.0
                            ),
                        }
                    )
                    self._append_heartbeat(
                        stage="predictive_rows",
                        step="variant_eval_complete",
                        repeat_seed=self.config.seed,
                        payload={
                            "label": label,
                            "variant_name": variant_name,
                            "mechanism": mechanism,
                            "missing_intensity": float(intensity),
                            "crps": metrics.get("crps"),
                            "gap_crps": gap_metrics.get("crps"),
                        },
                    )
                    runtime_rows.append(
                        {
                            "section": "predictive_missingness",
                            "variant_name": label,
                            "scenario": f"{mechanism}_{intensity:.2f}",
                            "fit_seconds": _format_float(fit_seconds),
                            "predict_seconds": "",
                            "selection_seconds": "",
                        }
                    )
        return rows, runtime_rows

    def _coverage_timeseries_rows(
        self,
        variant_name: str,
        evaluation: TensorBatch,
        lower: Tensor,
        upper: Tensor,
        shift_mask: Tensor,
    ) -> list[dict[str, Any]]:
        """Aggregate coverage and width by evaluation timestamp."""
        rows: list[dict[str, Any]] = []
        unique_times = torch.unique(evaluation.X[:, 2], sorted=True)
        for timestamp in unique_times.tolist():
            mask = evaluation.X[:, 2] == timestamp
            observed = mask & torch.isfinite(evaluation.y)
            if not observed.any():
                continue
            covered = (
                (evaluation.y[observed] >= lower[observed]) & (evaluation.y[observed] <= upper[observed])
            ).float()
            rows.append(
                {
                    "variant_name": variant_name,
                    "elapsed_hours": _format_float(float(timestamp)),
                    "coverage": _format_float(float(covered.mean().item())),
                    "interval_width": _format_float(
                        float((upper[observed] - lower[observed]).mean().item())
                    ),
                    "is_shift_region": int(bool(shift_mask[mask].any().item())),
                }
            )
        return rows

    def _coverage_recovery_speed(self, rows: list[dict[str, Any]]) -> float | None:
        """Estimate how quickly coverage recovers after the shift begins."""
        shift_rows = [row for row in rows if int(row["is_shift_region"]) == 1]
        if not shift_rows:
            return None
        shift_start = float(shift_rows[0]["elapsed_hours"])
        for row in shift_rows:
            if abs(float(row["coverage"]) - self.config.target_coverage) <= 0.05:
                return float(row["elapsed_hours"]) - shift_start
        return None

    def _reliability_rows(
        self,
        prepared: PreparedExperimentData,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Run chronological coverage-stress experiments for conformal methods."""
        rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        reliability_variants = {
            "no_conformal": "no_conformal",
            "split_conformal": "split_conformal",
            "adaptive_conformal": "adaptive_conformal",
            "relational_adaptive": "relational_adaptive",
            "graph_corel": "graph_corel",
            "full_model": "full_model",
        }
        shift_mask = self._shift_mask(prepared.evaluation)
        shift_indices = shift_mask.nonzero(as_tuple=False).squeeze(-1)
        stable_indices = (~shift_mask).nonzero(as_tuple=False).squeeze(-1)
        shift_batch = _slice_batch(prepared.evaluation, shift_indices) if shift_indices.numel() > 0 else None
        stable_batch = _slice_batch(prepared.evaluation, stable_indices) if stable_indices.numel() > 0 else None

        for label, variant_name in reliability_variants.items():
            self._append_heartbeat(
                stage="reliability_rows",
                step="variant_fit_start",
                repeat_seed=self.config.seed,
                payload={"label": label, "variant_name": variant_name},
            )
            pipeline, fit_seconds = self._fit_variant(variant_name, prepared.train, prepared.calibration)
            self._append_heartbeat(
                stage="reliability_rows",
                step="variant_fit_complete",
                repeat_seed=self.config.seed,
                payload={"label": label, "variant_name": variant_name, "fit_seconds": float(fit_seconds)},
            )
            predict_start = time.perf_counter()
            if pipeline.use_m5:
                lower, upper, metadata = pipeline.predict_interval(
                    prepared.evaluation.X,
                    y=prepared.evaluation.y,
                    M=prepared.evaluation.M,
                    S=prepared.evaluation.S,
                    y_true=prepared.evaluation.y,
                    context=prepared.evaluation.context,
                    sensor_metadata=prepared.evaluation.sensor_metadata,
                    batch_size=self.config.prediction_batch_size,
                )
            else:
                mu, var, _ = self._predictive_state_summary(pipeline, prepared.evaluation)
                lower, upper = self._nominal_interval(mu, var)
                metadata = {}
            predict_seconds = time.perf_counter() - predict_start
            self._append_heartbeat(
                stage="reliability_rows",
                step="interval_complete",
                repeat_seed=self.config.seed,
                payload={"label": label, "variant_name": variant_name, "predict_seconds": float(predict_seconds)},
            )
            overall = self._coverage_metrics(prepared.evaluation.y, lower, upper)
            stable = (
                self._coverage_metrics(stable_batch.y, lower[stable_indices], upper[stable_indices])
                if stable_batch is not None
                else {"coverage": float("nan"), "interval_width": float("nan")}
            )
            shift = (
                self._coverage_metrics(shift_batch.y, lower[shift_indices], upper[shift_indices])
                if shift_batch is not None
                else {"coverage": float("nan"), "interval_width": float("nan")}
            )
            metrics = pipeline.evaluate_predictions(
                prepared.evaluation.X,
                prepared.evaluation.y,
                context=prepared.evaluation.context,
                M=prepared.evaluation.M,
                S=prepared.evaluation.S,
                sensor_metadata=prepared.evaluation.sensor_metadata,
                integrate_missingness=pipeline.use_m3,
                batch_size=self.config.prediction_batch_size,
            )
            time_rows = self._coverage_timeseries_rows(label, prepared.evaluation, lower, upper, shift_mask)
            coverage_rows.extend(time_rows)
            rows.append(
                {
                    "variant_name": label,
                    "coverage": _format_float(overall["coverage"]),
                    "target_coverage_error": _format_float(self._calibration_error(float(overall["coverage"]))),
                    "interval_width": _format_float(overall["interval_width"]),
                    "stable_coverage": _format_float(stable["coverage"]),
                    "shift_coverage": _format_float(shift["coverage"]),
                    "stable_interval_width": _format_float(stable["interval_width"]),
                    "shift_interval_width": _format_float(shift["interval_width"]),
                    "crps": _format_float(metrics.get("crps")),
                    "log_score": _format_float(metrics.get("log_score")),
                    "recovery_speed_hours": _format_float(self._coverage_recovery_speed(time_rows)),
                    "final_adaptive_epsilon": _format_float(metadata.get("final_epsilon")),
                    "mean_graph_quantile": _format_float(metadata.get("mean_graph_quantile")),
                }
            )
            self._append_heartbeat(
                stage="reliability_rows",
                step="variant_eval_complete",
                repeat_seed=self.config.seed,
                payload={
                    "label": label,
                    "variant_name": variant_name,
                    "coverage": overall["coverage"],
                    "target_coverage_error": self._calibration_error(float(overall["coverage"])),
                },
            )
            runtime_rows.append(
                {
                    "section": "reliability_shift",
                    "variant_name": label,
                    "scenario": "chronological_shift",
                    "fit_seconds": _format_float(fit_seconds),
                    "predict_seconds": _format_float(predict_seconds),
                    "selection_seconds": "",
                }
            )
        return rows, coverage_rows, runtime_rows

    def _ablation_rows(
        self,
        prepared: PreparedExperimentData,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run the compact full-model ablation table."""
        rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        variants = (
            "base_gp_only",
            "gp_plus_dynamic_silence",
            "gp_plus_sensor_conditional_missingness",
            "gp_plus_joint_generative_jvi_training",
            "gp_plus_conformal_reliability",
            "myopic_policy_baseline",
            "full_model",
        )
        budget = self._policy_budget(prepared.evaluation.cost)
        for variant_name in variants:
            pipeline, fit_seconds = self._fit_variant(variant_name, prepared.train, prepared.calibration)
            metrics = pipeline.evaluate_predictions(
                prepared.evaluation.X,
                prepared.evaluation.y,
                context=prepared.evaluation.context,
                M=prepared.evaluation.M,
                S=prepared.evaluation.S,
                sensor_metadata=prepared.evaluation.sensor_metadata,
                integrate_missingness=pipeline.use_m3,
                batch_size=self.config.prediction_batch_size,
            )
            selection = None
            selection_seconds = None
            if prepared.evaluation.cost is not None:
                selection_start = time.perf_counter()
                selection = pipeline.select_sensors(
                    prepared.evaluation.X,
                    prepared.evaluation.cost,
                    context=prepared.evaluation.context,
                    M=prepared.evaluation.M,
                    S=prepared.evaluation.S,
                    sensor_metadata=prepared.evaluation.sensor_metadata,
                    budget=budget,
                    max_selections=self.config.max_selections,
                    batch_size=self.config.prediction_batch_size,
                )
                selection_seconds = time.perf_counter() - selection_start
            rows.append(
                {
                    "variant_name": variant_name,
                    "rmse": _format_float(metrics.get("rmse")),
                    "mae": _format_float(metrics.get("mae")),
                    "crps": _format_float(metrics.get("crps")),
                    "log_score": _format_float(metrics.get("log_score")),
                    "coverage": _format_float(metrics.get("coverage")),
                    "interval_width": _format_float(metrics.get("interval_width")),
                    "mean_missingness_proba": _format_float(metrics.get("mean_missingness_proba")),
                    "total_cost": _format_float(
                        None
                        if selection is None
                        else float(selection.get("total_operational_cost", selection.get("total_cost", 0.0)))
                    ),
                    "num_selected": 0 if selection is None else len(selection.get("selected_indices", [])),
                }
            )
            runtime_rows.append(
                {
                    "section": "ablation",
                    "variant_name": variant_name,
                    "scenario": "canonical",
                    "fit_seconds": _format_float(fit_seconds),
                    "predict_seconds": "",
                    "selection_seconds": _format_float(selection_seconds),
                }
            )
        return rows, runtime_rows

    def _region_holdout_rows(
        self,
        prepared: PreparedExperimentData,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Evaluate coarse region-holdout generalization for key predictive variants."""
        rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        if not self.config.region_holdout_variants:
            return rows, runtime_rows

        evaluation_regions = _region_labels_for_batch(prepared.evaluation)
        train_regions = _region_labels_for_batch(prepared.train)
        calibration_regions = _region_labels_for_batch(prepared.calibration)
        for region_name in sorted(set(evaluation_regions)):
            evaluation_indices = torch.tensor(
                [idx for idx, value in enumerate(evaluation_regions) if value == region_name],
                dtype=torch.long,
                device=prepared.evaluation.X.device,
            )
            train_indices = torch.tensor(
                [idx for idx, value in enumerate(train_regions) if value != region_name],
                dtype=torch.long,
                device=prepared.train.X.device,
            )
            calibration_indices = torch.tensor(
                [idx for idx, value in enumerate(calibration_regions) if value != region_name],
                dtype=torch.long,
                device=prepared.calibration.X.device,
            )
            if evaluation_indices.numel() == 0 or train_indices.numel() == 0 or calibration_indices.numel() == 0:
                continue
            train_batch = _slice_batch(prepared.train, train_indices)
            calibration_batch = _slice_batch(prepared.calibration, calibration_indices)
            evaluation_batch = _slice_batch(prepared.evaluation, evaluation_indices)
            station_count = (
                0
                if evaluation_batch.sensor_metadata.sensor_instance is None
                else int(torch.unique(evaluation_batch.sensor_metadata.sensor_instance).numel())
            )
            for variant_name in self.config.region_holdout_variants:
                pipeline, fit_seconds = self._fit_variant(variant_name, train_batch, calibration_batch)
                predict_start = time.perf_counter()
                metrics = pipeline.evaluate_predictions(
                    evaluation_batch.X,
                    evaluation_batch.y,
                    context=evaluation_batch.context,
                    M=evaluation_batch.M,
                    S=evaluation_batch.S,
                    sensor_metadata=evaluation_batch.sensor_metadata,
                    integrate_missingness=pipeline.use_m3,
                    batch_size=self.config.prediction_batch_size,
                )
                predict_seconds = time.perf_counter() - predict_start
                rows.append(
                    {
                        "variant_name": variant_name,
                        "holdout_region": region_name,
                        "station_count": station_count,
                        "evaluation_rows": int(evaluation_batch.X.shape[0]),
                        "rmse": _format_float(metrics.get("rmse")),
                        "mae": _format_float(metrics.get("mae")),
                        "crps": _format_float(metrics.get("crps")),
                        "coverage": _format_float(metrics.get("coverage")),
                        "interval_width": _format_float(metrics.get("interval_width")),
                    }
                )
                runtime_rows.append(
                    {
                        "section": "region_holdout",
                        "variant_name": variant_name,
                        "scenario": region_name,
                        "fit_seconds": _format_float(fit_seconds),
                        "predict_seconds": _format_float(predict_seconds),
                        "selection_seconds": "",
                    }
                )
        return rows, runtime_rows

    def _policy_rows(
        self,
        prepared: PreparedExperimentData,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Compare active-sensing baselines under one fixed budget."""
        rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        if prepared.evaluation.cost is None:
            return rows, runtime_rows

        budget = self._policy_budget(prepared.evaluation.cost)
        reference_pipeline, _ = self._fit_variant("full_model", prepared.train, prepared.calibration)
        _, reference_variance = reference_pipeline.predict_state(
            prepared.evaluation.X,
            batch_size=self.config.prediction_batch_size,
        )
        policy_variants = (
            "random",
            "variance_policy_baseline",
            "myopic_policy_baseline",
            "rollout_policy_baseline",
            "ppo_warmstart_baseline",
            "full_model",
        )
        for variant_name in policy_variants:
            if variant_name == "random":
                selection_start = time.perf_counter()
                selection = self._random_selection(
                    prepared.evaluation.cost,
                    budget=budget,
                    max_selections=self.config.max_selections,
                )
                selection_seconds = time.perf_counter() - selection_start
                selected = selection["selected_indices"]
                selected_x = (
                    prepared.evaluation.X[torch.tensor(selected, dtype=torch.long, device=prepared.evaluation.X.device)]
                    if selected
                    else prepared.evaluation.X.new_zeros((0, prepared.evaluation.X.shape[1]))
                )
                gain = self._selection_surrogate_gain(selection, reference_variance)
                total_cost = float(selection["total_cost"])
                rows.append(
                    {
                        "variant_name": "random",
                        "cumulative_information_gain": _format_float(gain),
                        "uncertainty_reduction": _format_float(gain),
                        "cumulative_energy_cost": _format_float(total_cost),
                        "routing_cost": _format_float(self._route_distance(selected_x)),
                        "total_operational_cost": _format_float(total_cost + self._route_distance(selected_x)),
                        "cost_normalized_gain": _format_float(
                            gain / max(total_cost + self._route_distance(selected_x), 1e-6)
                        ),
                        "num_selected": len(selected),
                    }
                )
                runtime_rows.append(
                    {
                        "section": "policy_runtime",
                        "variant_name": "random",
                        "scenario": "budgeted_selection",
                        "fit_seconds": "",
                        "predict_seconds": "",
                        "selection_seconds": _format_float(selection_seconds),
                    }
                )
                continue

            pipeline, fit_seconds = self._fit_variant(variant_name, prepared.train, prepared.calibration)
            predict_start = time.perf_counter()
            _, candidate_variance = pipeline.predict_state(
                prepared.evaluation.X,
                batch_size=self.config.prediction_batch_size,
            )
            predict_seconds = time.perf_counter() - predict_start
            selection_start = time.perf_counter()
            selection = pipeline.select_sensors(
                prepared.evaluation.X,
                prepared.evaluation.cost,
                context=prepared.evaluation.context,
                M=prepared.evaluation.M,
                S=prepared.evaluation.S,
                sensor_metadata=prepared.evaluation.sensor_metadata,
                budget=budget,
                max_selections=self.config.max_selections,
                batch_size=self.config.prediction_batch_size,
            )
            selection_seconds = time.perf_counter() - selection_start
            selected_x = (
                selection["selected_x"]
                if isinstance(selection.get("selected_x"), Tensor)
                else prepared.evaluation.X.new_zeros((0, prepared.evaluation.X.shape[1]))
            )
            gain = self._selection_surrogate_gain(selection, candidate_variance)
            total_cost = float(selection.get("total_cost", 0.0))
            routing_cost = float(selection.get("routing_cost", self._route_distance(selected_x)))
            total_operational_cost = float(selection.get("total_operational_cost", total_cost + routing_cost))
            rows.append(
                {
                    "variant_name": variant_name,
                    "cumulative_information_gain": _format_float(gain),
                    "uncertainty_reduction": _format_float(gain),
                    "cumulative_energy_cost": _format_float(total_cost),
                    "routing_cost": _format_float(routing_cost),
                    "total_operational_cost": _format_float(total_operational_cost),
                    "cost_normalized_gain": _format_float(gain / max(total_operational_cost, 1e-6)),
                    "num_selected": len(selection.get("selected_indices", [])),
                }
            )
            runtime_rows.append(
                {
                    "section": "policy_runtime",
                    "variant_name": variant_name,
                    "scenario": "budgeted_selection",
                    "fit_seconds": _format_float(fit_seconds),
                    "predict_seconds": _format_float(predict_seconds),
                    "selection_seconds": _format_float(selection_seconds),
                }
            )
        return rows, runtime_rows

    def run(self) -> BenchmarkResult:
        """Execute the full benchmark suite on the fixed canonical experiment."""
        self._write_progress(status="running", stage="starting")
        self._append_heartbeat(stage="starting", step="run_invoked")
        try:
            prepared = self.prepare_canonical_data()
            predictive_rows: list[dict[str, Any]] = []
            fault_rows: list[dict[str, Any]] = []
            reliability_rows: list[dict[str, Any]] = []
            coverage_rows: list[dict[str, Any]] = []
            region_holdout_rows: list[dict[str, Any]] = []
            ablation_rows: list[dict[str, Any]] = []
            policy_rows: list[dict[str, Any]] = []
            runtime_rows: list[dict[str, Any]] = []
            for repeat_seed in self.config.repeat_seeds:
                self.config.seed = int(repeat_seed)
                torch.manual_seed(self.config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.seed)
                self._write_progress(status="running", stage="repeat_start", repeat_seed=self.config.seed)
                self._append_heartbeat(stage="repeat", step="start", repeat_seed=self.config.seed)

                self._write_progress(status="running", stage="predictive_rows", repeat_seed=self.config.seed)
                section_predictive, section_predictive_runtime = self._predictive_rows(prepared)
                self._append_heartbeat(
                    stage="predictive_rows",
                    step="complete",
                    repeat_seed=self.config.seed,
                    payload={"rows": len(section_predictive)},
                )

                self._write_progress(status="running", stage="fault_rows", repeat_seed=self.config.seed)
                section_fault, section_fault_runtime = self._fault_rows(prepared)
                self._append_heartbeat(
                    stage="fault_rows",
                    step="complete",
                    repeat_seed=self.config.seed,
                    payload={"rows": len(section_fault)},
                )

                self._write_progress(status="running", stage="reliability_rows", repeat_seed=self.config.seed)
                section_reliability, section_coverage, section_reliability_runtime = self._reliability_rows(prepared)
                self._append_heartbeat(
                    stage="reliability_rows",
                    step="complete",
                    repeat_seed=self.config.seed,
                    payload={"rows": len(section_reliability), "coverage_rows": len(section_coverage)},
                )

                self._write_progress(status="running", stage="region_holdout_rows", repeat_seed=self.config.seed)
                section_region_holdout, section_region_holdout_runtime = self._region_holdout_rows(prepared)
                self._append_heartbeat(
                    stage="region_holdout_rows",
                    step="complete",
                    repeat_seed=self.config.seed,
                    payload={"rows": len(section_region_holdout)},
                )

                self._write_progress(status="running", stage="ablation_rows", repeat_seed=self.config.seed)
                section_ablation, section_ablation_runtime = self._ablation_rows(prepared)
                self._append_heartbeat(
                    stage="ablation_rows",
                    step="complete",
                    repeat_seed=self.config.seed,
                    payload={"rows": len(section_ablation)},
                )

                self._write_progress(status="running", stage="policy_rows", repeat_seed=self.config.seed)
                section_policy, section_policy_runtime = self._policy_rows(prepared)
                self._append_heartbeat(
                    stage="policy_rows",
                    step="complete",
                    repeat_seed=self.config.seed,
                    payload={"rows": len(section_policy)},
                )

                for collection in (
                    section_predictive,
                    section_fault,
                    section_reliability,
                    section_coverage,
                    section_region_holdout,
                    section_ablation,
                    section_policy,
                    section_predictive_runtime,
                    section_fault_runtime,
                    section_reliability_runtime,
                    section_region_holdout_runtime,
                    section_ablation_runtime,
                    section_policy_runtime,
                ):
                    for row in collection:
                        row["repeat_seed"] = self.config.seed

                predictive_rows.extend(section_predictive)
                fault_rows.extend(section_fault)
                reliability_rows.extend(section_reliability)
                coverage_rows.extend(section_coverage)
                region_holdout_rows.extend(section_region_holdout)
                ablation_rows.extend(section_ablation)
                policy_rows.extend(section_policy)
                runtime_rows.extend(
                    section_predictive_runtime
                    + section_fault_runtime
                    + section_reliability_runtime
                    + section_region_holdout_runtime
                    + section_ablation_runtime
                    + section_policy_runtime
                )
                self._write_progress(
                    status="running",
                    stage="repeat_complete",
                    repeat_seed=self.config.seed,
                    payload={
                        "predictive_rows": len(predictive_rows),
                        "fault_rows": len(fault_rows),
                        "reliability_rows": len(reliability_rows),
                        "region_holdout_rows": len(region_holdout_rows),
                        "ablation_rows": len(ablation_rows),
                        "policy_rows": len(policy_rows),
                    },
                )
                self._append_heartbeat(stage="repeat", step="complete", repeat_seed=self.config.seed)
            canonical_setup = {
            "framework_config_path": str(self.config.framework_config_path),
            "data_path": str(self.data_config.data_path),
            "target_column": self.data_config.target_column,
            "train_rows": int(prepared.train.X.shape[0]),
            "calibration_rows": int(prepared.calibration.X.shape[0]),
            "evaluation_rows": int(prepared.evaluation.X.shape[0]),
            "station_count": int(torch.unique(prepared.full.sensor_metadata.sensor_instance).numel()),
            "metadata_cardinalities": prepared.metadata_cardinalities,
            "temporal_stride_hours": self.config.temporal_stride_hours,
            "state_epochs": self.config.state_epochs,
            "missingness_epochs": self.config.missingness_epochs,
            "inducing_points": self.config.inducing_points,
            "target_coverage": self.config.target_coverage,
            "policy_budget": self._policy_budget(prepared.evaluation.cost),
            "max_selections": self.config.max_selections,
            "cache_prepared_data": self.config.cache_prepared_data,
            "repeat_seeds": list(self.config.repeat_seeds),
        }
            result = BenchmarkResult(
                canonical_setup=canonical_setup,
                predictive_rows=predictive_rows,
                fault_rows=fault_rows,
                reliability_rows=reliability_rows,
                coverage_timeseries_rows=coverage_rows,
                region_holdout_rows=region_holdout_rows,
                ablation_rows=ablation_rows,
                policy_rows=policy_rows,
                runtime_rows=runtime_rows,
            )
            self._write_progress(
                status="completed",
                stage="completed",
                payload={
                    "predictive_rows": len(result.predictive_rows),
                    "fault_rows": len(result.fault_rows),
                    "reliability_rows": len(result.reliability_rows),
                    "region_holdout_rows": len(result.region_holdout_rows),
                    "ablation_rows": len(result.ablation_rows),
                    "policy_rows": len(result.policy_rows),
                },
            )
            self._append_heartbeat(stage="completed", step="run_complete")
            return result
        except Exception as error:
            self._write_progress(status="failed", stage="failed", error=str(error))
            self._append_heartbeat(stage="failed", step="exception", payload={"error": str(error)})
            raise

    def write_artifacts(self, result: BenchmarkResult | None = None) -> BenchmarkArtifacts:
        """Write benchmark artifacts to disk."""
        resolved = self.run() if result is None else result
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.config.output_dir / "summary.json"
        predictive_path = self.config.output_dir / "predictive_mnar.csv"
        fault_path = self.config.output_dir / "fault_diagnosis.csv"
        reliability_path = self.config.output_dir / "reliability_shift.csv"
        coverage_timeseries_path = self.config.output_dir / "coverage_over_time.csv"
        region_holdout_path = self.config.output_dir / "region_holdout.csv"
        ablation_path = self.config.output_dir / "ablation.csv"
        policy_path = self.config.output_dir / "policy_runtime.csv"
        runtime_path = self.config.output_dir / "runtime.csv"
        significance_path = self.config.output_dir / "significance_summary.json"
        paper_tables_path = self.config.output_dir / "paper_tables.json"
        report_path = self.config.output_dir / "report.md"
        figure_dir = self.config.output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)

        summary_path.write_text(json.dumps(resolved.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(
            predictive_path,
            resolved.predictive_rows,
            [
                "repeat_seed",
                "variant_name",
                "pipeline_variant",
                "mechanism",
                "missing_intensity",
                "rmse",
                "mae",
                "crps",
                "log_score",
                "gaussian_coverage",
                "calibration_error",
                "gap_rmse",
                "gap_crps",
                "gap_fraction",
                "mean_synthetic_missingness_probability",
                "mean_predicted_missingness_probability",
            ],
        )
        _write_csv(
            fault_path,
            resolved.fault_rows,
            [
                "repeat_seed",
                "variant_name",
                "fault_scenario",
                "f1",
                "precision",
                "recall",
                "auroc",
                "detection_delay",
                "false_alarm_rate",
                "fault_fraction",
            ],
        )
        _write_csv(
            reliability_path,
            resolved.reliability_rows,
            [
                "repeat_seed",
                "variant_name",
                "coverage",
                "target_coverage_error",
                "interval_width",
                "stable_coverage",
                "shift_coverage",
                "stable_interval_width",
                "shift_interval_width",
                "crps",
                "log_score",
                "recovery_speed_hours",
                "final_adaptive_epsilon",
                "mean_graph_quantile",
            ],
        )
        _write_csv(
            region_holdout_path,
            resolved.region_holdout_rows,
            [
                "repeat_seed",
                "variant_name",
                "holdout_region",
                "station_count",
                "evaluation_rows",
                "rmse",
                "mae",
                "crps",
                "coverage",
                "interval_width",
            ],
        )
        _write_csv(
            coverage_timeseries_path,
            resolved.coverage_timeseries_rows,
            ["repeat_seed", "variant_name", "elapsed_hours", "coverage", "interval_width", "is_shift_region"],
        )
        _write_csv(
            ablation_path,
            resolved.ablation_rows,
            [
                "repeat_seed",
                "variant_name",
                "rmse",
                "mae",
                "crps",
                "log_score",
                "coverage",
                "interval_width",
                "mean_missingness_proba",
                "total_cost",
                "num_selected",
            ],
        )
        _write_csv(
            policy_path,
            resolved.policy_rows,
            [
                "repeat_seed",
                "variant_name",
                "cumulative_information_gain",
                "uncertainty_reduction",
                "cumulative_energy_cost",
                "routing_cost",
                "total_operational_cost",
                "cost_normalized_gain",
                "num_selected",
            ],
        )
        _write_csv(
            figure_dir / "figure_2_coverage_over_time.csv",
            resolved.coverage_timeseries_rows,
            ["repeat_seed", "variant_name", "elapsed_hours", "coverage", "interval_width", "is_shift_region"],
        )
        _write_csv(
            figure_dir / "figure_3_cost_vs_uncertainty.csv",
            resolved.policy_rows,
            [
                "repeat_seed",
                "variant_name",
                "cumulative_information_gain",
                "uncertainty_reduction",
                "cumulative_energy_cost",
                "routing_cost",
                "cost_normalized_gain",
                "num_selected",
            ],
        )
        _write_csv(
            figure_dir / "figure_5_mnar_sensitivity.csv",
            resolved.predictive_rows,
            [
                "repeat_seed",
                "variant_name",
                "mechanism",
                "missing_intensity",
                "crps",
                "gap_crps",
                "mean_predicted_missingness_probability",
            ],
        )
        _write_csv(
            runtime_path,
            resolved.runtime_rows,
            ["repeat_seed", "section", "variant_name", "scenario", "fit_seconds", "predict_seconds", "selection_seconds"],
        )
        table_1 = _aggregate_rows(
            resolved.predictive_rows,
            group_keys=("variant_name", "pipeline_variant", "mechanism", "missing_intensity"),
            numeric_keys=(
                "rmse",
                "mae",
                "crps",
                "log_score",
                "gaussian_coverage",
                "calibration_error",
                "gap_rmse",
                "gap_crps",
                "gap_fraction",
                "mean_synthetic_missingness_probability",
                "mean_predicted_missingness_probability",
            ),
            bootstrap_samples=self.config.bootstrap_samples,
            bootstrap_alpha=self.config.significance_alpha,
        )
        table_2 = _aggregate_rows(
            resolved.fault_rows,
            group_keys=("variant_name", "fault_scenario"),
            numeric_keys=("f1", "precision", "recall", "auroc", "detection_delay", "false_alarm_rate", "fault_fraction"),
            bootstrap_samples=self.config.bootstrap_samples,
            bootstrap_alpha=self.config.significance_alpha,
        )
        table_3 = _aggregate_rows(
            resolved.policy_rows,
            group_keys=("variant_name",),
            numeric_keys=(
                "cumulative_information_gain",
                "uncertainty_reduction",
                "cumulative_energy_cost",
                "routing_cost",
                "total_operational_cost",
                "cost_normalized_gain",
                "num_selected",
            ),
            bootstrap_samples=self.config.bootstrap_samples,
            bootstrap_alpha=self.config.significance_alpha,
        )
        table_4 = _aggregate_rows(
            resolved.reliability_rows,
            group_keys=("variant_name",),
            numeric_keys=(
                "coverage",
                "target_coverage_error",
                "interval_width",
                "stable_coverage",
                "shift_coverage",
                "stable_interval_width",
                "shift_interval_width",
                "crps",
                "log_score",
                "recovery_speed_hours",
                "final_adaptive_epsilon",
                "mean_graph_quantile",
            ),
            bootstrap_samples=self.config.bootstrap_samples,
            bootstrap_alpha=self.config.significance_alpha,
        )
        table_5 = _aggregate_rows(
            resolved.ablation_rows,
            group_keys=("variant_name",),
            numeric_keys=(
                "rmse",
                "mae",
                "crps",
                "log_score",
                "coverage",
                "interval_width",
                "mean_missingness_proba",
                "total_cost",
                "num_selected",
            ),
            bootstrap_samples=self.config.bootstrap_samples,
            bootstrap_alpha=self.config.significance_alpha,
        )
        table_region_holdout = _aggregate_rows(
            resolved.region_holdout_rows,
            group_keys=("variant_name", "holdout_region"),
            numeric_keys=("station_count", "evaluation_rows", "rmse", "mae", "crps", "coverage", "interval_width"),
            bootstrap_samples=self.config.bootstrap_samples,
            bootstrap_alpha=self.config.significance_alpha,
        )
        significance_tables = {
            "predictive_vs_base_gp": _significance_rows(
                resolved.predictive_rows,
                group_keys=("mechanism", "missing_intensity"),
                metric_key="crps",
                baseline_variant="gp_only",
                lower_is_better=True,
                bootstrap_samples=self.config.bootstrap_samples,
                significance_alpha=self.config.significance_alpha,
            ),
            "reliability_vs_conformal": _significance_rows(
                resolved.reliability_rows,
                group_keys=(),
                metric_key="target_coverage_error",
                baseline_variant="split_conformal",
                lower_is_better=True,
                bootstrap_samples=self.config.bootstrap_samples,
                significance_alpha=self.config.significance_alpha,
            ),
            "policy_vs_random": _significance_rows(
                resolved.policy_rows,
                group_keys=(),
                metric_key="cost_normalized_gain",
                baseline_variant="random",
                lower_is_better=False,
                bootstrap_samples=self.config.bootstrap_samples,
                significance_alpha=self.config.significance_alpha,
            ),
            "ablation_vs_base_gp": _significance_rows(
                resolved.ablation_rows,
                group_keys=(),
                metric_key="crps",
                baseline_variant="base_gp_only",
                lower_is_better=True,
                bootstrap_samples=self.config.bootstrap_samples,
                significance_alpha=self.config.significance_alpha,
            ),
        }
        paper_tables = {
            "table_1_predictive_mnar": table_1,
            "table_2_fault_diagnosis": table_2,
            "table_3_active_sensing": table_3,
            "table_4_reliability_shift": table_4,
            "table_5_ablation": table_5,
            "table_region_holdout": table_region_holdout,
            "significance": significance_tables,
        }
        significance_path.write_text(json.dumps(significance_tables, indent=2, ensure_ascii=False), encoding="utf-8")
        paper_tables_path.write_text(json.dumps(paper_tables, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(figure_dir / "table_1_predictive_mnar.csv", table_1, list(table_1[0].keys()) if table_1 else ["variant_name"])
        _write_csv(figure_dir / "table_2_fault_diagnosis.csv", table_2, list(table_2[0].keys()) if table_2 else ["variant_name"])
        _write_csv(figure_dir / "table_3_active_sensing.csv", table_3, list(table_3[0].keys()) if table_3 else ["variant_name"])
        _write_csv(figure_dir / "table_4_reliability_shift.csv", table_4, list(table_4[0].keys()) if table_4 else ["variant_name"])
        _write_csv(figure_dir / "table_5_ablation.csv", table_5, list(table_5[0].keys()) if table_5 else ["variant_name"])
        _write_csv(
            figure_dir / "table_region_holdout.csv",
            table_region_holdout,
            list(table_region_holdout[0].keys()) if table_region_holdout else ["variant_name"],
        )
        _write_csv(
            figure_dir / "table_1_predictive_significance.csv",
            significance_tables["predictive_vs_base_gp"],
            list(significance_tables["predictive_vs_base_gp"][0].keys())
            if significance_tables["predictive_vs_base_gp"]
            else ["variant_name"],
        )
        report_path.write_text(render_benchmark_report(resolved), encoding="utf-8")
        return BenchmarkArtifacts(
            summary_path=summary_path,
            predictive_path=predictive_path,
            fault_path=fault_path,
            reliability_path=reliability_path,
            coverage_timeseries_path=coverage_timeseries_path,
            region_holdout_path=region_holdout_path,
            ablation_path=ablation_path,
            policy_path=policy_path,
            runtime_path=runtime_path,
            significance_path=significance_path,
            paper_tables_path=paper_tables_path,
            report_path=report_path,
        )


def render_benchmark_report(result: BenchmarkResult) -> str:
    """Render a compact markdown report for benchmark review."""
    setup = result.canonical_setup
    best_predictive = min(
        result.predictive_rows,
        key=lambda row: float(row["crps"]) if row["crps"] else float("inf"),
    )
    best_reliability = min(
        result.reliability_rows,
        key=lambda row: float(row["target_coverage_error"]) if row["target_coverage_error"] else float("inf"),
    )
    best_fault = (
        max(
            result.fault_rows,
            key=lambda row: float(row["f1"]) if row["f1"] else float("-inf"),
        )
        if result.fault_rows
        else None
    )
    best_policy = (
        max(
            result.policy_rows,
            key=lambda row: float(row["cost_normalized_gain"]) if row["cost_normalized_gain"] else float("-inf"),
        )
        if result.policy_rows
        else None
    )
    best_region_holdout = (
        min(
            result.region_holdout_rows,
            key=lambda row: float(row["crps"]) if row["crps"] else float("inf"),
        )
        if result.region_holdout_rows
        else None
    )
    lines = [
        "# Silence-Aware IDS Benchmark Report",
        "",
        "## Canonical Setup",
        f"- Framework config: `{setup['framework_config_path']}`",
        f"- Data path: `{setup['data_path']}`",
        f"- Target: `{setup['target_column']}`",
        f"- Rows: train={setup['train_rows']}, calibration={setup['calibration_rows']}, evaluation={setup['evaluation_rows']}",
        f"- Stations: {setup['station_count']}, stride={setup['temporal_stride_hours']}h, inducing={setup['inducing_points']}",
        f"- Repeat seeds: {len(setup['repeat_seeds'])} ({', '.join(str(value) for value in setup['repeat_seeds'])})",
        "",
        "## Predictive MNAR Benchmark",
        (
            f"- Best CRPS: `{best_predictive['variant_name']}` under "
            f"`{best_predictive['mechanism']}` @ `{best_predictive['missing_intensity']}` with "
            f"CRPS `{best_predictive['crps']}` and gap CRPS `{best_predictive['gap_crps']}`"
        ),
        "",
        "## Reliability Under Shift",
        (
            f"- Best target-coverage error: `{best_reliability['variant_name']}` with coverage "
            f"`{best_reliability['coverage']}` and shift coverage `{best_reliability['shift_coverage']}`"
        ),
        "",
        "## Region Holdout",
        "",
        "## Fault Diagnosis",
    ]
    if best_region_holdout is not None:
        lines.append(
            f"- Best region holdout CRPS: `{best_region_holdout['variant_name']}` on "
            f"`{best_region_holdout['holdout_region']}` with CRPS `{best_region_holdout['crps']}`"
        )
    else:
        lines.append("- Region holdout was not enabled for this benchmark config.")
    if best_fault is not None:
        lines.append(
            f"- Best F1: `{best_fault['variant_name']}` on `{best_fault['fault_scenario']}` with "
            f"F1 `{best_fault['f1']}`, AUROC `{best_fault['auroc']}`, FAR `{best_fault['false_alarm_rate']}`"
        )
    lines.extend(
        [
        "",
        "## Ablation Snapshot",
        ]
    )
    for row in result.ablation_rows:
        lines.append(
            f"- `{row['variant_name']}`: RMSE `{row['rmse']}`, CRPS `{row['crps']}`, "
            f"Coverage `{row['coverage']}`, Cost `{row['total_cost']}`"
        )
    if best_policy is not None:
        lines.extend(
            [
                "",
                "## Policy Comparison",
                (
                    f"- Best cost-normalized gain: `{best_policy['variant_name']}` with "
                    f"`{best_policy['cost_normalized_gain']}`"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def run_benchmark(config_path: Path) -> BenchmarkArtifacts:
    """Run the benchmark suite from a JSON config path."""
    config = load_benchmark_config(config_path)
    runner = BenchmarkSuiteRunner(config)
    return runner.write_artifacts()


def launch_benchmark_detached(
    config_path: Path,
    *,
    python_executable: Path | None = None,
) -> dict[str, object]:
    """Launch one benchmark run in a detached subprocess."""
    resolved_config = config_path.resolve()
    config = load_benchmark_config(resolved_config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "benchmark_detached_stdout.log"
    stderr_path = output_dir / "benchmark_detached_stderr.log"
    launch_path = output_dir / "benchmark_detached_launch.json"
    chosen_python = (python_executable or Path(sys.executable)).resolve()
    stdout_handle = stdout_path.open("ab")
    stderr_handle = stderr_path.open("ab")
    creationflags = 0
    if hasattr(subprocess, "DETACHED_PROCESS"):
        creationflags |= int(subprocess.DETACHED_PROCESS)
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags |= int(subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        process = subprocess.Popen(
            [
                str(chosen_python),
                "-m",
                "task_cli",
                "benchmark-run",
                "--config",
                str(resolved_config),
            ],
            cwd=str(Path.cwd()),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    payload = {
        "pid": int(process.pid),
        "config_path": str(resolved_config),
        "output_dir": str(output_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "python_executable": str(chosen_python),
        "launched_at": datetime.now().isoformat(timespec="seconds"),
    }
    launch_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def benchmark_run_status(config_path: Path) -> dict[str, object]:
    """Read the latest persisted status for one benchmark config."""
    resolved_config = config_path.resolve()
    config = load_benchmark_config(resolved_config)
    output_dir = config.output_dir
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    runtime_path = output_dir / "runtime.csv"
    launch_path = output_dir / "benchmark_detached_launch.json"
    progress_path = output_dir / "benchmark_progress.json"
    heartbeat_path = output_dir / "benchmark_heartbeat.jsonl"

    summary_payload: dict[str, object] | None = None
    if summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary_payload = None

    launch_payload: dict[str, object] | None = None
    if launch_path.exists():
        launch_payload = json.loads(launch_path.read_text(encoding="utf-8"))

    progress_payload: dict[str, object] | None = None
    if progress_path.exists():
        try:
            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress_payload = None

    last_heartbeat: dict[str, object] | None = None
    if heartbeat_path.exists():
        try:
            lines = heartbeat_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                last_heartbeat = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    return {
        "config_path": str(resolved_config),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "runtime_path": str(runtime_path),
        "launch_path": str(launch_path),
        "progress_path": str(progress_path),
        "heartbeat_path": str(heartbeat_path),
        "summary": summary_payload,
        "launch": launch_payload,
        "progress": progress_payload,
        "last_heartbeat": last_heartbeat,
    }
