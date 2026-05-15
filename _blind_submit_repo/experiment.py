from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import gpytorch
import torch
from torch import Tensor

from models import SensorMetadataBatch
from pipeline import (
    AblationOutcome,
    PipelineFitSummary,
    SilenceAwareIDS,
    SilenceAwareIDSConfig,
    _slice_metadata,
    _slice_optional,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class TabularDataConfig:
    """Configuration for loading tabular weather-network datasets.

    Attributes:
        data_path: CSV path.
        timestamp_column: Timestamp column.
        station_id_column: Station identifier column.
        target_column: Observation target column.
        latitude_column: Latitude column.
        longitude_column: Longitude column.
        context_columns: Always-available numeric context columns.
        extra_input_columns: Extra numeric inputs appended to the GP input tensor.
        continuous_metadata_columns: Numeric metadata used by M3.
        cost_column: Optional sensor cost column.
        dynamic_silence_column: Optional precomputed dynamic silence column.
        missing_indicator_column: Optional structural missingness column.
        sensor_type_column: Optional categorical sensor-type column.
        sensor_group_column: Optional categorical sensor-group column.
        sensor_modality_column: Optional categorical modality column.
        installation_environment_column: Optional categorical installation-environment column.
        maintenance_state_column: Optional categorical maintenance-state column.
        train_ratio: Proportion of timestamps or rows used for training.
        calibration_ratio: Proportion of timestamps or rows used for calibration.
        split_strategy: Either `"temporal"` or `"row"`.
        timestamp_format: Optional explicit timestamp format.
        device: Tensor device.
        dtype: Tensor dtype for float tensors.
    """

    data_path: Path
    timestamp_column: str = "timestamp"
    station_id_column: str = "station_id"
    target_column: str = "temperature"
    latitude_column: str = "latitude"
    longitude_column: str = "longitude"
    context_columns: tuple[str, ...] = ()
    extra_input_columns: tuple[str, ...] = ()
    continuous_metadata_columns: tuple[str, ...] = ()
    cost_column: str | None = "cost"
    dynamic_silence_column: str | None = None
    missing_indicator_column: str | None = None
    sensor_type_column: str | None = None
    sensor_group_column: str | None = None
    sensor_modality_column: str | None = None
    installation_environment_column: str | None = None
    maintenance_state_column: str | None = None
    train_ratio: float = 0.7
    calibration_ratio: float = 0.15
    split_strategy: str = "temporal"
    timestamp_format: str | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must lie in (0, 1).")
        if not 0.0 <= self.calibration_ratio < 1.0:
            raise ValueError("calibration_ratio must lie in [0, 1).")
        if self.train_ratio + self.calibration_ratio >= 1.0:
            raise ValueError("train_ratio + calibration_ratio must be less than 1.0.")
        if self.split_strategy not in {"temporal", "row"}:
            raise ValueError("split_strategy must be 'temporal' or 'row'.")


@dataclass
class TensorBatch:
    """Tensor bundle for one dataset split."""

    X: Tensor
    y: Tensor
    context: Tensor | None
    M: Tensor | None
    S: Tensor | None
    missing_indicator: Tensor
    cost: Tensor | None
    sensor_metadata: SensorMetadataBatch
    indices: Tensor


@dataclass
class PreparedExperimentData:
    """Prepared tensorized dataset with train/calibration/evaluation splits."""

    full: TensorBatch
    train: TensorBatch
    calibration: TensorBatch
    evaluation: TensorBatch
    metadata_cardinalities: dict[str, int]
    feature_schema: dict[str, Any]


@dataclass
class ExperimentRunConfig:
    """Configuration for end-to-end ablation runs."""

    variant_names: tuple[str, ...] = ()
    sensitivity_logit_scales: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    policy_budget: float | None = None
    max_selections: int | None = None
    use_evaluation_as_candidate_pool: bool = True
    seed: int = 0
    deterministic_algorithms: bool = False
    torch_num_threads: int | None = None
    matmul_precision: str = "high"
    prediction_batch_size: int | None = None
    benchmark_expansion_factor: int = 1
    benchmark_candidate_pool_size: int | None = None
    enable_progress_artifacts: bool = True
    resume_from_checkpoint: bool = True
    max_train_rows: int | None = None
    max_calibration_rows: int | None = None
    max_evaluation_rows: int | None = None
    auto_scale_large_runs: bool = True

    def __post_init__(self) -> None:
        valid_precision = {"highest", "high", "medium"}
        if self.matmul_precision not in valid_precision:
            raise ValueError(f"matmul_precision must be one of {sorted(valid_precision)}.")
        if self.benchmark_expansion_factor <= 0:
            raise ValueError("benchmark_expansion_factor must be positive.")
        if self.prediction_batch_size is not None and self.prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive when provided.")
        if self.benchmark_candidate_pool_size is not None and self.benchmark_candidate_pool_size <= 0:
            raise ValueError("benchmark_candidate_pool_size must be positive when provided.")
        for name, value in (
            ("max_train_rows", self.max_train_rows),
            ("max_calibration_rows", self.max_calibration_rows),
            ("max_evaluation_rows", self.max_evaluation_rows),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided.")


@dataclass
class ExperimentResult:
    """Structured output of a full experiment run."""

    fit_summary: PipelineFitSummary
    base_metrics: dict[str, float]
    ablations: dict[str, AblationOutcome]
    sensitivity: dict[str, dict[str, float]]
    selection: dict[str, Any] | None
    dataset_summary: dict[str, Any]
    runtime_environment: dict[str, Any]
    reproducibility: dict[str, Any]
    benchmark: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        ablations: dict[str, Any] = {}
        for name, outcome in self.ablations.items():
            ablations[name] = {
                "variant_name": outcome.variant_name,
                "metrics": outcome.metrics,
                "report": outcome.report,
                "selection": _selection_to_summary(outcome.selection),
            }
        return {
            "fit_summary": {
                "state_history": self.fit_summary.state_history,
                "observation_history": self.fit_summary.observation_history,
                "dynamic_silence_threshold": self.fit_summary.dynamic_silence_threshold,
                "missingness_history": self.fit_summary.missingness_history,
                "conformal_quantile": self.fit_summary.conformal_quantile,
            },
            "base_metrics": self.base_metrics,
            "ablations": ablations,
            "sensitivity": self.sensitivity,
            "selection": self.selection,
            "dataset_summary": self.dataset_summary,
            "runtime_environment": self.runtime_environment,
            "reproducibility": self.reproducibility,
            "benchmark": self.benchmark,
        }


@dataclass
class ExperimentArtifacts:
    """Paths for persisted framework experiment outputs."""

    summary_path: Path
    ablations_path: Path
    sensitivity_path: Path
    selection_path: Path
    report_path: Path
    progress_path: Path | None = None
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class _CsvScanSummary:
    """Lightweight metadata collected before tensor allocation."""

    row_count: int
    base_time: datetime
    metadata_mappings: dict[str, dict[str, int]]


def _selection_to_summary(selection: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert selection outputs into a serializable summary."""
    if selection is None:
        return None
    summary: dict[str, Any] = {}
    for key, value in selection.items():
        if isinstance(value, Tensor):
            summary[key] = value.detach().cpu().tolist()
        else:
            summary[key] = value
    return summary


def _serialize_fit_summary(summary: PipelineFitSummary | None) -> dict[str, Any] | None:
    """Convert a fit summary into a JSON-friendly payload."""
    if summary is None:
        return None
    return {
        "state_history": summary.state_history,
        "observation_history": summary.observation_history,
        "dynamic_silence_threshold": summary.dynamic_silence_threshold,
        "missingness_history": summary.missingness_history,
        "conformal_quantile": summary.conformal_quantile,
    }


def _deserialize_fit_summary(payload: dict[str, Any] | None) -> PipelineFitSummary | None:
    """Restore a fit summary from serialized checkpoint state."""
    if payload is None:
        return None
    return PipelineFitSummary(
        state_history=dict(payload.get("state_history", {})),
        observation_history=payload.get("observation_history"),
        dynamic_silence_threshold=payload.get("dynamic_silence_threshold"),
        missingness_history=payload.get("missingness_history"),
        conformal_quantile=payload.get("conformal_quantile"),
    )


def _serialize_ablation_outcomes(outcomes: dict[str, AblationOutcome]) -> dict[str, Any]:
    """Convert ablation outcomes into checkpoint-friendly payloads."""
    serialized: dict[str, Any] = {}
    for name, outcome in outcomes.items():
        serialized[name] = {
            "variant_name": outcome.variant_name,
            "metrics": outcome.metrics,
            "report": outcome.report,
            "selection": _selection_to_summary(outcome.selection),
        }
    return serialized


def _deserialize_ablation_outcomes(payload: dict[str, Any] | None) -> dict[str, AblationOutcome]:
    """Restore ablation outcomes from serialized checkpoint state."""
    if not payload:
        return {}
    restored: dict[str, AblationOutcome] = {}
    for name, outcome in payload.items():
        restored[name] = AblationOutcome(
            variant_name=str(outcome.get("variant_name", name)),
            metrics=dict(outcome.get("metrics", {})),
            report=dict(outcome.get("report", {})),
            selection=outcome.get("selection"),
        )
    return restored


def _deserialize_experiment_result(payload: dict[str, Any]) -> ExperimentResult:
    """Restore an experiment result from serialized checkpoint state."""
    return ExperimentResult(
        fit_summary=_deserialize_fit_summary(payload.get("fit_summary")) or PipelineFitSummary(state_history={}),
        base_metrics=dict(payload.get("base_metrics", {})),
        ablations=_deserialize_ablation_outcomes(payload.get("ablations")),
        sensitivity=dict(payload.get("sensitivity", {})),
        selection=payload.get("selection"),
        dataset_summary=dict(payload.get("dataset_summary", {})),
        runtime_environment=dict(payload.get("runtime_environment", {})),
        reproducibility=dict(payload.get("reproducibility", {})),
        benchmark=payload.get("benchmark"),
    )


def _batch_to_cache_dict(batch: TensorBatch) -> dict[str, Any]:
    """Serialize a tensor batch for torch.save cache reuse."""
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
    """Restore a tensor batch from cached torch.save payloads."""
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


class WeatherDatasetAdapter:
    """Adapter that converts long-format CSV weather data into tensor splits."""

    def __init__(self, config: TabularDataConfig) -> None:
        self.config = config

    def prepare(self) -> PreparedExperimentData:
        """Load the configured CSV and convert it into tensor splits.

        Returns:
            Tensorized dataset with train, calibration, and evaluation splits.
        """
        scan_summary = self._scan_csv()
        encoded = self._encode_csv(scan_summary)
        split_indices = self._build_split_indices(encoded["timestamp_keys"])
        full_batch = self._build_full_batch(encoded)
        train_batch = self._build_batch(encoded, split_indices["train"])
        calibration_batch = self._build_batch(encoded, split_indices["calibration"])
        evaluation_batch = self._build_batch(encoded, split_indices["evaluation"])
        return PreparedExperimentData(
            full=full_batch,
            train=train_batch,
            calibration=calibration_batch,
            evaluation=evaluation_batch,
            metadata_cardinalities=encoded["metadata_cardinalities"],
            feature_schema=encoded["feature_schema"],
        )

    def _validate_fieldnames(self, fieldnames: list[str] | None) -> None:
        """Validate that the configured schema exists in the CSV header."""
        if fieldnames is None:
            raise ValueError(f"CSV is missing a header row: {self.config.data_path}")
        required = {
            self.config.timestamp_column,
            self.config.station_id_column,
            self.config.target_column,
            self.config.latitude_column,
            self.config.longitude_column,
        }
        for optional_column in [
            self.config.cost_column,
            self.config.dynamic_silence_column,
            self.config.missing_indicator_column,
            self.config.sensor_type_column,
            self.config.sensor_group_column,
            self.config.sensor_modality_column,
            self.config.installation_environment_column,
            self.config.maintenance_state_column,
        ]:
            if optional_column is not None:
                required.add(optional_column)
        required.update(self.config.context_columns)
        required.update(self.config.extra_input_columns)
        required.update(self.config.continuous_metadata_columns)
        missing = sorted(column for column in required if column not in fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    def _scan_csv(self) -> _CsvScanSummary:
        """Scan CSV metadata before allocating large tensors."""
        row_count = 0
        base_time: datetime | None = None
        category_sets: dict[str, set[str]] = {
            "sensor_type": set(),
            "sensor_group": set(),
            "sensor_modality": set(),
            "installation_environment": set(),
            "maintenance_state": set(),
        }
        with self.config.data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._validate_fieldnames(reader.fieldnames)
            for row in reader:
                timestamp = self._parse_timestamp(row[self.config.timestamp_column])
                if base_time is None or timestamp < base_time:
                    base_time = timestamp
                row_count += 1
                for key in category_sets:
                    column_name = getattr(self.config, f"{key}_column")
                    if column_name is not None:
                        category_sets[key].add(row[column_name].strip())
        if row_count == 0 or base_time is None:
            raise ValueError(f"Dataset is empty: {self.config.data_path}")
        metadata_mappings = {
            key: {value: index for index, value in enumerate(sorted(values))}
            for key, values in category_sets.items()
        }
        return _CsvScanSummary(
            row_count=row_count,
            base_time=base_time,
            metadata_mappings=metadata_mappings,
        )

    def _encode_csv(self, scan_summary: _CsvScanSummary) -> dict[str, Any]:
        """Encode CSV rows into aligned tensors without materializing row dictionaries."""
        device = torch.device(self.config.device)
        num_rows = scan_summary.row_count
        input_dim = 7 + len(self.config.extra_input_columns)
        context_dim = len(self.config.context_columns)
        continuous_dim = len(self.config.continuous_metadata_columns)

        X_tensor = torch.empty((num_rows, input_dim), dtype=self.config.dtype, device=device)
        y_tensor = torch.empty(num_rows, dtype=self.config.dtype, device=device)
        M_tensor = torch.empty(num_rows, dtype=self.config.dtype, device=device)
        S_tensor = torch.empty(num_rows, dtype=self.config.dtype, device=device)
        cost_tensor = torch.empty(num_rows, dtype=self.config.dtype, device=device)
        timestamp_keys = torch.empty(num_rows, dtype=torch.long, device=device)
        context_tensor = (
            None
            if context_dim == 0
            else torch.empty((num_rows, context_dim), dtype=self.config.dtype, device=device)
        )
        continuous_tensor = (
            None
            if continuous_dim == 0
            else torch.empty((num_rows, continuous_dim), dtype=self.config.dtype, device=device)
        )
        categorical_tensors: dict[str, Tensor | None] = {
            key: (
                None
                if not mapping
                else torch.empty(num_rows, dtype=torch.long, device=device)
            )
            for key, mapping in scan_summary.metadata_mappings.items()
        }
        with self.config.data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._validate_fieldnames(reader.fieldnames)
            for row_index, row in enumerate(reader):
                timestamp = self._parse_timestamp(row[self.config.timestamp_column])
                elapsed_hours = (timestamp - scan_summary.base_time).total_seconds() / 3600.0
                hour_angle = 2.0 * math.pi * (timestamp.hour / 24.0)
                day_angle = 2.0 * math.pi * ((timestamp.timetuple().tm_yday - 1) / 366.0)

                X_tensor[row_index, 0] = self._parse_float(row[self.config.latitude_column])
                X_tensor[row_index, 1] = self._parse_float(row[self.config.longitude_column])
                X_tensor[row_index, 2] = elapsed_hours
                X_tensor[row_index, 3] = math.sin(hour_angle)
                X_tensor[row_index, 4] = math.cos(hour_angle)
                X_tensor[row_index, 5] = math.sin(day_angle)
                X_tensor[row_index, 6] = math.cos(day_angle)
                for offset, column in enumerate(self.config.extra_input_columns, start=7):
                    X_tensor[row_index, offset] = self._parse_float(row[column])

                target_raw = row[self.config.target_column].strip()
                if target_raw:
                    y_tensor[row_index] = self._parse_float(target_raw)
                    derived_missing = 0.0
                else:
                    y_tensor[row_index] = float("nan")
                    derived_missing = 1.0

                if self.config.missing_indicator_column is not None:
                    M_tensor[row_index] = self._parse_float(row[self.config.missing_indicator_column])
                else:
                    M_tensor[row_index] = derived_missing

                if self.config.dynamic_silence_column is not None:
                    S_tensor[row_index] = self._parse_float(row[self.config.dynamic_silence_column])
                else:
                    S_tensor[row_index] = 0.0

                if context_tensor is not None:
                    for column_index, column in enumerate(self.config.context_columns):
                        context_tensor[row_index, column_index] = self._parse_float(row[column])
                if continuous_tensor is not None:
                    for column_index, column in enumerate(self.config.continuous_metadata_columns):
                        continuous_tensor[row_index, column_index] = self._parse_float(row[column])

                if self.config.cost_column is not None:
                    cost_tensor[row_index] = self._parse_float(row[self.config.cost_column])
                else:
                    cost_tensor[row_index] = 1.0

                timestamp_keys[row_index] = self._timestamp_sort_key(timestamp)
                for key, mapping in scan_summary.metadata_mappings.items():
                    tensor = categorical_tensors[key]
                    column_name = getattr(self.config, f"{key}_column")
                    if tensor is not None and column_name is not None:
                        tensor[row_index] = mapping[row[column_name].strip()]

        sensor_metadata = SensorMetadataBatch(
            sensor_instance=None,
            sensor_type=categorical_tensors["sensor_type"],
            sensor_group=categorical_tensors["sensor_group"],
            sensor_modality=categorical_tensors["sensor_modality"],
            installation_environment=categorical_tensors["installation_environment"],
            maintenance_state=categorical_tensors["maintenance_state"],
            continuous=continuous_tensor,
        )
        station_mapping: dict[str, int] = {}
        sensor_instance = torch.empty(scan_summary.row_count, dtype=torch.long, device=device)
        with self.config.data_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row_index, row in enumerate(reader):
                station_value = row[self.config.station_id_column].strip()
                station_mapping.setdefault(station_value, len(station_mapping))
                sensor_instance[row_index] = station_mapping[station_value]
        sensor_metadata.sensor_instance = sensor_instance

        metadata_cardinalities = {
            "sensor_type": len(scan_summary.metadata_mappings["sensor_type"]),
            "sensor_group": len(scan_summary.metadata_mappings["sensor_group"]),
            "sensor_modality": len(scan_summary.metadata_mappings["sensor_modality"]),
            "installation_environment": len(scan_summary.metadata_mappings["installation_environment"]),
            "maintenance_state": len(scan_summary.metadata_mappings["maintenance_state"]),
        }
        feature_schema = {
            "input_columns": [
                self.config.latitude_column,
                self.config.longitude_column,
                "elapsed_hours",
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
                *self.config.extra_input_columns,
            ],
            "context_columns": list(self.config.context_columns),
            "continuous_metadata_columns": list(self.config.continuous_metadata_columns),
        }
        return {
            "X": X_tensor,
            "y": y_tensor,
            "context": context_tensor,
            "M": M_tensor,
            "S": S_tensor,
            "missing_indicator": M_tensor,
            "cost": cost_tensor,
            "sensor_metadata": sensor_metadata,
            "timestamp_keys": timestamp_keys,
            "metadata_cardinalities": metadata_cardinalities,
            "feature_schema": feature_schema,
        }

    def _build_split_indices(self, timestamp_keys: Tensor) -> dict[str, Tensor]:
        """Build train/calibration/evaluation indices."""
        device = torch.device(self.config.device)
        if self.config.split_strategy == "row":
            num_rows = int(timestamp_keys.shape[0])
            indices = torch.arange(num_rows, dtype=torch.long, device=device)
            train_end = max(1, int(num_rows * self.config.train_ratio))
            cal_end = max(train_end + 1, int(num_rows * (self.config.train_ratio + self.config.calibration_ratio)))
            cal_end = min(cal_end, num_rows - 1)
            return {
                "train": indices[:train_end],
                "calibration": indices[train_end:cal_end],
                "evaluation": indices[cal_end:],
            }

        unique_timestamps = torch.unique(timestamp_keys)
        unique_timestamps, _ = torch.sort(unique_timestamps)
        if int(unique_timestamps.numel()) < 3:
            raise ValueError("Temporal split requires at least three unique timestamps.")
        train_end = max(1, int(unique_timestamps.numel() * self.config.train_ratio))
        cal_end = max(
            train_end + 1,
            int(unique_timestamps.numel() * (self.config.train_ratio + self.config.calibration_ratio)),
        )
        cal_end = min(cal_end, int(unique_timestamps.numel()) - 1)

        train_cutoff = unique_timestamps[train_end - 1]
        calibration_cutoff = unique_timestamps[cal_end - 1]
        return {
            "train": torch.nonzero(timestamp_keys <= train_cutoff, as_tuple=False).squeeze(-1).to(device=device),
            "calibration": torch.nonzero(
                (timestamp_keys > train_cutoff) & (timestamp_keys <= calibration_cutoff),
                as_tuple=False,
            ).squeeze(-1).to(device=device),
            "evaluation": torch.nonzero(timestamp_keys > calibration_cutoff, as_tuple=False).squeeze(-1).to(
                device=device
            ),
        }

    def _build_full_batch(self, encoded: dict[str, Any]) -> TensorBatch:
        """Build a full-dataset batch without duplicating the underlying tensors."""
        X = encoded["X"]
        sensor_metadata = encoded["sensor_metadata"]
        return TensorBatch(
            X=X,
            y=encoded["y"],
            context=encoded["context"],
            M=encoded["M"],
            S=encoded["S"],
            missing_indicator=encoded["missing_indicator"],
            cost=encoded["cost"],
            sensor_metadata=sensor_metadata,
            indices=torch.arange(X.shape[0], dtype=torch.long, device=X.device),
        )

    def _build_batch(self, encoded: dict[str, Any], indices: Tensor) -> TensorBatch:
        """Slice one split batch from encoded tensors."""
        sensor_metadata = encoded["sensor_metadata"]
        return TensorBatch(
            X=encoded["X"][indices],
            y=encoded["y"][indices],
            context=None if encoded["context"] is None else encoded["context"][indices],
            M=encoded["M"][indices],
            S=encoded["S"][indices],
            missing_indicator=encoded["missing_indicator"][indices],
            cost=None if encoded["cost"] is None else encoded["cost"][indices],
            sensor_metadata=SensorMetadataBatch(
                sensor_instance=None if sensor_metadata.sensor_instance is None else sensor_metadata.sensor_instance[indices],
                sensor_type=None if sensor_metadata.sensor_type is None else sensor_metadata.sensor_type[indices],
                sensor_group=None if sensor_metadata.sensor_group is None else sensor_metadata.sensor_group[indices],
                sensor_modality=(
                    None if sensor_metadata.sensor_modality is None else sensor_metadata.sensor_modality[indices]
                ),
                installation_environment=(
                    None
                    if sensor_metadata.installation_environment is None
                    else sensor_metadata.installation_environment[indices]
                ),
                maintenance_state=(
                    None if sensor_metadata.maintenance_state is None else sensor_metadata.maintenance_state[indices]
                ),
                continuous=None if sensor_metadata.continuous is None else sensor_metadata.continuous[indices],
            ),
            indices=indices,
        )

    def _parse_timestamp(self, raw_value: str) -> datetime:
        """Parse timestamps from ISO strings or an explicit format."""
        value = raw_value.strip()
        if self.config.timestamp_format is not None:
            return datetime.strptime(value, self.config.timestamp_format)
        return datetime.fromisoformat(value)

    @staticmethod
    def _timestamp_sort_key(timestamp: datetime) -> int:
        """Build a stable integer key for temporal ordering and splitting."""
        return (
            timestamp.toordinal() * 86_400
            + timestamp.hour * 3_600
            + timestamp.minute * 60
            + timestamp.second
        )

    @staticmethod
    def _parse_float(raw_value: str) -> float:
        """Parse numeric values with blank-safe fallback."""
        value = raw_value.strip()
        if not value:
            return 0.0
        return float(value)

class ResearchExperimentRunner:
    """Experiment runner for the silence-aware sensing framework."""

    def __init__(
        self,
        data_config: TabularDataConfig,
        pipeline_config: SilenceAwareIDSConfig,
        run_config: ExperimentRunConfig | None = None,
    ) -> None:
        self.data_config = data_config
        self.pipeline_config = pipeline_config
        self.run_config = run_config or ExperimentRunConfig()
        self.dataset_adapter = WeatherDatasetAdapter(data_config)

    def _configure_runtime(self) -> dict[str, Any]:
        """Apply reproducibility settings before running an experiment."""
        torch.manual_seed(self.run_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.run_config.seed)
        torch.use_deterministic_algorithms(self.run_config.deterministic_algorithms, warn_only=True)
        if self.run_config.torch_num_threads is not None:
            torch.set_num_threads(self.run_config.torch_num_threads)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(self.run_config.matmul_precision)
        return {
            "seed": self.run_config.seed,
            "deterministic_algorithms": self.run_config.deterministic_algorithms,
            "torch_num_threads": self.run_config.torch_num_threads,
            "matmul_precision": self.run_config.matmul_precision,
        }

    @staticmethod
    def _runtime_environment() -> dict[str, Any]:
        """Collect runtime metadata for reproducible experiment reports."""
        return {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "gpytorch_version": gpytorch.__version__,
        }

    @staticmethod
    def _tensor_footprint_bytes(*tensors: Tensor | None) -> int:
        """Estimate tensor storage footprint in bytes."""
        total = 0
        for tensor in tensors:
            if tensor is None:
                continue
            total += tensor.numel() * tensor.element_size()
        return int(total)

    def prepare_data(self) -> PreparedExperimentData:
        """Prepare tensorized data for the configured experiment."""
        return self.dataset_adapter.prepare()

    @staticmethod
    def _slice_metadata_rows(metadata: SensorMetadataBatch, indices: Tensor) -> SensorMetadataBatch:
        """Slice sensor metadata along the leading row dimension."""
        return SensorMetadataBatch(
            sensor_instance=metadata.sensor_instance[indices],
            sensor_type=metadata.sensor_type[indices],
            sensor_group=metadata.sensor_group[indices],
            sensor_modality=metadata.sensor_modality[indices],
            installation_environment=metadata.installation_environment[indices],
            maintenance_state=metadata.maintenance_state[indices],
            continuous=None if metadata.continuous is None else metadata.continuous[indices],
        )

    @classmethod
    def _slice_batch_rows(cls, batch: TensorBatch, indices: Tensor) -> TensorBatch:
        """Slice a tensor batch along rows while preserving alignment."""
        return TensorBatch(
            X=batch.X[indices],
            y=batch.y[indices],
            context=None if batch.context is None else batch.context[indices],
            M=None if batch.M is None else batch.M[indices],
            S=None if batch.S is None else batch.S[indices],
            missing_indicator=batch.missing_indicator[indices],
            cost=None if batch.cost is None else batch.cost[indices],
            sensor_metadata=cls._slice_metadata_rows(batch.sensor_metadata, indices),
            indices=batch.indices[indices],
        )

    @staticmethod
    def _row_cap_indices(num_rows: int, max_rows: int, *, device: torch.device) -> Tensor:
        """Build evenly spaced chronological indices for large-row caps."""
        if num_rows <= max_rows:
            return torch.arange(num_rows, device=device, dtype=torch.long)
        linspace = torch.linspace(
            0,
            num_rows - 1,
            steps=max_rows,
            device=device,
            dtype=torch.float64,
        )
        return torch.unique_consecutive(torch.round(linspace).to(dtype=torch.long))

    def _effective_row_caps(self, prepared: PreparedExperimentData) -> tuple[int | None, int | None, int | None]:
        """Resolve explicit or automatic row caps for large framework runs."""
        train_cap = self.run_config.max_train_rows
        calibration_cap = self.run_config.max_calibration_rows
        evaluation_cap = self.run_config.max_evaluation_rows
        if train_cap is not None or calibration_cap is not None or evaluation_cap is not None:
            return train_cap, calibration_cap, evaluation_cap
        if not self.run_config.auto_scale_large_runs:
            return None, None, None
        train_rows = int(prepared.train.X.shape[0])
        calibration_rows = int(prepared.calibration.X.shape[0])
        evaluation_rows = int(prepared.evaluation.X.shape[0])
        if train_rows <= 500_000 and calibration_rows <= 100_000 and evaluation_rows <= 100_000:
            return None, None, None
        return 131_072, 32_768, 32_768

    def _ablation_candidate_limit(self, variant: SilenceAwareIDS) -> int | None:
        """Resolve a safe candidate-pool cap for expensive policy ablations."""
        planning_strategy = variant.config.policy.planning_strategy
        if planning_strategy == "ppo_online":
            return min(2048, max(variant.config.policy.ppo_max_candidates, 1024))
        if planning_strategy == "ppo_warmstart":
            return min(3072, max(variant.config.policy.ppo_max_candidates * 2, 1536))
        if planning_strategy == "non_myopic_rollout":
            return 4096
        if planning_strategy == "lazy_greedy" and variant.config.use_m5:
            return 4096
        return None

    def _ablation_evaluation_limit(self, variant: SilenceAwareIDS) -> int | None:
        """Resolve a safe evaluation-row cap for heavy policy ablations."""
        planning_strategy = variant.config.policy.planning_strategy
        if planning_strategy == "ppo_online":
            return 2048
        if planning_strategy == "ppo_warmstart":
            return 2048
        if planning_strategy == "non_myopic_rollout":
            return 3072
        if planning_strategy == "lazy_greedy" and variant.config.use_m5:
            return 4096
        return None

    def _sensitivity_evaluation_limit(self, prepared: PreparedExperimentData) -> int | None:
        """Resolve a safe evaluation-row cap for post-ablation sensitivity sweeps."""
        evaluation_rows = int(prepared.evaluation.X.shape[0])
        if not self.run_config.auto_scale_large_runs:
            return None
        if evaluation_rows <= 8192:
            return None
        return 8192

    def _cap_sensitivity_evaluation(
        self,
        prepared: PreparedExperimentData,
        *,
        batch_size: int | None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        SensorMetadataBatch | dict[str, Tensor] | None,
        int | None,
        dict[str, Any],
    ]:
        """Downsample evaluation rows for sensitivity analysis on large runs."""
        X_eval = prepared.evaluation.X
        y_eval = prepared.evaluation.y
        context_eval = prepared.evaluation.context
        M_eval = prepared.evaluation.M
        S_eval = prepared.evaluation.S
        sensor_metadata_eval = prepared.evaluation.sensor_metadata
        limit = self._sensitivity_evaluation_limit(prepared)
        effective_batch_size = batch_size
        if limit is None or X_eval.shape[0] <= limit:
            if effective_batch_size is not None and limit is not None:
                effective_batch_size = min(int(effective_batch_size), 256)
            return (
                X_eval,
                y_eval,
                context_eval,
                M_eval,
                S_eval,
                sensor_metadata_eval,
                effective_batch_size,
                {
                    "applied": False,
                    "limit": limit,
                    "original_rows": int(X_eval.shape[0]),
                    "effective_rows": int(X_eval.shape[0]),
                    "evaluation_batch_size": None if effective_batch_size is None else int(effective_batch_size),
                },
            )
        index = self._row_cap_indices(int(X_eval.shape[0]), int(limit), device=X_eval.device)
        if effective_batch_size is not None:
            effective_batch_size = min(int(effective_batch_size), 256)
        metadata = None if sensor_metadata_eval is None else _slice_metadata(sensor_metadata_eval, index)
        return (
            X_eval[index],
            y_eval[index],
            _slice_optional(context_eval, index),
            _slice_optional(M_eval, index),
            _slice_optional(S_eval, index),
            metadata,
            effective_batch_size,
            {
                "applied": True,
                "limit": int(limit),
                "original_rows": int(X_eval.shape[0]),
                "effective_rows": int(index.shape[0]),
                "evaluation_batch_size": None if effective_batch_size is None else int(effective_batch_size),
            },
        )

    def _effective_benchmark_candidate_limit(self, prepared: PreparedExperimentData) -> int | None:
        """Resolve a safe benchmark candidate-pool cap for large framework runs."""
        if self.run_config.benchmark_candidate_pool_size is not None:
            return int(self.run_config.benchmark_candidate_pool_size)
        if not self.run_config.auto_scale_large_runs:
            return None
        evaluation_rows = int(prepared.evaluation.X.shape[0])
        if evaluation_rows < 16384:
            return None
        return 65536

    def _cap_ablation_evaluation(
        self,
        variant: SilenceAwareIDS,
        *,
        X_eval: Tensor,
        y_eval: Tensor,
        context_eval: Tensor | None,
        M_eval: Tensor | None,
        S_eval: Tensor | None,
        sensor_metadata_eval: SensorMetadataBatch | dict[str, Tensor] | None,
        batch_size: int | None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        SensorMetadataBatch | dict[str, Tensor] | None,
        int | None,
        dict[str, Any],
    ]:
        """Downsample evaluation rows for expensive policy ablations."""
        limit = self._ablation_evaluation_limit(variant)
        effective_batch_size = batch_size
        if limit is None or X_eval.shape[0] <= limit:
            if effective_batch_size is not None and limit is not None:
                effective_batch_size = min(int(effective_batch_size), 64)
            return (
                X_eval,
                y_eval,
                context_eval,
                M_eval,
                S_eval,
                sensor_metadata_eval,
                effective_batch_size,
                {
                    "applied": False,
                    "limit": limit,
                    "original_rows": int(X_eval.shape[0]),
                    "effective_rows": int(X_eval.shape[0]),
                    "evaluation_batch_size": None if effective_batch_size is None else int(effective_batch_size),
                    "planning_strategy": variant.config.policy.planning_strategy,
                },
            )
        index = self._row_cap_indices(int(X_eval.shape[0]), int(limit), device=X_eval.device)
        if effective_batch_size is not None:
            effective_batch_size = min(int(effective_batch_size), 64)
        metadata = None if sensor_metadata_eval is None else _slice_metadata(sensor_metadata_eval, index)
        return (
            X_eval[index],
            y_eval[index],
            _slice_optional(context_eval, index),
            _slice_optional(M_eval, index),
            _slice_optional(S_eval, index),
            metadata,
            effective_batch_size,
            {
                "applied": True,
                "limit": int(limit),
                "original_rows": int(X_eval.shape[0]),
                "effective_rows": int(index.shape[0]),
                "evaluation_batch_size": None if effective_batch_size is None else int(effective_batch_size),
                "planning_strategy": variant.config.policy.planning_strategy,
            },
        )

    def _cap_candidate_pool(
        self,
        variant: SilenceAwareIDS,
        *,
        candidate_x: Tensor | None,
        candidate_cost: Tensor | None,
        candidate_context: Tensor | None,
        candidate_M: Tensor | None,
        candidate_S: Tensor | None,
        candidate_sensor_metadata: SensorMetadataBatch | dict[str, Tensor] | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, SensorMetadataBatch | dict[str, Tensor] | None, dict[str, Any]]:
        """Downsample large candidate pools for heavy policy ablations."""
        if candidate_x is None or candidate_cost is None:
            return (
                candidate_x,
                candidate_cost,
                candidate_context,
                candidate_M,
                candidate_S,
                candidate_sensor_metadata,
                {"applied": False},
            )
        limit = self._ablation_candidate_limit(variant)
        if limit is None or candidate_x.shape[0] <= limit:
            return (
                candidate_x,
                candidate_cost,
                candidate_context,
                candidate_M,
                candidate_S,
                candidate_sensor_metadata,
                {"applied": False, "limit": limit, "rows": int(candidate_x.shape[0])},
            )
        index = self._row_cap_indices(int(candidate_x.shape[0]), int(limit), device=candidate_x.device)
        metadata = None if candidate_sensor_metadata is None else _slice_metadata(candidate_sensor_metadata, index)
        return (
            candidate_x[index],
            candidate_cost[index],
            _slice_optional(candidate_context, index),
            _slice_optional(candidate_M, index),
            _slice_optional(candidate_S, index),
            metadata,
            {
                "applied": True,
                "limit": int(limit),
                "original_rows": int(candidate_x.shape[0]),
                "effective_rows": int(index.shape[0]),
                "planning_strategy": variant.config.policy.planning_strategy,
            },
        )

    def _maybe_cap_prepared_data(self, prepared: PreparedExperimentData) -> tuple[PreparedExperimentData, dict[str, Any]]:
        """Downsample very large prepared splits to stable framework-run caps."""
        train_cap, calibration_cap, evaluation_cap = self._effective_row_caps(prepared)
        caps_payload = {
            "applied": False,
            "train_cap": train_cap,
            "calibration_cap": calibration_cap,
            "evaluation_cap": evaluation_cap,
            "original_train_rows": int(prepared.train.X.shape[0]),
            "original_calibration_rows": int(prepared.calibration.X.shape[0]),
            "original_evaluation_rows": int(prepared.evaluation.X.shape[0]),
        }
        if train_cap is None and calibration_cap is None and evaluation_cap is None:
            return prepared, caps_payload

        def cap_batch(batch: TensorBatch, max_rows: int | None) -> TensorBatch:
            if max_rows is None or batch.X.shape[0] <= max_rows:
                return batch
            indices = self._row_cap_indices(int(batch.X.shape[0]), int(max_rows), device=batch.X.device)
            return self._slice_batch_rows(batch, indices)

        retained_full = prepared.full
        retained_full_source = "full"
        if self.run_config.use_evaluation_as_candidate_pool:
            retained_full = cap_batch(prepared.evaluation, evaluation_cap)
            retained_full_source = "evaluation_proxy"

        capped = PreparedExperimentData(
            full=retained_full,
            train=cap_batch(prepared.train, train_cap),
            calibration=cap_batch(prepared.calibration, calibration_cap),
            evaluation=cap_batch(prepared.evaluation, evaluation_cap),
            metadata_cardinalities=prepared.metadata_cardinalities,
            feature_schema=prepared.feature_schema,
        )
        caps_payload["applied"] = True
        caps_payload["effective_train_rows"] = int(capped.train.X.shape[0])
        caps_payload["effective_calibration_rows"] = int(capped.calibration.X.shape[0])
        caps_payload["effective_evaluation_rows"] = int(capped.evaluation.X.shape[0])
        caps_payload["retained_full_rows"] = int(capped.full.X.shape[0])
        caps_payload["retained_full_source"] = retained_full_source
        return capped, caps_payload

    def _effective_inference_batch_size(self, row_caps: dict[str, Any]) -> int:
        """Resolve a memory-safe inference batch size for framework runs."""
        batch_size = self.run_config.prediction_batch_size or self.pipeline_config.state_training.batch_size
        if row_caps.get("applied"):
            batch_size = min(int(batch_size), 128)
        return max(int(batch_size), 1)

    @staticmethod
    def _repeat_rows(tensor: Tensor | None, repeats: int, *, limit: int | None = None) -> Tensor | None:
        """Repeat a tensor along the leading dimension for benchmark stress tests."""
        if tensor is None:
            return None
        if tensor.ndim == 1:
            repeated = tensor.repeat(repeats)
        else:
            repeated = tensor.repeat((repeats, 1))
        return repeated if limit is None else repeated[:limit]

    @classmethod
    def _repeat_metadata(
        cls,
        metadata: SensorMetadataBatch,
        repeats: int,
        *,
        limit: int | None = None,
    ) -> SensorMetadataBatch:
        """Repeat sensor metadata rows for benchmark-scale candidate pools."""
        return SensorMetadataBatch(
            sensor_instance=cls._repeat_rows(metadata.sensor_instance, repeats, limit=limit),
            sensor_type=cls._repeat_rows(metadata.sensor_type, repeats, limit=limit),
            sensor_group=cls._repeat_rows(metadata.sensor_group, repeats, limit=limit),
            sensor_modality=cls._repeat_rows(metadata.sensor_modality, repeats, limit=limit),
            installation_environment=cls._repeat_rows(
                metadata.installation_environment,
                repeats,
                limit=limit,
            ),
            maintenance_state=cls._repeat_rows(metadata.maintenance_state, repeats, limit=limit),
            continuous=cls._repeat_rows(metadata.continuous, repeats, limit=limit),
        )

    def _run_large_scale_benchmark(
        self,
        pipeline: SilenceAwareIDS,
        prepared: PreparedExperimentData,
        *,
        progress_callback: Any | None = None,
    ) -> dict[str, Any] | None:
        """Stress-test batched inference and policy selection on expanded candidate pools."""
        expansion_factor = self.run_config.benchmark_expansion_factor
        if expansion_factor <= 1 and self.run_config.benchmark_candidate_pool_size is None:
            return None

        evaluation = prepared.evaluation
        candidate_limit = self._effective_benchmark_candidate_limit(prepared)
        expanded_x = self._repeat_rows(evaluation.X, expansion_factor, limit=candidate_limit)
        expanded_context = self._repeat_rows(evaluation.context, expansion_factor, limit=candidate_limit)
        expanded_m = self._repeat_rows(evaluation.M, expansion_factor, limit=candidate_limit)
        expanded_s = self._repeat_rows(evaluation.S, expansion_factor, limit=candidate_limit)
        expanded_cost = self._repeat_rows(evaluation.cost, expansion_factor, limit=candidate_limit)
        expanded_metadata = self._repeat_metadata(
            evaluation.sensor_metadata,
            expansion_factor,
            limit=candidate_limit,
        )
        if expanded_x is None:
            return None

        batch_size = self.run_config.prediction_batch_size or self.pipeline_config.state_training.batch_size
        if progress_callback is not None:
            progress_callback(
                "benchmark_prepare_complete",
                {
                    "expansion_factor": int(expansion_factor),
                    "candidate_limit": candidate_limit,
                    "benchmark_rows": int(expanded_x.shape[0]),
                    "batch_size": int(batch_size),
                },
            )
        predict_start = time.perf_counter()
        if progress_callback is not None:
            progress_callback("benchmark_predict_start", {"rows": int(expanded_x.shape[0])})
        mean, variance = pipeline.predict_state(expanded_x, batch_size=batch_size)
        predict_seconds = time.perf_counter() - predict_start
        missingness_start = time.perf_counter()
        if progress_callback is not None:
            progress_callback("benchmark_missingness_start", {"rows": int(expanded_x.shape[0])})
        p_miss = pipeline.predict_missingness(
            expanded_x,
            context=expanded_context,
            M=expanded_m,
            S=expanded_s,
            sensor_metadata=expanded_metadata,
            batch_size=batch_size,
        )
        missingness_seconds = time.perf_counter() - missingness_start
        selection = None
        selection_seconds = 0.0
        if expanded_cost is not None:
            selection_start = time.perf_counter()
            if progress_callback is not None:
                progress_callback(
                    "benchmark_selection_start",
                    {"rows": int(expanded_x.shape[0]), "max_selections": self.run_config.max_selections},
                )
            selection = _selection_to_summary(
                pipeline.select_sensors(
                    expanded_x,
                    expanded_cost,
                    context=expanded_context,
                    M=expanded_m,
                    S=expanded_s,
                    sensor_metadata=expanded_metadata,
                    budget=self.run_config.policy_budget,
                    max_selections=self.run_config.max_selections,
                    batch_size=batch_size,
                )
            )
            selection_seconds = time.perf_counter() - selection_start
        if progress_callback is not None:
            progress_callback(
                "benchmark_complete",
                {
                    "predict_seconds": float(predict_seconds),
                    "missingness_seconds": float(missingness_seconds),
                    "selection_seconds": float(selection_seconds),
                },
            )

        return {
            "expansion_factor": expansion_factor,
            "benchmark_rows": int(expanded_x.shape[0]),
            "prediction_batch_size": int(batch_size),
            "predict_seconds": predict_seconds,
            "missingness_seconds": missingness_seconds,
            "selection_seconds": selection_seconds,
            "input_tensor_bytes": self._tensor_footprint_bytes(
                expanded_x,
                expanded_context,
                expanded_m,
                expanded_s,
                expanded_cost,
                expanded_metadata.sensor_type,
                expanded_metadata.sensor_group,
                expanded_metadata.sensor_modality,
                expanded_metadata.installation_environment,
                expanded_metadata.maintenance_state,
                expanded_metadata.continuous,
            ),
            "output_tensor_bytes": self._tensor_footprint_bytes(mean, variance, p_miss),
            "mean_predictive_variance": float(torch.clamp(variance, min=1e-6).mean().item()),
            "mean_predictive_mean_abs": float(mean.abs().mean().item()),
            "mean_missingness_proba": None if p_miss is None else float(p_miss.mean().item()),
            "selection": selection,
        }

    def _checkpoint_signature(self) -> dict[str, Any]:
        """Build a lightweight signature to validate resume checkpoints."""
        return {
            "data_path": str(self.data_config.data_path.resolve()),
            "target_column": self.data_config.target_column,
            "variant_names": list(self.run_config.variant_names),
            "train_ratio": self.data_config.train_ratio,
            "calibration_ratio": self.data_config.calibration_ratio,
            "split_strategy": self.data_config.split_strategy,
            "state_epochs": self.pipeline_config.state_training.epochs,
            "missingness_epochs": self.pipeline_config.missingness_training.epochs,
            "prediction_batch_size": self.run_config.prediction_batch_size,
            "benchmark_expansion_factor": self.run_config.benchmark_expansion_factor,
            "max_train_rows": self.run_config.max_train_rows,
            "max_calibration_rows": self.run_config.max_calibration_rows,
            "max_evaluation_rows": self.run_config.max_evaluation_rows,
            "auto_scale_large_runs": self.run_config.auto_scale_large_runs,
        }

    @staticmethod
    def _progress_paths(output_path: Path) -> tuple[Path, Path]:
        """Return progress and checkpoint paths for one framework run."""
        output_dir = output_path.parent
        return output_dir / "framework_progress.json", output_dir / "framework_run_checkpoint.pt"

    @staticmethod
    def _heartbeat_path(output_path: Path) -> Path:
        """Return the detached heartbeat trace path for one framework run."""
        return output_path.parent / "framework_heartbeat.jsonl"

    @staticmethod
    def _append_heartbeat(
        path: Path,
        *,
        stage: str,
        step: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append one JSONL heartbeat event for detached debugging."""
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "step": step,
            "payload": {} if payload is None else payload,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _prepared_cache_signature(self) -> dict[str, Any]:
        """Build a compatibility signature for prepared-data caches."""
        return {
            "checkpoint_signature": self._checkpoint_signature(),
            "data_mtime_ns": self.data_config.data_path.stat().st_mtime_ns,
        }

    def _prepared_cache_path(self, output_path: Path) -> Path:
        """Resolve the prepared-data cache path for one framework run."""
        output_dir = output_path.parent
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(
            json.dumps(self._prepared_cache_signature(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return cache_dir / f"prepared_{digest}.pt"

    def _try_load_prepared_cache(self, cache_path: Path) -> tuple[PreparedExperimentData, dict[str, Any]] | None:
        """Load cached prepared data when the signature matches."""
        if not cache_path.exists():
            return None
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("signature") != self._prepared_cache_signature():
            return None
        prepared = PreparedExperimentData(
            full=_batch_from_cache_dict(payload["full"]),
            train=_batch_from_cache_dict(payload["train"]),
            calibration=_batch_from_cache_dict(payload["calibration"]),
            evaluation=_batch_from_cache_dict(payload["evaluation"]),
            metadata_cardinalities=dict(payload["metadata_cardinalities"]),
            feature_schema=dict(payload["feature_schema"]),
        )
        row_caps = dict(payload.get("row_caps", {}))
        return prepared, row_caps

    def _save_prepared_cache(
        self,
        cache_path: Path,
        prepared: PreparedExperimentData,
        row_caps: dict[str, Any],
    ) -> None:
        """Persist capped prepared splits for fast framework-run resume."""
        torch.save(
            {
                "signature": self._prepared_cache_signature(),
                "full": _batch_to_cache_dict(prepared.full),
                "train": _batch_to_cache_dict(prepared.train),
                "calibration": _batch_to_cache_dict(prepared.calibration),
                "evaluation": _batch_to_cache_dict(prepared.evaluation),
                "metadata_cardinalities": prepared.metadata_cardinalities,
                "feature_schema": prepared.feature_schema,
                "row_caps": row_caps,
            },
            cache_path,
        )

    def _load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any] | None:
        """Load a compatible checkpoint when resume is enabled."""
        if not self.run_config.resume_from_checkpoint or not checkpoint_path.exists():
            return None
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("signature") != self._checkpoint_signature():
            return None
        return payload

    @staticmethod
    def _serialize_reliability_state(pipeline: SilenceAwareIDS) -> dict[str, Any] | None:
        """Serialize non-module reliability state that is not in pipeline.state_dict()."""
        reliability = pipeline.reliability_model
        if reliability is None:
            return None
        return {
            "q_hat": None if reliability._q_hat is None else float(reliability._q_hat.item()),
            "n_calibration": int(reliability._n_calibration),
            "adaptive_epsilon": float(reliability._adaptive_epsilon),
            "calibration_scores": None
            if reliability._calibration_scores is None
            else reliability._calibration_scores.detach().cpu(),
            "graph_feature_dim": reliability._graph_feature_dim,
            "graph_adapter_state_dict": None
            if reliability._graph_adapter is None
            else reliability._graph_adapter.state_dict(),
            "graph_calibration_features": None
            if reliability._graph_calibration_features is None
            else reliability._graph_calibration_features.detach().cpu(),
            "graph_calibration_scores": None
            if reliability._graph_calibration_scores is None
            else reliability._graph_calibration_scores.detach().cpu(),
        }

    @staticmethod
    def _restore_reliability_state(pipeline: SilenceAwareIDS, payload: dict[str, Any] | None) -> None:
        """Restore reliability state into a freshly constructed pipeline."""
        reliability = pipeline.reliability_model
        if reliability is None or payload is None:
            return
        device = next(pipeline.parameters()).device
        dtype = next(pipeline.parameters()).dtype
        q_hat = payload.get("q_hat")
        reliability._q_hat = None if q_hat is None else torch.tensor(float(q_hat), device=device, dtype=dtype)
        reliability._n_calibration = int(payload.get("n_calibration", 0))
        reliability._adaptive_epsilon = float(payload.get("adaptive_epsilon", reliability.config.epsilon))
        calibration_scores = payload.get("calibration_scores")
        reliability._calibration_scores = None if calibration_scores is None else calibration_scores.to(device=device, dtype=dtype)
        reliability._graph_feature_dim = payload.get("graph_feature_dim")
        reliability._graph_calibration_features = None
        graph_features = payload.get("graph_calibration_features")
        if graph_features is not None:
            reliability._graph_calibration_features = graph_features.to(device=device, dtype=dtype)
        reliability._graph_calibration_scores = None
        graph_scores = payload.get("graph_calibration_scores")
        if graph_scores is not None:
            reliability._graph_calibration_scores = graph_scores.to(device=device, dtype=dtype)
        graph_state = payload.get("graph_adapter_state_dict")
        graph_dim = payload.get("graph_feature_dim")
        if graph_state is not None and graph_dim is not None:
            reliability._ensure_graph_adapter(int(graph_dim), device=device, dtype=dtype)
            if reliability._graph_adapter is not None:
                reliability._graph_adapter.load_state_dict(graph_state)

    def _serialize_pipeline_checkpoint(self, pipeline: SilenceAwareIDS) -> dict[str, Any]:
        """Serialize fitted pipeline state for checkpoint/resume."""
        return {
            "module_state_dict": pipeline.state_dict(),
            "reliability_state": self._serialize_reliability_state(pipeline),
            "fault_detection_threshold": pipeline._fault_detection_threshold,
            "fault_component_stats": pipeline._fault_component_stats,
        }

    def _restore_pipeline_checkpoint(self, pipeline: SilenceAwareIDS, payload: dict[str, Any] | None) -> None:
        """Restore fitted pipeline state from a saved checkpoint."""
        if payload is None:
            return
        pipeline.load_state_dict(payload["module_state_dict"])
        pipeline._fault_detection_threshold = payload.get("fault_detection_threshold")
        pipeline._fault_component_stats = payload.get("fault_component_stats")
        self._restore_reliability_state(pipeline, payload.get("reliability_state"))

    def _write_progress_payload(
        self,
        *,
        path: Path,
        summary_path: Path,
        status: str,
        stage: str,
        runtime_environment: dict[str, Any],
        reproducibility: dict[str, Any],
        dataset_summary: dict[str, Any] | None = None,
        fit_summary: PipelineFitSummary | None = None,
        base_metrics: dict[str, float] | None = None,
        ablations: dict[str, AblationOutcome] | None = None,
        sensitivity: dict[str, dict[str, float]] | None = None,
        selection: dict[str, Any] | None = None,
        benchmark: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Write a stage-progress snapshot and partial summary."""
        payload = {
            "status": status,
            "stage": stage,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "error": error,
            "fit_summary": _serialize_fit_summary(fit_summary),
            "base_metrics": {} if base_metrics is None else base_metrics,
            "ablations": _serialize_ablation_outcomes({} if ablations is None else ablations),
            "completed_ablation_variants": [] if ablations is None else list(ablations.keys()),
            "sensitivity": {} if sensitivity is None else sensitivity,
            "selection": selection,
            "dataset_summary": {} if dataset_summary is None else dataset_summary,
            "runtime_environment": runtime_environment,
            "reproducibility": reproducibility,
            "benchmark": benchmark,
            "is_partial": status != "completed",
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        stage: str,
        pipeline: SilenceAwareIDS | None = None,
        include_pipeline_state: bool = True,
        fit_summary: PipelineFitSummary | None = None,
        base_metrics: dict[str, float] | None = None,
        ablations: dict[str, AblationOutcome] | None = None,
        sensitivity: dict[str, dict[str, float]] | None = None,
        selection: dict[str, Any] | None = None,
        dataset_summary: dict[str, Any] | None = None,
        runtime_environment: dict[str, Any] | None = None,
        reproducibility: dict[str, Any] | None = None,
        benchmark: dict[str, Any] | None = None,
        completed_result: ExperimentResult | None = None,
    ) -> None:
        """Persist pipeline and metric state for resume."""
        pipeline_state: dict[str, Any] | None = None
        if pipeline is not None and include_pipeline_state:
            pipeline_state = self._serialize_pipeline_checkpoint(pipeline)
        elif checkpoint_path.exists():
            try:
                existing_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception:
                existing_payload = None
            if (
                isinstance(existing_payload, dict)
                and existing_payload.get("signature") == self._checkpoint_signature()
            ):
                pipeline_state = existing_payload.get("pipeline_state")
        payload = {
            "signature": self._checkpoint_signature(),
            "stage": stage,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "pipeline_state": pipeline_state,
            "fit_summary": _serialize_fit_summary(fit_summary),
            "base_metrics": {} if base_metrics is None else base_metrics,
            "ablations": _serialize_ablation_outcomes({} if ablations is None else ablations),
            "sensitivity": {} if sensitivity is None else sensitivity,
            "selection": selection,
            "dataset_summary": {} if dataset_summary is None else dataset_summary,
            "runtime_environment": {} if runtime_environment is None else runtime_environment,
            "reproducibility": {} if reproducibility is None else reproducibility,
            "benchmark": benchmark,
            "completed_result": None if completed_result is None else completed_result.to_dict(),
        }
        torch.save(payload, checkpoint_path)

    def _ensure_dynamic_silence(
        self,
        pipeline: SilenceAwareIDS,
        batch: TensorBatch,
        *,
        batch_size: int | None,
        heartbeat_path: Path,
        stage: str,
    ) -> Tensor | None:
        """Return a usable dynamic-silence tensor, recomputing it only when required.

        Resume runs used to re-enter base evaluation because silence detection happened
        before the `base_metrics is None` guard. This helper makes the recomputation
        explicit so later stages such as sensitivity can request it without forcing the
        full base-evaluation path to run again.
        """
        dynamic_s = batch.S
        if not pipeline.config.use_m2 or dynamic_s is None:
            return dynamic_s
        if torch.count_nonzero(dynamic_s).item() != 0:
            return dynamic_s
        self._append_heartbeat(
            heartbeat_path,
            stage=stage,
            step="evaluation_silence_start",
            payload={"rows": int(batch.X.shape[0])},
        )
        dynamic_s = pipeline.detect_silence(
            batch.X,
            batch.y,
            context=batch.context,
            sensor_metadata=batch.sensor_metadata,
            batch_size=batch_size,
        )["dynamic_silence"].float()
        self._append_heartbeat(
            heartbeat_path,
            stage=stage,
            step="evaluation_silence_complete",
            payload={"flagged": int(dynamic_s.sum().item())},
        )
        return dynamic_s

    def _run_ablation_suite_incremental(
        self,
        base_pipeline: SilenceAwareIDS,
        X_train: Tensor,
        y_train: Tensor,
        X_eval: Tensor,
        y_eval: Tensor,
        *,
        missing_indicator_train: Tensor | None,
        context_train: Tensor | None,
        M_train: Tensor | None,
        S_train: Tensor | None,
        sensor_metadata_train: SensorMetadataBatch | dict[str, Tensor] | None,
        context_eval: Tensor | None,
        M_eval: Tensor | None,
        S_eval: Tensor | None,
        sensor_metadata_eval: SensorMetadataBatch | dict[str, Tensor] | None,
        X_cal: Tensor | None,
        y_cal: Tensor | None,
        context_cal: Tensor | None,
        M_cal: Tensor | None,
        S_cal: Tensor | None,
        sensor_metadata_cal: SensorMetadataBatch | dict[str, Tensor] | None,
        candidate_x: Tensor | None,
        candidate_cost: Tensor | None,
        candidate_context: Tensor | None,
        candidate_M: Tensor | None,
        candidate_S: Tensor | None,
        candidate_sensor_metadata: SensorMetadataBatch | dict[str, Tensor] | None,
        budget: float | None,
        max_selections: int | None,
        variant_names: list[str] | None,
        batch_size: int | None,
        existing_results: dict[str, AblationOutcome] | None = None,
        on_variant_complete: Any | None = None,
        on_variant_progress: Any | None = None,
    ) -> dict[str, AblationOutcome]:
        """Run ablations one variant at a time so progress can be checkpointed."""
        variants = base_pipeline.build_ablation_configs()
        active_names = variant_names or list(variants.keys())
        results = dict(existing_results or {})
        for variant_name in active_names:
            if variant_name in results:
                continue
            if variant_name not in variants:
                raise KeyError(f"Unknown ablation variant: {variant_name}")
            variant = SilenceAwareIDS(variants[variant_name])
            (
                X_eval_variant,
                y_eval_variant,
                context_eval_variant,
                M_eval_variant,
                S_eval_variant,
                sensor_metadata_eval_variant,
                eval_batch_size,
                evaluation_cap_info,
            ) = self._cap_ablation_evaluation(
                variant,
                X_eval=X_eval,
                y_eval=y_eval,
                context_eval=context_eval,
                M_eval=M_eval,
                S_eval=S_eval,
                sensor_metadata_eval=sensor_metadata_eval,
                batch_size=batch_size,
            )
            if on_variant_progress is not None:
                on_variant_progress(variant_name, "fit_start", evaluation_cap_info)
            fit_summary = variant.fit(
                X_train,
                y_train,
                missing_indicator_train=missing_indicator_train,
                context_train=context_train,
                M_train=M_train,
                S_train=S_train,
                sensor_metadata_train=sensor_metadata_train,
                X_cal=X_cal,
                y_cal=y_cal,
                context_cal=context_cal,
                M_cal=M_cal,
                S_cal=S_cal,
                sensor_metadata_cal=sensor_metadata_cal,
            )
            if on_variant_progress is not None:
                on_variant_progress(
                    variant_name,
                    "fit_complete",
                    {
                        "state_epochs": len(fit_summary.state_history.get("loss", [])),
                        "missingness_epochs": (
                            0
                            if fit_summary.missingness_history is None
                            else len(fit_summary.missingness_history.get("loss", []))
                        ),
                    },
                )
            eval_s = S_eval
            if evaluation_cap_info["applied"]:
                eval_s = S_eval_variant
            if eval_s is None and variant.config.use_m2:
                if on_variant_progress is not None:
                    on_variant_progress(variant_name, "silence_start", evaluation_cap_info)
                eval_s = variant.detect_silence(
                    X_eval_variant,
                    y_eval_variant,
                    context=context_eval_variant,
                    sensor_metadata=sensor_metadata_eval_variant,
                    batch_size=eval_batch_size,
                )["dynamic_silence"].float()
                if on_variant_progress is not None:
                    on_variant_progress(
                        variant_name,
                        "silence_complete",
                        {
                            **evaluation_cap_info,
                            "flagged": int(eval_s.sum().item()),
                        },
                    )
            if on_variant_progress is not None:
                on_variant_progress(variant_name, "evaluate_start", evaluation_cap_info)
            metrics = variant.evaluate_predictions(
                X_eval_variant,
                y_eval_variant,
                context=context_eval_variant,
                M=M_eval_variant,
                S=eval_s,
                sensor_metadata=sensor_metadata_eval_variant,
                integrate_missingness=variant.use_m3,
                batch_size=eval_batch_size,
                progress_callback=(
                    None
                    if on_variant_progress is None
                    else lambda step, payload, name=variant_name: on_variant_progress(
                        name,
                        f"evaluate::{step}",
                        payload,
                    )
                ),
            )
            if on_variant_progress is not None:
                on_variant_progress(
                    variant_name,
                    "evaluate_complete",
                    {
                        **evaluation_cap_info,
                        "rmse": float(metrics.get("rmse", 0.0)),
                        "crps": float(metrics.get("crps", 0.0)),
                    },
                )
            selection: dict[str, Tensor | list[int] | float] | None = None
            if candidate_x is not None and candidate_cost is not None:
                (
                    selection_x,
                    selection_cost,
                    selection_context,
                    selection_m,
                    selection_s,
                    selection_metadata,
                    candidate_cap_info,
                ) = self._cap_candidate_pool(
                    variant,
                    candidate_x=candidate_x,
                    candidate_cost=candidate_cost,
                    candidate_context=candidate_context,
                    candidate_M=candidate_M,
                    candidate_S=candidate_S,
                    candidate_sensor_metadata=candidate_sensor_metadata,
                )
                if on_variant_progress is not None:
                    on_variant_progress(
                        variant_name,
                        "selection_start",
                        candidate_cap_info,
                    )
                selection = variant.select_sensors(
                    selection_x,
                    selection_cost,
                    context=selection_context,
                    M=selection_m,
                    S=selection_s,
                    sensor_metadata=selection_metadata,
                    budget=budget,
                    max_selections=max_selections,
                    batch_size=batch_size,
                )
                if on_variant_progress is not None:
                    on_variant_progress(
                        variant_name,
                        "selection_complete",
                        {
                            "selected": len(selection.get("selected_indices", [])),
                            "planning_strategy": variant.config.policy.planning_strategy,
                        },
                    )
            report = variant.ablation_report()
            report["ablation_evaluation"] = evaluation_cap_info
            report["fit_summary"] = {
                "state_epochs": len(fit_summary.state_history.get("loss", [])),
                "observation_steps": (
                    None
                    if fit_summary.observation_history is None
                    else len(fit_summary.observation_history.get("loss", []))
                ),
                "dynamic_silence_threshold": fit_summary.dynamic_silence_threshold,
                "missingness_epochs": (
                    None
                    if fit_summary.missingness_history is None
                    else len(fit_summary.missingness_history.get("loss", []))
                ),
                "conformal_quantile": fit_summary.conformal_quantile,
            }
            results[variant_name] = AblationOutcome(
                variant_name=variant_name,
                metrics=metrics,
                report=report,
                selection=selection,
            )
            if on_variant_complete is not None:
                on_variant_complete(variant_name, results)
        return results

    def run(self) -> ExperimentResult:
        """Run fitting, ablation study, sensitivity analysis, and policy selection.

        Returns:
            Experiment result bundle.
        """
        reproducibility = self._configure_runtime()
        prepared_raw = self.prepare_data()
        prepared, row_caps = self._maybe_cap_prepared_data(prepared_raw)
        del prepared_raw
        gc.collect()
        resolved_pipeline_config = self._resolve_pipeline_config(prepared)
        resolved_pipeline_config = self._stabilize_pipeline_config_for_large_runs(
            resolved_pipeline_config,
            row_caps=row_caps,
        )
        pipeline = SilenceAwareIDS(resolved_pipeline_config)
        inference_batch_size = self._effective_inference_batch_size(row_caps)

        fit_summary = pipeline.fit(
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
        gc.collect()

        evaluation_s = prepared.evaluation.S
        if pipeline.config.use_m2 and evaluation_s is not None and torch.count_nonzero(evaluation_s).item() == 0:
            evaluation_s = pipeline.detect_silence(
                prepared.evaluation.X,
                prepared.evaluation.y,
                context=prepared.evaluation.context,
                sensor_metadata=prepared.evaluation.sensor_metadata,
                batch_size=inference_batch_size,
            )["dynamic_silence"].float()

        base_metrics = pipeline.evaluate_predictions(
            prepared.evaluation.X,
            prepared.evaluation.y,
            context=prepared.evaluation.context,
            M=prepared.evaluation.M,
            S=evaluation_s,
            sensor_metadata=prepared.evaluation.sensor_metadata,
            integrate_missingness=pipeline.use_m3,
            batch_size=inference_batch_size,
        )

        candidate_batch = prepared.evaluation if self.run_config.use_evaluation_as_candidate_pool else prepared.full
        selection = None
        if candidate_batch.cost is not None:
            raw_selection = pipeline.select_sensors(
                candidate_batch.X,
                candidate_batch.cost,
                context=candidate_batch.context,
                M=candidate_batch.M,
                S=candidate_batch.S,
                sensor_metadata=candidate_batch.sensor_metadata,
                budget=self.run_config.policy_budget,
                max_selections=self.run_config.max_selections,
                batch_size=inference_batch_size,
            )
            selection = _selection_to_summary(raw_selection)

        ablations = pipeline.run_ablation_suite(
            prepared.train.X,
            prepared.train.y,
            prepared.evaluation.X,
            prepared.evaluation.y,
            missing_indicator_train=prepared.train.missing_indicator,
            context_train=prepared.train.context,
            M_train=prepared.train.M,
            S_train=prepared.train.S,
            sensor_metadata_train=prepared.train.sensor_metadata,
            context_eval=prepared.evaluation.context,
            M_eval=prepared.evaluation.M,
            S_eval=evaluation_s,
            sensor_metadata_eval=prepared.evaluation.sensor_metadata,
            X_cal=prepared.calibration.X,
            y_cal=prepared.calibration.y,
            context_cal=prepared.calibration.context,
            M_cal=prepared.calibration.M,
            S_cal=prepared.calibration.S,
            sensor_metadata_cal=prepared.calibration.sensor_metadata,
            candidate_x=candidate_batch.X,
            candidate_cost=candidate_batch.cost,
            candidate_context=candidate_batch.context,
            candidate_M=candidate_batch.M,
            candidate_S=candidate_batch.S,
            candidate_sensor_metadata=candidate_batch.sensor_metadata,
            budget=self.run_config.policy_budget,
            max_selections=self.run_config.max_selections,
            variant_names=list(self.run_config.variant_names) if self.run_config.variant_names else None,
            batch_size=inference_batch_size,
        )

        sensitivity: dict[str, dict[str, float]]
        if pipeline.use_m3:
            sensitivity = pipeline.run_missingness_sensitivity_analysis(
                prepared.evaluation.X,
                prepared.evaluation.y,
                logit_scales=list(self.run_config.sensitivity_logit_scales),
                context=prepared.evaluation.context,
                M=prepared.evaluation.M,
                S=evaluation_s,
                sensor_metadata=prepared.evaluation.sensor_metadata,
                batch_size=inference_batch_size,
            )
        else:
            sensitivity = {}

        dataset_summary = {
            "train_rows": int(prepared.train.X.shape[0]),
            "calibration_rows": int(prepared.calibration.X.shape[0]),
            "evaluation_rows": int(prepared.evaluation.X.shape[0]),
            "row_caps": row_caps,
            "effective_reliability_mode": resolved_pipeline_config.reliability.mode,
            "input_dim": int(prepared.full.X.shape[1]),
            "context_dim": 0 if prepared.full.context is None else int(prepared.full.context.shape[1]),
            "metadata_cardinalities": prepared.metadata_cardinalities,
            "feature_schema": prepared.feature_schema,
        }
        benchmark = self._run_large_scale_benchmark(pipeline, prepared)
        return ExperimentResult(
            fit_summary=fit_summary,
            base_metrics=base_metrics,
            ablations=ablations,
            sensitivity=sensitivity,
            selection=selection,
            dataset_summary=dataset_summary,
            runtime_environment=self._runtime_environment(),
            reproducibility=reproducibility,
            benchmark=benchmark,
        )

    def _run_with_checkpoint(self, output_path: Path) -> tuple[ExperimentResult, Path, Path]:
        """Run the framework with incremental progress artifacts and resume support."""
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path, checkpoint_path = self._progress_paths(output_path)
        heartbeat_path = self._heartbeat_path(output_path)
        prepared_cache_path = self._prepared_cache_path(output_path)
        checkpoint = self._load_checkpoint(checkpoint_path)
        if checkpoint is not None and checkpoint.get("stage") == "completed" and checkpoint.get("completed_result"):
            result = _deserialize_experiment_result(checkpoint["completed_result"])
            self._write_progress_payload(
                path=progress_path,
                summary_path=output_path,
                status="completed",
                stage="completed",
                runtime_environment=result.runtime_environment,
                reproducibility=result.reproducibility,
                dataset_summary=result.dataset_summary,
                fit_summary=result.fit_summary,
                base_metrics=result.base_metrics,
                ablations=result.ablations,
                sensitivity=result.sensitivity,
                selection=result.selection,
                benchmark=result.benchmark,
            )
            return result, progress_path, checkpoint_path

        reproducibility = self._configure_runtime()
        runtime_environment = self._runtime_environment()
        fit_summary: PipelineFitSummary | None = None
        base_metrics: dict[str, float] | None = None
        ablations: dict[str, AblationOutcome] = {}
        sensitivity: dict[str, dict[str, float]] = {}
        selection: dict[str, Any] | None = None
        benchmark: dict[str, Any] | None = None
        dataset_summary: dict[str, Any] | None = None

        try:
            self._write_progress_payload(
                path=progress_path,
                summary_path=output_path,
                status="running",
                stage="preparing_data",
                runtime_environment=runtime_environment,
                reproducibility=reproducibility,
            )
            cached_prepared = self._try_load_prepared_cache(prepared_cache_path)
            if cached_prepared is None:
                prepared_raw = self.prepare_data()
                prepared, row_caps = self._maybe_cap_prepared_data(prepared_raw)
                del prepared_raw
                gc.collect()
                self._save_prepared_cache(prepared_cache_path, prepared, row_caps)
            else:
                prepared, row_caps = cached_prepared
                gc.collect()
            resolved_pipeline_config = self._resolve_pipeline_config(prepared)
            resolved_pipeline_config = self._stabilize_pipeline_config_for_large_runs(
                resolved_pipeline_config,
                row_caps=row_caps,
            )
            pipeline = SilenceAwareIDS(resolved_pipeline_config)
            inference_batch_size = self._effective_inference_batch_size(row_caps)
            dataset_summary = {
                "train_rows": int(prepared.train.X.shape[0]),
                "calibration_rows": int(prepared.calibration.X.shape[0]),
                "evaluation_rows": int(prepared.evaluation.X.shape[0]),
                "row_caps": row_caps,
                "effective_reliability_mode": resolved_pipeline_config.reliability.mode,
                "input_dim": int(prepared.full.X.shape[1]),
                "context_dim": 0 if prepared.full.context is None else int(prepared.full.context.shape[1]),
                "metadata_cardinalities": prepared.metadata_cardinalities,
                "feature_schema": prepared.feature_schema,
            }

            if checkpoint is not None:
                fit_summary = _deserialize_fit_summary(checkpoint.get("fit_summary"))
                base_metrics = checkpoint.get("base_metrics") or None
                ablations = _deserialize_ablation_outcomes(checkpoint.get("ablations"))
                sensitivity = dict(checkpoint.get("sensitivity") or {})
                selection = checkpoint.get("selection")
                benchmark = checkpoint.get("benchmark")
                if checkpoint.get("pipeline_state") is not None:
                    self._restore_pipeline_checkpoint(pipeline, checkpoint.get("pipeline_state"))

            if fit_summary is None:
                self._write_progress_payload(
                    path=progress_path,
                    summary_path=output_path,
                    status="running",
                    stage="fitting_base_pipeline",
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                    dataset_summary=dataset_summary,
                )
                fit_summary = pipeline.fit(
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
                gc.collect()
                self._save_checkpoint(
                    checkpoint_path,
                    stage="fit_complete",
                    pipeline=pipeline,
                    include_pipeline_state=True,
                    fit_summary=fit_summary,
                    dataset_summary=dataset_summary,
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                )

            evaluation_s = prepared.evaluation.S

            if base_metrics is None:
                evaluation_s = self._ensure_dynamic_silence(
                    pipeline,
                    prepared.evaluation,
                    batch_size=inference_batch_size,
                    heartbeat_path=heartbeat_path,
                    stage="evaluating_base_pipeline",
                )
                self._write_progress_payload(
                    path=progress_path,
                    summary_path=output_path,
                    status="running",
                    stage="evaluating_base_pipeline",
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                    dataset_summary=dataset_summary,
                    fit_summary=fit_summary,
                )
                self._append_heartbeat(
                    heartbeat_path,
                    stage="evaluating_base_pipeline",
                    step="dispatch",
                    payload={
                        "rows": int(prepared.evaluation.X.shape[0]),
                        "batch_size": int(inference_batch_size),
                        "use_m3": bool(pipeline.use_m3),
                        "use_m5": bool(pipeline.use_m5),
                    },
                )
                base_metrics = pipeline.evaluate_predictions(
                    prepared.evaluation.X,
                    prepared.evaluation.y,
                    context=prepared.evaluation.context,
                    M=prepared.evaluation.M,
                    S=evaluation_s,
                    sensor_metadata=prepared.evaluation.sensor_metadata,
                    integrate_missingness=pipeline.use_m3,
                    batch_size=inference_batch_size,
                    progress_callback=lambda step, payload: self._append_heartbeat(
                        heartbeat_path,
                        stage="evaluating_base_pipeline",
                        step=step,
                        payload=payload,
                    ),
                )
                self._append_heartbeat(
                    heartbeat_path,
                    stage="evaluating_base_pipeline",
                    step="persist_base_metrics",
                    payload={"metric_keys": sorted(base_metrics.keys())},
                )
                self._write_progress_payload(
                    path=progress_path,
                    summary_path=output_path,
                    status="running",
                    stage="base_metrics_complete",
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                    dataset_summary=dataset_summary,
                    fit_summary=fit_summary,
                    base_metrics=base_metrics,
                )
                self._save_checkpoint(
                    checkpoint_path,
                    stage="base_metrics_complete",
                    pipeline=pipeline,
                    include_pipeline_state=False,
                    fit_summary=fit_summary,
                    base_metrics=base_metrics,
                    dataset_summary=dataset_summary,
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                )

            candidate_batch = prepared.evaluation if self.run_config.use_evaluation_as_candidate_pool else prepared.full
            if selection is None and candidate_batch.cost is not None:
                self._write_progress_payload(
                    path=progress_path,
                    summary_path=output_path,
                    status="running",
                    stage="selecting_policy_candidates",
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                    dataset_summary=dataset_summary,
                    fit_summary=fit_summary,
                    base_metrics=base_metrics,
                )
                raw_selection = pipeline.select_sensors(
                    candidate_batch.X,
                    candidate_batch.cost,
                    context=candidate_batch.context,
                    M=candidate_batch.M,
                    S=candidate_batch.S,
                    sensor_metadata=candidate_batch.sensor_metadata,
                    budget=self.run_config.policy_budget,
                    max_selections=self.run_config.max_selections,
                    batch_size=inference_batch_size,
                )
                selection = _selection_to_summary(raw_selection)
                self._save_checkpoint(
                    checkpoint_path,
                    stage="selection_complete",
                    pipeline=pipeline,
                    include_pipeline_state=False,
                    fit_summary=fit_summary,
                    base_metrics=base_metrics,
                    selection=selection,
                    dataset_summary=dataset_summary,
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                )

            self._write_progress_payload(
                path=progress_path,
                summary_path=output_path,
                status="running",
                stage="running_ablations",
                runtime_environment=runtime_environment,
                reproducibility=reproducibility,
                dataset_summary=dataset_summary,
                fit_summary=fit_summary,
                base_metrics=base_metrics,
                ablations=ablations,
                selection=selection,
            )
            ablations = self._run_ablation_suite_incremental(
                pipeline,
                prepared.train.X,
                prepared.train.y,
                prepared.evaluation.X,
                prepared.evaluation.y,
                missing_indicator_train=prepared.train.missing_indicator,
                context_train=prepared.train.context,
                M_train=prepared.train.M,
                S_train=prepared.train.S,
                sensor_metadata_train=prepared.train.sensor_metadata,
                context_eval=prepared.evaluation.context,
                M_eval=prepared.evaluation.M,
                S_eval=evaluation_s,
                sensor_metadata_eval=prepared.evaluation.sensor_metadata,
                X_cal=prepared.calibration.X,
                y_cal=prepared.calibration.y,
                context_cal=prepared.calibration.context,
                M_cal=prepared.calibration.M,
                S_cal=prepared.calibration.S,
                sensor_metadata_cal=prepared.calibration.sensor_metadata,
                candidate_x=candidate_batch.X,
                candidate_cost=candidate_batch.cost,
                candidate_context=candidate_batch.context,
                candidate_M=candidate_batch.M,
                candidate_S=candidate_batch.S,
                candidate_sensor_metadata=candidate_batch.sensor_metadata,
                budget=self.run_config.policy_budget,
                max_selections=self.run_config.max_selections,
                variant_names=list(self.run_config.variant_names) if self.run_config.variant_names else None,
                batch_size=inference_batch_size,
                existing_results=ablations,
                on_variant_complete=lambda _name, results: (
                    self._save_checkpoint(
                        checkpoint_path,
                        stage="ablations_in_progress",
                        pipeline=pipeline,
                        include_pipeline_state=False,
                        fit_summary=fit_summary,
                        base_metrics=base_metrics,
                        ablations=results,
                        selection=selection,
                        dataset_summary=dataset_summary,
                        runtime_environment=runtime_environment,
                        reproducibility=reproducibility,
                    ),
                    self._write_progress_payload(
                        path=progress_path,
                        summary_path=output_path,
                        status="running",
                        stage="running_ablations",
                        runtime_environment=runtime_environment,
                        reproducibility=reproducibility,
                        dataset_summary=dataset_summary,
                        fit_summary=fit_summary,
                        base_metrics=base_metrics,
                        ablations=results,
                        selection=selection,
                    ),
                ),
                on_variant_progress=lambda name, step, payload: self._append_heartbeat(
                    heartbeat_path,
                    stage="running_ablations",
                    step=f"{name}:{step}",
                    payload=payload,
                ),
            )

            if pipeline.use_m3 and not sensitivity:
                (
                    sensitivity_x,
                    sensitivity_y,
                    sensitivity_context,
                    sensitivity_m,
                    sensitivity_s,
                    sensitivity_metadata,
                    sensitivity_batch_size,
                    sensitivity_cap_info,
                ) = self._cap_sensitivity_evaluation(
                    prepared,
                    batch_size=inference_batch_size,
                )
                sensitivity_batch = TensorBatch(
                    X=sensitivity_x,
                    y=sensitivity_y,
                    context=sensitivity_context,
                    M=sensitivity_m,
                    S=sensitivity_s,
                    missing_indicator=sensitivity_m if sensitivity_m is not None else torch.zeros_like(sensitivity_y),
                    cost=None,
                    sensor_metadata=(
                        sensitivity_metadata
                        if isinstance(sensitivity_metadata, SensorMetadataBatch)
                        else prepared.evaluation.sensor_metadata
                    ),
                    indices=torch.arange(sensitivity_x.shape[0], dtype=torch.long, device=sensitivity_x.device),
                )
                evaluation_s = self._ensure_dynamic_silence(
                    pipeline,
                    sensitivity_batch,
                    batch_size=sensitivity_batch_size,
                    heartbeat_path=heartbeat_path,
                    stage="running_sensitivity",
                )
                self._write_progress_payload(
                    path=progress_path,
                    summary_path=output_path,
                    status="running",
                    stage="running_sensitivity",
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                    dataset_summary=dataset_summary,
                    fit_summary=fit_summary,
                    base_metrics=base_metrics,
                    ablations=ablations,
                    selection=selection,
                )
                self._append_heartbeat(
                    heartbeat_path,
                    stage="running_sensitivity",
                    step="dispatch",
                    payload=sensitivity_cap_info,
                )
                sensitivity = pipeline.run_missingness_sensitivity_analysis(
                    sensitivity_x,
                    sensitivity_y,
                    logit_scales=list(self.run_config.sensitivity_logit_scales),
                    context=sensitivity_context,
                    M=sensitivity_m,
                    S=evaluation_s,
                    sensor_metadata=sensitivity_batch.sensor_metadata,
                    batch_size=sensitivity_batch_size,
                    progress_callback=lambda step, payload: self._append_heartbeat(
                        heartbeat_path,
                        stage="running_sensitivity",
                        step=step,
                        payload=payload,
                    ),
                )
                self._append_heartbeat(
                    heartbeat_path,
                    stage="running_sensitivity",
                    step="persist_sensitivity",
                    payload={"keys": sorted(sensitivity.keys())},
                )
                self._save_checkpoint(
                    checkpoint_path,
                    stage="sensitivity_complete",
                    pipeline=pipeline,
                    include_pipeline_state=False,
                    fit_summary=fit_summary,
                    base_metrics=base_metrics,
                    ablations=ablations,
                    sensitivity=sensitivity,
                    selection=selection,
                    dataset_summary=dataset_summary,
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                )

            if benchmark is None:
                self._write_progress_payload(
                    path=progress_path,
                    summary_path=output_path,
                    status="running",
                    stage="running_benchmark",
                    runtime_environment=runtime_environment,
                    reproducibility=reproducibility,
                    dataset_summary=dataset_summary,
                    fit_summary=fit_summary,
                    base_metrics=base_metrics,
                    ablations=ablations,
                    sensitivity=sensitivity,
                    selection=selection,
                )
                benchmark = self._run_large_scale_benchmark(
                    pipeline,
                    prepared,
                    progress_callback=lambda step, payload: self._append_heartbeat(
                        heartbeat_path,
                        stage="running_benchmark",
                        step=step,
                        payload=payload,
                    ),
                )

            result = ExperimentResult(
                fit_summary=fit_summary,
                base_metrics=base_metrics or {},
                ablations=ablations,
                sensitivity=sensitivity,
                selection=selection,
                dataset_summary=dataset_summary or {},
                runtime_environment=runtime_environment,
                reproducibility=reproducibility,
                benchmark=benchmark,
            )
            self._save_checkpoint(
                checkpoint_path,
                stage="completed",
                pipeline=pipeline,
                include_pipeline_state=False,
                fit_summary=fit_summary,
                base_metrics=base_metrics,
                ablations=ablations,
                sensitivity=sensitivity,
                selection=selection,
                dataset_summary=dataset_summary,
                runtime_environment=runtime_environment,
                reproducibility=reproducibility,
                benchmark=benchmark,
                completed_result=result,
            )
            self._write_progress_payload(
                path=progress_path,
                summary_path=output_path,
                status="completed",
                stage="completed",
                runtime_environment=runtime_environment,
                reproducibility=reproducibility,
                dataset_summary=dataset_summary,
                fit_summary=fit_summary,
                base_metrics=base_metrics,
                ablations=ablations,
                sensitivity=sensitivity,
                selection=selection,
                benchmark=benchmark,
            )
            return result, progress_path, checkpoint_path
        except Exception as exc:
            self._append_heartbeat(
                heartbeat_path,
                stage="failed",
                step="exception",
                payload={"error": str(exc)},
            )
            self._write_progress_payload(
                path=progress_path,
                summary_path=output_path,
                status="failed",
                stage="failed",
                runtime_environment=runtime_environment,
                reproducibility=reproducibility,
                dataset_summary=dataset_summary,
                fit_summary=fit_summary,
                base_metrics=base_metrics,
                ablations=ablations,
                sensitivity=sensitivity,
                selection=selection,
                benchmark=benchmark,
                error=str(exc),
            )
            raise

    def write_summary(self, output_path: Path, result: ExperimentResult | None = None) -> Path:
        """Write an experiment summary to JSON.

        Args:
            output_path: Output JSON path.
            result: Optional precomputed result. If omitted, the experiment is run.

        Returns:
            The written output path.
        """
        resolved_result = self.run() if result is None else result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(resolved_result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_artifacts(
        self,
        output_path: Path,
        result: ExperimentResult | None = None,
    ) -> ExperimentArtifacts:
        """Write framework experiment artifacts to disk.

        Args:
            output_path: Target JSON summary path. Sibling CSV/Markdown files are
                written to the same directory.
            result: Optional precomputed experiment result.

        Returns:
            Paths to all written artifacts.
        """
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path, checkpoint_path = self._progress_paths(output_path)

        if result is None:
            resolved_result, progress_path, checkpoint_path = self._run_with_checkpoint(output_path)
        else:
            resolved_result = result

        summary_path = self.write_summary(output_path, resolved_result)
        ablations_path = output_dir / "ablations.csv"
        sensitivity_path = output_dir / "sensitivity.csv"
        selection_path = output_dir / "selection.csv"
        report_path = output_dir / "report.md"

        _write_csv(
            ablations_path,
            _ablation_rows(resolved_result),
            [
                "variant_name",
                "use_m2",
                "use_m3",
                "use_m5",
                "missingness_assumption",
                "missingness_mode",
                "missingness_inference_strategy",
                "state_training_strategy",
                "diagnosis_mode",
                "diagnosis_representation",
                "diagnosis_temporal_model",
                "diagnosis_curriculum",
                "diagnosis_latent_dynamics",
                "reliability_mode",
                "reliability_relational",
                "reliability_graph_corel",
                "reliability_graph_message_passing_steps",
                "missingness_sensor_health_latent",
                "policy_surrogate",
                "policy_planning_strategy",
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
            sensitivity_path,
            _sensitivity_rows(resolved_result),
            [
                "setting",
                "rmse",
                "mae",
                "crps",
                "log_score",
                "coverage",
                "interval_width",
                "mean_missingness_proba",
            ],
        )
        _write_csv(
            selection_path,
            _selection_rows(resolved_result),
            [
                "scope",
                "variant_name",
                "rank",
                "selected_index",
                "utility",
                "total_cost",
            ],
        )
        report_path.write_text(render_framework_report(resolved_result), encoding="utf-8")
        return ExperimentArtifacts(
            summary_path=summary_path,
            ablations_path=ablations_path,
            sensitivity_path=sensitivity_path,
            selection_path=selection_path,
            report_path=report_path,
            progress_path=progress_path if self.run_config.enable_progress_artifacts else None,
            checkpoint_path=checkpoint_path if self.run_config.resume_from_checkpoint else None,
        )

    def _resolve_pipeline_config(self, prepared: PreparedExperimentData) -> SilenceAwareIDSConfig:
        """Inject dataset-derived cardinalities into the pipeline configuration."""
        missingness = self.pipeline_config.missingness
        observation = self.pipeline_config.observation
        resolved_missingness = replace(
            missingness,
            context_dim=missingness.context_dim or (
                0 if prepared.full.context is None else int(prepared.full.context.shape[1])
            ),
            x_dim=missingness.x_dim or int(prepared.full.X.shape[1]),
            continuous_metadata_dim=(
                missingness.continuous_metadata_dim
                or (0 if prepared.full.sensor_metadata.continuous is None else int(prepared.full.sensor_metadata.continuous.shape[1]))
            ),
            num_sensor_types=missingness.num_sensor_types or prepared.metadata_cardinalities["sensor_type"],
            num_sensor_groups=missingness.num_sensor_groups or prepared.metadata_cardinalities["sensor_group"],
            num_sensor_modalities=(
                missingness.num_sensor_modalities or prepared.metadata_cardinalities["sensor_modality"]
            ),
            num_installation_environments=(
                missingness.num_installation_environments
                or prepared.metadata_cardinalities["installation_environment"]
            ),
            num_maintenance_states=(
                missingness.num_maintenance_states or prepared.metadata_cardinalities["maintenance_state"]
            ),
        )
        resolved_observation = replace(
            observation,
            context_dim=observation.context_dim or (
                0 if prepared.full.context is None else int(prepared.full.context.shape[1])
            ),
            continuous_metadata_dim=(
                observation.continuous_metadata_dim
                or (
                    0
                    if prepared.full.sensor_metadata.continuous is None
                    else int(prepared.full.sensor_metadata.continuous.shape[1])
                )
            ),
            num_sensor_types=observation.num_sensor_types or prepared.metadata_cardinalities["sensor_type"],
            num_sensor_groups=observation.num_sensor_groups or prepared.metadata_cardinalities["sensor_group"],
            num_sensor_modalities=(
                observation.num_sensor_modalities or prepared.metadata_cardinalities["sensor_modality"]
            ),
            num_installation_environments=(
                observation.num_installation_environments
                or prepared.metadata_cardinalities["installation_environment"]
            ),
            num_maintenance_states=(
                observation.num_maintenance_states or prepared.metadata_cardinalities["maintenance_state"]
            ),
        )
        return replace(
            self.pipeline_config,
            observation=resolved_observation,
            missingness=resolved_missingness,
        )

    def _stabilize_pipeline_config_for_large_runs(
        self,
        config: SilenceAwareIDSConfig,
        *,
        row_caps: dict[str, Any],
    ) -> SilenceAwareIDSConfig:
        """Apply conservative reliability overrides after large-run downsampling."""
        if not bool(row_caps.get("applied")):
            return config
        reliability = config.reliability
        if reliability.mode == "graph_corel":
            return replace(
                config,
                reliability=replace(
                    reliability,
                    mode="relational_adaptive",
                    adaptation_rate=max(reliability.adaptation_rate, 0.05),
                    relational_neighbor_weight=max(reliability.relational_neighbor_weight, 0.1),
                ),
            )
        return config


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable columns."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: Any) -> str:
    """Format scalar metrics for compact artifact tables."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _ablation_rows(result: ExperimentResult) -> list[dict[str, Any]]:
    """Flatten ablation outcomes into CSV rows."""
    rows: list[dict[str, Any]] = []
    for variant_name, outcome in result.ablations.items():
        report = outcome.report
        selection_summary = _selection_to_summary(outcome.selection)
        selected_indices = [] if selection_summary is None else selection_summary.get("selected_indices", [])
        rows.append(
            {
                "variant_name": variant_name,
                "use_m2": report.get("use_m2"),
                "use_m3": report.get("use_m3"),
                "use_m5": report.get("use_m5"),
                "missingness_assumption": report.get("missingness_assumption"),
                "missingness_mode": report.get("missingness_mode"),
                "missingness_inference_strategy": report.get("missingness_inference_strategy"),
                "state_training_strategy": report.get("state_training_strategy"),
                "diagnosis_mode": report.get("diagnosis_mode"),
                "diagnosis_representation": report.get("diagnosis_representation"),
                "diagnosis_temporal_model": report.get("diagnosis_temporal_model"),
                "diagnosis_curriculum": report.get("diagnosis_curriculum"),
                "diagnosis_latent_dynamics": report.get("diagnosis_latent_dynamics"),
                "reliability_mode": report.get("reliability_mode"),
                "reliability_relational": report.get("reliability_relational"),
                "reliability_graph_corel": report.get("reliability_graph_corel"),
                "reliability_graph_message_passing_steps": report.get("reliability_graph_message_passing_steps"),
                "missingness_sensor_health_latent": report.get("missingness_sensor_health_latent"),
                "policy_surrogate": report.get("policy_surrogate"),
                "policy_planning_strategy": report.get("policy_planning_strategy"),
                "rmse": _format_metric(outcome.metrics.get("rmse")),
                "mae": _format_metric(outcome.metrics.get("mae")),
                "crps": _format_metric(outcome.metrics.get("crps")),
                "log_score": _format_metric(outcome.metrics.get("log_score")),
                "coverage": _format_metric(outcome.metrics.get("coverage")),
                "interval_width": _format_metric(outcome.metrics.get("interval_width")),
                "mean_missingness_proba": _format_metric(outcome.metrics.get("mean_missingness_proba")),
                "total_cost": _format_metric(None if selection_summary is None else selection_summary.get("total_cost")),
                "num_selected": len(selected_indices),
            }
        )
    return rows


def _sensitivity_rows(result: ExperimentResult) -> list[dict[str, Any]]:
    """Flatten sensitivity metrics into CSV rows."""
    rows: list[dict[str, Any]] = []
    for setting, metrics in result.sensitivity.items():
        rows.append(
            {
                "setting": setting,
                "rmse": _format_metric(metrics.get("rmse")),
                "mae": _format_metric(metrics.get("mae")),
                "crps": _format_metric(metrics.get("crps")),
                "log_score": _format_metric(metrics.get("log_score")),
                "coverage": _format_metric(metrics.get("coverage")),
                "interval_width": _format_metric(metrics.get("interval_width")),
                "mean_missingness_proba": _format_metric(metrics.get("mean_missingness_proba")),
            }
        )
    return rows


def _selection_rows(result: ExperimentResult) -> list[dict[str, Any]]:
    """Flatten base and per-ablation policy selections into CSV rows."""
    rows: list[dict[str, Any]] = []
    rows.extend(_selection_rows_for_scope("base_policy", "", result.selection))
    for variant_name, outcome in result.ablations.items():
        rows.extend(_selection_rows_for_scope("ablation", variant_name, _selection_to_summary(outcome.selection)))
    return rows


def _selection_rows_for_scope(
    scope: str,
    variant_name: str,
    selection: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Convert one selection result into per-rank CSV rows."""
    if selection is None:
        return []
    indices = selection.get("selected_indices", [])
    utilities = selection.get("utility_trace", [])
    total_cost = selection.get("total_cost")
    rows: list[dict[str, Any]] = []
    for rank, selected_index in enumerate(indices, start=1):
        utility = utilities[rank - 1] if rank - 1 < len(utilities) else None
        rows.append(
            {
                "scope": scope,
                "variant_name": variant_name,
                "rank": rank,
                "selected_index": selected_index,
                "utility": _format_metric(utility),
                "total_cost": _format_metric(total_cost),
            }
        )
    return rows


def render_framework_report(result: ExperimentResult) -> str:
    """Render a compact markdown summary for framework experiments."""
    lines = [
        "# Silence-Aware IDS Framework Report",
        "",
        "## Runtime",
        f"- Python executable: {result.runtime_environment.get('python_executable', '')}",
        f"- Python version: {result.runtime_environment.get('python_version', '').splitlines()[0]}",
        f"- Torch: {result.runtime_environment.get('torch_version', '')}",
        f"- GPyTorch: {result.runtime_environment.get('gpytorch_version', '')}",
        "",
        "## Reproducibility",
        f"- Seed: {result.reproducibility.get('seed', '')}",
        f"- Deterministic algorithms: {result.reproducibility.get('deterministic_algorithms', '')}",
        f"- Torch threads: {result.reproducibility.get('torch_num_threads', '')}",
        f"- Matmul precision: {result.reproducibility.get('matmul_precision', '')}",
        "",
        "## Dataset",
        f"- Train rows: {result.dataset_summary.get('train_rows', 0)}",
        f"- Calibration rows: {result.dataset_summary.get('calibration_rows', 0)}",
        f"- Evaluation rows: {result.dataset_summary.get('evaluation_rows', 0)}",
        f"- Input dimension: {result.dataset_summary.get('input_dim', 0)}",
        f"- Context dimension: {result.dataset_summary.get('context_dim', 0)}",
        f"- Metadata cardinalities: {json.dumps(result.dataset_summary.get('metadata_cardinalities', {}), ensure_ascii=False)}",
        "",
        "## Base Metrics",
        f"- RMSE: {_format_metric(result.base_metrics.get('rmse'))}",
        f"- MAE: {_format_metric(result.base_metrics.get('mae'))}",
        f"- CRPS: {_format_metric(result.base_metrics.get('crps'))}",
        f"- Log-Score: {_format_metric(result.base_metrics.get('log_score'))}",
        f"- Coverage: {_format_metric(result.base_metrics.get('coverage'))}",
        f"- Interval Width: {_format_metric(result.base_metrics.get('interval_width'))}",
        f"- Observation-link steps: {len(result.fit_summary.observation_history.get('loss', [])) if result.fit_summary.observation_history else 0}",
        "",
        "## Ablations",
        "| Variant | Assumption | Mode | Inference | State Train | Policy | Planner | RMSE | CRPS | Log-Score | Coverage |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _ablation_rows(result):
        lines.append(
            f"| {row['variant_name']} | {row['missingness_assumption']} | {row['missingness_mode']} | "
            f"{row['missingness_inference_strategy']} | {row['state_training_strategy']} | {row['policy_surrogate']} | "
            f"{row['policy_planning_strategy']} | {row['rmse']} | "
            f"{row['crps']} | {row['log_score']} | {row['coverage']} |"
        )
    lines.extend(
        [
            "",
            "## Missingness Sensitivity",
            "| Setting | RMSE | CRPS | Log-Score | Mean Missingness |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    sensitivity_rows = _sensitivity_rows(result)
    if sensitivity_rows:
        for row in sensitivity_rows:
            lines.append(
                f"| {row['setting']} | {row['rmse']} | {row['crps']} | {row['log_score']} | "
                f"{row['mean_missingness_proba']} |"
            )
    else:
        lines.append("| disabled |  |  |  |  |")
    lines.extend(["", "## Benchmark"])
    if result.benchmark is None:
        lines.append("- Disabled")
    else:
        lines.append(f"- Expansion factor: {result.benchmark.get('expansion_factor', '')}")
        lines.append(f"- Benchmark rows: {result.benchmark.get('benchmark_rows', '')}")
        lines.append(f"- Prediction batch size: {result.benchmark.get('prediction_batch_size', '')}")
        lines.append(f"- Predict seconds: {_format_metric(result.benchmark.get('predict_seconds'))}")
        lines.append(f"- Missingness seconds: {_format_metric(result.benchmark.get('missingness_seconds'))}")
        lines.append(f"- Selection seconds: {_format_metric(result.benchmark.get('selection_seconds'))}")
        lines.append(f"- Input tensor bytes: {result.benchmark.get('input_tensor_bytes', '')}")
        lines.append(f"- Output tensor bytes: {result.benchmark.get('output_tensor_bytes', '')}")
        lines.append(
            f"- Mean predictive variance: {_format_metric(result.benchmark.get('mean_predictive_variance'))}"
        )
        lines.append(
            f"- Mean missingness probability: {_format_metric(result.benchmark.get('mean_missingness_proba'))}"
        )
    lines.extend(
        [
            "",
            "## Selection",
        ]
    )
    selection_rows = _selection_rows(result)
    if selection_rows:
        for row in selection_rows[:12]:
            label = row["variant_name"] or "base_policy"
            lines.append(
                f"- {row['scope']}:{label} rank {row['rank']} -> index {row['selected_index']} "
                f"(utility={row['utility']}, total_cost={row['total_cost']})"
            )
    else:
        lines.append("- No selection output was generated.")
    return "\n".join(lines) + "\n"
