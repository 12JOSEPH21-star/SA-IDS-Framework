from __future__ import annotations

import csv
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
from pipeline import AblationOutcome, PipelineFitSummary, SilenceAwareIDS, SilenceAwareIDSConfig


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
            sensor_type=categorical_tensors["sensor_type"],
            sensor_group=categorical_tensors["sensor_group"],
            sensor_modality=categorical_tensors["sensor_modality"],
            installation_environment=categorical_tensors["installation_environment"],
            maintenance_state=categorical_tensors["maintenance_state"],
            continuous=continuous_tensor,
        )

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
    ) -> dict[str, Any] | None:
        """Stress-test batched inference and policy selection on expanded candidate pools."""
        expansion_factor = self.run_config.benchmark_expansion_factor
        if expansion_factor <= 1 and self.run_config.benchmark_candidate_pool_size is None:
            return None

        evaluation = prepared.evaluation
        candidate_limit = self.run_config.benchmark_candidate_pool_size
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
        predict_start = time.perf_counter()
        mean, variance = pipeline.predict_state(expanded_x, batch_size=batch_size)
        predict_seconds = time.perf_counter() - predict_start
        missingness_start = time.perf_counter()
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

    def run(self) -> ExperimentResult:
        """Run fitting, ablation study, sensitivity analysis, and policy selection.

        Returns:
            Experiment result bundle.
        """
        reproducibility = self._configure_runtime()
        prepared = self.prepare_data()
        resolved_pipeline_config = self._resolve_pipeline_config(prepared)
        pipeline = SilenceAwareIDS(resolved_pipeline_config)
        inference_batch_size = self.run_config.prediction_batch_size

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
        resolved_result = self.run() if result is None else result
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

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
