from __future__ import annotations

import csv
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


JOINT_FIELDNAMES: tuple[str, ...] = (
    "timestamp",
    "station_id",
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "dew_point_temperature",
    "elevation",
    "cost",
    "sensor_type",
    "sensor_group",
    "sensor_modality",
    "site_type",
    "maintenance_state",
    "maintenance_age",
    "station_age_years",
    "source",
    "source_station_code",
    "source_station_name",
    "era5_temperature",
    "era5_pressure",
    "era5_u10",
    "era5_v10",
    "era5_precipitation",
    "era5_orography",
    "era5_land_sea_mask",
)


@dataclass(frozen=True)
class JointBuildConfig:
    """Configuration for building a joint AWS/NOAA/ERA5 experiment dataset.

    Attributes:
        aws_csv_path: Framework-ready AWS CSV path.
        noaa_csv_path: Framework-ready NOAA ISD CSV path.
        era5_csv_path: Optional ERA5 reference CSV path used for context enrichment.
        output_dir: Directory for outputs.
        output_csv_path: Optional output CSV path.
        framework_config_path: Optional framework config path.
        overwrite: Whether to overwrite existing outputs.
        era5_grid_step: ERA5 grid step in degrees for nearest-grid lookup.
    """

    aws_csv_path: Path
    noaa_csv_path: Path
    era5_csv_path: Path | None
    output_dir: Path
    output_csv_path: Path | None = None
    framework_config_path: Path | None = None
    overwrite: bool = False
    era5_grid_step: float = 0.25


@dataclass(frozen=True)
class JointBuildArtifacts:
    """Generated artifacts for a joint multisource dataset."""

    output_dir: Path
    output_csv_path: Path
    manifest_path: Path
    framework_config_path: Path | None
    row_count: int
    source_counts: dict[str, int]


def build_joint_dataset(config: JointBuildConfig) -> JointBuildArtifacts:
    """Build a joint AWS/NOAA experiment dataset with optional ERA5 enrichment.

    Args:
        config: Joint dataset builder configuration.

    Returns:
        Paths and summary counts for generated outputs.
    """

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = config.output_csv_path or output_dir / "joint_weather_network_q1.csv"
    manifest_path = output_dir / "joint_build_manifest.json"
    protected_paths = [output_csv_path, manifest_path]
    if config.framework_config_path is not None:
        protected_paths.append(config.framework_config_path)
    for path in protected_paths:
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"{path} already exists. Re-run with --overwrite to replace it.")

    aws_iter = _iter_aws_rows(config.aws_csv_path)
    noaa_iter = _iter_noaa_rows(config.noaa_csv_path)
    observation_iter = heapq.merge(aws_iter, noaa_iter, key=lambda row: row["timestamp"])
    era5_lookup = None if config.era5_csv_path is None else _Era5Lookup(config.era5_csv_path, config.era5_grid_step)

    row_count = 0
    source_counts = {"aws": 0, "noaa_isd_lite": 0}
    timestamp_min = None
    timestamp_max = None
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(JOINT_FIELDNAMES))
        writer.writeheader()
        for row in observation_iter:
            if era5_lookup is not None:
                row.update(
                    era5_lookup.lookup(
                        timestamp=row["timestamp"],
                        latitude=row["latitude"],
                        longitude=row["longitude"],
                    )
                )
            writer.writerow(row)
            row_count += 1
            source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
            if timestamp_min is None:
                timestamp_min = row["timestamp"]
            timestamp_max = row["timestamp"]
    if era5_lookup is not None:
        era5_lookup.close()

    if config.framework_config_path is not None:
        _write_joint_framework_config(config.framework_config_path, output_csv_path)

    manifest = {
        "output_csv_path": str(output_csv_path.resolve()),
        "framework_config_path": None
        if config.framework_config_path is None
        else str(config.framework_config_path.resolve()),
        "row_count": row_count,
        "source_counts": source_counts,
        "timestamp_range": {
            "start": timestamp_min,
            "end": timestamp_max,
        },
        "input_paths": {
            "aws_csv_path": str(config.aws_csv_path.resolve()),
            "noaa_csv_path": str(config.noaa_csv_path.resolve()),
            "era5_csv_path": None if config.era5_csv_path is None else str(config.era5_csv_path.resolve()),
        },
        "era5_grid_step": config.era5_grid_step,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return JointBuildArtifacts(
        output_dir=output_dir,
        output_csv_path=output_csv_path,
        manifest_path=manifest_path,
        framework_config_path=config.framework_config_path,
        row_count=row_count,
        source_counts=source_counts,
    )


def _iter_aws_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            temperature = _clean_numeric(row.get("temperature"))
            humidity = _clean_numeric(row.get("humidity"))
            dew_point = _compute_dew_point_celsius(temperature, humidity)
            yield {
                "timestamp": _normalize_timestamp(row.get("timestamp", "")),
                "station_id": f"aws_{(row.get('station_id') or '').strip()}",
                "latitude": _clean_numeric(row.get("latitude")),
                "longitude": _clean_numeric(row.get("longitude")),
                "temperature": temperature,
                "humidity": humidity,
                "pressure": _clean_numeric(row.get("pressure")),
                "wind_speed": _clean_numeric(row.get("wind_speed")),
                "wind_direction": _clean_numeric(row.get("wind_direction")),
                "precipitation": _clean_numeric(row.get("precipitation")),
                "dew_point_temperature": dew_point,
                "elevation": _clean_numeric(row.get("elevation")),
                "cost": _clean_numeric(row.get("cost")) or "1.0",
                "sensor_type": (row.get("sensor_type") or "aws").strip(),
                "sensor_group": (row.get("sensor_group") or "surface").strip(),
                "sensor_modality": (row.get("sensor_modality") or "automatic_weather_station").strip(),
                "site_type": (row.get("site_type") or "").strip(),
                "maintenance_state": (row.get("maintenance_state") or "").strip(),
                "maintenance_age": _clean_numeric(row.get("maintenance_age")),
                "station_age_years": "",
                "source": "aws",
                "source_station_code": (row.get("station_id") or "").strip(),
                "source_station_name": "",
                "era5_temperature": "",
                "era5_pressure": "",
                "era5_u10": "",
                "era5_v10": "",
                "era5_precipitation": "",
                "era5_orography": "",
                "era5_land_sea_mask": "",
            }


def _iter_noaa_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            temperature = _clean_numeric(row.get("air_temperature"))
            dew_point = _clean_numeric(row.get("dew_point_temperature"))
            humidity = _compute_relative_humidity(temperature, dew_point)
            yield {
                "timestamp": _normalize_timestamp(row.get("timestamp", "")),
                "station_id": f"noaa_{(row.get('station_id') or '').strip()}",
                "latitude": _clean_numeric(row.get("latitude")),
                "longitude": _clean_numeric(row.get("longitude")),
                "temperature": temperature,
                "humidity": humidity,
                "pressure": _clean_numeric(row.get("sea_level_pressure")),
                "wind_speed": _clean_numeric(row.get("wind_speed")),
                "wind_direction": _clean_numeric(row.get("wind_direction")),
                "precipitation": _clean_numeric(row.get("precipitation_1hr")),
                "dew_point_temperature": dew_point,
                "elevation": _clean_numeric(row.get("elevation")),
                "cost": _clean_numeric(row.get("sensor_cost")) or "1.0",
                "sensor_type": (row.get("sensor_type") or "isd_station").strip(),
                "sensor_group": (row.get("sensor_group") or "surface").strip(),
                "sensor_modality": (row.get("sensor_modality") or "synoptic_surface").strip(),
                "site_type": (row.get("site_type") or "").strip(),
                "maintenance_state": (row.get("maintenance_state") or "").strip(),
                "maintenance_age": "",
                "station_age_years": _clean_numeric(row.get("station_age_years")),
                "source": "noaa_isd_lite",
                "source_station_code": (row.get("station_id") or "").strip(),
                "source_station_name": (row.get("raw_station_name") or "").strip(),
                "era5_temperature": "",
                "era5_pressure": "",
                "era5_u10": "",
                "era5_v10": "",
                "era5_precipitation": "",
                "era5_orography": "",
                "era5_land_sea_mask": "",
            }


class _Era5Lookup:
    """Streaming ERA5 lookup keyed by timestamp and nearest grid point."""

    def __init__(self, era5_csv_path: Path, grid_step: float) -> None:
        self.grid_step = grid_step
        self.handle = era5_csv_path.open("r", encoding="utf-8", newline="")
        self.reader = csv.DictReader(self.handle)
        self.buffer_row = next(self.reader, None)
        self.current_timestamp = ""
        self.current_grid: dict[tuple[str, str], dict[str, str]] = {}

    def close(self) -> None:
        """Close the underlying ERA5 CSV handle."""
        self.handle.close()

    def lookup(self, *, timestamp: str, latitude: str, longitude: str) -> dict[str, str]:
        """Return nearest ERA5 context for a timestamp and coordinate."""
        self._advance_to(timestamp)
        if self.current_timestamp != timestamp or not latitude or not longitude:
            return {
                "era5_temperature": "",
                "era5_pressure": "",
                "era5_u10": "",
                "era5_v10": "",
                "era5_precipitation": "",
                "era5_orography": "",
                "era5_land_sea_mask": "",
            }
        key = (
            _format_grid_coord(_round_to_step(float(latitude), self.grid_step)),
            _format_grid_coord(_round_to_step(float(longitude), self.grid_step)),
        )
        row = self.current_grid.get(key)
        if row is None:
            return {
                "era5_temperature": "",
                "era5_pressure": "",
                "era5_u10": "",
                "era5_v10": "",
                "era5_precipitation": "",
                "era5_orography": "",
                "era5_land_sea_mask": "",
            }
        return {
            "era5_temperature": _kelvin_to_celsius(row.get("t2m", "")),
            "era5_pressure": _pascal_to_hectopascal(row.get("sp", "")),
            "era5_u10": _clean_numeric(row.get("u10")),
            "era5_v10": _clean_numeric(row.get("v10")),
            "era5_precipitation": _meters_to_millimeters(row.get("tp", "")),
            "era5_orography": _clean_numeric(row.get("orography")),
            "era5_land_sea_mask": _clean_numeric(row.get("land_sea_mask")),
        }

    def _advance_to(self, timestamp: str) -> None:
        while self.buffer_row is not None:
            buffer_ts = _normalize_timestamp(self.buffer_row.get("time", ""))
            if buffer_ts < timestamp:
                self._consume_timestamp(buffer_ts)
                continue
            if buffer_ts == timestamp:
                if self.current_timestamp != timestamp:
                    self._consume_timestamp(timestamp)
                return
            break
        self.current_timestamp = ""
        self.current_grid = {}

    def _consume_timestamp(self, timestamp: str) -> None:
        self.current_timestamp = timestamp
        self.current_grid = {}
        while self.buffer_row is not None and _normalize_timestamp(self.buffer_row.get("time", "")) == timestamp:
            lat_key = _format_grid_coord(float(self.buffer_row["latitude"]))
            lon_key = _format_grid_coord(float(self.buffer_row["longitude"]))
            self.current_grid[(lat_key, lon_key)] = dict(self.buffer_row)
            self.buffer_row = next(self.reader, None)


def _write_joint_framework_config(config_path: Path, data_path: Path) -> None:
    from .framework import write_framework_template

    write_framework_template(config_path, data_path, preset="aws_network", force=True)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["preset"] = "joint_weather_network"
    payload["data"]["context_columns"] = [
        "humidity",
        "pressure",
        "wind_speed",
        "dew_point_temperature",
        "era5_temperature",
        "era5_pressure",
        "era5_u10",
        "era5_v10",
        "era5_precipitation",
    ]
    payload["data"]["continuous_metadata_columns"] = [
        "elevation",
        "maintenance_age",
        "station_age_years",
        "era5_orography",
        "era5_land_sea_mask",
    ]
    payload["data"]["cost_column"] = "cost"
    payload["data"]["sensor_type_column"] = "sensor_type"
    payload["data"]["sensor_group_column"] = "sensor_group"
    payload["data"]["sensor_modality_column"] = "sensor_modality"
    payload["data"]["installation_environment_column"] = "site_type"
    payload["data"]["maintenance_state_column"] = "maintenance_state"
    payload["pipeline"]["observation"]["diagnosis_mode"] = "temporal_nwp"
    payload["pipeline"]["observation"]["use_latent_ode"] = True
    payload["pipeline"]["observation"]["corruption_probability_start"] = 0.05
    payload["pipeline"]["observation"]["corruption_probability_end"] = 0.2
    payload["pipeline"]["observation"]["latent_ode_weight"] = 0.05
    payload["pipeline"]["observation"]["nwp_context_index"] = 4
    payload["pipeline"]["observation"]["nwp_anchor_weight"] = 0.5
    payload["pipeline"]["missingness"]["inference_strategy"] = "joint_variational"
    payload["pipeline"]["missingness"]["reconstruction_weight"] = 1.0
    payload["pipeline"]["missingness"]["kl_weight"] = 0.01
    payload["pipeline"]["policy"]["planning_strategy"] = "ppo_online"
    payload["pipeline"]["policy"]["future_context_index"] = 4
    payload["pipeline"]["policy"]["planning_horizon"] = 4
    payload["pipeline"]["policy"]["future_discount"] = 0.85
    payload["pipeline"]["policy"]["lookahead_strength"] = 0.6
    payload["pipeline"]["policy"]["ppo_epochs"] = 10
    payload["pipeline"]["policy"]["ppo_policy_weight"] = 0.25
    payload["pipeline"]["policy"]["ppo_max_candidates"] = 1024
    payload["pipeline"]["reliability"]["mode"] = "graph_corel"
    payload["pipeline"]["reliability"]["adaptation_rate"] = 0.02
    payload["pipeline"]["reliability"]["relational_neighbor_weight"] = 0.25
    payload["pipeline"]["reliability"]["relational_temperature"] = 1.0
    payload["pipeline"]["reliability"]["graph_k_neighbors"] = 8
    payload["pipeline"]["reliability"]["graph_message_passing_steps"] = 2
    payload["pipeline"]["reliability"]["graph_training_steps"] = 50
    payload["pipeline"]["reliability"]["graph_score_weight"] = 0.5
    payload["pipeline"]["reliability"]["graph_covariance_weight"] = 0.5
    payload["output_path"] = "outputs/framework_joint_run/summary.json"
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_timestamp(raw_value: str) -> str:
    value = raw_value.strip()
    if "T" in value:
        return value[:16]
    if " " in value:
        return value.replace(" ", "T")[:16]
    return value[:16]


def _clean_numeric(raw_value: str | None) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""
    return value


def _compute_dew_point_celsius(temperature: str, humidity: str) -> str:
    if not temperature or not humidity:
        return ""
    try:
        temp_c = float(temperature)
        rh = max(min(float(humidity), 100.0), 1e-3)
    except ValueError:
        return ""
    gamma = math.log(rh / 100.0) + (17.625 * temp_c) / (243.04 + temp_c)
    dew_point = (243.04 * gamma) / (17.625 - gamma)
    return f"{dew_point:.1f}"


def _compute_relative_humidity(temperature: str, dew_point: str) -> str:
    if not temperature or not dew_point:
        return ""
    try:
        temp_c = float(temperature)
        dew_c = float(dew_point)
    except ValueError:
        return ""
    exponent = (17.625 * dew_c) / (243.04 + dew_c) - (17.625 * temp_c) / (243.04 + temp_c)
    rh = max(min(100.0 * math.exp(exponent), 100.0), 0.0)
    return f"{rh:.1f}"


def _kelvin_to_celsius(value: str) -> str:
    cleaned = _clean_numeric(value)
    if not cleaned:
        return ""
    return f"{float(cleaned) - 273.15:.2f}"


def _pascal_to_hectopascal(value: str) -> str:
    cleaned = _clean_numeric(value)
    if not cleaned:
        return ""
    return f"{float(cleaned) / 100.0:.2f}"


def _meters_to_millimeters(value: str) -> str:
    cleaned = _clean_numeric(value)
    if not cleaned:
        return ""
    return f"{float(cleaned) * 1000.0:.3f}"


def _round_to_step(value: float, step: float) -> float:
    return round(value / step) * step


def _format_grid_coord(value: float) -> str:
    return f"{value:.3f}"
