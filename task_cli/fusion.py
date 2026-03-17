from __future__ import annotations

import csv
import heapq
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import torch


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
    "ldaps_temperature",
    "rdaps_temperature",
    "event_warning_active",
    "event_warning_count",
    "event_information_count",
    "event_max_warning_level",
    "event_station_active",
    "event_station_count",
    "event_station_max_warning_level",
    "qc_hour_bucket_availability_ratio",
    "qc_status_or_qc_flag_count",
    "qc_suspect_value_count",
    "qc_raw_rows_per_observed_hour",
)


@dataclass(frozen=True)
class JointBuildConfig:
    """Configuration for building a multisource joint experiment dataset."""

    aws_csv_path: Path
    noaa_csv_path: Path
    output_dir: Path
    asos_csv_path: Path | None = None
    era5_csv_path: Path | None = None
    event_history_csv_path: Path | None = None
    event_information_csv_path: Path | None = None
    event_station_csv_path: Path | None = None
    qc_metadata_csv_path: Path | None = None
    nwp_ldaps_summary_csv_path: Path | None = None
    nwp_rdaps_summary_csv_path: Path | None = None
    nwp_ldaps_grid_csv_path: Path | None = None
    nwp_rdaps_grid_csv_path: Path | None = None
    output_csv_path: Path | None = None
    framework_config_path: Path | None = None
    overwrite: bool = False
    era5_grid_step: float = 0.25


@dataclass(frozen=True)
class JointBuildArtifacts:
    """Generated artifacts for a multisource dataset."""

    output_dir: Path
    output_csv_path: Path
    manifest_path: Path
    framework_config_path: Path | None
    row_count: int
    source_counts: dict[str, int]


def build_joint_dataset(config: JointBuildConfig) -> JointBuildArtifacts:
    """Build a joint AWS/ASOS/NOAA dataset with optional enrichments."""
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

    station_coordinates = _collect_station_coordinates(
        config.aws_csv_path,
        config.noaa_csv_path,
        config.asos_csv_path,
    )
    era5_lookup = None if config.era5_csv_path is None else _Era5Lookup(config.era5_csv_path, config.era5_grid_step)
    event_lookup = _EventTruthLookup(
        history_csv_path=config.event_history_csv_path,
        information_csv_path=config.event_information_csv_path,
    )
    station_event_lookup = _StationEventLookup(config.event_station_csv_path)
    qc_lookup = _QcMetadataLookup(config.qc_metadata_csv_path)
    nwp_ldaps_lookup = _NwpAnchorLookup(
        summary_csv_path=config.nwp_ldaps_summary_csv_path,
        grid_csv_path=config.nwp_ldaps_grid_csv_path,
        station_coordinates=station_coordinates,
        output_field="ldaps_temperature",
    )
    nwp_rdaps_lookup = _NwpAnchorLookup(
        summary_csv_path=config.nwp_rdaps_summary_csv_path,
        grid_csv_path=config.nwp_rdaps_grid_csv_path,
        station_coordinates=station_coordinates,
        output_field="rdaps_temperature",
    )

    iterators: list[Iterator[dict[str, str]]] = [
        _iter_kma_rows(
            config.aws_csv_path,
            source="aws",
            station_prefix="aws",
            default_sensor_type="aws",
            default_sensor_modality="automatic_weather_station",
        ),
        _iter_noaa_rows(config.noaa_csv_path),
    ]
    if config.asos_csv_path is not None:
        iterators.append(
            _iter_kma_rows(
                config.asos_csv_path,
                source="asos",
                station_prefix="asos",
                default_sensor_type="asos",
                default_sensor_modality="synoptic_surface",
            )
        )
    observation_iter = heapq.merge(*iterators, key=lambda row: row["timestamp"])

    row_count = 0
    source_counts: dict[str, int] = {}
    enrichment_counts = {
        "era5_rows": 0,
        "event_warning_positive_rows": 0,
        "event_information_positive_rows": 0,
        "qc_rows": 0,
        "ldaps_rows": 0,
        "rdaps_rows": 0,
    }
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
            row.update(event_lookup.lookup(row["timestamp"]))
            row.update(
                station_event_lookup.lookup(
                    timestamp=row["timestamp"],
                    source=row["source"],
                    source_station_code=row["source_station_code"],
                )
            )
            row.update(qc_lookup.lookup(source=row["source"], source_station_code=row["source_station_code"]))
            row.update(nwp_ldaps_lookup.lookup(timestamp=row["timestamp"], station_id=row["station_id"]))
            row.update(nwp_rdaps_lookup.lookup(timestamp=row["timestamp"], station_id=row["station_id"]))
            writer.writerow(row)
            row_count += 1
            source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
            if row["era5_temperature"]:
                enrichment_counts["era5_rows"] += 1
            if row["event_warning_active"] not in {"", "0", "0.0"}:
                enrichment_counts["event_warning_positive_rows"] += 1
            if row["event_information_count"] not in {"", "0", "0.0"}:
                enrichment_counts["event_information_positive_rows"] += 1
            if row["qc_hour_bucket_availability_ratio"]:
                enrichment_counts["qc_rows"] += 1
            if row["ldaps_temperature"]:
                enrichment_counts["ldaps_rows"] += 1
            if row["rdaps_temperature"]:
                enrichment_counts["rdaps_rows"] += 1
            if timestamp_min is None:
                timestamp_min = row["timestamp"]
            timestamp_max = row["timestamp"]

    if era5_lookup is not None:
        era5_lookup.close()
    nwp_ldaps_lookup.close()
    nwp_rdaps_lookup.close()

    if config.framework_config_path is not None:
        _write_joint_framework_config(
            config.framework_config_path,
            output_csv_path,
            include_era5=enrichment_counts["era5_rows"] > 0,
            include_event=(config.event_history_csv_path is not None or config.event_information_csv_path is not None),
            include_station_event=config.event_station_csv_path is not None,
            include_qc=config.qc_metadata_csv_path is not None,
            include_nwp=(enrichment_counts["ldaps_rows"] > 0 or enrichment_counts["rdaps_rows"] > 0),
        )

    manifest = {
        "output_csv_path": str(output_csv_path.resolve()),
        "framework_config_path": None
        if config.framework_config_path is None
        else str(config.framework_config_path.resolve()),
        "row_count": row_count,
        "source_counts": source_counts,
        "timestamp_range": {"start": timestamp_min, "end": timestamp_max},
        "enrichment_counts": enrichment_counts,
        "input_paths": {
            "aws_csv_path": str(config.aws_csv_path.resolve()),
            "asos_csv_path": None if config.asos_csv_path is None else str(config.asos_csv_path.resolve()),
            "noaa_csv_path": str(config.noaa_csv_path.resolve()),
            "era5_csv_path": None if config.era5_csv_path is None else str(config.era5_csv_path.resolve()),
            "event_history_csv_path": None
            if config.event_history_csv_path is None
            else str(config.event_history_csv_path.resolve()),
            "event_information_csv_path": None
            if config.event_information_csv_path is None
            else str(config.event_information_csv_path.resolve()),
            "event_station_csv_path": None
            if config.event_station_csv_path is None
            else str(config.event_station_csv_path.resolve()),
            "qc_metadata_csv_path": None
            if config.qc_metadata_csv_path is None
            else str(config.qc_metadata_csv_path.resolve()),
            "nwp_ldaps_summary_csv_path": None
            if config.nwp_ldaps_summary_csv_path is None
            else str(config.nwp_ldaps_summary_csv_path.resolve()),
            "nwp_rdaps_summary_csv_path": None
            if config.nwp_rdaps_summary_csv_path is None
            else str(config.nwp_rdaps_summary_csv_path.resolve()),
            "nwp_ldaps_grid_csv_path": None
            if config.nwp_ldaps_grid_csv_path is None
            else str(config.nwp_ldaps_grid_csv_path.resolve()),
            "nwp_rdaps_grid_csv_path": None
            if config.nwp_rdaps_grid_csv_path is None
            else str(config.nwp_rdaps_grid_csv_path.resolve()),
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


def _collect_station_coordinates(
    aws_csv_path: Path,
    noaa_csv_path: Path,
    asos_csv_path: Path | None,
) -> dict[str, tuple[float, float]]:
    """Collect one coordinate pair per prefixed station id."""
    coordinates: dict[str, tuple[float, float]] = {}
    for path, prefix in (
        (aws_csv_path, "aws"),
        (noaa_csv_path, "noaa"),
        (asos_csv_path, "asos"),
    ):
        if path is None:
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                station_code = (row.get("station_id") or "").strip()
                if not station_code:
                    continue
                latitude = _parse_float(row.get("latitude"))
                longitude = _parse_float(row.get("longitude"))
                if latitude is None or longitude is None:
                    continue
                coordinates.setdefault(f"{prefix}_{station_code}", (latitude, longitude))
    return coordinates


def _iter_kma_rows(
    path: Path,
    *,
    source: str,
    station_prefix: str,
    default_sensor_type: str,
    default_sensor_modality: str,
) -> Iterator[dict[str, str]]:
    """Yield AWS or ASOS rows in the joint schema."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            temperature = _clean_numeric(row.get("temperature"))
            humidity = _clean_numeric(row.get("humidity"))
            dew_point = _compute_dew_point_celsius(temperature, humidity)
            yield {
                "timestamp": _normalize_timestamp(row.get("timestamp", "")),
                "station_id": f"{station_prefix}_{(row.get('station_id') or '').strip()}",
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
                "sensor_type": (row.get("sensor_type") or default_sensor_type).strip(),
                "sensor_group": (row.get("sensor_group") or "surface").strip(),
                "sensor_modality": (row.get("sensor_modality") or default_sensor_modality).strip(),
                "site_type": (row.get("site_type") or "").strip(),
                "maintenance_state": (row.get("maintenance_state") or "").strip(),
                "maintenance_age": _clean_numeric(row.get("maintenance_age")),
                "station_age_years": "",
                "source": source,
                "source_station_code": (row.get("station_id") or "").strip(),
                "source_station_name": "",
                "era5_temperature": "",
                "era5_pressure": "",
                "era5_u10": "",
                "era5_v10": "",
                "era5_precipitation": "",
                "era5_orography": "",
                "era5_land_sea_mask": "",
                "ldaps_temperature": "",
                "rdaps_temperature": "",
                "event_warning_active": "",
                "event_warning_count": "",
                "event_information_count": "",
                "event_max_warning_level": "",
                "event_station_active": "",
                "event_station_count": "",
                "event_station_max_warning_level": "",
                "qc_hour_bucket_availability_ratio": "",
                "qc_status_or_qc_flag_count": "",
                "qc_suspect_value_count": "",
                "qc_raw_rows_per_observed_hour": "",
            }


def _iter_noaa_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield NOAA ISD rows in the joint schema."""
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
                "ldaps_temperature": "",
                "rdaps_temperature": "",
                "event_warning_active": "",
                "event_warning_count": "",
                "event_information_count": "",
                "event_max_warning_level": "",
                "event_station_active": "",
                "event_station_count": "",
                "event_station_max_warning_level": "",
                "qc_hour_bucket_availability_ratio": "",
                "qc_status_or_qc_flag_count": "",
                "qc_suspect_value_count": "",
                "qc_raw_rows_per_observed_hour": "",
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
            return _empty_era5_context()
        key = (
            _format_grid_coord(_round_to_step(float(latitude), self.grid_step)),
            _format_grid_coord(_round_to_step(float(longitude), self.grid_step)),
        )
        row = self.current_grid.get(key)
        if row is None:
            return _empty_era5_context()
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


class _EventTruthLookup:
    """Lookup KMA event-truth counts by observation hour."""

    def __init__(
        self,
        *,
        history_csv_path: Path | None,
        information_csv_path: Path | None,
    ) -> None:
        self.history_by_hour: dict[str, dict[str, float]] = {}
        self.info_by_hour: dict[str, float] = {}
        if history_csv_path is not None and history_csv_path.exists():
            self.history_by_hour = self._load_history(history_csv_path)
        if information_csv_path is not None and information_csv_path.exists():
            self.info_by_hour = self._load_information(information_csv_path)

    def lookup(self, timestamp: str) -> dict[str, str]:
        """Return hour-level event features for one observation timestamp."""
        hour_key = _floor_to_hour(timestamp)
        history = self.history_by_hour.get(hour_key, {})
        info_count = self.info_by_hour.get(hour_key, 0.0)
        warning_count = history.get("event_warning_count", 0.0)
        return {
            "event_warning_active": "1.0" if warning_count > 0.0 else "0.0",
            "event_warning_count": _format_float_string(warning_count),
            "event_information_count": _format_float_string(info_count),
            "event_max_warning_level": _format_float_string(history.get("event_max_warning_level", 0.0)),
        }

    def _load_history(self, path: Path) -> dict[str, dict[str, float]]:
        by_hour: dict[str, dict[str, float]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                # `wrn_reg.php` may return static region-registry rows; ignore those
                # unless they carry actual warning event metadata.
                if not (
                    (row.get("warning_code") or "").strip()
                    or (row.get("issuing_station_id") or "").strip()
                    or (row.get("publication_time") or "").strip()
                    or (row.get("effective_time") or "").strip()
                ):
                    continue
                start_raw = row.get("start_time") or row.get("effective_time") or row.get("publication_time") or ""
                end_raw = row.get("end_time") or start_raw
                start_dt = _parse_iso_datetime(start_raw)
                end_dt = _parse_iso_datetime(end_raw)
                if start_dt is None or end_dt is None:
                    continue
                if end_dt < start_dt:
                    end_dt = start_dt
                warning_level = _parse_float(row.get("warning_level")) or 0.0
                cursor = _truncate_to_hour(start_dt)
                end_hour = _truncate_to_hour(end_dt)
                while cursor <= end_hour:
                    key = cursor.isoformat(timespec="minutes")
                    bucket = by_hour.setdefault(
                        key,
                        {"event_warning_count": 0.0, "event_max_warning_level": 0.0},
                    )
                    bucket["event_warning_count"] += 1.0
                    bucket["event_max_warning_level"] = max(bucket["event_max_warning_level"], warning_level)
                    cursor += timedelta(hours=1)
        return by_hour

    def _load_information(self, path: Path) -> dict[str, float]:
        by_hour: dict[str, float] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                time_raw = row.get("effective_time") or row.get("publication_time") or row.get("input_time") or ""
                parsed = _parse_iso_datetime(time_raw)
                if parsed is None:
                    continue
                key = _truncate_to_hour(parsed).isoformat(timespec="minutes")
                by_hour[key] = by_hour.get(key, 0.0) + 1.0
        return by_hour


class _StationEventLookup:
    """Lookup station-local event features keyed by source/station/hour."""

    def __init__(self, event_station_csv_path: Path | None) -> None:
        self.by_key: dict[tuple[str, str, str], dict[str, str]] = {}
        if event_station_csv_path is None or not event_station_csv_path.exists():
            return
        with event_station_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source = (row.get("source") or "").strip()
                station_id = (row.get("station_id") or "").strip()
                hour_timestamp = _normalize_timestamp(row.get("hour_timestamp", ""))
                if not source or not station_id or not hour_timestamp:
                    continue
                self.by_key[(source, station_id, hour_timestamp)] = {
                    "event_station_active": _clean_numeric(row.get("event_station_active")) or "0.0",
                    "event_station_count": _clean_numeric(row.get("event_station_count")) or "0.0",
                    "event_station_max_warning_level": _clean_numeric(row.get("event_station_max_warning_level")) or "0.0",
                }

    def lookup(self, *, timestamp: str, source: str, source_station_code: str) -> dict[str, str]:
        """Return station-local event features for one observation row."""
        key = (source, source_station_code, _floor_to_hour(timestamp))
        return self.by_key.get(
            key,
            {
                "event_station_active": "",
                "event_station_count": "",
                "event_station_max_warning_level": "",
            },
        )


class _QcMetadataLookup:
    """Lookup precomputed station-level QC and operational summaries."""

    def __init__(self, qc_metadata_csv_path: Path | None) -> None:
        self.by_station: dict[tuple[str, str], dict[str, str]] = {}
        if qc_metadata_csv_path is None or not qc_metadata_csv_path.exists():
            return
        with qc_metadata_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source = (row.get("source") or "").strip()
                station_id = (row.get("station_id") or "").strip()
                if not source or not station_id:
                    continue
                self.by_station[(source, station_id)] = {
                    "qc_hour_bucket_availability_ratio": _clean_numeric(row.get("hour_bucket_availability_ratio")),
                    "qc_status_or_qc_flag_count": _clean_numeric(row.get("status_or_qc_flag_count")),
                    "qc_suspect_value_count": _clean_numeric(row.get("suspect_value_count")),
                    "qc_raw_rows_per_observed_hour": _clean_numeric(row.get("raw_rows_per_observed_hour")),
                }

    def lookup(self, *, source: str, source_station_code: str) -> dict[str, str]:
        """Return QC features for one source/station pair."""
        return self.by_station.get(
            (source, source_station_code),
            {
                "qc_hour_bucket_availability_ratio": "",
                "qc_status_or_qc_flag_count": "",
                "qc_suspect_value_count": "",
                "qc_raw_rows_per_observed_hour": "",
            },
        )


class _NwpAnchorLookup:
    """Lookup station-aligned NWP anchor values from summary/raw/grid artifacts."""

    def __init__(
        self,
        *,
        summary_csv_path: Path | None,
        grid_csv_path: Path | None,
        station_coordinates: dict[str, tuple[float, float]],
        output_field: str,
    ) -> None:
        self.output_field = output_field
        self.summary_by_time: dict[str, Path] = {}
        self.station_index_by_id: dict[str, int] = {}
        self.value_cache: dict[str, dict[str, str]] = {}
        if summary_csv_path is None or grid_csv_path is None:
            return
        if not summary_csv_path.exists() or not grid_csv_path.exists():
            return
        lon_tensor, lat_tensor = _load_nwp_grid(grid_csv_path)
        if lon_tensor is None or lat_tensor is None:
            return
        self.station_index_by_id = _nearest_grid_indices(
            station_coordinates,
            lon_tensor=lon_tensor,
            lat_tensor=lat_tensor,
        )
        with summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_path = Path((row.get("raw_path") or "").strip())
                if not raw_path.exists():
                    continue
                forecast_time = _resolve_nwp_forecast_time(row, raw_path)
                if forecast_time:
                    self.summary_by_time.setdefault(forecast_time, raw_path)

    def close(self) -> None:
        """Release cached values."""
        self.value_cache.clear()

    def lookup(self, *, timestamp: str, station_id: str) -> dict[str, str]:
        """Return one NWP anchor value for an observation row."""
        hour_key = _floor_to_hour(timestamp)
        if not hour_key:
            return {self.output_field: ""}
        raw_path = self.summary_by_time.get(hour_key)
        if raw_path is None:
            return {self.output_field: ""}
        if hour_key not in self.value_cache:
            self.value_cache[hour_key] = self._load_station_values(raw_path)
        return {self.output_field: self.value_cache[hour_key].get(station_id, "")}

    def _load_station_values(self, raw_path: Path) -> dict[str, str]:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        header = (((payload.get("response") or {}).get("header")) or {})
        result_code = str(header.get("resultCode", "00"))
        if result_code not in {"00", "0"}:
            return {}
        items = _extract_nwp_items(payload)
        if not items:
            return {}
        values = _parse_value_tokens(items[0].get("value", ""))
        if not values:
            return {}
        station_values: dict[str, str] = {}
        for station_id, index in self.station_index_by_id.items():
            if 0 <= index < len(values):
                station_values[station_id] = values[index]
        return station_values


def _load_nwp_grid(grid_csv_path: Path) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load lon/lat arrays from one KMA grid CSV."""
    lon_values: dict[int, float] = {}
    lat_values: dict[int, float] = {}
    with grid_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                grid_index = int((row.get("grid_index") or "").strip())
                value = float((row.get("value") or "").strip())
            except (TypeError, ValueError):
                continue
            coordinate_type = (row.get("coordinate_type") or "").strip()
            if coordinate_type == "lon":
                lon_values[grid_index] = value
            elif coordinate_type == "lat":
                lat_values[grid_index] = value
    if not lon_values or not lat_values:
        return None, None
    max_index = max(max(lon_values), max(lat_values))
    lon_series = [lon_values.get(index, math.nan) for index in range(max_index + 1)]
    lat_series = [lat_values.get(index, math.nan) for index in range(max_index + 1)]
    return (
        torch.tensor(lon_series, dtype=torch.float32),
        torch.tensor(lat_series, dtype=torch.float32),
    )


def _nearest_grid_indices(
    station_coordinates: dict[str, tuple[float, float]],
    *,
    lon_tensor: torch.Tensor,
    lat_tensor: torch.Tensor,
) -> dict[str, int]:
    """Map each station to its nearest grid index."""
    valid_mask = torch.isfinite(lon_tensor) & torch.isfinite(lat_tensor)
    valid_lon = lon_tensor[valid_mask]
    valid_lat = lat_tensor[valid_mask]
    valid_indices = torch.arange(lon_tensor.numel(), dtype=torch.long)[valid_mask]
    lookup: dict[str, int] = {}
    if valid_indices.numel() == 0:
        return lookup
    for station_id, (latitude, longitude) in station_coordinates.items():
        distance = (valid_lat - float(latitude)).pow(2) + (valid_lon - float(longitude)).pow(2)
        nearest = int(valid_indices[torch.argmin(distance)].item())
        lookup[station_id] = nearest
    return lookup


def _extract_nwp_items(payload: Any) -> list[dict[str, Any]]:
    """Extract nested item dictionaries from one typ02 JSON payload."""
    if isinstance(payload, dict):
        if "item" in payload:
            item = payload["item"]
            if isinstance(item, list):
                return [value for value in item if isinstance(value, dict)]
            if isinstance(item, dict):
                return [item]
        items: list[dict[str, Any]] = []
        for value in payload.values():
            items.extend(_extract_nwp_items(value))
        return items
    if isinstance(payload, list):
        items: list[dict[str, Any]] = []
        for value in payload:
            items.extend(_extract_nwp_items(value))
        return items
    return []


def _parse_value_tokens(raw_value: Any) -> list[str]:
    """Parse one NWP value vector into string tokens."""
    if isinstance(raw_value, list):
        return [str(value) for value in raw_value]
    if isinstance(raw_value, str):
        return [token for token in re.split(r"[\s,]+", raw_value.strip()) if token]
    return []


def _resolve_nwp_forecast_time(row: dict[str, str], raw_path: Path) -> str:
    """Resolve forecast time from summary rows or raw-path labels."""
    forecast_time = _normalize_timestamp(row.get("forecast_time", ""))
    if forecast_time:
        return forecast_time
    base_time = _normalize_timestamp(row.get("base_time", ""))
    lead_hour = _parse_float(row.get("lead_hour"))
    if base_time and lead_hour is not None:
        base_dt = _parse_iso_datetime(base_time)
        if base_dt is not None:
            return (base_dt + timedelta(hours=int(lead_hour))).isoformat(timespec="minutes")
    match = re.search(r"_(\d{12})_h(\d{3})\.json$", raw_path.name)
    if not match:
        return ""
    base_dt = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    return (base_dt + timedelta(hours=int(match.group(2)))).isoformat(timespec="minutes")


def _write_joint_framework_config(
    config_path: Path,
    data_path: Path,
    *,
    include_era5: bool,
    include_event: bool,
    include_station_event: bool,
    include_qc: bool,
    include_nwp: bool,
) -> None:
    """Write a framework config aligned with the fused schema."""
    from .framework import write_framework_template

    write_framework_template(config_path, data_path, preset="aws_network", force=True)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["preset"] = "joint_weather_network"
    context_columns = ["humidity", "pressure", "wind_speed", "dew_point_temperature"]
    if include_era5:
        context_columns.extend(
            ["era5_temperature", "era5_pressure", "era5_u10", "era5_v10", "era5_precipitation"]
        )
    if include_nwp:
        context_columns.extend(["ldaps_temperature", "rdaps_temperature"])
    if include_event:
        context_columns.extend(
            ["event_warning_active", "event_warning_count", "event_information_count", "event_max_warning_level"]
        )
    if include_station_event:
        context_columns.extend(
            ["event_station_active", "event_station_count", "event_station_max_warning_level"]
        )
    payload["data"]["context_columns"] = context_columns

    continuous_columns = ["elevation", "maintenance_age", "station_age_years"]
    if include_era5:
        continuous_columns.extend(["era5_orography", "era5_land_sea_mask"])
    if include_qc:
        continuous_columns.extend(
            [
                "qc_hour_bucket_availability_ratio",
                "qc_status_or_qc_flag_count",
                "qc_suspect_value_count",
                "qc_raw_rows_per_observed_hour",
            ]
        )
    payload["data"]["continuous_metadata_columns"] = continuous_columns
    payload["data"]["cost_column"] = "cost"
    payload["data"]["sensor_type_column"] = "sensor_type"
    payload["data"]["sensor_group_column"] = "sensor_group"
    payload["data"]["sensor_modality_column"] = "sensor_modality"
    payload["data"]["installation_environment_column"] = "site_type"
    payload["data"]["maintenance_state_column"] = "maintenance_state"

    anchor_column = None
    if include_nwp and "ldaps_temperature" in context_columns:
        anchor_column = "ldaps_temperature"
    elif include_era5 and "era5_temperature" in context_columns:
        anchor_column = "era5_temperature"
    payload["pipeline"]["observation"]["diagnosis_mode"] = "temporal_nwp" if anchor_column is not None else "temporal"
    if anchor_column is not None:
        anchor_index = context_columns.index(anchor_column)
        payload["pipeline"]["observation"]["nwp_context_index"] = anchor_index
        payload["pipeline"]["observation"]["nwp_anchor_weight"] = 0.5
        payload["pipeline"]["policy"]["future_context_index"] = anchor_index
    payload["output_path"] = "outputs/framework_joint_run/summary.json"
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_timestamp(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if "T" in value:
        return value[:16]
    if " " in value:
        return value.replace(" ", "T")[:16]
    if len(value) == 12 and value.isdigit():
        return datetime.strptime(value, "%Y%m%d%H%M").isoformat(timespec="minutes")
    return value[:16]


def _parse_iso_datetime(raw_value: str) -> datetime | None:
    value = _normalize_timestamp(raw_value)
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _truncate_to_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def _floor_to_hour(timestamp: str) -> str:
    parsed = _parse_iso_datetime(timestamp)
    if parsed is None:
        return ""
    return _truncate_to_hour(parsed).isoformat(timespec="minutes")


def _clean_numeric(raw_value: str | None) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""
    return value


def _parse_float(raw_value: str | None) -> float | None:
    value = _clean_numeric(raw_value)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_float_string(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


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


def _empty_era5_context() -> dict[str, str]:
    return {
        "era5_temperature": "",
        "era5_pressure": "",
        "era5_u10": "",
        "era5_v10": "",
        "era5_precipitation": "",
        "era5_orography": "",
        "era5_land_sea_mask": "",
    }
