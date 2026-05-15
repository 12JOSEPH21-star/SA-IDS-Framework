from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StationSummary:
    """Aggregated per-station registry row."""

    station_id: str
    source: str
    latitude: float
    longitude: float
    elevation: float
    row_count: int
    observed_count: int
    start_timestamp: str
    end_timestamp: str
    qc_availability_ratio_mean: float
    era5_overlap_count: int
    event_overlap_count: int
    region_bin: str
    climate_zone: str

    def to_row(self) -> dict[str, str]:
        """Serialize the summary for CSV export."""
        coverage_ratio = 0.0 if self.row_count == 0 else self.observed_count / self.row_count
        return {
            "station_id": self.station_id,
            "source": self.source,
            "latitude": f"{self.latitude:.6f}",
            "longitude": f"{self.longitude:.6f}",
            "elevation": f"{self.elevation:.3f}",
            "row_count": str(self.row_count),
            "observed_count": str(self.observed_count),
            "coverage_ratio": f"{coverage_ratio:.6f}",
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "qc_availability_ratio_mean": f"{self.qc_availability_ratio_mean:.6f}",
            "era5_overlap_count": str(self.era5_overlap_count),
            "event_overlap_count": str(self.event_overlap_count),
            "region_bin": self.region_bin,
            "climate_zone": self.climate_zone,
        }


def _parse_float(raw_value: str) -> float:
    """Parse a possibly empty CSV cell as float."""
    value = raw_value.strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _derive_region(latitude: float, longitude: float) -> str:
    """Derive a coarse Korean region bin from coordinates."""
    north = latitude >= 36.5
    east = longitude >= 127.75
    if north and east:
        return "northeast"
    if north and not east:
        return "northwest"
    if not north and east:
        return "southeast"
    return "southwest"


def _derive_climate_zone(
    *,
    latitude: float,
    longitude: float,
    elevation: float,
    era5_land_sea_mask: float,
    source: str,
) -> str:
    """Derive a coarse climate zone using available public covariates."""
    if elevation >= 500.0:
        return "highland"
    if era5_land_sea_mask and era5_land_sea_mask < 0.9:
        return "coastal"
    if longitude >= 128.3 or longitude <= 126.2 or latitude <= 34.9:
        return "coastal"
    if source == "noaa_isd_lite":
        return "international_synoptic"
    return "inland"


def build_station_registry(
    input_csv_path: Path,
    output_csv_path: Path,
    summary_json_path: Path,
) -> None:
    """Build one station-level registry from the fused Q1 joint dataset."""
    aggregate: dict[str, dict[str, object]] = {}
    with input_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            station_id = row["station_id"].strip()
            source = row.get("source", "").strip()
            latitude = _parse_float(row.get("latitude", ""))
            longitude = _parse_float(row.get("longitude", ""))
            elevation = _parse_float(row.get("elevation", ""))
            timestamp = row.get("timestamp", "").strip()
            target_present = bool(row.get("temperature", "").strip())
            qc_ratio = _parse_float(row.get("qc_hour_bucket_availability_ratio", ""))
            era5_overlap = bool(row.get("era5_temperature", "").strip())
            event_overlap = bool(row.get("event_station_active", "").strip() and _parse_float(row.get("event_station_active", "")) > 0.0)
            era5_land_sea_mask = _parse_float(row.get("era5_land_sea_mask", ""))

            payload = aggregate.setdefault(
                station_id,
                {
                    "station_id": station_id,
                    "source": source,
                    "latitude": latitude,
                    "longitude": longitude,
                    "elevation": elevation,
                    "row_count": 0,
                    "observed_count": 0,
                    "start_timestamp": timestamp,
                    "end_timestamp": timestamp,
                    "qc_sum": 0.0,
                    "qc_count": 0,
                    "era5_overlap_count": 0,
                    "event_overlap_count": 0,
                    "region_bin": _derive_region(latitude, longitude),
                    "climate_zone": _derive_climate_zone(
                        latitude=latitude,
                        longitude=longitude,
                        elevation=elevation,
                        era5_land_sea_mask=era5_land_sea_mask,
                        source=source,
                    ),
                },
            )
            payload["row_count"] = int(payload["row_count"]) + 1
            payload["observed_count"] = int(payload["observed_count"]) + int(target_present)
            payload["start_timestamp"] = min(str(payload["start_timestamp"]), timestamp)
            payload["end_timestamp"] = max(str(payload["end_timestamp"]), timestamp)
            payload["qc_sum"] = float(payload["qc_sum"]) + qc_ratio
            payload["qc_count"] = int(payload["qc_count"]) + 1
            payload["era5_overlap_count"] = int(payload["era5_overlap_count"]) + int(era5_overlap)
            payload["event_overlap_count"] = int(payload["event_overlap_count"]) + int(event_overlap)

    rows: list[dict[str, str]] = []
    by_source: dict[str, int] = {}
    by_region: dict[str, int] = {}
    by_climate: dict[str, int] = {}
    for station_id in sorted(aggregate):
        payload = aggregate[station_id]
        qc_count = max(int(payload["qc_count"]), 1)
        summary = StationSummary(
            station_id=str(payload["station_id"]),
            source=str(payload["source"]),
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            elevation=float(payload["elevation"]),
            row_count=int(payload["row_count"]),
            observed_count=int(payload["observed_count"]),
            start_timestamp=str(payload["start_timestamp"]),
            end_timestamp=str(payload["end_timestamp"]),
            qc_availability_ratio_mean=float(payload["qc_sum"]) / qc_count,
            era5_overlap_count=int(payload["era5_overlap_count"]),
            event_overlap_count=int(payload["event_overlap_count"]),
            region_bin=str(payload["region_bin"]),
            climate_zone=str(payload["climate_zone"]),
        )
        rows.append(summary.to_row())
        by_source[summary.source] = by_source.get(summary.source, 0) + 1
        by_region[summary.region_bin] = by_region.get(summary.region_bin, 0) + 1
        by_climate[summary.climate_zone] = by_climate.get(summary.climate_zone, 0) + 1

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "station_id",
                "source",
                "latitude",
                "longitude",
                "elevation",
                "row_count",
                "observed_count",
                "coverage_ratio",
                "start_timestamp",
                "end_timestamp",
                "qc_availability_ratio_mean",
                "era5_overlap_count",
                "event_overlap_count",
                "region_bin",
                "climate_zone",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_json_path.write_text(
        json.dumps(
            {
                "input_csv_path": str(input_csv_path.resolve()),
                "output_csv_path": str(output_csv_path.resolve()),
                "station_count": len(rows),
                "source_counts": by_source,
                "region_counts": by_region,
                "climate_zone_counts": by_climate,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    root = Path("data/joint_weather_network_q1_2025")
    build_station_registry(
        input_csv_path=root / "joint_weather_network_q1.csv",
        output_csv_path=root / "station_registry_q1.csv",
        summary_json_path=root / "station_registry_summary.json",
    )
