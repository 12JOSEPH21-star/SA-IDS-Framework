from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


STATION_INFO_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php"
USER_AGENT = "silence-aware-ids/1.0"


@dataclass(frozen=True)
class StationRow:
    """Parsed KMA station-info row."""

    source: str
    station_id: str
    longitude: str
    latitude: str
    stn_sp: str
    elevation: str
    extra_1: str
    extra_2: str
    extra_3: str
    extra_4: str
    station_name_ko: str
    station_name_en: str
    managing_office: str
    forecast_id: str
    law_id: str
    basin: str
    law_address: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download official KMA station metadata and derive Q1 coverage/QC sidecars.",
    )
    parser.add_argument("--auth-key", required=True, help="KMA API Hub authKey.")
    parser.add_argument(
        "--reference-time",
        default="202503010000",
        help="Reference time for station-info snapshots in YYYYMMDDHHMM format.",
    )
    parser.add_argument(
        "--aws-raw-csv",
        type=Path,
        default=Path("data/aws_korea_q1_2025/aws_raw.csv"),
        help="AWS raw CSV path.",
    )
    parser.add_argument(
        "--asos-raw-csv",
        type=Path,
        default=Path("data/asos_korea_q1_2025/asos_hourly_apihub_raw.csv"),
        help="ASOS raw CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/qc_maintenance_metadata_q1_2025"),
        help="Output directory for station metadata and QC sidecars.",
    )
    parser.add_argument(
        "--noaa-raw-csv",
        type=Path,
        default=Path("data/noaa_isd_korea_q1_2025/noaa_isd_raw.csv"),
        help="Optional NOAA raw CSV path.",
    )
    parser.add_argument(
        "--noaa-metadata-csv",
        type=Path,
        default=Path("data/noaa_isd_korea_q1_2025/isd_station_metadata.csv"),
        help="Optional NOAA station metadata CSV path.",
    )
    return parser.parse_args()


def fetch_text(url: str, timeout: int = 60) -> str:
    """Fetch text payload."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore").replace("\ufeff", "")


def build_station_info_url(*, auth_key: str, inf: str, reference_time: str) -> str:
    """Build station-info request URL."""
    params = {
        "authKey": auth_key,
        "inf": inf,
        "stn": "",
        "tm": reference_time,
        "help": "1",
    }
    return f"{STATION_INFO_ENDPOINT}?{urllib.parse.urlencode(params)}"


def parse_station_info(text: str, *, source: str) -> list[StationRow]:
    """Parse KMA station-info whitespace payload."""
    rows: list[StationRow] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 12:
            continue
        prefix = tokens[:10]
        address = " ".join(tokens[10:])
        rows.append(
            StationRow(
                source=source,
                station_id=prefix[0],
                longitude=prefix[1],
                latitude=prefix[2],
                stn_sp=prefix[3],
                elevation=prefix[4],
                extra_1=prefix[5],
                extra_2=prefix[6],
                extra_3=prefix[7],
                extra_4=prefix[8],
                station_name_ko=prefix[9],
                station_name_en=prefix[10] if len(prefix) > 10 else "",
                managing_office=prefix[7] if source == "aws" else prefix[9],
                forecast_id=prefix[8] if source == "aws" else prefix[10],
                law_id=prefix[9] if source == "aws" else prefix[11] if len(tokens) > 11 else "",
                basin=prefix[10] if source == "aws" else prefix[12] if len(tokens) > 12 else "",
                law_address=address,
            )
        )
    return rows


def parse_station_info_flexible(text: str, *, source: str) -> list[dict[str, str]]:
    """Parse station-info text into CSV-friendly dicts.

    AWS and ASOS station-info tables have slightly different numeric columns. This
    parser preserves the first fixed set of columns and stores the remaining text
    columns separately.
    """
    parsed: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 10:
            continue
        if source == "aws":
            if len(tokens) < 12:
                continue
            parsed.append(
                {
                    "source": "aws",
                    "station_id": tokens[0],
                    "longitude": tokens[1],
                    "latitude": tokens[2],
                    "stn_sp": tokens[3],
                    "elevation": tokens[4],
                    "height_wind": tokens[5],
                    "lau_id": tokens[6],
                    "station_admin": tokens[7],
                    "station_name_ko": tokens[8],
                    "station_name_en": tokens[9],
                    "forecast_id": tokens[10],
                    "law_id": tokens[11],
                    "basin": tokens[12] if len(tokens) > 12 else "",
                    "law_address": " ".join(tokens[13:]) if len(tokens) > 13 else "",
                }
            )
        else:
            if len(tokens) < 15:
                continue
            parsed.append(
                {
                    "source": "asos",
                    "station_id": tokens[0],
                    "longitude": tokens[1],
                    "latitude": tokens[2],
                    "stn_sp": tokens[3],
                    "elevation": tokens[4],
                    "height_pressure": tokens[5],
                    "height_temperature": tokens[6],
                    "height_wind": tokens[7],
                    "height_rain": tokens[8],
                    "station_admin": tokens[9],
                    "station_name_ko": tokens[10],
                    "station_name_en": tokens[11],
                    "forecast_id": tokens[12],
                    "law_id": tokens[13],
                    "basin": tokens[14] if len(tokens) > 14 else "",
                    "law_address": " ".join(tokens[15:]) if len(tokens) > 15 else "",
                }
            )
    return parsed


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _safe_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= -90.0:
        return None
    return parsed


def summarize_raw_csv(
    raw_csv_path: Path,
    *,
    source: str,
    start_time: datetime,
    end_time: datetime,
    timestamp_column: str,
    station_column: str,
    status_column: str | None,
    status_missing_values: set[str],
    suspect_keys: tuple[str, ...],
) -> list[dict[str, str]]:
    """Summarize observed coverage and simple QC proxies per station."""
    if not raw_csv_path.exists():
        return []
    expected_hours = int(((end_time - start_time).total_seconds() // 3600)) + 1
    observed_by_station: dict[str, set[str]] = defaultdict(set)
    observed_hours_by_station: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, str] = {}
    last_seen: dict[str, str] = {}
    status_non_equal: dict[str, int] = defaultdict(int)
    suspect_values: dict[str, int] = defaultdict(int)
    total_rows: dict[str, int] = defaultdict(int)

    with raw_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            station_id = (row.get(station_column) or "").strip()
            timestamp = (row.get(timestamp_column) or "").strip()
            if not station_id or not timestamp:
                continue
            total_rows[station_id] += 1
            observed_by_station[station_id].add(timestamp)
            hour_bucket = timestamp[:13] if "T" in timestamp else timestamp[:10]
            observed_hours_by_station[station_id].add(hour_bucket)
            first_seen.setdefault(station_id, timestamp)
            last_seen[station_id] = timestamp
            if status_column is not None:
                if (row.get(status_column) or "").strip() not in status_missing_values:
                    status_non_equal[station_id] += 1
            for key in suspect_keys:
                value = (row.get(key) or "").strip()
                if value in {"", "-99", "-99.0", "-99.9", "-999", "-9", "-9.0"}:
                    suspect_values[station_id] += 1

    summaries: list[dict[str, str]] = []
    for station_id in sorted(observed_by_station):
        observed_count = len(observed_by_station[station_id])
        observed_hours = len(observed_hours_by_station[station_id])
        missing_hours = max(0, expected_hours - observed_hours)
        hour_availability_ratio = observed_hours / expected_hours if expected_hours else 0.0
        raw_per_hour = total_rows.get(station_id, 0) / max(1, observed_hours)
        summaries.append(
            {
                "source": source,
                "station_id": station_id,
                "first_observation": first_seen.get(station_id, ""),
                "last_observation": last_seen.get(station_id, ""),
                "expected_hours": str(expected_hours),
                "observed_hour_buckets": str(observed_hours),
                "missing_hour_buckets": str(missing_hours),
                "hour_bucket_availability_ratio": f"{hour_availability_ratio:.6f}",
                "observed_raw_timestamps": str(observed_count),
                "raw_rows_per_observed_hour": f"{raw_per_hour:.6f}",
                "status_or_qc_flag_count": str(status_non_equal.get(station_id, 0)),
                "suspect_value_count": str(suspect_values.get(station_id, 0)),
                "total_rows": str(total_rows.get(station_id, 0)),
            }
        )
    return summaries


def main() -> int:
    """Run the downloader and sidecar builder."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    aws_url = build_station_info_url(auth_key=args.auth_key, inf="AWS", reference_time=args.reference_time)
    asos_url = build_station_info_url(auth_key=args.auth_key, inf="SFC", reference_time=args.reference_time)
    aws_text = fetch_text(aws_url)
    asos_text = fetch_text(asos_url)

    aws_text_path = output_dir / "aws_station_info_full.txt"
    asos_text_path = output_dir / "asos_station_info_full.txt"
    aws_text_path.write_text(aws_text, encoding="utf-8")
    asos_text_path.write_text(asos_text, encoding="utf-8")

    aws_station_rows = parse_station_info_flexible(aws_text, source="aws")
    asos_station_rows = parse_station_info_flexible(asos_text, source="asos")
    write_csv(
        output_dir / "aws_station_info_full.csv",
        list(aws_station_rows[0].keys()) if aws_station_rows else [],
        aws_station_rows,
    )
    write_csv(
        output_dir / "asos_station_info_full.csv",
        list(asos_station_rows[0].keys()) if asos_station_rows else [],
        asos_station_rows,
    )

    q1_start = datetime(2025, 1, 1, 0, 0)
    q1_end = datetime(2025, 3, 31, 23, 0)
    aws_summary = summarize_raw_csv(
        args.aws_raw_csv,
        source="aws",
        start_time=q1_start,
        end_time=q1_end,
        timestamp_column="tm",
        station_column="stn",
        status_column="status",
        status_missing_values={"", "="},
        suspect_keys=("ta", "hm", "ps", "ws10", "wd10", "rn_15m"),
    )
    asos_summary = summarize_raw_csv(
        args.asos_raw_csv,
        source="asos",
        start_time=q1_start,
        end_time=q1_end,
        timestamp_column="tm",
        station_column="stn",
        status_column="ix",
        status_missing_values={"", "-9"},
        suspect_keys=("ta", "hm", "ps", "ws", "wd", "rn"),
    )
    noaa_summary = summarize_raw_csv(
        args.noaa_raw_csv,
        source="noaa_isd_lite",
        start_time=q1_start,
        end_time=q1_end,
        timestamp_column="timestamp",
        station_column="station_id",
        status_column=None,
        status_missing_values=set(),
        suspect_keys=("air_temperature_raw", "dew_point_raw", "sea_level_pressure_raw", "wind_direction_raw"),
    )
    coverage_fieldnames = [
        "source",
        "station_id",
        "first_observation",
        "last_observation",
        "expected_hours",
        "observed_hour_buckets",
        "missing_hour_buckets",
        "hour_bucket_availability_ratio",
        "observed_raw_timestamps",
        "raw_rows_per_observed_hour",
        "status_or_qc_flag_count",
        "suspect_value_count",
        "total_rows",
    ]
    write_csv(output_dir / "aws_q1_operational_qc_summary.csv", coverage_fieldnames, aws_summary)
    write_csv(output_dir / "asos_q1_operational_qc_summary.csv", coverage_fieldnames, asos_summary)
    write_csv(output_dir / "noaa_q1_operational_qc_summary.csv", coverage_fieldnames, noaa_summary)
    write_csv(output_dir / "q1_operational_qc_summary.csv", coverage_fieldnames, aws_summary + asos_summary + noaa_summary)
    if args.noaa_metadata_csv.exists():
        target_path = output_dir / "noaa_isd_station_metadata.csv"
        target_path.write_text(args.noaa_metadata_csv.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "reference_time": args.reference_time,
        "aws_station_info_url": aws_url.replace(args.auth_key, "***"),
        "asos_station_info_url": asos_url.replace(args.auth_key, "***"),
        "aws_raw_csv": str(args.aws_raw_csv.resolve()) if args.aws_raw_csv.exists() else None,
        "asos_raw_csv": str(args.asos_raw_csv.resolve()) if args.asos_raw_csv.exists() else None,
        "noaa_raw_csv": str(args.noaa_raw_csv.resolve()) if args.noaa_raw_csv.exists() else None,
        "noaa_metadata_csv": str(args.noaa_metadata_csv.resolve()) if args.noaa_metadata_csv.exists() else None,
        "aws_station_count": len(aws_station_rows),
        "asos_station_count": len(asos_station_rows),
        "aws_summary_station_count": len(aws_summary),
        "asos_summary_station_count": len(asos_summary),
        "noaa_summary_station_count": len(noaa_summary),
    }
    (output_dir / "qc_maintenance_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote QC/maintenance metadata sidecars to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
