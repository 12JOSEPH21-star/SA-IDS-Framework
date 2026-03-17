from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


EVENT_FIELDNAMES: tuple[str, ...] = (
    "source",
    "station_id",
    "station_admin",
    "hour_timestamp",
    "event_station_active",
    "event_station_count",
    "event_station_max_warning_level",
    "first_event_time",
    "last_input_time",
)


@dataclass(frozen=True)
class EventRecord:
    """Parsed station-admin event record from warning-information raw text."""

    office_code: str
    event_time: datetime
    warning_level: float
    input_time: datetime | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build a station-local event-truth sidecar from KMA warning-information raw data.",
    )
    parser.add_argument(
        "--warning-information-raw-csv",
        type=Path,
        default=Path("data/event_truth_q1_2025_info/warning_information_raw.csv"),
        help="Raw warning-information CSV downloaded from KMA API Hub.",
    )
    parser.add_argument(
        "--aws-station-info-csv",
        type=Path,
        default=Path("data/qc_maintenance_metadata_q1_2025/aws_station_info_full.csv"),
        help="AWS station metadata CSV with station_admin codes.",
    )
    parser.add_argument(
        "--asos-station-info-csv",
        type=Path,
        default=Path("data/qc_maintenance_metadata_q1_2025/asos_station_info_full.csv"),
        help="ASOS station metadata CSV with station_admin codes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/event_truth_q1_2025_station"),
        help="Output directory for the sidecar CSV and manifest.",
    )
    return parser.parse_args()


def _parse_compact_timestamp(value: str) -> datetime | None:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) != 12:
        return None
    try:
        return datetime.strptime(digits, "%Y%m%d%H%M")
    except ValueError:
        return None


def _load_station_admin_map(path: Path, *, source: str) -> dict[str, list[str]]:
    """Load station ids grouped by station_admin code."""
    by_admin: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            station_id = (row.get("station_id") or "").strip()
            station_admin = (row.get("station_admin") or "").strip()
            if not station_id or not station_admin:
                continue
            by_admin[station_admin].append(station_id)
    return dict(by_admin)


def _iter_warning_information_events(path: Path) -> list[EventRecord]:
    """Extract warning events from the first encoded field of raw warning-information CSV."""
    records: list[EventRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return records
        primary_field = reader.fieldnames[0]
        for row in reader:
            payload = (row.get(primary_field) or "").strip()
            if not payload.startswith("$0#"):
                continue
            parts = payload.split("#")
            if len(parts) < 5:
                continue
            event_time = _parse_compact_timestamp(parts[2])
            if event_time is None:
                continue
            input_time = _parse_compact_timestamp(parts[4])
            try:
                warning_level = float(parts[3])
            except ValueError:
                warning_level = 0.0
            office_code = parts[1].strip()
            if not office_code:
                continue
            records.append(
                EventRecord(
                    office_code=office_code,
                    event_time=event_time,
                    warning_level=warning_level,
                    input_time=input_time,
                )
            )
    return records


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVENT_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    """Build the station-local event sidecar."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "station_event_truth_q1.csv"
    manifest_path = output_dir / "station_event_truth_manifest.json"

    aws_by_admin = _load_station_admin_map(args.aws_station_info_csv, source="aws")
    asos_by_admin = _load_station_admin_map(args.asos_station_info_csv, source="asos")
    events = _iter_warning_information_events(args.warning_information_raw_csv)

    aggregates: dict[tuple[str, str, str], dict[str, object]] = {}
    unmatched_office_codes: set[str] = set()
    for event in events:
        matched = False
        hour_timestamp = event.event_time.replace(minute=0, second=0, microsecond=0).isoformat(timespec="minutes")
        input_time_text = "" if event.input_time is None else event.input_time.isoformat(timespec="minutes")
        for source, station_map in (("aws", aws_by_admin), ("asos", asos_by_admin)):
            station_ids = station_map.get(event.office_code, [])
            if not station_ids:
                continue
            matched = True
            for station_id in station_ids:
                key = (source, station_id, hour_timestamp)
                bucket = aggregates.setdefault(
                    key,
                    {
                        "source": source,
                        "station_id": station_id,
                        "station_admin": event.office_code,
                        "hour_timestamp": hour_timestamp,
                        "event_station_active": "1.0",
                        "event_station_count": 0,
                        "event_station_max_warning_level": 0.0,
                        "first_event_time": event.event_time.isoformat(timespec="minutes"),
                        "last_input_time": input_time_text,
                    },
                )
                bucket["event_station_count"] = int(bucket["event_station_count"]) + 1
                bucket["event_station_max_warning_level"] = max(
                    float(bucket["event_station_max_warning_level"]),
                    event.warning_level,
                )
                first_event_time = str(bucket["first_event_time"])
                if event.event_time.isoformat(timespec="minutes") < first_event_time:
                    bucket["first_event_time"] = event.event_time.isoformat(timespec="minutes")
                if input_time_text and input_time_text > str(bucket["last_input_time"]):
                    bucket["last_input_time"] = input_time_text
        if not matched:
            unmatched_office_codes.add(event.office_code)

    rows = [
        {
            "source": str(bucket["source"]),
            "station_id": str(bucket["station_id"]),
            "station_admin": str(bucket["station_admin"]),
            "hour_timestamp": str(bucket["hour_timestamp"]),
            "event_station_active": "1.0",
            "event_station_count": str(bucket["event_station_count"]),
            "event_station_max_warning_level": f"{float(bucket['event_station_max_warning_level']):.6f}".rstrip("0").rstrip("."),
            "first_event_time": str(bucket["first_event_time"]),
            "last_input_time": str(bucket["last_input_time"]),
        }
        for bucket in sorted(
            aggregates.values(),
            key=lambda row: (str(row["source"]), str(row["station_id"]), str(row["hour_timestamp"])),
        )
    ]
    _write_csv(output_csv_path, rows)

    manifest = {
        "warning_information_raw_csv": str(args.warning_information_raw_csv.resolve()),
        "aws_station_info_csv": str(args.aws_station_info_csv.resolve()),
        "asos_station_info_csv": str(args.asos_station_info_csv.resolve()),
        "output_csv_path": str(output_csv_path.resolve()),
        "row_count": len(rows),
        "event_record_count": len(events),
        "matched_office_code_count": len(set(aws_by_admin).intersection({event.office_code for event in events}) | set(asos_by_admin).intersection({event.office_code for event in events})),
        "unmatched_office_codes": sorted(unmatched_office_codes),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built station event sidecar rows={len(rows)} at {output_csv_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
