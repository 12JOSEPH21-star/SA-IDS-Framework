from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_cli.kma import STANDARDIZED_FIELDNAMES, write_kma_framework_config


OBS_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
STATION_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php"
RAW_HEADER = (
    "tm",
    "stn",
    "wd1",
    "ws1",
    "wds",
    "wss",
    "wd10",
    "ws10",
    "ta",
    "re",
    "rn_15m",
    "rn_60m",
    "rn_12h",
    "rn_day",
    "hm",
    "pa",
    "ps",
    "td",
    "status",
)


class DailyQuotaExceededError(RuntimeError):
    """Raised when the KMA API Hub daily quota has been exhausted."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AWS API Hub range data into framework-ready CSV files.")
    parser.add_argument("--start", required=True, help="Inclusive start timestamp in YYYY-MM-DDTHH:MM format.")
    parser.add_argument("--end", required=True, help="Inclusive end timestamp in YYYY-MM-DDTHH:MM format.")
    parser.add_argument("--interval-minutes", type=int, default=60, help="Sampling interval in minutes.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=24,
        help="Number of timestamps to request per API call. With hourly data, 24 means one day per request.",
    )
    parser.add_argument("--auth-key", required=True, help="KMA API Hub authKey.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--station-id", default="0", help="Station selector. Use 0 for nationwide AWS snapshots.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=6, help="Retry budget per timestamp.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Throttle delay between timestamp requests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted download by appending new rows.")
    return parser.parse_args()


def iter_timestamps(start: datetime, end: datetime, interval_minutes: int) -> list[datetime]:
    stamps: list[datetime] = []
    cursor = start
    step = timedelta(minutes=interval_minutes)
    while cursor <= end:
        stamps.append(cursor)
        cursor += step
    return stamps


def iter_chunk_ranges(
    timestamps: list[datetime],
    chunk_size: int,
) -> list[tuple[datetime, datetime, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk-size must be positive.")
    chunks: list[tuple[datetime, datetime, int]] = []
    for offset in range(0, len(timestamps), chunk_size):
        chunk = timestamps[offset : offset + chunk_size]
        chunks.append((chunk[0], chunk[-1], len(chunk)))
    return chunks


def fetch_text(url: str, timeout: int, max_retries: int) -> str:
    attempts = 0
    request = urllib.request.Request(url, headers={"User-Agent": "silence-aware-ids/1.0"})
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="ignore").replace("\ufeff", "")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="ignore")
            if error.code == 403 and "일일 최대 호출 건수 제한" in body:
                raise DailyQuotaExceededError("KMA API Hub daily quota exceeded.") from error
            if attempts >= max_retries:
                raise
            attempts += 1
            if error.code == 403:
                time.sleep(min(180.0, 15.0 * attempts))
            else:
                time.sleep(min(10.0, 1.0 * attempts))
        except Exception:
            if attempts >= max_retries:
                raise
            attempts += 1
            time.sleep(min(10.0, 1.0 * attempts))


def build_url(endpoint: str, params: dict[str, str], auth_key: str) -> str:
    query = urllib.parse.urlencode({"authKey": auth_key, **params})
    return f"{endpoint}?{query}"


def fetch_station_metadata(auth_key: str, tm: datetime, timeout: int, max_retries: int) -> dict[str, dict[str, str]]:
    url = build_url(
        STATION_ENDPOINT,
        {"inf": "AWS", "stn": "", "tm": tm.strftime("%Y%m%d%H%M"), "help": "0"},
        auth_key,
    )
    text = fetch_text(url, timeout=timeout, max_retries=max_retries)
    metadata: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        metadata[parts[0]] = {
            "station_id": parts[0],
            "longitude": parts[1],
            "latitude": parts[2],
            "elevation": parts[4],
            "sensor_type": "aws",
            "sensor_group": "surface",
            "sensor_modality": "automatic_weather_station",
        }
    return metadata


def parse_observation_lines(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [value.strip() for value in next(csv.reader([stripped]))]
        if len(parts) != len(RAW_HEADER):
            continue
        rows.append({RAW_HEADER[index]: parts[index] for index in range(len(RAW_HEADER))})
    return rows


def standardize_row(row: dict[str, str], metadata: dict[str, dict[str, str]]) -> dict[str, str]:
    station_id = row["stn"]
    station_meta = metadata.get(station_id, {})
    return {
        "timestamp": datetime.strptime(row["tm"], "%Y%m%d%H%M").isoformat(timespec="minutes"),
        "station_id": station_id,
        "latitude": station_meta.get("latitude", ""),
        "longitude": station_meta.get("longitude", ""),
        "temperature": row.get("ta", ""),
        "humidity": row.get("hm", ""),
        "pressure": row.get("ps", ""),
        "wind_speed": row.get("ws10", ""),
        "wind_direction": row.get("wd10", ""),
        "precipitation": row.get("rn_15m", ""),
        "elevation": station_meta.get("elevation", ""),
        "cost": "",
        "sensor_type": station_meta.get("sensor_type", "aws"),
        "sensor_group": station_meta.get("sensor_group", "surface"),
        "sensor_modality": station_meta.get("sensor_modality", "automatic_weather_station"),
        "site_type": "",
        "maintenance_state": "",
        "maintenance_age": "",
        "source": "aws_recent_1min",
        "raw_timestamp": row.get("tm", ""),
    }


def count_existing_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_last_timestamp(csv_path: Path) -> datetime | None:
    if not csv_path.exists():
        return None
    with csv_path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = bytearray()
        while position > 0:
            position -= 1
            handle.seek(position)
            chunk = handle.read(1)
            if chunk == b"\n" and buffer:
                break
            if chunk not in {b"\n", b"\r"}:
                buffer.extend(chunk)
        if not buffer:
            return None
    last_line = bytes(reversed(buffer)).decode("utf-8", errors="ignore").strip()
    if not last_line:
        return None
    values = next(csv.reader([last_line]))
    if not values or values[0] == "tm":
        return None
    return datetime.strptime(values[0], "%Y%m%d%H%M")


def load_existing_metadata(metadata_csv_path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    if not metadata_csv_path.exists():
        return metadata
    with metadata_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            station_id = (row.get("station_id") or "").strip()
            if not station_id:
                continue
            metadata[station_id] = {
                "station_id": station_id,
                "longitude": (row.get("longitude") or "").strip(),
                "latitude": (row.get("latitude") or "").strip(),
                "elevation": (row.get("elevation") or "").strip(),
                "sensor_type": (row.get("sensor_type") or "aws").strip(),
                "sensor_group": (row.get("sensor_group") or "surface").strip(),
                "sensor_modality": (row.get("sensor_modality") or "automatic_weather_station").strip(),
            }
    return metadata


def main() -> int:
    args = parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    if end < start:
        raise ValueError("end must be on or after start.")
    if args.interval_minutes <= 0:
        raise ValueError("interval-minutes must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_dir / "aws_raw.csv"
    standardized_csv_path = output_dir / "aws_framework.csv"
    metadata_csv_path = output_dir / "aws_station_metadata_auto.csv"
    manifest_path = output_dir / "kma_download_manifest.json"
    framework_config_path = output_dir / "framework_aws_q1.json"

    if args.overwrite and args.resume:
        raise ValueError("overwrite and resume cannot be enabled at the same time.")

    protected_paths = (raw_csv_path, standardized_csv_path, metadata_csv_path, manifest_path, framework_config_path)
    for path in protected_paths:
        if path.exists() and not args.overwrite and not args.resume:
            raise FileExistsError(f"{path} already exists. Re-run with --overwrite to replace it.")

    timestamps = iter_timestamps(start, end, args.interval_minutes)
    last_timestamp = read_last_timestamp(raw_csv_path) if args.resume else None
    remaining_timestamps = timestamps
    if last_timestamp is not None:
        remaining_timestamps = [stamp for stamp in timestamps if stamp > last_timestamp]
    remaining_chunks = iter_chunk_ranges(remaining_timestamps, args.chunk_size)

    if args.resume and metadata_csv_path.exists():
        metadata = load_existing_metadata(metadata_csv_path)
    else:
        metadata = fetch_station_metadata(args.auth_key, start, args.timeout, args.max_retries)
        with metadata_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "station_id",
                    "latitude",
                    "longitude",
                    "elevation",
                    "sensor_type",
                    "sensor_group",
                    "sensor_modality",
                ],
            )
            writer.writeheader()
            for row in metadata.values():
                writer.writerow(row)

    total_rows = count_existing_rows(raw_csv_path) if args.resume else 0
    missing_metadata_rows = 0
    raw_mode = "a" if args.resume and raw_csv_path.exists() else "w"
    std_mode = "a" if args.resume and standardized_csv_path.exists() else "w"
    with raw_csv_path.open(raw_mode, encoding="utf-8", newline="") as raw_handle, standardized_csv_path.open(
        std_mode,
        encoding="utf-8",
        newline="",
    ) as std_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=list(RAW_HEADER))
        std_writer = csv.DictWriter(std_handle, fieldnames=list(STANDARDIZED_FIELDNAMES))
        if raw_mode == "w":
            raw_writer.writeheader()
        if std_mode == "w":
            std_writer.writeheader()

        started = time.time()
        processed_before_resume = len(timestamps) - len(remaining_timestamps)
        for chunk_start, chunk_end, chunk_len in remaining_chunks:
            url = build_url(
                OBS_ENDPOINT,
                {
                    "tm1": chunk_start.strftime("%Y%m%d%H%M"),
                    "tm2": chunk_end.strftime("%Y%m%d%H%M"),
                    "stn": str(args.station_id),
                    "disp": "1",
                    "help": "0",
                },
                args.auth_key,
            )
            try:
                text = fetch_text(url, args.timeout, args.max_retries)
            except DailyQuotaExceededError:
                print(
                    f"Daily quota exceeded while requesting {chunk_start.isoformat(timespec='minutes')} "
                    f"to {chunk_end.isoformat(timespec='minutes')}. Resume later.",
                    file=sys.stderr,
                )
                return 75
            rows = parse_observation_lines(text)
            for row in rows:
                raw_writer.writerow(row)
                standardized = standardize_row(row, metadata)
                if not standardized["latitude"] or not standardized["longitude"]:
                    missing_metadata_rows += 1
                std_writer.writerow(standardized)
            total_rows += len(rows)
            if args.sleep_seconds > 0.0:
                time.sleep(args.sleep_seconds)
            processed_before_resume += chunk_len
            index = processed_before_resume
            if index % 24 == 0 or index == len(timestamps):
                elapsed = time.time() - started
                print(
                    f"[{index}/{len(timestamps)}] {chunk_end.isoformat(timespec='minutes')} "
                    f"rows={len(rows)} total_rows={total_rows} elapsed={elapsed:.1f}s"
                )

    write_kma_framework_config(
        config_path=framework_config_path,
        data_path=standardized_csv_path,
        overwrite=True,
    )

    manifest = {
        "source": "aws_recent_1min",
        "start": start.isoformat(timespec="minutes"),
        "end": end.isoformat(timespec="minutes"),
        "interval_minutes": args.interval_minutes,
        "station_id": str(args.station_id),
        "raw_csv_path": str(raw_csv_path.resolve()),
        "standardized_csv_path": str(standardized_csv_path.resolve()),
        "resolved_metadata_csv_path": str(metadata_csv_path.resolve()),
        "framework_config_path": str(framework_config_path.resolve()),
        "timestamp_count": len(timestamps),
        "raw_row_count": total_rows,
        "missing_metadata_rows": missing_metadata_rows,
        "request_url_template": _redacted_template(args.station_id),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print(f"Framework config: {framework_config_path}")
    return 0


def _redacted_template(station_id: str) -> str:
    return (
        "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
        f"?authKey=***&tm2=<YYYYMMDDHHMM>&stn={station_id}&disp=1&help=0"
    )


if __name__ == "__main__":
    raise SystemExit(main())
