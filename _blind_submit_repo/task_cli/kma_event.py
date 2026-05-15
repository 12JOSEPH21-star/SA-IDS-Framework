from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable


EVENT_SOURCE_CHOICES = ("warning_history", "warning_information", "warning_now_snapshot")
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 2
DEFAULT_WARNING_INTERVAL_MINUTES = 60

WARNING_API_PAGE = "https://apihub.kma.go.kr/apiList.do?apiMov=%ED%8A%B9.%EC%A0%95%EB%B3%B4+%EC%9E%90%EB%A3%8C+%EC%A1%B0%ED%9A%8C&seqApi=10&seqApiSub=288"
API_HUB_GUIDE = "https://apihub.kma.go.kr/static/file/%EA%B8%B0%EC%83%81%EC%B2%AD_API%ED%97%88%EB%B8%8C_%EC%82%AC%EC%9A%A9_%EB%B0%A9%EB%B2%95_%EC%95%88%EB%82%B4.pdf"
WARNING_HISTORY_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/url/wrn_reg.php"
WARNING_INFORMATION_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/url/wrn_inf_rpt.php"
WARNING_NOW_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/url/wrn_now_data_new.php"

STANDARDIZED_FIELDNAMES: tuple[str, ...] = (
    "source",
    "basis",
    "publication_time",
    "effective_time",
    "input_time",
    "start_time",
    "end_time",
    "region_id",
    "region_parent_id",
    "region_short_name",
    "region_name",
    "issuing_office",
    "issuing_station_id",
    "warning_code",
    "warning_level",
    "command",
    "grade",
    "status_count",
    "report_flag",
    "sequence",
    "forecaster",
    "operator",
    "raw_text",
)

BytesFetcher = Callable[[str, int], bytes]


@dataclass(frozen=True)
class KmaEventDownloadConfig:
    """Configuration for KMA event-truth downloads.

    Attributes:
        source: Event source identifier.
        output_dir: Directory for manifests and CSV outputs.
        start_datetime: Inclusive event-window start in KST.
        end_datetime: Inclusive event-window end in KST.
        raw_csv_path: Optional raw merged CSV output path.
        standardized_csv_path: Optional normalized CSV output path.
        service_key: Optional KMA API Hub authKey.
        warning_codes: Optional weather-warning codes such as `R` or `T`.
        region_codes: Optional KMA region ids.
        basis: `f` for publication-time basis or `e` for effective-time basis.
        interval_minutes: Snapshot interval for `warning_now_snapshot`.
        disp: API display level.
        overwrite: Whether to overwrite existing outputs.
        dry_run: Whether to write only the manifest.
        timeout: HTTP timeout in seconds.
        max_retries: Retry budget per request.
    """

    source: str
    output_dir: Path
    start_datetime: datetime
    end_datetime: datetime
    raw_csv_path: Path | None = None
    standardized_csv_path: Path | None = None
    service_key: str | None = None
    warning_codes: tuple[str, ...] = ()
    region_codes: tuple[str, ...] = ()
    basis: str = "f"
    interval_minutes: int = DEFAULT_WARNING_INTERVAL_MINUTES
    disp: int = 0
    overwrite: bool = False
    dry_run: bool = False
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_RETRIES

    def validate(self) -> None:
        """Validate the downloader configuration."""
        if self.source not in EVENT_SOURCE_CHOICES:
            raise ValueError(f"source must be one of {EVENT_SOURCE_CHOICES}.")
        if self.end_datetime < self.start_datetime:
            raise ValueError("end_datetime must be on or after start_datetime.")
        if self.basis not in {"f", "e"}:
            raise ValueError("basis must be 'f' or 'e'.")
        if self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive.")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be zero or greater.")


@dataclass(frozen=True)
class KmaEventDownloadArtifacts:
    """Artifacts written by the KMA event downloader."""

    output_dir: Path
    manifest_path: Path
    raw_csv_path: Path | None
    standardized_csv_path: Path | None
    row_count: int
    dry_run: bool


def download_kma_events(
    config: KmaEventDownloadConfig,
    *,
    fetcher: BytesFetcher | None = None,
) -> KmaEventDownloadArtifacts:
    """Download KMA warning/event records and normalize them.

    Args:
        config: Event download configuration.
        fetcher: Optional HTTP fetch override used for tests.

    Returns:
        Dataclass describing generated files.
    """
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = config.raw_csv_path or (config.output_dir / f"{config.source}_raw.csv")
    standardized_csv_path = config.standardized_csv_path or (config.output_dir / f"{config.source}_standardized.csv")
    manifest_path = config.output_dir / "kma_event_manifest.json"

    protected_paths = [manifest_path]
    if not config.dry_run:
        protected_paths.extend([raw_csv_path, standardized_csv_path])
    for path in protected_paths:
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"{path} already exists. Re-run with --overwrite to replace it.")

    auth_key, auth_source = _resolve_service_key(config)
    request_plan = _build_request_plan(config, auth_key=auth_key)
    request_status: list[dict[str, Any]] = []
    raw_rows: list[dict[str, str]] = []
    standardized_rows: list[dict[str, str]] = []

    fetch = fetcher or _default_fetcher
    if not config.dry_run:
        for request in request_plan:
            payload = _fetch_with_retries(
                fetch,
                url=request["url"],
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
            rows = _parse_whitespace_table(payload.decode("utf-8", errors="ignore"))
            for row in rows:
                enriched = dict(row)
                enriched["request_url"] = request["url"]
                enriched["request_label"] = request["label"]
                raw_rows.append(enriched)
                standardized_rows.append(_standardize_event_row(config.source, config.basis, row))
            request_status.append({"label": request["label"], "url": request["url"], "rows": len(rows)})
        _write_raw_csv(raw_csv_path, raw_rows)
        _write_standardized_csv(standardized_csv_path, standardized_rows)
        row_count = len(raw_rows)
    else:
        raw_csv_path = None
        standardized_csv_path = None
        row_count = 0
        for request in request_plan:
            request_status.append({"label": request["label"], "url": request["url"], "rows": None})

    manifest = {
        "source": config.source,
        "start_datetime": config.start_datetime.isoformat(timespec="minutes"),
        "end_datetime": config.end_datetime.isoformat(timespec="minutes"),
        "warning_codes": list(config.warning_codes),
        "region_codes": list(config.region_codes),
        "basis": config.basis,
        "interval_minutes": config.interval_minutes,
        "disp": config.disp,
        "auth_source": auth_source,
        "request_status": request_status,
        "raw_csv_path": None if raw_csv_path is None else str(raw_csv_path.resolve()),
        "standardized_csv_path": None if standardized_csv_path is None else str(standardized_csv_path.resolve()),
        "official_sources": [WARNING_API_PAGE, API_HUB_GUIDE],
        "dry_run": config.dry_run,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return KmaEventDownloadArtifacts(
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        raw_csv_path=raw_csv_path,
        standardized_csv_path=standardized_csv_path,
        row_count=row_count,
        dry_run=config.dry_run,
    )


def _resolve_service_key(config: KmaEventDownloadConfig) -> tuple[str | None, str]:
    if config.service_key:
        return config.service_key, "cli"
    for name in ("KMA_APIHUB_AUTH_KEY", "KMA_AUTH_KEY"):
        value = os.environ.get(name)
        if value:
            return value, f"env:{name}"
    return None, "missing"


def _build_request_plan(config: KmaEventDownloadConfig, *, auth_key: str | None) -> list[dict[str, str]]:
    if config.source == "warning_now_snapshot":
        return _build_warning_now_plan(config, auth_key=auth_key)
    if config.source == "warning_information":
        return _build_range_plan(config, auth_key=auth_key, endpoint=WARNING_INFORMATION_ENDPOINT)
    return _build_range_plan(config, auth_key=auth_key, endpoint=WARNING_HISTORY_ENDPOINT)


def _build_range_plan(
    config: KmaEventDownloadConfig,
    *,
    auth_key: str | None,
    endpoint: str,
) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    current = config.start_datetime
    chunk = timedelta(days=31)
    warning_codes = config.warning_codes or ("",)
    region_codes = config.region_codes or ("",)
    while current <= config.end_datetime:
        chunk_end = min(config.end_datetime, current + chunk - timedelta(minutes=1))
        for warning_code in warning_codes:
            for region_code in region_codes:
                params = {
                    "tmfc1": current.strftime("%Y%m%d%H%M"),
                    "tmfc2": chunk_end.strftime("%Y%m%d%H%M"),
                    "disp": str(config.disp),
                    "help": "1",
                }
                if warning_code:
                    params["wrn"] = warning_code
                if region_code:
                    params["reg"] = region_code
                if auth_key:
                    params["authKey"] = auth_key
                label_parts = [current.strftime("%Y%m%d%H%M"), chunk_end.strftime("%Y%m%d%H%M")]
                if warning_code:
                    label_parts.append(warning_code)
                if region_code:
                    label_parts.append(region_code)
                plan.append({"label": "_".join(label_parts), "url": f"{endpoint}?{urllib.parse.urlencode(params)}"})
        current = chunk_end + timedelta(minutes=1)
    return plan


def _build_warning_now_plan(config: KmaEventDownloadConfig, *, auth_key: str | None) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    current = config.start_datetime
    step = timedelta(minutes=config.interval_minutes)
    while current <= config.end_datetime:
        params = {
            "fe": config.basis,
            "tm": current.strftime("%Y%m%d%H%M"),
            "disp": str(config.disp),
            "help": "1",
        }
        if auth_key:
            params["authKey"] = auth_key
        plan.append(
            {
                "label": current.strftime("%Y%m%d%H%M"),
                "url": f"{WARNING_NOW_ENDPOINT}?{urllib.parse.urlencode(params)}",
            }
        )
        current += step
    return plan


def _fetch_with_retries(fetcher: BytesFetcher, *, url: str, timeout: int, max_retries: int) -> bytes:
    attempt = 0
    while True:
        try:
            return fetcher(url, timeout)
        except Exception:
            if attempt >= max_retries:
                raise
            attempt += 1
            time.sleep(min(2.0 * attempt, 5.0))


def _default_fetcher(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "silence-aware-ids/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="ignore")
        if error.code == 403 and "일일 최대 호출 건수 제한" in body:
            raise RuntimeError("KMA API Hub daily quota exceeded. Retry after the next quota reset.") from error
        raise RuntimeError(f"KMA event request failed with HTTP {error.code}: {body[:200]}") from error


def _parse_whitespace_table(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in lines:
        if line.startswith("#"):
            candidate = line.lstrip("# ").strip()
            if candidate:
                tokens = candidate.split()
                if len(tokens) >= 3 and any(any(ch.isalpha() for ch in token) for token in tokens):
                    header = [_normalize_token(token) for token in tokens]
            continue
        if header is None:
            continue
        tokens = line.split()
        if len(tokens) < len(header):
            tokens.extend([""] * (len(header) - len(tokens)))
        rows.append({header[index]: tokens[index] for index in range(len(header))})
    return rows


def _normalize_token(token: str) -> str:
    return token.strip().strip(",").upper()


def _format_timestamp(value: str) -> str:
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) != 12:
        return value
    try:
        return datetime.strptime(digits, "%Y%m%d%H%M").isoformat(timespec="minutes")
    except ValueError:
        return value


def _standardize_event_row(source: str, basis: str, row: dict[str, str]) -> dict[str, str]:
    return {
        "source": source,
        "basis": basis,
        "publication_time": _format_timestamp(row.get("TM_FC", "")),
        "effective_time": _format_timestamp(row.get("TM_EF", "")),
        "input_time": _format_timestamp(row.get("TM_IN", "")),
        "start_time": _format_timestamp(row.get("TM_ST", "")),
        "end_time": _format_timestamp(row.get("TM_ED", "")),
        "region_id": row.get("REG_ID", ""),
        "region_parent_id": row.get("REG_UP", ""),
        "region_short_name": row.get("REG_KO", ""),
        "region_name": row.get("REG_NAME", row.get("REG_UP_KO", "")),
        "issuing_office": row.get("STN", ""),
        "issuing_station_id": row.get("STN_ID", ""),
        "warning_code": row.get("WRN", ""),
        "warning_level": row.get("LVL", ""),
        "command": row.get("CMD", ""),
        "grade": row.get("GRD", ""),
        "status_count": row.get("CNT", ""),
        "report_flag": row.get("RPT", ""),
        "sequence": row.get("TM_SEQ", ""),
        "forecaster": row.get("MAN_FC", ""),
        "operator": row.get("MAN_IN", ""),
        "raw_text": json.dumps(row, ensure_ascii=False, sort_keys=True),
    }


def _write_raw_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_standardized_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(STANDARDIZED_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
