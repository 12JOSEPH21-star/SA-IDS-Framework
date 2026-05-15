from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable


NWP_SOURCE_CHOICES = ("ldaps_unis_all", "rdaps_unis_all", "nwp_latlon_grid")
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 2
DEFAULT_PAGE_SIZE = 1000
DEFAULT_BASE_INTERVAL_HOURS = 6

NWP_API_PAGE = "https://apihub.kma.go.kr/apiList.do?seqApi=9"
API_HUB_GUIDE = "https://apihub.kma.go.kr/static/file/%EA%B8%B0%EC%83%81%EC%B2%AD_API%ED%97%88%EB%B8%8C_%EC%82%AC%EC%9A%A9_%EB%B0%A9%EB%B2%95_%EC%95%88%EB%82%B4.pdf"
LDAPS_UNIS_ALL_ENDPOINT = "https://apihub.kma.go.kr/api/typ02/openApi/NwpModelInfoService/getLdapsUnisAll"
RDAPS_UNIS_ALL_ENDPOINT = "https://apihub.kma.go.kr/api/typ02/openApi/NwpModelInfoService/getRdapsUnisAll"
NWP_LATLON_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-nwp_latlon_api"

SUMMARY_FIELDNAMES: tuple[str, ...] = (
    "source",
    "base_time",
    "forecast_time",
    "lead_hour",
    "data_type_code",
    "grid_km",
    "xdim",
    "ydim",
    "x0",
    "y0",
    "unit",
    "lon",
    "lat",
    "item_index",
    "item_count",
    "value_count",
    "value_preview",
    "raw_path",
)
GRID_FIELDNAMES: tuple[str, ...] = ("nwp_code", "coordinate_type", "grid_index", "value")

BytesFetcher = Callable[[str, int], bytes]


@dataclass(frozen=True)
class KmaNwpDownloadConfig:
    """Configuration for KMA NWP-anchor downloads.

    Attributes:
        source: NWP source identifier.
        output_dir: Directory for manifests and raw/parsed outputs.
        start_base_time: Inclusive base forecast time in KST.
        end_base_time: Inclusive end base time in KST.
        data_type_code: NWP variable code, for example `Temp`.
        lead_hours: Forecast lead hours requested for typ02 sources.
        base_interval_hours: Step between base times.
        service_key: Optional KMA API Hub authKey.
        raw_dir: Optional raw payload directory.
        summary_csv_path: Optional request-summary CSV path.
        grid_csv_path: Optional grid-coordinate CSV path for `nwp_latlon_grid`.
        data_type: typ02 response data type.
        num_rows: typ02 page size.
        nwp_code: Grid code for `nwp_latlon_grid`, for example `u015` or `u120`.
        coordinate_types: Coordinate types requested from `nwp_latlon_grid`.
        overwrite: Whether to overwrite existing outputs.
        dry_run: Whether to write only the manifest.
        timeout: HTTP timeout in seconds.
        max_retries: Retry budget per request.
    """

    source: str
    output_dir: Path
    start_base_time: datetime
    end_base_time: datetime
    data_type_code: str = "Temp"
    lead_hours: tuple[int, ...] = (0, 6, 12)
    base_interval_hours: int = DEFAULT_BASE_INTERVAL_HOURS
    service_key: str | None = None
    raw_dir: Path | None = None
    summary_csv_path: Path | None = None
    grid_csv_path: Path | None = None
    data_type: str = "JSON"
    num_rows: int = DEFAULT_PAGE_SIZE
    nwp_code: str = "u015"
    coordinate_types: tuple[str, ...] = ("lon", "lat")
    overwrite: bool = False
    dry_run: bool = False
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_RETRIES

    def validate(self) -> None:
        """Validate the NWP downloader configuration."""
        if self.source not in NWP_SOURCE_CHOICES:
            raise ValueError(f"source must be one of {NWP_SOURCE_CHOICES}.")
        if self.end_base_time < self.start_base_time:
            raise ValueError("end_base_time must be on or after start_base_time.")
        if self.base_interval_hours <= 0:
            raise ValueError("base_interval_hours must be positive.")
        if not self.lead_hours:
            raise ValueError("lead_hours must not be empty.")
        if self.num_rows <= 0:
            raise ValueError("num_rows must be positive.")
        if self.data_type not in {"JSON", "XML"}:
            raise ValueError("data_type must be JSON or XML.")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be zero or greater.")
        if self.source == "nwp_latlon_grid":
            for coordinate_type in self.coordinate_types:
                if coordinate_type not in {"lon", "lat"}:
                    raise ValueError("coordinate_types must contain only 'lon' and/or 'lat'.")


@dataclass(frozen=True)
class KmaNwpDownloadArtifacts:
    """Artifacts written by the KMA NWP downloader."""

    output_dir: Path
    manifest_path: Path
    raw_dir: Path | None
    summary_csv_path: Path | None
    grid_csv_path: Path | None
    row_count: int
    dry_run: bool


def download_kma_nwp(
    config: KmaNwpDownloadConfig,
    *,
    fetcher: BytesFetcher | None = None,
) -> KmaNwpDownloadArtifacts:
    """Download KMA NWP anchor payloads and summarize them.

    Args:
        config: NWP download configuration.
        fetcher: Optional HTTP fetch override used for tests.

    Returns:
        Dataclass describing written artifacts.
    """
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = config.raw_dir or (config.output_dir / "raw")
    summary_csv_path = config.summary_csv_path or (config.output_dir / "kma_nwp_summary.csv")
    grid_csv_path = config.grid_csv_path or (config.output_dir / "kma_nwp_grid.csv")
    manifest_path = config.output_dir / "kma_nwp_manifest.json"

    protected_paths = [manifest_path]
    if not config.dry_run:
        protected_paths.append(summary_csv_path)
        if config.source == "nwp_latlon_grid":
            protected_paths.append(grid_csv_path)
    for path in protected_paths:
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"{path} already exists. Re-run with --overwrite to replace it.")
    if raw_dir.exists() and not config.overwrite and not config.dry_run:
        raise FileExistsError(f"{raw_dir} already exists. Re-run with --overwrite to replace it.")

    auth_key, auth_source = _resolve_service_key(config)
    request_plan = _build_request_plan(config, auth_key=auth_key)
    request_status: list[dict[str, Any]] = []
    summary_rows: list[dict[str, str]] = []
    grid_rows: list[dict[str, str]] = []
    fetch = fetcher or _default_fetcher

    if not config.dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for request in request_plan:
            payload = _fetch_with_retries(
                fetch,
                url=request["url"],
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
            suffix = ".txt" if request["response_format"] == "text" else ".json"
            raw_path = raw_dir / f"{request['label']}{suffix}"
            raw_path.write_bytes(payload)
            if request["response_format"] == "json":
                parsed_rows = _summarize_json_payload(
                    payload=payload,
                    source=config.source,
                    data_type_code=config.data_type_code,
                    lead_hour=request.get("lead_hour"),
                    raw_path=raw_path,
                )
                summary_rows.extend(parsed_rows)
                request_status.append(
                    {
                        "label": request["label"],
                        "url": request["url"],
                        "items": len(parsed_rows),
                        "raw_path": str(raw_path.resolve()),
                    }
                )
            else:
                values = _parse_numeric_text_payload(payload.decode("utf-8", errors="ignore"))
                for index, value in enumerate(values):
                    grid_rows.append(
                        {
                            "nwp_code": config.nwp_code,
                            "coordinate_type": request["coordinate_type"],
                            "grid_index": str(index),
                            "value": value,
                        }
                    )
                summary_rows.append(
                    {
                        "source": config.source,
                        "base_time": "",
                        "forecast_time": "",
                        "lead_hour": "",
                        "data_type_code": config.data_type_code,
                        "grid_km": "",
                        "xdim": "",
                        "ydim": "",
                        "x0": "",
                        "y0": "",
                        "unit": request["coordinate_type"],
                        "lon": "",
                        "lat": "",
                        "item_index": "",
                        "item_count": "1",
                        "value_count": str(len(values)),
                        "value_preview": ",".join(values[:3]),
                        "raw_path": str(raw_path.resolve()),
                    }
                )
                request_status.append(
                    {
                        "label": request["label"],
                        "url": request["url"],
                        "items": len(values),
                        "raw_path": str(raw_path.resolve()),
                    }
                )
        _write_csv(summary_csv_path, SUMMARY_FIELDNAMES, summary_rows)
        if config.source == "nwp_latlon_grid":
            _write_csv(grid_csv_path, GRID_FIELDNAMES, grid_rows)
        else:
            grid_csv_path = None
        row_count = len(summary_rows)
    else:
        raw_dir = None
        summary_csv_path = None
        grid_csv_path = None
        row_count = 0
        for request in request_plan:
            request_status.append({"label": request["label"], "url": request["url"]})

    manifest = {
        "source": config.source,
        "start_base_time": config.start_base_time.isoformat(timespec="minutes"),
        "end_base_time": config.end_base_time.isoformat(timespec="minutes"),
        "data_type_code": config.data_type_code,
        "lead_hours": list(config.lead_hours),
        "base_interval_hours": config.base_interval_hours,
        "nwp_code": config.nwp_code,
        "coordinate_types": list(config.coordinate_types),
        "auth_source": auth_source,
        "request_status": request_status,
        "summary_csv_path": None if summary_csv_path is None else str(summary_csv_path.resolve()),
        "grid_csv_path": None if grid_csv_path is None else str(grid_csv_path.resolve()),
        "official_sources": [NWP_API_PAGE, API_HUB_GUIDE],
        "dry_run": config.dry_run,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return KmaNwpDownloadArtifacts(
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        raw_dir=raw_dir,
        summary_csv_path=summary_csv_path,
        grid_csv_path=grid_csv_path,
        row_count=row_count,
        dry_run=config.dry_run,
    )


def _resolve_service_key(config: KmaNwpDownloadConfig) -> tuple[str | None, str]:
    if config.service_key:
        return config.service_key, "cli"
    for name in ("KMA_APIHUB_AUTH_KEY", "KMA_AUTH_KEY"):
        value = os.environ.get(name)
        if value:
            return value, f"env:{name}"
    return None, "missing"


def _build_request_plan(config: KmaNwpDownloadConfig, *, auth_key: str | None) -> list[dict[str, Any]]:
    if config.source == "nwp_latlon_grid":
        plan: list[dict[str, Any]] = []
        for coordinate_type in config.coordinate_types:
            params = {"nwp": config.nwp_code, "latlon": coordinate_type, "help": "1"}
            if auth_key:
                params["authKey"] = auth_key
            plan.append(
                {
                    "label": f"{config.source}_{config.nwp_code}_{coordinate_type}",
                    "url": f"{NWP_LATLON_ENDPOINT}?{urllib.parse.urlencode(params)}",
                    "response_format": "text",
                    "coordinate_type": coordinate_type,
                }
            )
        return plan

    endpoint = LDAPS_UNIS_ALL_ENDPOINT if config.source == "ldaps_unis_all" else RDAPS_UNIS_ALL_ENDPOINT
    plan: list[dict[str, Any]] = []
    current = config.start_base_time
    step = timedelta(hours=config.base_interval_hours)
    while current <= config.end_base_time:
        for lead_hour in config.lead_hours:
            params = {
                "pageNo": "1",
                "numOfRows": str(config.num_rows),
                "dataType": config.data_type,
                "baseTime": current.strftime("%Y%m%d%H%M"),
                "leadHour": str(lead_hour),
                "dataTypeCd": config.data_type_code,
            }
            if auth_key:
                params["authKey"] = auth_key
            plan.append(
                {
                    "label": f"{config.source}_{current.strftime('%Y%m%d%H%M')}_h{lead_hour:03d}",
                    "url": f"{endpoint}?{urllib.parse.urlencode(params)}",
                    "response_format": "json" if config.data_type == "JSON" else "text",
                    "lead_hour": lead_hour,
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
        raise RuntimeError(f"KMA NWP request failed with HTTP {error.code}: {body[:200]}") from error


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "item" in payload:
            item = payload["item"]
            if isinstance(item, list):
                return [value for value in item if isinstance(value, dict)]
            if isinstance(item, dict):
                return [item]
        items: list[dict[str, Any]] = []
        for value in payload.values():
            items.extend(_extract_items(value))
        return items
    if isinstance(payload, list):
        items: list[dict[str, Any]] = []
        for value in payload:
            items.extend(_extract_items(value))
        return items
    return []


def _summarize_json_payload(
    *,
    payload: bytes,
    source: str,
    data_type_code: str,
    lead_hour: int | None,
    raw_path: Path,
) -> list[dict[str, str]]:
    content = json.loads(payload.decode("utf-8"))
    response = content.get("response", content) if isinstance(content, dict) else content
    header = response.get("header", {}) if isinstance(response, dict) else {}
    result_code = str(header.get("resultCode", "00"))
    if result_code not in {"00", "0"}:
        return []
    body = response.get("body", response) if isinstance(response, dict) else response
    items = _extract_items(body)
    rows: list[dict[str, str]] = []
    item_count = len(items)
    for index, item in enumerate(items):
        raw_value = item.get("value", "")
        rows.append(
            {
                "source": source,
                "base_time": _format_timestamp(str(item.get("baseTime", ""))),
                "forecast_time": _format_timestamp(str(item.get("fcstTime", ""))),
                "lead_hour": "" if lead_hour is None else str(lead_hour),
                "data_type_code": str(item.get("dataTypeCd", data_type_code)),
                "grid_km": str(item.get("gridKm", "")),
                "xdim": str(item.get("xdim", "")),
                "ydim": str(item.get("ydim", "")),
                "x0": str(item.get("x0", "")),
                "y0": str(item.get("y0", "")),
                "unit": str(item.get("unit", "")),
                "lon": str(item.get("lon", "")),
                "lat": str(item.get("lat", "")),
                "item_index": str(index),
                "item_count": str(item_count),
                "value_count": str(_count_values(raw_value)),
                "value_preview": _preview_value(raw_value),
                "raw_path": str(raw_path.resolve()),
            }
        )
    return rows


def _count_values(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        return len([token for token in re.split(r"[\s,]+", value.strip()) if token])
    return 1 if value not in ("", None) else 0


def _preview_value(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value[:3])
    if isinstance(value, str):
        tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
        return ",".join(tokens[:3])
    return str(value)


def _parse_numeric_text_payload(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def _format_timestamp(value: str) -> str:
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) == 12:
        return datetime.strptime(digits, "%Y%m%d%H%M").isoformat(timespec="minutes")
    return value


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
