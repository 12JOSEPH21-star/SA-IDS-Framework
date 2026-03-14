from __future__ import annotations

import csv
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable


KMA_SOURCE_CHOICES = ("asos_hourly", "aws_recent_1min")
KMA_DATA_TYPE_CHOICES = ("JSON", "XML")
DEFAULT_KMA_PAGE_SIZE = 999
DEFAULT_KMA_TIMEOUT = 30
DEFAULT_KMA_RETRIES = 2
ASOS_HOURLY_ENDPOINT = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
AWS_RECENT_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
ASOS_OFFICIAL_PAGE = "https://www.data.go.kr/data/15057210/openapi.do"
AWS_API_HUB_PAGE = "https://www.data.go.kr/data/15139433/openapi.do"
AWS_API_HUB_GUIDE = "https://apihub.kma.go.kr/static/file/%EA%B8%B0%EC%83%81%EC%B2%AD_API%ED%97%88%EB%B8%8C_%EC%82%AC%EC%9A%A9_%EB%B0%A9%EB%B2%95_%EC%95%88%EB%82%B4.pdf"
AWS_STATION_INFO_ENDPOINT = "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php"
STANDARDIZED_FIELDNAMES = (
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
    "elevation",
    "cost",
    "sensor_type",
    "sensor_group",
    "sensor_modality",
    "site_type",
    "maintenance_state",
    "maintenance_age",
    "source",
    "raw_timestamp",
)
AWS_APIHUB_COMPACT_HEADER = (
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
    "hm",
    "pa",
    "ps",
    "td",
    "status",
)
AWS_APIHUB_FULL_HEADER = (
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
)
AWS_APIHUB_FULL_HEADER_WITH_STATUS = AWS_APIHUB_FULL_HEADER + ("status",)


@dataclass(frozen=True)
class KmaDownloadConfig:
    """Configuration for Korean weather-observation data downloads.

    Attributes:
        source: Data source identifier. Supported values are `"asos_hourly"`
            and `"aws_recent_1min"`.
        output_dir: Directory for manifests and CSV outputs.
        raw_csv_path: Optional raw API CSV output. Defaults to
            `<output_dir>/<source>_raw.csv`.
        standardized_csv_path: Optional framework-oriented CSV output.
        framework_config_path: Optional `aws_network` framework config output.
        metadata_csv_path: Optional station metadata CSV used to populate
            latitude, longitude, elevation, and sensor metadata columns.
        auto_station_metadata: Whether to fetch station coordinates and
            elevation automatically from the KMA API Hub when metadata_csv_path
            is omitted for AWS downloads.
        metadata_template_path: Optional CSV template path written from the
            provided station ids.
        service_key: Optional source-specific credential. For ASOS, this is a
            data.go.kr service key. For AWS API Hub, this is the `authKey`.
        start_date: Inclusive start date for ASOS hourly downloads.
        end_date: Inclusive end date for ASOS hourly downloads.
        start_hour: Inclusive start hour in `HH` format for ASOS hourly data.
        end_hour: Inclusive end hour in `HH` format for ASOS hourly data.
        station_ids: Station ids requested from the API.
        aws_datetime: Observation timestamp for the AWS recent 1-minute API.
        data_type: Response format, `"JSON"` or `"XML"`.
        num_rows: Page size for paginated requests.
        overwrite: Whether to overwrite existing outputs.
        dry_run: Whether to skip network calls and only write the manifest.
        timeout: Request timeout in seconds.
        max_retries: Retry budget per request.
    """

    source: str
    output_dir: Path
    raw_csv_path: Path | None = None
    standardized_csv_path: Path | None = None
    framework_config_path: Path | None = None
    metadata_csv_path: Path | None = None
    auto_station_metadata: bool = True
    metadata_template_path: Path | None = None
    service_key: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    start_hour: str = "00"
    end_hour: str = "23"
    station_ids: tuple[str, ...] = ()
    aws_datetime: datetime | None = None
    data_type: str = "JSON"
    num_rows: int = DEFAULT_KMA_PAGE_SIZE
    overwrite: bool = False
    dry_run: bool = False
    timeout: int = DEFAULT_KMA_TIMEOUT
    max_retries: int = DEFAULT_KMA_RETRIES

    def validate(self) -> None:
        """Validate the download configuration."""
        if self.source not in KMA_SOURCE_CHOICES:
            raise ValueError(f"source must be one of {KMA_SOURCE_CHOICES}.")
        if self.data_type not in KMA_DATA_TYPE_CHOICES:
            raise ValueError(f"data_type must be one of {KMA_DATA_TYPE_CHOICES}.")
        if not self.station_ids:
            raise ValueError("At least one station id is required.")
        if self.num_rows <= 0:
            raise ValueError("num_rows must be positive.")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be zero or greater.")
        if self.framework_config_path is not None and self.standardized_csv_path is None:
            raise ValueError("Framework config generation requires --standardized-csv.")
        if self.source == "asos_hourly":
            if self.start_date is None or self.end_date is None:
                raise ValueError("ASOS hourly downloads require start_date and end_date.")
            if self.start_date > self.end_date:
                raise ValueError("start_date must be on or before end_date.")
            if len(self.start_hour) != 2 or len(self.end_hour) != 2:
                raise ValueError("start_hour and end_hour must use HH format.")
        if self.source == "aws_recent_1min" and self.aws_datetime is None:
            raise ValueError("AWS recent 1-minute downloads require --aws-datetime.")


@dataclass(frozen=True)
class KmaDownloadArtifacts:
    """Artifacts written by the Korean observation downloader."""

    manifest_path: Path
    output_dir: Path
    raw_csv_path: Path | None
    standardized_csv_path: Path | None
    framework_config_path: Path | None
    resolved_metadata_csv_path: Path | None
    metadata_template_path: Path | None
    row_count: int
    dry_run: bool


def download_kma(
    config: KmaDownloadConfig,
    *,
    fetcher: Callable[[str, int], bytes] | None = None,
) -> KmaDownloadArtifacts:
    """Download Korean ASOS or AWS observations and optionally standardize them.

    Args:
        config: Downloader configuration.
        fetcher: Optional HTTP fetch function used for testing.

    Returns:
        Paths to written artifacts and row counts.
    """
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = config.raw_csv_path or (config.output_dir / f"{config.source}_raw.csv")
    standardized_csv_path = config.standardized_csv_path
    metadata_template_path = config.metadata_template_path
    framework_config_path = config.framework_config_path
    resolved_metadata_csv_path = config.metadata_csv_path

    for path in (raw_csv_path, standardized_csv_path, framework_config_path, metadata_template_path):
        if path is not None and path.exists() and not config.overwrite:
            raise FileExistsError(f"{path} already exists. Re-run with --overwrite to replace it.")

    service_key, auth_source = _resolve_service_key(config)
    request_plan = _build_request_plan(config)
    warnings: list[str] = []

    if metadata_template_path is not None:
        write_station_metadata_template(
            metadata_template_path,
            station_ids=config.station_ids,
            source=config.source,
            overwrite=config.overwrite,
        )

    rows: list[dict[str, str]] = []
    request_status: list[dict[str, Any]] = []
    if config.dry_run:
        request_status = [
            {
                "url": _redact_secret_url(_build_request_url(request["endpoint"], request["params"], service_key)),
                "station_id": request["station_id"],
                "page_count": 0,
                "row_count": 0,
                "status": "planned",
            }
            for request in request_plan
        ]
    else:
        if service_key is None:
            if config.source == "aws_recent_1min":
                raise ValueError(
                    "A KMA API Hub authKey is required. Pass --service-key or set KMA_APIHUB_AUTH_KEY."
                )
            raise ValueError(
                "A KMA service key is required. Pass --service-key or set DATA_GO_KR_SERVICE_KEY."
            )
        active_fetcher = fetcher or _fetch_bytes
        for request in request_plan:
            request_rows, page_count = _download_request_rows(
                endpoint=request["endpoint"],
                base_params=request["params"],
                service_key=service_key,
                data_type=config.data_type,
                page_size=config.num_rows,
                timeout=config.timeout,
                max_retries=config.max_retries,
                fetcher=active_fetcher,
                response_format=request.get("response_format"),
                paginated=bool(request.get("paginated", True)),
            )
            rows.extend(request_rows)
            request_status.append(
                {
                    "url": _redact_secret_url(_build_request_url(request["endpoint"], request["params"], service_key)),
                    "station_id": request["station_id"],
                    "page_count": page_count,
                    "row_count": len(request_rows),
                    "status": "downloaded",
                }
            )

        _write_csv(raw_csv_path, rows)

    standardized_row_count = 0
    if standardized_csv_path is not None and not config.dry_run:
        metadata = _load_station_metadata(config.metadata_csv_path) if config.metadata_csv_path is not None else {}
        if not metadata and config.source == "aws_recent_1min" and config.auto_station_metadata and service_key is not None:
            metadata = _fetch_aws_station_metadata(
                service_key=service_key,
                observation_time=config.aws_datetime,
                timeout=config.timeout,
                max_retries=config.max_retries,
                fetcher=active_fetcher,
                station_ids=config.station_ids,
            )
            if metadata:
                resolved_metadata_csv_path = config.output_dir / "aws_station_metadata_auto.csv"
                if resolved_metadata_csv_path.exists() and not config.overwrite:
                    raise FileExistsError(
                        f"{resolved_metadata_csv_path} already exists. Re-run with --overwrite to replace it."
                    )
                _write_csv(
                    resolved_metadata_csv_path,
                    list(metadata.values()),
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
        standardized_rows, standardization_warnings = _standardize_rows(
            rows,
            source=config.source,
            metadata=metadata,
        )
        warnings.extend(standardization_warnings)
        _write_csv(standardized_csv_path, standardized_rows, fieldnames=list(STANDARDIZED_FIELDNAMES))
        standardized_row_count = len(standardized_rows)
        if framework_config_path is not None:
            write_kma_framework_config(
                config_path=framework_config_path,
                data_path=standardized_csv_path,
                overwrite=config.overwrite,
            )

    manifest_path = config.output_dir / "kma_download_manifest.json"
    manifest_payload = {
        "source": config.source,
        "dry_run": config.dry_run,
        "auth_source": auth_source,
        "official_sources": _official_source_links(config.source),
        "raw_csv_path": str(raw_csv_path.resolve()) if raw_csv_path is not None else None,
        "standardized_csv_path": (
            str(standardized_csv_path.resolve()) if standardized_csv_path is not None else None
        ),
        "framework_config_path": (
            str(framework_config_path.resolve()) if framework_config_path is not None else None
        ),
        "metadata_csv_path": str(config.metadata_csv_path.resolve()) if config.metadata_csv_path is not None else None,
        "resolved_metadata_csv_path": (
            str(resolved_metadata_csv_path.resolve()) if resolved_metadata_csv_path is not None else None
        ),
        "metadata_template_path": (
            str(metadata_template_path.resolve()) if metadata_template_path is not None else None
        ),
        "station_ids": list(config.station_ids),
        "request_status": request_status,
        "raw_row_count": len(rows),
        "standardized_row_count": standardized_row_count,
        "warnings": warnings,
        "parameters": _manifest_parameters(config),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return KmaDownloadArtifacts(
        manifest_path=manifest_path,
        output_dir=config.output_dir,
        raw_csv_path=raw_csv_path if not config.dry_run else None,
        standardized_csv_path=standardized_csv_path if not config.dry_run else None,
        framework_config_path=framework_config_path,
        resolved_metadata_csv_path=resolved_metadata_csv_path,
        metadata_template_path=metadata_template_path,
        row_count=len(rows),
        dry_run=config.dry_run,
    )


def write_station_metadata_template(
    output_path: Path,
    *,
    station_ids: tuple[str, ...],
    source: str,
    overwrite: bool = False,
) -> Path:
    """Write a station metadata template aligned with framework ingestion.

    Args:
        output_path: Destination CSV path.
        station_ids: Station ids to seed the template with.
        source: Source name used for default sensor metadata.
        overwrite: Whether to overwrite an existing file.

    Returns:
        The written CSV path.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Re-run with --overwrite to replace it.")
    default_sensor_type = "asos" if source == "asos_hourly" else "aws"
    default_modality = "synoptic_surface" if source == "asos_hourly" else "automatic_weather_station"
    rows = []
    for station_id in station_ids:
        rows.append(
            {
                "station_id": station_id,
                "latitude": "",
                "longitude": "",
                "elevation": "",
                "cost": "",
                "sensor_type": default_sensor_type,
                "sensor_group": "surface",
                "sensor_modality": default_modality,
                "site_type": "",
                "maintenance_state": "",
                "maintenance_age": "",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_path,
        rows,
        fieldnames=[
            "station_id",
            "latitude",
            "longitude",
            "elevation",
            "cost",
            "sensor_type",
            "sensor_group",
            "sensor_modality",
            "site_type",
            "maintenance_state",
            "maintenance_age",
        ],
    )
    return output_path


def write_kma_framework_config(
    config_path: Path,
    data_path: Path,
    *,
    overwrite: bool,
) -> Path:
    """Write an aws_network framework config for standardized KMA data."""
    from .framework import write_framework_template

    write_framework_template(
        config_path=config_path,
        data_path=data_path,
        preset="aws_network",
        force=overwrite,
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    unique_times = _count_unique_csv_values(data_path, column_name="timestamp")
    if unique_times < 3:
        payload.setdefault("data", {})["split_strategy"] = "row"
        config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def _resolve_service_key(config: KmaDownloadConfig) -> tuple[str | None, str]:
    if config.service_key:
        return config.service_key.strip(), "cli"
    if config.source == "aws_recent_1min":
        for env_name in ("KMA_APIHUB_AUTH_KEY", "KMA_AUTH_KEY"):
            env_value = os.getenv(env_name)
            if env_value:
                return env_value.strip(), f"env:{env_name}"
        return None, "missing"
    for env_name in ("DATA_GO_KR_SERVICE_KEY", "DATA_GO_KR_API_KEY"):
        env_value = os.getenv(env_name)
        if env_value:
            return env_value.strip(), f"env:{env_name}"
    return None, "missing"


def _build_request_plan(config: KmaDownloadConfig) -> list[dict[str, Any]]:
    if config.source == "asos_hourly":
        return [
            {
                "endpoint": ASOS_HOURLY_ENDPOINT,
                "station_id": ",".join(config.station_ids),
                "params": {
                    "pageNo": "1",
                    "numOfRows": str(config.num_rows),
                    "dataType": config.data_type,
                    "dataCd": "ASOS",
                    "dateCd": "HR",
                    "startDt": config.start_date.strftime("%Y%m%d") if config.start_date is not None else "",
                    "startHh": config.start_hour,
                    "endDt": config.end_date.strftime("%Y%m%d") if config.end_date is not None else "",
                    "endHh": config.end_hour,
                    "stnIds": ",".join(config.station_ids),
                },
            }
        ]
    return [
        {
            "endpoint": AWS_RECENT_ENDPOINT,
            "station_id": station_id,
            "params": {
                "tm2": config.aws_datetime.strftime("%Y%m%d%H%M") if config.aws_datetime is not None else "",
                "stn": station_id,
                "disp": "1",
                "help": "0",
            },
            "response_format": "apihub_text_csv",
            "paginated": False,
        }
        for station_id in config.station_ids
    ]


def _manifest_parameters(config: KmaDownloadConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": config.source,
        "data_type": config.data_type,
        "num_rows": config.num_rows,
        "auto_station_metadata": config.auto_station_metadata,
    }
    if config.source == "asos_hourly":
        payload.update(
            {
                "start_date": config.start_date.isoformat() if config.start_date is not None else None,
                "end_date": config.end_date.isoformat() if config.end_date is not None else None,
                "start_hour": config.start_hour,
                "end_hour": config.end_hour,
            }
        )
    if config.source == "aws_recent_1min":
        payload["aws_datetime"] = (
            config.aws_datetime.isoformat(timespec="minutes") if config.aws_datetime is not None else None
        )
    return payload


def _official_source_links(source: str) -> list[str]:
    if source == "asos_hourly":
        return [ASOS_OFFICIAL_PAGE]
    return [AWS_API_HUB_PAGE, AWS_API_HUB_GUIDE]


def _build_request_url(endpoint: str, params: dict[str, Any], service_key: str | None) -> str:
    clean_params = {key: value for key, value in params.items() if value not in (None, "")}
    query = urllib.parse.urlencode(clean_params, doseq=True)
    if service_key is not None:
        key_name = "ServiceKey"
        if endpoint.startswith("https://apihub.kma.go.kr"):
            key_name = "authKey"
            encoded_key = urllib.parse.quote(service_key, safe="")
        else:
            encoded_key = service_key if "%" in service_key else urllib.parse.quote(service_key, safe="")
        query = f"{key_name}={encoded_key}&{query}" if query else f"{key_name}={encoded_key}"
    return f"{endpoint}?{query}" if query else endpoint


def _download_request_rows(
    *,
    endpoint: str,
    base_params: dict[str, Any],
    service_key: str,
    data_type: str,
    page_size: int,
    timeout: int,
    max_retries: int,
    fetcher: Callable[[str, int], bytes],
    response_format: str | None = None,
    paginated: bool = True,
) -> tuple[list[dict[str, str]], int]:
    if not paginated:
        url = _build_request_url(endpoint, base_params, service_key)
        response_bytes = _fetch_with_retries(url, timeout=timeout, max_retries=max_retries, fetcher=fetcher)
        page_rows, _ = _parse_kma_response(response_bytes, data_type=data_type, response_format=response_format)
        return page_rows, 1
    rows: list[dict[str, str]] = []
    total_count: int | None = None
    page = 1
    while True:
        params = dict(base_params)
        params["pageNo"] = str(page)
        params["numOfRows"] = str(page_size)
        url = _build_request_url(endpoint, params, service_key)
        response_bytes = _fetch_with_retries(url, timeout=timeout, max_retries=max_retries, fetcher=fetcher)
        page_rows, page_total = _parse_kma_response(
            response_bytes,
            data_type=data_type,
            response_format=response_format,
        )
        rows.extend(page_rows)
        if total_count is None:
            total_count = page_total
        if not page_rows:
            break
        if total_count is not None and page * page_size >= total_count:
            break
        if total_count is None and len(page_rows) < page_size:
            break
        page += 1
    return rows, page


def _fetch_with_retries(
    url: str,
    *,
    timeout: int,
    max_retries: int,
    fetcher: Callable[[str, int], bytes],
) -> bytes:
    attempts = 0
    while True:
        try:
            return fetcher(url, timeout)
        except Exception:
            if attempts >= max_retries:
                raise
            attempts += 1
            time.sleep(min(2.0, 0.5 * attempts))


def _fetch_bytes(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "silence-aware-ids/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _parse_kma_response(
    response_bytes: bytes,
    *,
    data_type: str,
    response_format: str | None = None,
) -> tuple[list[dict[str, str]], int | None]:
    if response_format == "apihub_text_csv":
        return _parse_apihub_text_csv_response(response_bytes)
    if data_type == "JSON":
        return _parse_json_response(response_bytes)
    return _parse_xml_response(response_bytes)


def _parse_apihub_text_csv_response(response_bytes: bytes) -> tuple[list[dict[str, str]], int | None]:
    text = response_bytes.decode("utf-8", errors="ignore").replace("\ufeff", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    csv_lines = [line for line in lines if "," in line]
    if not csv_lines:
        raise RuntimeError(f"KMA API Hub returned no CSV payload: {text[:200]}")
    header_index: int | None = None
    for index, line in enumerate(csv_lines):
        tokens = [token.strip() for token in next(csv.reader([line]))]
        if _looks_like_header(tokens):
            header_index = index
            break
    data_start = 0
    if header_index is None:
        header_tokens = _infer_apihub_header([token.strip() for token in next(csv.reader([csv_lines[0]]))])
    else:
        header_tokens = _normalize_header_tokens(next(csv.reader([csv_lines[header_index]])))
        data_start = header_index + 1
    rows: list[dict[str, str]] = []
    for line in csv_lines[data_start:]:
        values = [token.strip() for token in next(csv.reader([line]))]
        if len(values) != len(header_tokens):
            continue
        row = {header_tokens[index]: values[index] for index in range(len(header_tokens))}
        if any(value for value in row.values()):
            rows.append(row)
    return rows, len(rows)


def _looks_like_header(tokens: list[str]) -> bool:
    if not tokens:
        return False
    alpha_count = sum(1 for token in tokens if any(character.isalpha() for character in token))
    numeric_count = sum(1 for token in tokens if token.replace(".", "", 1).replace("-", "", 1).isdigit())
    return alpha_count >= max(1, numeric_count)


def _normalize_header_tokens(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    for token in tokens:
        cleaned = token.strip().lower()
        cleaned = cleaned.replace("/", "_").replace("-", "_").replace(" ", "_")
        normalized.append(cleaned)
    return normalized


def _infer_apihub_header(values: list[str]) -> list[str]:
    if len(values) == len(AWS_APIHUB_COMPACT_HEADER):
        return list(AWS_APIHUB_COMPACT_HEADER)
    if len(values) == len(AWS_APIHUB_FULL_HEADER):
        return list(AWS_APIHUB_FULL_HEADER)
    if len(values) == len(AWS_APIHUB_FULL_HEADER_WITH_STATUS):
        return list(AWS_APIHUB_FULL_HEADER_WITH_STATUS)
    raise RuntimeError(f"Unsupported AWS API Hub row width: {len(values)} columns.")


def _redact_secret_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    redacted = []
    for key, value in query:
        if key.lower() in {"authkey", "servicekey"}:
            redacted.append((key, "***"))
        else:
            redacted.append((key, value))
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(redacted, doseq=True), parts.fragment)
    )


def _parse_json_response(response_bytes: bytes) -> tuple[list[dict[str, str]], int | None]:
    payload = json.loads(response_bytes.decode("utf-8"))
    response = payload.get("response", payload)
    header = response.get("header", {})
    result_code = str(header.get("resultCode", "00"))
    if result_code not in {"00", "0"}:
        message = header.get("resultMsg", "KMA API returned an error.")
        raise RuntimeError(f"KMA API error {result_code}: {message}")
    body = response.get("body", {})
    total_count = body.get("totalCount")
    items = body.get("items", {})
    raw_items = items.get("item", []) if isinstance(items, dict) else items
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    rows = [_stringify_mapping(item) for item in raw_items]
    return rows, int(total_count) if total_count not in (None, "") else None


def _parse_xml_response(response_bytes: bytes) -> tuple[list[dict[str, str]], int | None]:
    root = ET.fromstring(response_bytes)
    result_code = root.findtext(".//header/resultCode", default="00")
    if result_code not in {"00", "0"}:
        message = root.findtext(".//header/resultMsg", default="KMA API returned an error.")
        raise RuntimeError(f"KMA API error {result_code}: {message}")
    total_count_text = root.findtext(".//body/totalCount", default="")
    total_count = int(total_count_text) if total_count_text else None
    rows: list[dict[str, str]] = []
    for item in root.findall(".//body/items/item"):
        rows.append({child.tag: (child.text or "") for child in list(item)})
    return rows, total_count


def _stringify_mapping(item: dict[str, Any]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in item.items()}


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    names.append(str(key))
                    seen.add(str(key))
        fieldnames = names
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _load_station_metadata(metadata_csv_path: Path) -> dict[str, dict[str, str]]:
    with metadata_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Station metadata CSV is missing a header row: {metadata_csv_path}")
        if "station_id" not in reader.fieldnames:
            raise ValueError(f"Station metadata CSV must contain station_id: {metadata_csv_path}")
        metadata: dict[str, dict[str, str]] = {}
        for row in reader:
            station_id = row.get("station_id", "").strip()
            if station_id:
                metadata[station_id] = {key: value.strip() for key, value in row.items() if key is not None}
    return metadata


def _fetch_aws_station_metadata(
    *,
    service_key: str,
    observation_time: datetime | None,
    timeout: int,
    max_retries: int,
    fetcher: Callable[[str, int], bytes],
    station_ids: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    params = {
        "inf": "AWS",
        "stn": "",
        "tm": (
            observation_time.strftime("%Y%m%d%H%M")
            if observation_time is not None
            else datetime.utcnow().strftime("%Y%m%d%H%M")
        ),
        "help": "0",
    }
    url = _build_request_url(AWS_STATION_INFO_ENDPOINT, params, service_key)
    response_bytes = _fetch_with_retries(url, timeout=timeout, max_retries=max_retries, fetcher=fetcher)
    requested = {station_id for station_id in station_ids if station_id != "0"}
    metadata: dict[str, dict[str, str]] = {}
    text = response_bytes.decode("utf-8", errors="ignore").replace("\ufeff", "")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 5:
            continue
        station_id = tokens[0]
        if requested and station_id not in requested:
            continue
        metadata[station_id] = {
            "station_id": station_id,
            "longitude": tokens[1],
            "latitude": tokens[2],
            "elevation": tokens[4],
            "sensor_type": "aws",
            "sensor_group": "surface",
            "sensor_modality": "automatic_weather_station",
        }
    return metadata


def _standardize_rows(
    rows: list[dict[str, str]],
    *,
    source: str,
    metadata: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    standardized_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    missing_coordinates = 0
    missing_metadata = 0
    for row in rows:
        station_id = _first_present(row, "station_id", "stnId", "awsId", "stn", "station")
        metadata_row = metadata.get(station_id, {})
        if not metadata_row:
            missing_metadata += 1
        normalized = {
            "timestamp": _normalize_timestamp(_first_present(row, "tm", "tm2", "awsDt", "time", "datetime")),
            "station_id": station_id,
            "latitude": _prefer_metadata(metadata_row, row, "latitude", ("lat", "latitude")),
            "longitude": _prefer_metadata(metadata_row, row, "longitude", ("lon", "longitude")),
            "temperature": _first_present(row, "ta", "temperature", "temp", "ta_1m"),
            "humidity": _first_present(row, "hm", "humidity", "hm_1m"),
            "pressure": _first_present(row, "ps", "pa", "pressure", "ps_1m"),
            "wind_speed": _first_present(
                row,
                "ws",
                "ws10",
                "windSpeed",
                "wind_speed",
                "ws10m",
                "ws_10m",
            ),
            "wind_direction": _first_present(
                row,
                "wd",
                "wd10",
                "windDir",
                "wind_direction",
                "wd10m",
                "wd_10m",
            ),
            "precipitation": _first_present(
                row,
                "rn",
                "rn_15m",
                "rn_60m",
                "rn_day",
                "rnDay",
                "precipitation",
                "rn_1m",
            ),
            "elevation": _prefer_metadata(metadata_row, row, "elevation", ("alt", "elevation", "height")),
            "cost": metadata_row.get("cost", ""),
            "sensor_type": metadata_row.get("sensor_type", "asos" if source == "asos_hourly" else "aws"),
            "sensor_group": metadata_row.get("sensor_group", "surface"),
            "sensor_modality": metadata_row.get(
                "sensor_modality",
                "synoptic_surface" if source == "asos_hourly" else "automatic_weather_station",
            ),
            "site_type": metadata_row.get("site_type", ""),
            "maintenance_state": metadata_row.get("maintenance_state", ""),
            "maintenance_age": metadata_row.get("maintenance_age", ""),
            "source": source,
            "raw_timestamp": _first_present(row, "tm", "tm2", "awsDt", "time", "datetime"),
        }
        if not normalized["latitude"] or not normalized["longitude"]:
            missing_coordinates += 1
        standardized_rows.append(normalized)
    if missing_metadata:
        warnings.append(
            f"{missing_metadata} row(s) had no matching station metadata; sensor fields used source defaults."
        )
    if missing_coordinates:
        warnings.append(
            f"{missing_coordinates} row(s) are missing latitude/longitude. Provide --metadata-csv for framework-ready runs."
        )
    return standardized_rows, warnings


def _prefer_metadata(
    metadata_row: dict[str, str],
    source_row: dict[str, str],
    metadata_key: str,
    source_keys: tuple[str, ...],
) -> str:
    if metadata_key in metadata_row and metadata_row[metadata_key]:
        return metadata_row[metadata_key]
    return _first_present(source_row, *source_keys)


def _first_present(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_timestamp(value: str) -> str:
    if not value:
        return ""
    patterns = (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d%H%M",
        "%Y%m%d%H",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
    )
    for pattern in patterns:
        try:
            parsed = datetime.strptime(value, pattern)
            return parsed.isoformat(timespec="minutes")
        except ValueError:
            continue
    return value


def _count_unique_csv_values(path: Path, *, column_name: str) -> int:
    values: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().strip().split(",")
        if column_name not in header:
            return 0
        index = header.index(column_name)
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if index < len(parts):
                values.add(parts[index])
    return len(values)
