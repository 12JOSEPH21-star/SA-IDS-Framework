from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any


EARTH_GRAVITY = 9.80665
DEFAULT_ERA5_DATASET = "reanalysis-era5-single-levels"
DEFAULT_CDS_API_URL = "https://cds.climate.copernicus.eu/api"
DEFAULT_ERA5_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
)
DEFAULT_ERA5_STATIC_VARIABLES = (
    "geopotential",
    "land_sea_mask",
)
DEFAULT_ERA5_TIMES = tuple(f"{hour:02d}:00" for hour in range(24))


@dataclass(frozen=True)
class Era5DownloadConfig:
    start_date: date
    end_date: date
    output_dir: Path
    csv_path: Path | None = None
    framework_config_path: Path | None = None
    cds_url: str | None = None
    cds_key: str | None = None
    dataset: str = DEFAULT_ERA5_DATASET
    variables: tuple[str, ...] = DEFAULT_ERA5_VARIABLES
    static_variables: tuple[str, ...] = DEFAULT_ERA5_STATIC_VARIABLES
    times: tuple[str, ...] = DEFAULT_ERA5_TIMES
    area: tuple[float, float, float, float] | None = None
    grid: tuple[float, float] | None = None
    data_format: str = "netcdf"
    overwrite: bool = False
    dry_run: bool = False
    timeout: int = 600
    max_retries: int = 2

    def validate(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date.")
        if not self.variables:
            raise ValueError("At least one ERA5 variable is required.")
        if not self.times:
            raise ValueError("At least one hour is required.")
        if self.area is not None and len(self.area) != 4:
            raise ValueError("area must be (north, west, south, east).")
        if self.grid is not None and len(self.grid) != 2:
            raise ValueError("grid must be (lat_step, lon_step).")
        if self.data_format not in {"netcdf", "grib"}:
            raise ValueError("data_format must be one of: netcdf, grib.")
        if self.csv_path is not None and self.data_format != "netcdf":
            raise ValueError("CSV conversion requires data_format=netcdf.")
        if self.framework_config_path is not None and self.csv_path is None:
            raise ValueError("Framework config generation requires CSV conversion. Remove --no-csv or omit --framework-config.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be zero or greater.")


@dataclass(frozen=True)
class Era5DownloadTask:
    kind: str
    target: Path
    request: dict[str, Any]
    variables: tuple[str, ...]
    year: int | None = None
    month: int | None = None


@dataclass(frozen=True)
class Era5DownloadArtifacts:
    manifest_path: Path
    output_dir: Path
    monthly_files: tuple[Path, ...]
    static_file: Path | None
    csv_path: Path | None
    framework_config_path: Path | None
    dry_run: bool


def build_download_plan(config: Era5DownloadConfig) -> tuple[Era5DownloadTask, ...]:
    config.validate()
    tasks: list[Era5DownloadTask] = []
    suffix = ".nc" if config.data_format == "netcdf" else ".grib"

    if config.static_variables:
        static_request = _base_request(
            variables=config.static_variables,
            year=config.start_date.year,
            month=config.start_date.month,
            days=(f"{config.start_date.day:02d}",),
            times=(config.times[0],),
            data_format=config.data_format,
            area=config.area,
            grid=config.grid,
        )
        tasks.append(
            Era5DownloadTask(
                kind="static",
                target=config.output_dir / f"era5_static{suffix}",
                request=static_request,
                variables=tuple(config.static_variables),
            )
        )

    month_cursor = date(config.start_date.year, config.start_date.month, 1)
    while month_cursor <= config.end_date:
        window_end = min(_month_end(month_cursor), config.end_date)
        day_start = config.start_date.day if month_cursor.year == config.start_date.year and month_cursor.month == config.start_date.month else 1
        day_stop = window_end.day
        request = _base_request(
            variables=config.variables,
            year=month_cursor.year,
            month=month_cursor.month,
            days=tuple(f"{day:02d}" for day in range(day_start, day_stop + 1)),
            times=config.times,
            data_format=config.data_format,
            area=config.area,
            grid=config.grid,
        )
        tasks.append(
            Era5DownloadTask(
                kind="dynamic",
                target=config.output_dir / f"era5_{month_cursor.year}_{month_cursor.month:02d}{suffix}",
                request=request,
                variables=tuple(config.variables),
                year=month_cursor.year,
                month=month_cursor.month,
            )
        )
        month_cursor = _next_month(month_cursor)
    return tuple(tasks)


def download_era5(config: Era5DownloadConfig) -> Era5DownloadArtifacts:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_download_plan(config)
    monthly_files = tuple(task.target for task in plan if task.kind == "dynamic")
    static_file = next((task.target for task in plan if task.kind == "static"), None)

    task_status: list[dict[str, Any]] = []
    if config.dry_run:
        task_status = [_task_status_payload(task, status="planned") for task in plan]
    else:
        client = _build_cds_client(
            timeout=config.timeout,
            cds_url=config.cds_url,
            cds_key=config.cds_key,
        )
        for task in plan:
            if task.target.exists() and not config.overwrite:
                task_status.append(_task_status_payload(task, status="existing"))
                continue
            task.target.parent.mkdir(parents=True, exist_ok=True)
            _retrieve_with_retries(
                client=client,
                dataset=config.dataset,
                task=task,
                max_retries=config.max_retries,
            )
            task_status.append(_task_status_payload(task, status="downloaded"))

    csv_path = config.csv_path
    if csv_path is not None and not config.dry_run:
        convert_era5_to_reference_csv(
            dynamic_paths=monthly_files,
            csv_path=csv_path,
            static_path=static_file,
        )

    framework_config_path = config.framework_config_path
    if framework_config_path is not None:
        write_era5_framework_config(
            config_path=framework_config_path,
            data_path=csv_path or config.output_dir / "era5_reference.csv",
            overwrite=config.overwrite,
            expected_unique_timestamps=_expected_unique_timestamps(config),
        )

    manifest_path = config.output_dir / "era5_download_manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest_payload(config, task_status, csv_path=csv_path, static_file=static_file),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return Era5DownloadArtifacts(
        manifest_path=manifest_path,
        output_dir=config.output_dir,
        monthly_files=monthly_files,
        static_file=static_file,
        csv_path=csv_path,
        framework_config_path=framework_config_path,
        dry_run=config.dry_run,
    )


def convert_era5_to_reference_csv(
    dynamic_paths: tuple[Path, ...],
    csv_path: Path,
    *,
    static_path: Path | None = None,
) -> Path:
    if not dynamic_paths:
        raise ValueError("At least one dynamic ERA5 NetCDF file is required for CSV conversion.")
    pd, xr = _import_conversion_stack()

    dynamic_frames = []
    for path in dynamic_paths:
        frame = _load_dynamic_frame(path, xr=xr, pd=pd)
        frame["source_file"] = path.name
        dynamic_frames.append(frame)

    merged = pd.concat(dynamic_frames, ignore_index=True)

    if static_path is not None and static_path.exists():
        static_frame = _load_static_frame(static_path, xr=xr, pd=pd)
        if not static_frame.empty:
            merged = merged.merge(static_frame, on=["latitude", "longitude"], how="left")

    for column in ("orography", "land_sea_mask"):
        if column not in merged.columns:
            merged[column] = float("nan")

    merged["grid_id"] = merged.apply(
        lambda row: f"era5_{row['latitude']:.3f}_{row['longitude']:.3f}",
        axis=1,
    )
    merged["time"] = merged["time"].astype(str)
    ordered = [
        "time",
        "grid_id",
        "latitude",
        "longitude",
        "t2m",
        "u10",
        "v10",
        "sp",
        "tp",
        "orography",
        "land_sea_mask",
        "source_file",
    ]
    for column in ordered:
        if column not in merged.columns:
            merged[column] = float("nan")
    merged = merged[ordered].sort_values(["time", "latitude", "longitude"]).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)
    return csv_path


def write_era5_framework_config(
    config_path: Path,
    data_path: Path,
    *,
    overwrite: bool,
    expected_unique_timestamps: int | None = None,
) -> Path:
    from .framework import write_framework_template

    write_framework_template(
        config_path=config_path,
        data_path=data_path,
        preset="era5_reference",
        force=overwrite,
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    unique_times = expected_unique_timestamps
    if data_path.exists():
        unique_times = _count_unique_csv_values(data_path, column_name="time")
    if unique_times is not None and unique_times < 3:
        payload.setdefault("data", {})["split_strategy"] = "row"
        config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def _base_request(
    *,
    variables: tuple[str, ...],
    year: int,
    month: int,
    days: tuple[str, ...],
    times: tuple[str, ...],
    data_format: str,
    area: tuple[float, float, float, float] | None,
    grid: tuple[float, float] | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "product_type": ["reanalysis"],
        "variable": list(variables),
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": list(days),
        "time": list(times),
        "data_format": data_format,
    }
    if area is not None:
        request["area"] = [float(value) for value in area]
    if grid is not None:
        request["grid"] = [float(value) for value in grid]
    return request


def _month_end(value: date) -> date:
    return _next_month(value) - timedelta(days=1)


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _build_cds_client(*, timeout: int, cds_url: str | None, cds_key: str | None) -> Any:
    try:
        import cdsapi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ERA5 download requires cdsapi. Install cdsapi and configure your CDS API credentials first."
        ) from exc
    resolved_url = cds_url or os.environ.get("CDSAPI_URL") or DEFAULT_CDS_API_URL
    resolved_key = cds_key or os.environ.get("CDSAPI_KEY")
    if resolved_url and resolved_key:
        return cdsapi.Client(url=resolved_url, key=resolved_key, timeout=timeout)
    if _default_cdsapirc_path().exists():
        return cdsapi.Client(timeout=timeout)
    raise RuntimeError(
        f"CDS credentials were not found. Set {_default_cdsapirc_path()} or pass --cds-url and --cds-key "
        "(or CDSAPI_URL/CDSAPI_KEY) before running a real ERA5 download."
    )


def _retrieve_with_retries(*, client: Any, dataset: str, task: Era5DownloadTask, max_retries: int) -> None:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            client.retrieve(dataset, task.request, str(task.target))
            return
        except Exception as exc:  # pragma: no cover - network path is exercised manually.
            last_error = exc
            if attempt == max_retries:
                break
    raise RuntimeError(f"Failed to download {task.target.name}: {last_error}") from last_error


def _manifest_payload(
    config: Era5DownloadConfig,
    task_status: list[dict[str, Any]],
    *,
    csv_path: Path | None,
    static_file: Path | None,
) -> dict[str, Any]:
    return {
        "dataset": config.dataset,
        "start_date": config.start_date.isoformat(),
        "end_date": config.end_date.isoformat(),
        "dry_run": config.dry_run,
        "output_dir": str(config.output_dir.resolve()),
        "data_format": config.data_format,
        "variables": list(config.variables),
        "static_variables": list(config.static_variables),
        "times": list(config.times),
        "area": list(config.area) if config.area is not None else None,
        "grid": list(config.grid) if config.grid is not None else None,
        "csv_path": str(csv_path.resolve()) if csv_path is not None else None,
        "framework_config_path": (
            str(config.framework_config_path.resolve()) if config.framework_config_path is not None else None
        ),
        "auth_source": _auth_source(config.cds_url, config.cds_key),
        "static_file": str(static_file.resolve()) if static_file is not None else None,
        "tasks": task_status,
    }


def _task_status_payload(task: Era5DownloadTask, *, status: str) -> dict[str, Any]:
    return {
        "kind": task.kind,
        "status": status,
        "target": str(task.target.resolve()),
        "year": task.year,
        "month": task.month,
        "variables": list(task.variables),
        "request": task.request,
    }


def _import_conversion_stack() -> tuple[Any, Any]:
    try:
        import pandas as pd
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ERA5 CSV conversion requires pandas, xarray, and a NetCDF backend such as netCDF4."
        ) from exc
    return pd, xr


def _load_dynamic_frame(path: Path, *, xr: Any, pd: Any) -> Any:
    frames = [_dynamic_reference_frame(dataset) for dataset in _open_datasets(path, xr=xr)]
    if not frames:
        raise ValueError(f"No readable NetCDF members were found in {path}.")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["time", "latitude", "longitude"], how="outer")
    return merged


def _load_static_frame(path: Path, *, xr: Any, pd: Any) -> Any:
    frames = [_static_reference_frame(dataset) for dataset in _open_datasets(path, xr=xr)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "orography", "land_sea_mask"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["latitude", "longitude"], how="outer")
    return merged


def _open_datasets(path: Path, *, xr: Any) -> list[Any]:
    if zipfile.is_zipfile(path):
        datasets: list[Any] = []
        temp_root = Path.cwd() / ".tmp_era5_unzip"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="era5_unzip_", dir=temp_root) as temp_dir:
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if not member.lower().endswith(".nc"):
                        continue
                    extracted = Path(archive.extract(member, path=temp_dir))
                    with xr.open_dataset(extracted, engine="netcdf4") as dataset:
                        datasets.append(dataset.load())
        return datasets
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        return [dataset.load()]


def _normalize_dataset(dataset: Any) -> Any:
    rename_map: dict[str, str] = {}
    if "valid_time" in dataset.coords:
        rename_map["valid_time"] = "time"
    if "lat" in dataset.coords:
        rename_map["lat"] = "latitude"
    if "lon" in dataset.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        dataset = dataset.rename(rename_map)
    return dataset


def _dynamic_reference_frame(dataset: Any) -> Any:
    dataset = _normalize_dataset(dataset)
    variable_map = {
        "t2m": _first_available(dataset, "t2m", "2m_temperature"),
        "u10": _first_available(dataset, "u10", "10m_u_component_of_wind"),
        "v10": _first_available(dataset, "v10", "10m_v_component_of_wind"),
        "sp": _first_available(dataset, "sp", "surface_pressure"),
        "tp": _first_available(dataset, "tp", "total_precipitation"),
    }
    available = {target: source for target, source in variable_map.items() if source is not None}
    if not available:
        raise ValueError("No recognized ERA5 reference variables were found in the NetCDF file.")

    frame = dataset[list(available.values())].to_dataframe().reset_index()
    coord_map = {}
    if "latitude" not in frame.columns and "lat" in frame.columns:
        coord_map["lat"] = "latitude"
    if "longitude" not in frame.columns and "lon" in frame.columns:
        coord_map["lon"] = "longitude"
    if "time" not in frame.columns and "valid_time" in frame.columns:
        coord_map["valid_time"] = "time"
    if coord_map:
        frame = frame.rename(columns=coord_map)
    frame = frame.rename(columns={source: target for target, source in available.items()})
    keep = ["time", "latitude", "longitude", *available.keys()]
    return frame[keep]


def _static_reference_frame(dataset: Any) -> Any:
    dataset = _normalize_dataset(dataset)
    variable_map = {
        "geopotential": _first_available(dataset, "geopotential", "z"),
        "orography": _first_available(dataset, "orography"),
        "land_sea_mask": _first_available(dataset, "land_sea_mask", "lsm"),
    }
    available = {target: source for target, source in variable_map.items() if source is not None}
    if not available:
        return dataset.to_dataframe().reset_index().iloc[0:0]

    frame = dataset[list(available.values())].to_dataframe().reset_index()
    coord_map = {}
    if "latitude" not in frame.columns and "lat" in frame.columns:
        coord_map["lat"] = "latitude"
    if "longitude" not in frame.columns and "lon" in frame.columns:
        coord_map["lon"] = "longitude"
    if coord_map:
        frame = frame.rename(columns=coord_map)

    geopotential_source = available.get("geopotential")
    if geopotential_source is not None:
        frame["orography"] = frame[geopotential_source] / EARTH_GRAVITY
    orography_source = available.get("orography")
    if orography_source is not None:
        frame["orography"] = frame[orography_source]
    land_mask_source = available.get("land_sea_mask")
    if land_mask_source is not None:
        frame["land_sea_mask"] = frame[land_mask_source]

    keep = ["latitude", "longitude", "orography", "land_sea_mask"]
    for column in keep:
        if column not in frame.columns:
            frame[column] = float("nan")
    return frame[keep].drop_duplicates(subset=["latitude", "longitude"])


def _first_available(dataset: Any, *names: str) -> str | None:
    for name in names:
        if name in dataset.data_vars:
            return name
    return None


def _default_cdsapirc_path() -> Path:
    return Path.home() / ".cdsapirc"


def _expected_unique_timestamps(config: Era5DownloadConfig) -> int:
    return ((config.end_date - config.start_date).days + 1) * len(config.times)


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


def _auth_source(cds_url: str | None, cds_key: str | None) -> str:
    if cds_url or cds_key:
        return "cli"
    if os.environ.get("CDSAPI_URL") or os.environ.get("CDSAPI_KEY"):
        return "environment"
    if _default_cdsapirc_path().exists():
        return "cdsapirc"
    return "missing"
