from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .research import load_config, run_study, write_template_project


def _iso_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sensor-sparse weather observation experiments from a CSV dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a ready-to-run config and demo dataset.")
    init_parser.add_argument(
        "--config",
        type=Path,
        default=Path("research_config.json"),
        help="Where to write the JSON config.",
    )
    init_parser.add_argument(
        "--data",
        type=Path,
        default=Path("sample_weather.csv"),
        help="Where to write the demo CSV dataset.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target files if they already exist.",
    )

    run_parser = subparsers.add_parser("run", help="Run the full MSM experiment pipeline.")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("research_config.json"),
        help="Path to the JSON config file.",
    )

    framework_init_parser = subparsers.add_parser(
        "framework-init",
        help="Create a research-framework config and demo dataset.",
    )
    framework_init_parser.add_argument(
        "--config",
        type=Path,
        default=Path("framework_config.json"),
        help="Where to write the framework JSON config.",
    )
    framework_init_parser.add_argument(
        "--data",
        type=Path,
        default=Path("sample_weather.csv"),
        help="Where to write the demo CSV dataset.",
    )
    framework_init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target files if they already exist.",
    )
    framework_init_parser.add_argument(
        "--preset",
        default="demo",
        choices=("demo", "isd_hourly", "aws_network", "era5_reference"),
        help="Configuration preset to write.",
    )

    framework_run_parser = subparsers.add_parser(
        "framework-run",
        help="Run the silence-aware research framework from a config.",
    )
    framework_run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("framework_config.json"),
        help="Path to the framework JSON config file.",
    )

    era5_parser = subparsers.add_parser(
        "era5-download",
        help="Download ERA5 reference data in monthly chunks and optionally build a framework-ready CSV.",
    )
    era5_parser.add_argument(
        "--start-date",
        type=_iso_date,
        required=True,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    era5_parser.add_argument(
        "--end-date",
        type=_iso_date,
        required=True,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    era5_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/era5"),
        help="Directory for monthly ERA5 files and the manifest.",
    )
    era5_parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Framework-ready CSV output path. Defaults to <output-dir>/era5_reference.csv.",
    )
    era5_parser.add_argument(
        "--framework-config",
        type=Path,
        default=None,
        help="Optional era5_reference framework config output path.",
    )
    era5_parser.add_argument(
        "--cds-url",
        default=None,
        help="Optional CDS API URL. If omitted, uses CDSAPI_URL or ~/.cdsapirc.",
    )
    era5_parser.add_argument(
        "--cds-key",
        default=None,
        help="Optional CDS API key or personal access token. If omitted, uses CDSAPI_KEY or ~/.cdsapirc.",
    )
    era5_parser.add_argument(
        "--dataset",
        default="reanalysis-era5-single-levels",
        help="CDS dataset identifier.",
    )
    era5_parser.add_argument(
        "--variables",
        nargs="+",
        default=[
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "total_precipitation",
        ],
        help="ERA5 dynamic variables to request.",
    )
    era5_parser.add_argument(
        "--static-variables",
        nargs="*",
        default=["geopotential", "land_sea_mask"],
        help="Static variables requested once at the beginning of the range.",
    )
    era5_parser.add_argument(
        "--hours",
        nargs="+",
        default=[f"{hour:02d}:00" for hour in range(24)],
        help="Hourly timesteps to request, for example 00:00 06:00 12:00 18:00.",
    )
    era5_parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=None,
        help="Optional bounding box in CDS order: north west south east.",
    )
    era5_parser.add_argument(
        "--grid",
        nargs=2,
        type=float,
        metavar=("LAT_STEP", "LON_STEP"),
        default=None,
        help="Optional output grid spacing in degrees.",
    )
    era5_parser.add_argument(
        "--data-format",
        choices=("netcdf", "grib"),
        default="netcdf",
        help="Download format used by CDS.",
    )
    era5_parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="CDS API timeout in seconds.",
    )
    era5_parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per monthly request.",
    )
    era5_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing monthly files, CSV, and framework config.",
    )
    era5_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the request manifest and optional framework config without downloading data.",
    )
    era5_parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip NetCDF to CSV conversion even when using data_format=netcdf.",
    )

    kma_parser = subparsers.add_parser(
        "kma-download",
        help="Download Korean ASOS hourly or AWS recent observations and optionally standardize them.",
    )
    kma_parser.add_argument(
        "--source",
        required=True,
        choices=("asos_hourly", "aws_recent_1min"),
        help="Official KMA source to download.",
    )
    kma_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/kma"),
        help="Directory for manifests and CSV outputs.",
    )
    kma_parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Optional raw API CSV output path. Defaults to <output-dir>/<source>_raw.csv.",
    )
    kma_parser.add_argument(
        "--standardized-csv",
        type=Path,
        default=None,
        help="Optional framework-oriented CSV output path.",
    )
    kma_parser.add_argument(
        "--framework-config",
        type=Path,
        default=None,
        help="Optional aws_network framework config output path.",
    )
    kma_parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional station metadata CSV used to populate coordinates and sensor metadata.",
    )
    kma_parser.add_argument(
        "--no-auto-station-metadata",
        action="store_true",
        help="Disable automatic AWS station metadata fetch from the KMA API Hub when --metadata-csv is omitted.",
    )
    kma_parser.add_argument(
        "--metadata-template",
        type=Path,
        default=None,
        help="Optional station metadata template output path seeded with the requested station ids.",
    )
    kma_parser.add_argument(
        "--service-key",
        default=None,
        help="Optional source credential. Uses DATA_GO_KR_SERVICE_KEY for ASOS or KMA_APIHUB_AUTH_KEY for AWS when omitted.",
    )
    kma_parser.add_argument(
        "--station-ids",
        nargs="+",
        required=True,
        help="Station ids to request from the API.",
    )
    kma_parser.add_argument(
        "--start-date",
        type=_iso_date,
        default=None,
        help="Inclusive start date for ASOS hourly data in YYYY-MM-DD format.",
    )
    kma_parser.add_argument(
        "--end-date",
        type=_iso_date,
        default=None,
        help="Inclusive end date for ASOS hourly data in YYYY-MM-DD format.",
    )
    kma_parser.add_argument(
        "--start-hour",
        default="00",
        help="Inclusive ASOS start hour in HH format.",
    )
    kma_parser.add_argument(
        "--end-hour",
        default="23",
        help="Inclusive ASOS end hour in HH format.",
    )
    kma_parser.add_argument(
        "--aws-datetime",
        default=None,
        help="AWS recent observation time in YYYY-MM-DDTHH:MM or YYYYMMDDHHMM format.",
    )
    kma_parser.add_argument(
        "--data-type",
        choices=("JSON", "XML"),
        default="JSON",
        help="Response format requested from the API.",
    )
    kma_parser.add_argument(
        "--num-rows",
        type=int,
        default=999,
        help="Page size for paginated requests.",
    )
    kma_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    kma_parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry budget per request.",
    )
    kma_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    kma_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the manifest and optional metadata template.",
    )

    noaa_parser = subparsers.add_parser(
        "noaa-download",
        help="Download NOAA ISD-Lite station data and optionally standardize it for the framework.",
    )
    noaa_parser.add_argument(
        "--start-date",
        type=_iso_date,
        required=True,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    noaa_parser.add_argument(
        "--end-date",
        type=_iso_date,
        required=True,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    noaa_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/noaa_isd"),
        help="Directory for NOAA manifests and CSV outputs.",
    )
    noaa_parser.add_argument(
        "--country-code",
        default="KS",
        help="NOAA country code used when station ids are not provided. Default: KS.",
    )
    noaa_parser.add_argument(
        "--station-ids",
        nargs="*",
        default=(),
        help="Optional explicit station ids formatted as USAF-WBAN.",
    )
    noaa_parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Optional cap on the number of selected stations after filtering.",
    )
    noaa_parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Optional merged raw ISD-Lite CSV output path.",
    )
    noaa_parser.add_argument(
        "--standardized-csv",
        type=Path,
        default=None,
        help="Optional framework-oriented CSV output path.",
    )
    noaa_parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional station metadata CSV output path.",
    )
    noaa_parser.add_argument(
        "--framework-config",
        type=Path,
        default=None,
        help="Optional isd_hourly framework config output path.",
    )
    noaa_parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    noaa_parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry budget per request.",
    )
    noaa_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    noaa_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only metadata, manifest, and optional config without downloading station files.",
    )

    joint_parser = subparsers.add_parser(
        "joint-build",
        help="Build a joint AWS/NOAA dataset and optionally enrich it with ERA5 reference context.",
    )
    joint_parser.add_argument(
        "--aws-csv",
        type=Path,
        required=True,
        help="Framework-ready AWS CSV path.",
    )
    joint_parser.add_argument(
        "--noaa-csv",
        type=Path,
        required=True,
        help="Framework-ready NOAA ISD CSV path.",
    )
    joint_parser.add_argument(
        "--era5-csv",
        type=Path,
        default=None,
        help="Optional ERA5 reference CSV path used for enrichment.",
    )
    joint_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/joint_weather_network"),
        help="Directory for joint outputs.",
    )
    joint_parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional joint CSV output path.",
    )
    joint_parser.add_argument(
        "--framework-config",
        type=Path,
        default=None,
        help="Optional joint framework config output path.",
    )
    joint_parser.add_argument(
        "--era5-grid-step",
        type=float,
        default=0.25,
        help="ERA5 grid step used for nearest-grid enrichment.",
    )
    joint_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            config_path, data_path = write_template_project(args.config, args.data, force=args.force)
            print(f"Created config: {config_path}")
            print(f"Created demo dataset: {data_path}")
            print(f"Next: python -m task_cli run --config {config_path}")
            return 0

        if args.command == "run":
            config = load_config(args.config)
            artifacts = run_study(config)
            print(f"Study finished: {artifacts.output_dir}")
            print(f"Metrics: {artifacts.metrics_path}")
            print(f"Summary: {artifacts.summary_path}")
            print(f"Plot: {artifacts.tradeoff_plot_path}")
            return 0
        if args.command == "framework-init":
            from .framework import write_framework_template

            config_path, data_path = write_framework_template(
                args.config,
                args.data,
                preset=args.preset,
                force=args.force,
            )
            print(f"Created framework config: {config_path}")
            if args.preset == "demo":
                print(f"Created demo dataset: {data_path}")
            else:
                print(f"Configured dataset path: {data_path}")
            print(f"Next: python -m task_cli framework-run --config {config_path}")
            return 0

        if args.command == "framework-run":
            from .framework import run_framework

            artifacts = run_framework(args.config)
            print(f"Framework experiment finished: {artifacts.summary_path.parent}")
            print(f"Summary: {artifacts.summary_path}")
            print(f"Ablations: {artifacts.ablations_path}")
            print(f"Sensitivity: {artifacts.sensitivity_path}")
            print(f"Selection: {artifacts.selection_path}")
            print(f"Report: {artifacts.report_path}")
            return 0
        if args.command == "era5-download":
            from .era5 import Era5DownloadConfig, download_era5

            csv_path = None if args.no_csv else (args.csv_path or args.output_dir / "era5_reference.csv")
            artifacts = download_era5(
                Era5DownloadConfig(
                    start_date=args.start_date,
                    end_date=args.end_date,
                    output_dir=args.output_dir,
                    csv_path=csv_path,
                    framework_config_path=args.framework_config,
                    cds_url=args.cds_url,
                    cds_key=args.cds_key,
                    dataset=str(args.dataset),
                    variables=tuple(args.variables),
                    static_variables=tuple(args.static_variables),
                    times=tuple(args.hours),
                    area=tuple(args.area) if args.area is not None else None,
                    grid=tuple(args.grid) if args.grid is not None else None,
                    data_format=str(args.data_format),
                    overwrite=bool(args.overwrite),
                    dry_run=bool(args.dry_run),
                    timeout=int(args.timeout),
                    max_retries=int(args.retries),
                )
            )
            if artifacts.dry_run:
                print(f"Planned {len(artifacts.monthly_files)} monthly ERA5 requests in {artifacts.output_dir}")
            else:
                print(f"Downloaded {len(artifacts.monthly_files)} monthly ERA5 files to {artifacts.output_dir}")
            print(f"Manifest: {artifacts.manifest_path}")
            if artifacts.static_file is not None:
                print(f"Static file: {artifacts.static_file}")
            if artifacts.csv_path is not None:
                label = "Planned CSV" if artifacts.dry_run else "Reference CSV"
                print(f"{label}: {artifacts.csv_path}")
            if artifacts.framework_config_path is not None:
                print(f"Framework config: {artifacts.framework_config_path}")
            return 0
        if args.command == "kma-download":
            from datetime import datetime

            from .kma import KmaDownloadConfig, download_kma

            aws_datetime = None
            if args.aws_datetime is not None:
                try:
                    aws_datetime = datetime.fromisoformat(args.aws_datetime)
                except ValueError:
                    aws_datetime = datetime.strptime(args.aws_datetime, "%Y%m%d%H%M")
            artifacts = download_kma(
                KmaDownloadConfig(
                    source=str(args.source),
                    output_dir=args.output_dir,
                    raw_csv_path=args.raw_csv,
                    standardized_csv_path=args.standardized_csv,
                    framework_config_path=args.framework_config,
                    metadata_csv_path=args.metadata_csv,
                    auto_station_metadata=not bool(args.no_auto_station_metadata),
                    metadata_template_path=args.metadata_template,
                    service_key=args.service_key,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    start_hour=str(args.start_hour),
                    end_hour=str(args.end_hour),
                    station_ids=tuple(str(value) for value in args.station_ids),
                    aws_datetime=aws_datetime,
                    data_type=str(args.data_type),
                    num_rows=int(args.num_rows),
                    overwrite=bool(args.overwrite),
                    dry_run=bool(args.dry_run),
                    timeout=int(args.timeout),
                    max_retries=int(args.retries),
                )
            )
            if artifacts.dry_run:
                print(f"Planned KMA {args.source} download in {artifacts.output_dir}")
            else:
                print(f"Downloaded {artifacts.row_count} KMA row(s) into {artifacts.output_dir}")
            print(f"Manifest: {artifacts.manifest_path}")
            if artifacts.raw_csv_path is not None:
                print(f"Raw CSV: {artifacts.raw_csv_path}")
            if artifacts.standardized_csv_path is not None:
                print(f"Standardized CSV: {artifacts.standardized_csv_path}")
            if artifacts.framework_config_path is not None:
                print(f"Framework config: {artifacts.framework_config_path}")
            if artifacts.resolved_metadata_csv_path is not None:
                print(f"Resolved metadata CSV: {artifacts.resolved_metadata_csv_path}")
            if artifacts.metadata_template_path is not None:
                print(f"Metadata template: {artifacts.metadata_template_path}")
            return 0
        if args.command == "noaa-download":
            from .noaa import NoaaDownloadConfig, download_noaa_isd

            artifacts = download_noaa_isd(
                NoaaDownloadConfig(
                    start_date=args.start_date,
                    end_date=args.end_date,
                    output_dir=args.output_dir,
                    country_code=str(args.country_code),
                    station_ids=tuple(str(value) for value in args.station_ids),
                    max_stations=args.max_stations,
                    raw_csv_path=args.raw_csv,
                    standardized_csv_path=args.standardized_csv,
                    metadata_csv_path=args.metadata_csv,
                    framework_config_path=args.framework_config,
                    overwrite=bool(args.overwrite),
                    dry_run=bool(args.dry_run),
                    timeout=int(args.timeout),
                    max_retries=int(args.retries),
                )
            )
            if artifacts.dry_run:
                print(f"Planned NOAA ISD download for {artifacts.station_count} station(s) in {artifacts.output_dir}")
            else:
                print(
                    f"Downloaded NOAA ISD rows={artifacts.row_count} "
                    f"for {artifacts.station_count} station(s) into {artifacts.output_dir}"
                )
            print(f"Manifest: {artifacts.manifest_path}")
            print(f"Metadata CSV: {artifacts.metadata_csv_path}")
            if artifacts.raw_csv_path is not None:
                print(f"Raw CSV: {artifacts.raw_csv_path}")
            if artifacts.standardized_csv_path is not None:
                print(f"Standardized CSV: {artifacts.standardized_csv_path}")
            if artifacts.framework_config_path is not None:
                print(f"Framework config: {artifacts.framework_config_path}")
            return 0
        if args.command == "joint-build":
            from .fusion import JointBuildConfig, build_joint_dataset

            artifacts = build_joint_dataset(
                JointBuildConfig(
                    aws_csv_path=args.aws_csv,
                    noaa_csv_path=args.noaa_csv,
                    era5_csv_path=args.era5_csv,
                    output_dir=args.output_dir,
                    output_csv_path=args.output_csv,
                    framework_config_path=args.framework_config,
                    overwrite=bool(args.overwrite),
                    era5_grid_step=float(args.era5_grid_step),
                )
            )
            print(f"Built joint dataset rows={artifacts.row_count} in {artifacts.output_dir}")
            print(f"Joint CSV: {artifacts.output_csv_path}")
            print(f"Manifest: {artifacts.manifest_path}")
            if artifacts.framework_config_path is not None:
                print(f"Framework config: {artifacts.framework_config_path}")
            return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc))
        return 1

    parser.error("Unsupported command.")
    return 2
