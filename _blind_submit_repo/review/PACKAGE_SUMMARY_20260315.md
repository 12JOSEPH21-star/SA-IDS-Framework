# SA-IDS Review Package Summary

## Scope

This package is a clean review bundle for the current `SA-IDS-Framework` state as of `2026-03-15`.

It includes:

- core research code
- CLI and downloader modules
- tests
- benchmark configs
- lightweight data manifests/configs
- latest benchmark and framework artifacts

It excludes:

- raw ERA5/AWS/NOAA/joint CSV and NetCDF files
- virtual environments
- cache directories
- old zip bundles

## Current Result State

- Latest fixed benchmark is complete:
  - `outputs/benchmark_joint_q1/summary.json`
  - `outputs/benchmark_joint_q1/report.md`
- Latest legacy framework run artifacts are present:
  - `outputs/framework_run/summary.json`
  - `outputs/framework_run/report.md`
- Joint full `framework-run` restart was attempted but did not finish with a new summary before packaging.

## Data Collection State

- ERA5 Q1 manifest/config included
- NOAA ISD Q1 manifest/config included
- Joint AWS+NOAA+ERA5 build manifest/config included
- AWS Q1 remainder is still quota-blocked on KMA API Hub
- ASOS Q1 downloader/config is included, but live collection is currently quota-blocked
- KMA event-truth and NWP-anchor download plans are included as manifests

## Key Files

- `models.py`
- `policy.py`
- `reliability.py`
- `pipeline.py`
- `benchmark_suite.py`
- `task_cli/`
- `tests/`
- `outputs/benchmark_joint_q1/`
- `review/REVIEW_SUMMARY.md`
