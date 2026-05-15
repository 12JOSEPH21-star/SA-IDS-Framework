# SA-IDS Slim Review Package

This is the slim review package for paper-oriented inspection.

Included:
- Source code and configs
- CLI, scripts, tests, and review notes
- Benchmark outputs and framework summaries/reports
- Dataset manifests, framework configs, and lightweight metadata sidecars

Intentionally excluded:
- Raw downloaded observation files
  - AWS/ASOS/NOAA/ERA5 raw CSV/NetCDF
  - Joint merged raw observation CSV
- Large cache/checkpoint artifacts
  - benchmark cache `.pt`
  - large framework checkpoint binaries
- Virtual environments, git metadata, Python caches, and previous zip bundles

Recommended starting points:
- `README.md`
- `review/REVIEW_SUMMARY.md`
- `review/PACKAGE_SUMMARY_20260315.md`
- `outputs/benchmark_joint_q1/report.md`
- `outputs/benchmark_joint_q1_medium/report.md`
- `data/joint_weather_network_q1_2025/framework_joint_q1.json`
- `data/joint_weather_network_q1_2025/outputs/framework_joint_run/framework_progress.json`
