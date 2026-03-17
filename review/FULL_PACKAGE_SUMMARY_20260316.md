# SA-IDS Full Paper Package

This bundle is the full paper-oriented package as of 2026-03-16.

Included:
- Core code: `models.py`, `policy.py`, `reliability.py`, `pipeline.py`, `experiment.py`, `benchmark_suite.py`
- CLI and download/build tooling: `task_cli/`, `scripts/`
- Tests: `tests/`
- Configs: framework and benchmark JSON files in the repository root and `data/.../framework_*.json`
- Data: `data/` including AWS, ASOS, NOAA, ERA5, event-truth, QC sidecar, NWP anchor support files, and joint datasets
- Outputs: `outputs/` and `data/.../outputs/`
- Review/support docs: `review/`, `README.md`, `requirements-research.txt`, `pytest.ini`

Current experiment status:
- Benchmark small completed: `outputs/benchmark_joint_q1/`
- Benchmark medium completed: `outputs/benchmark_joint_q1_medium/`
- Full framework run is partial/resumable:
  - `data/joint_weather_network_q1_2025/outputs/framework_joint_run/framework_progress.json`
  - `data/joint_weather_network_q1_2025/outputs/framework_joint_run/framework_run_checkpoint.pt`
  - Last saved checkpoint stage: `fit_complete`
  - Last progress stage: `evaluating_base_pipeline`

Large-run stabilization currently applied in the code/config:
- Row caps:
  - train: `131072`
  - calibration: `32768`
  - evaluation: `32768`
- Large-run reliability fallback:
  - `graph_corel` -> `relational_adaptive` for stabilized framework runs

Excluded from the full package:
- `.git/`
- `.venv311/`, `.venv311a/`, `.venv311b/`
- `__pycache__/`, `.pytest_cache/`
- temporary directories such as `tmp*` and `.tmp_era5_unzip/`
- previous zip bundles under `dist/` and `dist/archive/`
