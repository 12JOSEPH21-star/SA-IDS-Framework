# Silence-Aware Information-Driven Sensing

Research-grade code for the framework:

`Silence-Aware Information-Driven Sensing: A Cost-Reliability Optimization Framework`

The current implementation also includes the first manuscript-alignment slice for the
newer submission draft:

- sequence-aware `M2` diagnosis with optional NWP-anchored scoring
- adaptive conformal `M5` updates for chronological stress testing
- `M3` plug-in versus joint-variational latent-missingness comparison
- `M4` myopic versus non-myopic rollout policy comparison
- PI-SSD-style diagnosis embeddings, DBN-lite state inference, and CoRel-lite relational adaptive conformal
- preserved lazy-greedy / plug-in baselines so older ablations still run

The repository currently contains two runnable paths:

- `python -m task_cli run`: the earlier CSV-based sparse-scenario study CLI
- `python -m task_cli framework-run`: the modular `M1`-`M5` PyTorch/GPyTorch framework

## Framework Modules

- `M1`: sparse variational spatiotemporal GP in [models.py](./models.py)
- `M2`: sequence-aware dynamic silence diagnosis with PI-SSD-style embeddings and DBN-lite state posteriors in [models.py](./models.py)
- `M3`: heterogeneous MNAR missingness modeling with `plug_in`, `joint_variational`, and latent sensor-health summaries in [models.py](./models.py)
- `M4`: myopic lazy-greedy, rollout, and PPO-warm-start policy surrogates in [policy.py](./policy.py)
- `M5`: normalized conformal reliability with adaptive, relational-adaptive, and iterative graph-aware CoRel-lite updates in [reliability.py](./reliability.py)
- unified orchestration in [pipeline.py](./pipeline.py) and [experiment.py](./experiment.py)

## Quick Start

Create a framework config and demo dataset:

```bash
python -m task_cli framework-init
python -m task_cli framework-init --preset isd_hourly --data isd_hourly.csv --config framework_isd.json
```

Run the framework experiment:

```bash
python -m task_cli framework-run --config framework_config.json
```

Install research dependencies first:

```bash
powershell -ExecutionPolicy Bypass -File scripts/setup_py311_env.ps1
```

The setup script installs official CPython `3.11` when needed, creates `.venv311`,
installs the CPU PyTorch wheel pinned in [requirements-research.txt](./requirements-research.txt),
and verifies `torch` / `gpytorch` imports.

## Default Framework Outputs

`framework-run` writes to `outputs/framework_run/` by default:

- `summary.json`
- `ablations.csv`
- `sensitivity.csv`
- `selection.csv`
- `report.md`

The default ablation suite includes:

- `base_gp_only`
- `gp_plus_dynamic_silence`
- `gp_plus_homogeneous_missingness`
- `gp_plus_sensor_conditional_missingness`
- `gp_plus_joint_variational_missingness`
- `gp_plus_joint_jvi_training`
- `gp_plus_pattern_mixture_missingness`
- `gp_plus_conformal_reliability`
- `relational_reliability_baseline`
- `myopic_policy_baseline`
- `ppo_warmstart_baseline`
- `rollout_policy_baseline`
- `variance_policy_baseline`
- `full_model`

Default framework configs now enable:

- `diagnosis_mode="temporal"` for standard station-network presets
- `diagnosis_mode="temporal_nwp"` for the joint AWS+NOAA+ERA5 build
- `observation.use_pi_ssd=True` for self-supervised diagnosis embeddings
- `observation.use_dbn=True` for temporal diagnostic state smoothing
- `observation.use_latent_ode=True` for latent dynamics regularization over diagnosis embeddings
- linear corruption curriculum for stronger PI-SSD pretext training
- `missingness.inference_strategy="joint_variational"` for sensor-network presets
- `state_training.training_strategy="joint_variational"` for coupled M1+M3 training
- `missingness.use_sensor_health_latent=True` for latent health-aware MNAR modeling
- `policy.planning_strategy="ppo_online"` for research presets
- `reliability.mode="graph_corel"` with graph-aware local quantiles, iterative message passing, and adaptive updates

The larger manuscript items that still remain future slices are:

- full GP-VAE/JVI replacing the current sparse-GP plus latent-adapter approximation
- full PI-SSD beyond the current corruption-curriculum and latent-ODE-lite regularization
- full environment-trained DRL / Schur-MI policy learning beyond the current capped PPO-online surrogate
- full STGNN-grade CoRel beyond the current lightweight iterative graph local-quantile adapter

The default `run` payload also includes:

- fixed seed / deterministic-runtime knobs
- batched inference size for large candidate pools
- an expanded candidate-pool benchmark profile
- benchmark timing and tensor-footprint reporting

Available `framework-init` presets:

- `demo`
- `isd_hourly`
- `aws_network`
- `era5_reference`

## Korean ASOS / AWS Download

The CLI includes an official Korean observation downloader for the domestic
surface sources used in the draft methodology:

- `asos_hourly`: hourly ASOS observations from data.go.kr
- `aws_recent_1min`: recent 1-minute AWS observations from KMA API Hub

Dry-run the request plan:

```bash
python -m task_cli kma-download \
  --source asos_hourly \
  --station-ids 108 159 \
  --start-date 2025-01-01 \
  --end-date 2025-01-07 \
  --dry-run
```

Download raw ASOS observations, generate a framework-oriented CSV, and emit an
`aws_network` framework config:

```bash
python -m task_cli kma-download \
  --source asos_hourly \
  --station-ids 108 159 \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --service-key "<DATA_GO_KR_SERVICE_KEY>" \
  --metadata-csv station_metadata.csv \
  --standardized-csv data/kma/asos_framework.csv \
  --framework-config framework_kma.json
```

For recent 1-minute AWS:

```bash
python -m task_cli kma-download \
  --source aws_recent_1min \
  --station-ids 400 401 \
  --aws-datetime 2025-01-02T09:00 \
  --service-key "<KMA_APIHUB_AUTH_KEY>" \
  --metadata-template station_metadata_template.csv
```

Long AWS historical pulls can hit temporary API Hub `403` responses. To keep
resuming the same range until the target end time is reached, use the retry
wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\retry_aws_apihub_range.ps1 `
  -StartTimestamp 2025-01-01T00:00 `
  -EndTimestamp 2025-03-31T23:00 `
  -AuthKey "<KMA_APIHUB_AUTH_KEY>" `
  -OutputDir data\aws_korea_q1_2025 `
  -RetryWaitSeconds 900
```

The wrapper checks the last timestamp already stored in `aws_raw.csv`, reruns
the downloader with `--resume`, and sleeps between retries until the requested
end time is present locally.

Outputs:

- raw API CSV in `<output-dir>/<source>_raw.csv`
- `kma_download_manifest.json`
- auto-fetched AWS station metadata CSV when `--metadata-csv` is omitted
- optional station metadata template
- optional standardized CSV with the `aws_network` schema
- optional framework config for `python -m task_cli framework-run`

Metadata note:

- `--metadata-csv` should contain `station_id`, `latitude`, `longitude`
- recommended extra columns: `elevation`, `cost`, `sensor_type`, `sensor_group`,
  `sensor_modality`, `site_type`, `maintenance_state`, `maintenance_age`
- AWS downloads auto-fetch `latitude`, `longitude`, and `elevation` from the
  KMA API Hub unless `--no-auto-station-metadata` is passed
- without metadata, the downloader still writes raw data, but standardized CSVs
  may be missing coordinates required by the framework

Official sources:

- ASOS hourly: <https://www.data.go.kr/data/15057210/openapi.do>
- KMA AWS API Hub landing page: <https://www.data.go.kr/data/15139433/openapi.do>
- KMA API Hub usage guide: <https://apihub.kma.go.kr/static/file/%EA%B8%B0%EC%83%81%EC%B2%AD_API%ED%97%88%EB%B8%8C_%EC%82%AC%EC%9A%A9_%EB%B0%A9%EB%B2%95_%EC%95%88%EB%82%B4.pdf>

## NOAA ISD Download

The CLI also includes a NOAA ISD-Lite downloader for `isd_hourly`-style experiments.

Download active South Korea (`KS`) stations for Q1 2025 and emit a framework-ready CSV:

```bash
python -m task_cli noaa-download \
  --start-date 2025-01-01 \
  --end-date 2025-03-31 \
  --country-code KS \
  --output-dir data/noaa_isd_korea_q1_2025 \
  --framework-config data/noaa_isd_korea_q1_2025/framework_isd_q1.json
```

Outputs:

- `isd_station_metadata.csv`
- `noaa_isd_raw.csv`
- `noaa_isd_framework.csv`
- `noaa_download_manifest.json`
- optional `framework_isd_q1.json`

Official sources:

- NOAA ISD overview: <https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database>
- NOAA station history: <https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv>

## Joint Q1 Build

To build a single Q1 2025 observation-network dataset from AWS and NOAA, with
ERA5 attached as nearest-grid reference context:

```bash
python -m task_cli joint-build \
  --aws-csv data/aws_korea_q1_2025/aws_framework.csv \
  --noaa-csv data/noaa_isd_korea_q1_2025/noaa_isd_framework.csv \
  --era5-csv data/era5_korea_q1_2025/era5_reference.csv \
  --output-dir data/joint_weather_network_q1_2025 \
  --framework-config data/joint_weather_network_q1_2025/framework_joint_q1.json
```

This writes a joint long-format CSV plus a framework config that uses observed
temperature as the target and includes ERA5 fields as additional context.

## ERA5 Reference Download

The CLI now includes an ERA5 downloader that plans monthly CDS requests, downloads raw NetCDF files,
optionally merges them into the `era5_reference` tabular schema, and can emit a matching framework config.

Preview the request plan without calling CDS:

```bash
python -m task_cli era5-download \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --area 39 124 32 132 \
  --grid 0.25 0.25 \
  --framework-config framework_era5.json \
  --dry-run
```

Run the download and produce a framework-ready CSV:

```bash
python -m task_cli era5-download \
  --start-date 2025-01-01 \
  --end-date 2025-03-31 \
  --area 39 124 32 132 \
  --grid 0.25 0.25 \
  --output-dir data/era5_korea \
  --csv-path data/era5_korea/era5_reference.csv \
  --framework-config framework_era5.json
```

Authentication:

- default: `C:\Users\이요셉\.cdsapirc`
- alternative: pass `--cds-url` and `--cds-key`
- alternative: set `CDSAPI_URL` and `CDSAPI_KEY`
- `--cds-key` may be either the full CDS key string or a personal access token

Defaults:

- dataset: `reanalysis-era5-single-levels`
- dynamic variables: `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `surface_pressure`, `total_precipitation`
- static variables: `geopotential`, `land_sea_mask`

The merged CSV writes `orography` by converting ERA5 `geopotential` to meters with `z / 9.80665`,
which matches the `era5_reference` preset columns.

Downloader-generated `era5_reference` configs automatically fall back to `row` splits when the
requested range contains fewer than three unique timestamps, which keeps small smoke runs valid.

## Expected CSV Columns

The framework demo config uses long-format tabular weather data with:

```text
timestamp,station_id,latitude,longitude,cost,temperature,humidity,pressure,elevation,sensor_type,sensor_group,sensor_modality,site_type,maintenance_state,maintenance_age
```

The sensor metadata fields are used to compare:

- homogeneous missingness assumptions
- sensor-conditional heterogeneous missingness assumptions

## Notes

- The GP implementation uses sparse variational GPs only. `ExactGP` is intentionally not used.
- Active sensing uses a heap-based lazy greedy approximation and does not iteratively materialize full candidate-pool covariance matrices.
- Public APIs are typed and organized for ablation-heavy experimentation.

## Tests

```bash
python -m unittest discover -s tests -v
pytest -q
```

`pytest.ini` and `tests/conftest.py` keep the repo root on the import path so
the CLI and framework tests run directly from a fresh checkout.
