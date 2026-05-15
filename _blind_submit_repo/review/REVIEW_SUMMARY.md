# Silence-Aware IDS Review Summary

## Scope

This review bundle contains:

- current source code and CLI tooling
- test suite and environment/setup scripts
- framework configs used for ERA5-only, NOAA-only, and joint AWS+NOAA+ERA5 runs
- compact output artifacts: `summary.json`, `ablations.csv`, `sensitivity.csv`, `selection.csv`, `report.md`
- dataset/manifold metadata for NOAA, AWS status, and the joint Q1 2025 build

This bundle intentionally excludes:

- virtual environments
- `__pycache__` and compiled files
- large raw/reference CSV and NetCDF files
- generated benchmark outputs not needed for review

## Current Data Status

### ERA5 Q1 2025

Reference dataset and config were already built for `2025-01-01` to `2025-03-31`.

- Config: `data/era5_korea_q1_2025/framework_era5_q1.json`
- Run outputs: `data/era5_korea_q1_2025/outputs/framework_run/`

Key dataset summary:

- train rows: `1,446,984`
- calibration rows: `310,068`
- evaluation rows: `310,068`
- context dim: `4`
- sensor metadata cardinalities: all `0` because ERA5 is a gridded reference source

Key ERA5-only base metrics:

- RMSE: `253.0008`
- CRPS: `249.5345`
- Log-Score: `872.6105`
- Coverage: `0.3752`
- Interval Width: `502.5826`

Interpretation:

- This run is a gridded reference baseline, not a heterogeneous sensor-network test.
- M3 is effectively disabled because no sensor metadata is present.
- It is still useful as a scale/reference sanity check for the sparse GP and reliability path.

### AWS Q1 2025

AWS API Hub download progressed but is not yet complete for the full quarter because the API started returning temporary `403` responses.

Current local status:

- first timestamp: `2025-01-01T00:00`
- last timestamp: `2025-02-10T20:00`
- downloaded rows: `715,149`

Relevant files:

- station metadata: `data/aws_korea_q1_2025/aws_station_metadata_auto.csv`
- retry wrapper: `scripts/retry_aws_apihub_range.ps1`
- downloader: `scripts/download_aws_apihub_range.py`

Interpretation:

- The AWS side is operational and resumable.
- The joint dataset below is therefore a valid multisource smoke/integration dataset, but it is not yet a complete full-quarter AWS network dataset.

### NOAA ISD-Lite Q1 2025

NOAA ISD-Lite download for South Korea (`KS`) completed successfully.

- Metadata-selected stations: `83`
- Stations with actual rows in the merged framework CSV: `79`
- Downloaded rows: `131,488`

Relevant files:

- metadata: `data/noaa_isd_korea_q1_2025/isd_station_metadata.csv`
- manifest: `data/noaa_isd_korea_q1_2025/noaa_download_manifest.json`
- config: `data/noaa_isd_korea_q1_2025/framework_isd_q1.json`

Interpretation:

- NOAA provides a useful heterogeneous station layer to complement AWS.
- This path is complete for Q1 2025 and currently acts as the stable observational backbone in the joint build.

## Joint Q1 2025 Dataset

The joint build merges AWS and NOAA observations and attaches nearest-grid ERA5 context.

Relevant files:

- config: `data/joint_weather_network_q1_2025/framework_joint_q1.json`
- smoke config: `data/joint_weather_network_q1_2025/framework_joint_q1_smoke.json`
- build manifest: `data/joint_weather_network_q1_2025/joint_build_manifest.json`

Joint build summary:

- total rows: `846,637`
- source counts:
  - AWS: `715,149`
  - NOAA ISD-Lite: `131,488`
- timestamp range:
  - start: `2025-01-01T00:00`
  - end: `2025-03-31T23:00`

Important caveat:

- Because AWS currently stops at `2025-02-10T20:00`, the joint dataset is temporally complete only through NOAA after that point.
- This means current joint results should be read as an end-to-end integration check, not yet the final balanced Q1 multisource benchmark.

## Joint Framework Smoke Run

An end-to-end framework run was executed on the full joint dataset using a reduced smoke config:

- runtime: `.venv311` / Python `3.11.9`
- torch: `2.10.0+cpu`
- gpytorch: `1.15.2`
- output directory: `data/joint_weather_network_q1_2025/outputs/framework_joint_smoke/`

Dataset summary for the smoke run:

- train rows: `807,243`
- calibration rows: `20,008`
- evaluation rows: `19,386`
- context dim: `9`
- metadata cardinalities:
  - sensor_type: `2`
  - sensor_group: `1`
  - sensor_modality: `2`
  - installation_environment: `3`
  - maintenance_state: `1`

Base metrics:

- RMSE: `12.1764`
- MAE: `10.3445`
- CRPS: `7.8361`
- Log-Score: `4.7629`
- Coverage: `0.6848`
- Interval Width: `26.5684`

Selected ablation results:

- `base_gp_only`
  - RMSE: `12.1886`
  - CRPS: `7.8423`
- `gp_plus_sensor_conditional_missingness`
  - RMSE: `12.0612`
  - CRPS: `7.7361`
- `gp_plus_conformal_reliability`
  - RMSE: `12.1944`
  - CRPS: `7.8553`
  - Coverage: `0.6848`
- `full_model`
  - RMSE: `12.2305`
  - CRPS: `7.8832`
  - Coverage: `0.6744`

Interpretation:

- The joint pipeline runs end-to-end on real multisource data.
- In this 1-epoch smoke setting, `sensor-conditional missingness` improves over `base_gp_only`.
- The `full_model` is not yet best under the smoke budget; this should be read as under-training, not as a final negative result.
- The correct next step is a longer run after AWS completion, not a conclusion that M3/M5 are ineffective.

## Code State

Implemented and exercised components in this bundle:

- `M1` sparse variational spatiotemporal GP
- `M2` sequence-aware dynamic silence diagnosis in observation space, with optional NWP-anchored scoring, DBN-lite state posteriors, and latent-ODE-lite diagnosis regularization
- `M3` homogeneous vs sensor-conditional missingness with selection/pattern-mixture support and latent sensor-health summaries
- `M4` lazy greedy, rollout, PPO-warm-start, and PPO-online information policies with approximate Gaussian surrogates
- `M5` normalized conformal reliability with adaptive, relational-adaptive, and iterative graph-aware CoRel-lite updates
- domestic KMA AWS/ASOS downloaders
- NOAA ISD-Lite downloader
- joint AWS+NOAA+ERA5 dataset builder

First-slice manuscript alignment already landed:

- temporal and `temporal_nwp` diagnosis modes in `M2`
- adaptive conformal updates in `M5`
- `plug_in` versus `joint_variational` versus `joint_generative` missingness inference in `M3`, including station-level temporal transition priors for the generative path
- sequential versus joint-ELBO sparse-GP training for coupled `M1+M3`
- `lazy_greedy` versus `non_myopic_rollout` versus `ppo_warmstart` versus `ppo_online` policy planning in `M4`
- PI-SSD-style diagnosis embeddings with corruption curriculum and latent-ODE-lite regularization, DBN-lite state inference, latent health-aware missingness, and iterative graph-aware CoRel-lite conformal updates
- framework presets updated to enable those modes by default

Known open items:

- Full-quarter AWS completion remains blocked by temporary API Hub `403` responses. The downloader and retry wrapper are already in place.
- The newer draft's larger architectural changes are not fully implemented yet. Recommended priority:
  - 1. full GP-VAE / JVI instead of the current sparse-GP plus latent-adapter approximation
  - 2. full environment-trained PPO / DRL Schur-MI policy learning beyond the current capped PPO-online surrogate
  - 3. graph-aware STGNN CoRel beyond the current lightweight iterative graph-corel local quantile path
  - 4. stronger PI-SSD sensor-health representation learning beyond the current curriculum plus latent-ODE-lite regularization

## Verification

Latest repository test result at packaging time:

- `python -m unittest discover -s tests -v`
- Result: `59` tests total, `OK`, `1` skipped
- `python -m pytest -q`
- Result: `58 passed, 1 skipped`

The skips are expected for tests that require the full research runtime stack in the active interpreter.
