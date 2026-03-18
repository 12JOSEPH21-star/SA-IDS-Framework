# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\noaa_isd_korea_q2_2025\framework_isd_q2.json`
- Data path: `C:\Programming Prctice\data\noaa_isd_korea_q2_2025\noaa_isd_framework.csv`
- Target: `air_temperature`
- Rows: train=2544, calibration=524, evaluation=542
- Stations: 11, stride=6h, inducing=16
- Repeat seeds: 5 (7, 11, 19, 23, 29)

## Predictive MNAR Benchmark
- Best CRPS: `joint_generative_missingness` under `state_dependent_mnar` @ `0.700000` with CRPS `7.395180` and gap CRPS `7.496500`

## Reliability Under Shift
- Best target-coverage error: `adaptive_conformal` with coverage `0.891144` and shift coverage `0.860696`

## Region Holdout

## Fault Diagnosis
- Region holdout was not enabled for this benchmark config.
- Best F1: `pointwise_threshold_baseline` on `random_dropout` with F1 `0.673913`, AUROC `0.978068`, FAR `0.119497`

## Ablation Snapshot
- `base_gp_only`: RMSE `22.814865`, CRPS `22.061563`, Coverage ``, Cost `9.591967`
- `gp_plus_dynamic_silence`: RMSE `22.814127`, CRPS `22.061150`, Coverage ``, Cost `9.591967`
- `gp_plus_sensor_conditional_missingness`: RMSE `22.812796`, CRPS `22.060204`, Coverage ``, Cost `9.591967`
- `gp_plus_joint_generative_jvi_training`: RMSE `17.873230`, CRPS `16.576208`, Coverage ``, Cost `9.591967`
- `gp_plus_conformal_reliability`: RMSE `22.813276`, CRPS `22.060701`, Coverage `0.892989`, Cost `9.591967`
- `myopic_policy_baseline`: RMSE `15.202304`, CRPS `13.581460`, Coverage `0.863469`, Cost `9.591967`
- `full_model`: RMSE `15.940004`, CRPS `14.531897`, Coverage `0.861624`, Cost `9.591967`
- `base_gp_only`: RMSE `22.813454`, CRPS `22.061174`, Coverage ``, Cost `9.591967`
- `gp_plus_dynamic_silence`: RMSE `22.816717`, CRPS `22.063248`, Coverage ``, Cost `9.591967`
- `gp_plus_sensor_conditional_missingness`: RMSE `22.812220`, CRPS `22.059475`, Coverage ``, Cost `9.591967`
- `gp_plus_joint_generative_jvi_training`: RMSE `17.971735`, CRPS `16.543304`, Coverage ``, Cost `9.591967`
- `gp_plus_conformal_reliability`: RMSE `22.809122`, CRPS `22.057999`, Coverage `0.892989`, Cost `9.591967`
- `myopic_policy_baseline`: RMSE `15.178620`, CRPS `13.424287`, Coverage `0.928044`, Cost `9.591967`
- `full_model`: RMSE `17.050022`, CRPS `15.587725`, Coverage `0.902214`, Cost `9.591967`
- `base_gp_only`: RMSE `22.812969`, CRPS `22.060352`, Coverage ``, Cost `9.591967`
- `gp_plus_dynamic_silence`: RMSE `22.815401`, CRPS `22.061750`, Coverage ``, Cost `9.591967`
- `gp_plus_sensor_conditional_missingness`: RMSE `22.812979`, CRPS `22.060238`, Coverage ``, Cost `9.591967`
- `gp_plus_joint_generative_jvi_training`: RMSE `15.261295`, CRPS `14.360895`, Coverage ``, Cost `9.591967`
- `gp_plus_conformal_reliability`: RMSE `22.816734`, CRPS `22.062853`, Coverage `0.892989`, Cost `9.591967`
- `myopic_policy_baseline`: RMSE `16.998219`, CRPS `15.676537`, Coverage `0.867159`, Cost `9.591967`
- `full_model`: RMSE `14.648219`, CRPS `11.019667`, Coverage `0.880074`, Cost `9.591967`
- `base_gp_only`: RMSE `22.815008`, CRPS `22.062410`, Coverage ``, Cost `9.591967`
- `gp_plus_dynamic_silence`: RMSE `22.812401`, CRPS `22.060644`, Coverage ``, Cost `9.591967`
- `gp_plus_sensor_conditional_missingness`: RMSE `22.814857`, CRPS `22.061510`, Coverage ``, Cost `9.591967`
- `gp_plus_joint_generative_jvi_training`: RMSE `16.114653`, CRPS `14.600599`, Coverage ``, Cost `9.591967`
- `gp_plus_conformal_reliability`: RMSE `22.816498`, CRPS `22.063028`, Coverage `0.892989`, Cost `9.591967`
- `myopic_policy_baseline`: RMSE `15.931828`, CRPS `14.236260`, Coverage `0.880074`, Cost `9.591967`
- `full_model`: RMSE `18.966629`, CRPS `18.016273`, Coverage `0.892989`, Cost `9.591967`
- `base_gp_only`: RMSE `22.817677`, CRPS `22.063810`, Coverage ``, Cost `9.591967`
- `gp_plus_dynamic_silence`: RMSE `22.816435`, CRPS `22.063091`, Coverage ``, Cost `9.591967`
- `gp_plus_sensor_conditional_missingness`: RMSE `22.816261`, CRPS `22.062719`, Coverage ``, Cost `9.591967`
- `gp_plus_joint_generative_jvi_training`: RMSE `16.312517`, CRPS `15.632397`, Coverage ``, Cost `9.591967`
- `gp_plus_conformal_reliability`: RMSE `22.815895`, CRPS `22.062588`, Coverage `0.892989`, Cost `9.591967`
- `myopic_policy_baseline`: RMSE `13.635681`, CRPS `10.164140`, Coverage `0.881919`, Cost `9.591967`
- `full_model`: RMSE `15.938079`, CRPS `14.287108`, Coverage `0.892989`, Cost `9.591967`

## Policy Comparison
- Best cost-normalized gain: `random` with `0.393110`
