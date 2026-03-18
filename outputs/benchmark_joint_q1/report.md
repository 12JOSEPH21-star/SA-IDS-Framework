# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\joint_weather_network_q1_2025\framework_joint_q1.json`
- Data path: `C:\Programming Prctice\data\joint_weather_network_q1_2025\joint_weather_network_q1.csv`
- Target: `temperature`
- Rows: train=1536, calibration=384, evaluation=384
- Stations: 8, stride=6h, inducing=16

## Predictive MNAR Benchmark
- Best CRPS: `joint_generative_missingness` under `value_dependent_mnar` @ `0.700000` with CRPS `9.452142` and gap CRPS `9.806243`

## Reliability Under Shift
- Best target-coverage error: `relational_adaptive` with coverage `0.906250` and shift coverage `0.892857`

## Fault Diagnosis
- Best F1: `pointwise_threshold_baseline` on `random_dropout` with F1 `0.400000`, AUROC `0.767687`, FAR `0.041420`

## Ablation Snapshot
- `base_gp_only`: RMSE `14.681655`, CRPS `12.946574`, Coverage ``, Cost `6.574634`
- `gp_plus_dynamic_silence`: RMSE `14.731496`, CRPS `13.001004`, Coverage ``, Cost `6.574634`
- `gp_plus_sensor_conditional_missingness`: RMSE `14.718198`, CRPS `13.055133`, Coverage ``, Cost `6.574951`
- `gp_plus_joint_generative_jvi_training`: RMSE `23.067392`, CRPS `21.206038`, Coverage ``, Cost `6.574951`
- `gp_plus_conformal_reliability`: RMSE `14.675926`, CRPS `12.939316`, Coverage `0.005208`, Cost `6.574634`
- `myopic_policy_baseline`: RMSE `42.023075`, CRPS `34.052433`, Coverage `0.695312`, Cost `6.267728`
- `full_model`: RMSE `15.151605`, CRPS `12.129514`, Coverage `0.388021`, Cost `6.360423`

## Policy Comparison
- Best cost-normalized gain: `ppo_warmstart_baseline` with `0.775243`
