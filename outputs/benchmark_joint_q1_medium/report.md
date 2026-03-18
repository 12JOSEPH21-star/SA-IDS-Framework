# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\joint_weather_network_q1_2025\framework_joint_q1.json`
- Data path: `C:\Programming Prctice\data\joint_weather_network_q1_2025\joint_weather_network_q1.csv`
- Target: `temperature`
- Rows: train=8192, calibration=2048, evaluation=2048
- Stations: 16, stride=3h, inducing=24

## Predictive MNAR Benchmark
- Best CRPS: `joint_generative_missingness` under `value_dependent_mnar` @ `0.700000` with CRPS `7.211614` and gap CRPS `7.691085`

## Reliability Under Shift
- Best target-coverage error: `relational_adaptive` with coverage `0.893066` and shift coverage `0.891185`

## Fault Diagnosis
- Best F1: `pi_ssd_only` on `random_dropout` with F1 `0.761719`, AUROC `0.953443`, FAR `0.039933`

## Ablation Snapshot
- `base_gp_only`: RMSE `11.753034`, CRPS `9.675175`, Coverage ``, Cost `13.782031`
- `gp_plus_dynamic_silence`: RMSE `11.767048`, CRPS `9.675213`, Coverage ``, Cost `13.782031`
- `gp_plus_sensor_conditional_missingness`: RMSE `11.758461`, CRPS `9.670602`, Coverage ``, Cost `13.782031`
- `gp_plus_joint_generative_jvi_training`: RMSE `11.963173`, CRPS `9.909698`, Coverage ``, Cost `13.782031`
- `gp_plus_conformal_reliability`: RMSE `11.767965`, CRPS `9.672903`, Coverage `0.843750`, Cost `13.782031`
- `myopic_policy_baseline`: RMSE `11.932215`, CRPS `9.565239`, Coverage `0.893066`, Cost `13.633778`
- `full_model`: RMSE `10.236119`, CRPS `7.531391`, Coverage `0.812988`, Cost `13.734293`

## Policy Comparison
- Best cost-normalized gain: `ppo_warmstart_baseline` with `1.096588`
