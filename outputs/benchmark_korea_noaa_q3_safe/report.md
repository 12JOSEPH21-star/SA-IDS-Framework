# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\noaa_isd_korea_q3_2025\framework_isd_q3.json`
- Data path: `C:\Programming Prctice\data\noaa_isd_korea_q3_2025\noaa_isd_framework.csv`
- Target: `air_temperature`
- Rows: train=1386, calibration=297, evaluation=298
- Stations: 11, stride=6h, inducing=16
- Repeat seeds: 5 (7, 11, 19, 23, 29)

## Predictive MNAR Benchmark
- Best CRPS: `joint_generative_missingness` under `state_dependent_mnar` @ `0.300000` with CRPS `4.759351` and gap CRPS `4.157133`

## Reliability Under Shift
- Best target-coverage error: `relational_adaptive` with coverage `0.909396` and shift coverage `0.888889`

## Region Holdout

## Fault Diagnosis
- Region holdout was not enabled for this benchmark config.
- Best F1: `pointwise_threshold_baseline` on `random_dropout` with F1 `0.752941`, AUROC `0.973710`, FAR `0.068441`

## Ablation Snapshot
- `base_gp_only`: RMSE `27.044405`, CRPS `26.384504`, Coverage ``, Cost `9.973104`
- `gp_plus_dynamic_silence`: RMSE `27.044470`, CRPS `26.384909`, Coverage ``, Cost `9.973104`
- `gp_plus_sensor_conditional_missingness`: RMSE `27.043571`, CRPS `26.383961`, Coverage ``, Cost `9.973104`
- `gp_plus_joint_generative_jvi_training`: RMSE `28.966825`, CRPS `27.203159`, Coverage ``, Cost `9.973104`
- `gp_plus_conformal_reliability`: RMSE `27.043480`, CRPS `26.384062`, Coverage `0.889262`, Cost `9.973104`
- `myopic_policy_baseline`: RMSE `13.089530`, CRPS `10.340708`, Coverage `0.845638`, Cost `9.973104`
- `full_model`: RMSE `9.640521`, CRPS `5.172875`, Coverage `0.892617`, Cost `9.973104`
- `base_gp_only`: RMSE `27.044067`, CRPS `26.384388`, Coverage ``, Cost `9.973104`
- `gp_plus_dynamic_silence`: RMSE `27.044006`, CRPS `26.384417`, Coverage ``, Cost `9.973104`
- `gp_plus_sensor_conditional_missingness`: RMSE `27.043911`, CRPS `26.384501`, Coverage ``, Cost `9.973104`
- `gp_plus_joint_generative_jvi_training`: RMSE `17.263515`, CRPS `13.751060`, Coverage ``, Cost `9.973104`
- `gp_plus_conformal_reliability`: RMSE `27.043751`, CRPS `26.384209`, Coverage `0.889262`, Cost `9.973104`
- `myopic_policy_baseline`: RMSE `10.507230`, CRPS `6.166630`, Coverage `0.922819`, Cost `9.973104`
- `full_model`: RMSE `14.967470`, CRPS `10.997034`, Coverage `0.845638`, Cost `9.973104`
- `base_gp_only`: RMSE `27.044664`, CRPS `26.384872`, Coverage ``, Cost `9.973104`
- `gp_plus_dynamic_silence`: RMSE `27.043461`, CRPS `26.384022`, Coverage ``, Cost `9.973104`
- `gp_plus_sensor_conditional_missingness`: RMSE `27.043537`, CRPS `26.384054`, Coverage ``, Cost `9.973104`
- `gp_plus_joint_generative_jvi_training`: RMSE `22.223772`, CRPS `21.771570`, Coverage ``, Cost `9.973104`
- `gp_plus_conformal_reliability`: RMSE `27.043537`, CRPS `26.384026`, Coverage `0.889262`, Cost `9.973104`
- `myopic_policy_baseline`: RMSE `14.152939`, CRPS `11.431059`, Coverage `0.852349`, Cost `9.973104`
- `full_model`: RMSE `9.843144`, CRPS `5.586045`, Coverage `0.936242`, Cost `9.973104`
- `base_gp_only`: RMSE `27.044128`, CRPS `26.384748`, Coverage ``, Cost `9.973104`
- `gp_plus_dynamic_silence`: RMSE `27.044447`, CRPS `26.384401`, Coverage ``, Cost `9.973104`
- `gp_plus_sensor_conditional_missingness`: RMSE `27.044750`, CRPS `26.384674`, Coverage ``, Cost `9.973104`
- `gp_plus_joint_generative_jvi_training`: RMSE `19.618645`, CRPS `18.277811`, Coverage ``, Cost `9.973104`
- `gp_plus_conformal_reliability`: RMSE `27.044373`, CRPS `26.384733`, Coverage `0.889262`, Cost `9.973104`
- `myopic_policy_baseline`: RMSE `12.667878`, CRPS `10.407317`, Coverage `0.845638`, Cost `9.973104`
- `full_model`: RMSE `21.311478`, CRPS `20.454618`, Coverage `0.872483`, Cost `9.973104`
- `base_gp_only`: RMSE `27.044672`, CRPS `26.384554`, Coverage ``, Cost `9.973104`
- `gp_plus_dynamic_silence`: RMSE `27.044010`, CRPS `26.384346`, Coverage ``, Cost `9.973104`
- `gp_plus_sensor_conditional_missingness`: RMSE `27.044050`, CRPS `26.384165`, Coverage ``, Cost `9.973104`
- `gp_plus_joint_generative_jvi_training`: RMSE `12.140935`, CRPS `9.841254`, Coverage ``, Cost `9.973104`
- `gp_plus_conformal_reliability`: RMSE `27.044304`, CRPS `26.384684`, Coverage `0.889262`, Cost `9.973104`
- `myopic_policy_baseline`: RMSE `10.207344`, CRPS `5.452138`, Coverage `0.949664`, Cost `9.973104`
- `full_model`: RMSE `12.920411`, CRPS `10.307572`, Coverage `0.869128`, Cost `9.973104`

## Policy Comparison
- Best cost-normalized gain: `random` with `0.569060`
