# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\joint_weather_network_q1_2025\framework_joint_q1.json`
- Data path: `C:\Programming Prctice\data\joint_weather_network_q1_2025\joint_weather_network_q1.csv`
- Target: `temperature`
- Rows: train=32768, calibration=8192, evaluation=8192
- Stations: 96, stride=1h, inducing=32
- Repeat seeds: 5 (7, 11, 19, 23, 29)

## Predictive MNAR Benchmark
- Best CRPS: `full_joint_jvi_training` under `state_dependent_mnar` @ `0.300000` with CRPS `3.160321` and gap CRPS `3.093582`

## Reliability Under Shift
- Best target-coverage error: `graph_corel` with coverage `0.782837` and shift coverage `0.821045`

## Region Holdout

## Fault Diagnosis
- Best region holdout CRPS: `gp_plus_joint_generative_jvi_training` on `southwest` with CRPS `1.558408`
- Best F1: `pi_ssd_only` on `random_dropout` with F1 `0.737347`, AUROC `0.964513`, FAR `0.064364`

## Ablation Snapshot
- `base_gp_only`: RMSE `16.253141`, CRPS `14.327386`, Coverage ``, Cost `17.259787`
- `gp_plus_dynamic_silence`: RMSE `16.451736`, CRPS `14.553152`, Coverage ``, Cost `17.259787`
- `gp_plus_sensor_conditional_missingness`: RMSE `16.651911`, CRPS `14.758986`, Coverage ``, Cost `17.259787`
- `gp_plus_joint_generative_jvi_training`: RMSE `5.179099`, CRPS `3.291051`, Coverage ``, Cost `17.259787`
- `gp_plus_conformal_reliability`: RMSE `16.393549`, CRPS `14.477301`, Coverage `0.039917`, Cost `17.259787`
- `myopic_policy_baseline`: RMSE `5.968249`, CRPS `4.734331`, Coverage `0.310303`, Cost `15.424854`
- `full_model`: RMSE `4.756909`, CRPS `3.392877`, Coverage `0.799561`, Cost `16.682882`
- `base_gp_only`: RMSE `16.147234`, CRPS `14.220286`, Coverage ``, Cost `17.259787`
- `gp_plus_dynamic_silence`: RMSE `16.522669`, CRPS `14.622375`, Coverage ``, Cost `17.259787`
- `gp_plus_sensor_conditional_missingness`: RMSE `16.565041`, CRPS `14.670681`, Coverage ``, Cost `17.259787`
- `gp_plus_joint_generative_jvi_training`: RMSE `12.722323`, CRPS `11.553467`, Coverage ``, Cost `17.259787`
- `gp_plus_conformal_reliability`: RMSE `16.526928`, CRPS `14.634304`, Coverage `0.039917`, Cost `17.259787`
- `myopic_policy_baseline`: RMSE `6.348012`, CRPS `4.703622`, Coverage `0.497070`, Cost `15.424854`
- `full_model`: RMSE `5.771153`, CRPS `3.339438`, Coverage `0.881836`, Cost `16.682882`
- `base_gp_only`: RMSE `16.544491`, CRPS `14.648958`, Coverage ``, Cost `17.259787`
- `gp_plus_dynamic_silence`: RMSE `16.479528`, CRPS `14.577324`, Coverage ``, Cost `17.259787`
- `gp_plus_sensor_conditional_missingness`: RMSE `16.591448`, CRPS `14.694572`, Coverage ``, Cost `17.259787`
- `gp_plus_joint_generative_jvi_training`: RMSE `4.047961`, CRPS `2.920749`, Coverage ``, Cost `17.259787`
- `gp_plus_conformal_reliability`: RMSE `16.516228`, CRPS `14.618254`, Coverage `0.039917`, Cost `17.259787`
- `myopic_policy_baseline`: RMSE `7.081217`, CRPS `5.073261`, Coverage `0.685547`, Cost `15.424854`
- `full_model`: RMSE `6.270697`, CRPS `4.523432`, Coverage `0.883423`, Cost `16.682882`
- `base_gp_only`: RMSE `16.401560`, CRPS `14.488935`, Coverage ``, Cost `17.259787`
- `gp_plus_dynamic_silence`: RMSE `16.561430`, CRPS `14.657104`, Coverage ``, Cost `17.259787`
- `gp_plus_sensor_conditional_missingness`: RMSE `16.427284`, CRPS `14.523212`, Coverage ``, Cost `17.259787`
- `gp_plus_joint_generative_jvi_training`: RMSE `11.132210`, CRPS `9.473741`, Coverage ``, Cost `17.259787`
- `gp_plus_conformal_reliability`: RMSE `16.680815`, CRPS `14.790403`, Coverage `0.039917`, Cost `17.259787`
- `myopic_policy_baseline`: RMSE `5.107518`, CRPS `3.721302`, Coverage `0.698975`, Cost `15.424854`
- `full_model`: RMSE `6.718814`, CRPS `4.149080`, Coverage `0.889160`, Cost `16.682882`
- `base_gp_only`: RMSE `16.526005`, CRPS `14.622721`, Coverage ``, Cost `17.259787`
- `gp_plus_dynamic_silence`: RMSE `16.562828`, CRPS `14.665216`, Coverage ``, Cost `17.259787`
- `gp_plus_sensor_conditional_missingness`: RMSE `16.235003`, CRPS `14.309911`, Coverage ``, Cost `16.902115`
- `gp_plus_joint_generative_jvi_training`: RMSE `6.968758`, CRPS `5.087026`, Coverage ``, Cost `17.259787`
- `gp_plus_conformal_reliability`: RMSE `16.637495`, CRPS `14.742915`, Coverage `0.039917`, Cost `17.259787`
- `myopic_policy_baseline`: RMSE `5.642488`, CRPS `4.480699`, Coverage `0.190918`, Cost `15.424854`
- `full_model`: RMSE `7.736532`, CRPS `5.805083`, Coverage `0.238770`, Cost `16.682882`

## Policy Comparison
- Best cost-normalized gain: `random` with `1.591876`
