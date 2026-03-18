# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\noaa_isd_china_q1_2025\framework_isd_q1.json`
- Data path: `C:\Programming Prctice\data\noaa_isd_china_q1_2025\noaa_isd_framework.csv`
- Target: `air_temperature`
- Rows: train=7793, calibration=1665, evaluation=1655
- Stations: 16, stride=3h, inducing=24
- Repeat seeds: 5 (7, 11, 19, 23, 29)

## Predictive MNAR Benchmark
- Best CRPS: `full_joint_jvi_training` under `value_dependent_mnar` @ `0.300000` with CRPS `4.265300` and gap CRPS `4.215453`

## Reliability Under Shift
- Best target-coverage error: `graph_corel` with coverage `0.904474` and shift coverage `0.897391`

## Region Holdout

## Fault Diagnosis
- Region holdout was not enabled for this benchmark config.
- Best F1: `pointwise_threshold_baseline` on `random_dropout` with F1 `0.725061`, AUROC `0.940167`, FAR `0.043926`

## Ablation Snapshot
- `base_gp_only`: RMSE `7.229191`, CRPS `5.835285`, Coverage ``, Cost `8.608436`
- `gp_plus_dynamic_silence`: RMSE `7.220513`, CRPS `5.828903`, Coverage ``, Cost `8.608436`
- `gp_plus_sensor_conditional_missingness`: RMSE `7.196317`, CRPS `5.800590`, Coverage ``, Cost `8.608436`
- `gp_plus_joint_generative_jvi_training`: RMSE `6.560259`, CRPS `4.876052`, Coverage ``, Cost `8.608436`
- `gp_plus_conformal_reliability`: RMSE `7.207981`, CRPS `5.814856`, Coverage `0.897219`, Cost `8.608436`
- `myopic_policy_baseline`: RMSE `5.546408`, CRPS `3.925396`, Coverage `0.891778`, Cost `8.120148`
- `full_model`: RMSE `6.452160`, CRPS `4.741419`, Coverage `0.884522`, Cost `7.818929`
- `base_gp_only`: RMSE `7.237229`, CRPS `5.843485`, Coverage ``, Cost `8.608436`
- `gp_plus_dynamic_silence`: RMSE `7.190478`, CRPS `5.796535`, Coverage ``, Cost `8.608436`
- `gp_plus_sensor_conditional_missingness`: RMSE `7.196515`, CRPS `5.800642`, Coverage ``, Cost `8.608436`
- `gp_plus_joint_generative_jvi_training`: RMSE `6.902174`, CRPS `5.378623`, Coverage ``, Cost `8.608436`
- `gp_plus_conformal_reliability`: RMSE `7.217757`, CRPS `5.823961`, Coverage `0.897219`, Cost `8.608436`
- `myopic_policy_baseline`: RMSE `6.391582`, CRPS `4.737782`, Coverage `0.908102`, Cost `8.120148`
- `full_model`: RMSE `6.668537`, CRPS `5.196549`, Coverage `0.879686`, Cost `7.818929`
- `base_gp_only`: RMSE `7.195190`, CRPS `5.798285`, Coverage ``, Cost `8.608436`
- `gp_plus_dynamic_silence`: RMSE `7.201308`, CRPS `5.807620`, Coverage ``, Cost `8.608436`
- `gp_plus_sensor_conditional_missingness`: RMSE `7.213708`, CRPS `5.819353`, Coverage ``, Cost `8.608436`
- `gp_plus_joint_generative_jvi_training`: RMSE `7.333810`, CRPS `6.241620`, Coverage ``, Cost `8.608436`
- `gp_plus_conformal_reliability`: RMSE `7.207867`, CRPS `5.812270`, Coverage `0.897219`, Cost `8.608436`
- `myopic_policy_baseline`: RMSE `6.568888`, CRPS `4.945817`, Coverage `0.892987`, Cost `8.120148`
- `full_model`: RMSE `6.349356`, CRPS `4.814246`, Coverage `0.883313`, Cost `7.818929`
- `base_gp_only`: RMSE `7.222320`, CRPS `5.825741`, Coverage ``, Cost `8.608436`
- `gp_plus_dynamic_silence`: RMSE `7.238393`, CRPS `5.844057`, Coverage ``, Cost `8.608436`
- `gp_plus_sensor_conditional_missingness`: RMSE `7.249380`, CRPS `5.854328`, Coverage ``, Cost `8.608436`
- `gp_plus_joint_generative_jvi_training`: RMSE `6.249286`, CRPS `4.634585`, Coverage ``, Cost `8.608436`
- `gp_plus_conformal_reliability`: RMSE `7.231081`, CRPS `5.838191`, Coverage `0.897219`, Cost `8.608436`
- `myopic_policy_baseline`: RMSE `6.714586`, CRPS `5.056236`, Coverage `0.892987`, Cost `8.120148`
- `full_model`: RMSE `6.995877`, CRPS `5.307842`, Coverage `0.876058`, Cost `7.818929`
- `base_gp_only`: RMSE `7.203700`, CRPS `5.810788`, Coverage ``, Cost `8.608436`
- `gp_plus_dynamic_silence`: RMSE `7.216016`, CRPS `5.822373`, Coverage ``, Cost `8.608436`
- `gp_plus_sensor_conditional_missingness`: RMSE `7.240239`, CRPS `5.844687`, Coverage ``, Cost `8.402580`
- `gp_plus_joint_generative_jvi_training`: RMSE `6.996527`, CRPS `5.580640`, Coverage ``, Cost `8.608436`
- `gp_plus_conformal_reliability`: RMSE `7.230766`, CRPS `5.838599`, Coverage `0.897219`, Cost `8.608436`
- `myopic_policy_baseline`: RMSE `7.019483`, CRPS `5.631348`, Coverage `0.883918`, Cost `8.120148`
- `full_model`: RMSE `6.044444`, CRPS `4.362093`, Coverage `0.884522`, Cost `7.818929`

## Policy Comparison
- Best cost-normalized gain: `variance_policy_baseline` with `0.279419`
