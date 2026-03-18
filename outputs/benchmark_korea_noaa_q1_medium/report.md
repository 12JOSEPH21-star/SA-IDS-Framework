# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\noaa_isd_korea_q1_2025\framework_isd_q1.json`
- Data path: `C:\Programming Prctice\data\noaa_isd_korea_q1_2025\noaa_isd_framework.csv`
- Target: `air_temperature`
- Rows: train=5904, calibration=1286, evaluation=1273
- Stations: 16, stride=3h, inducing=24
- Repeat seeds: 5 (7, 11, 19, 23, 29)

## Predictive MNAR Benchmark
- Best CRPS: `joint_variational_missingness` under `mar` @ `0.300000` with CRPS `4.539888` and gap CRPS `4.383013`

## Reliability Under Shift
- Best target-coverage error: `graph_corel` with coverage `0.897013` and shift coverage `0.995506`

## Region Holdout

## Fault Diagnosis
- Region holdout was not enabled for this benchmark config.
- Best F1: `pointwise_threshold_baseline` on `random_dropout` with F1 `0.654762`, AUROC `0.947222`, FAR `0.066012`

## Ablation Snapshot
- `base_gp_only`: RMSE `11.264709`, CRPS `9.154143`, Coverage ``, Cost `10.487021`
- `gp_plus_dynamic_silence`: RMSE `11.240766`, CRPS `9.130675`, Coverage ``, Cost `10.487021`
- `gp_plus_sensor_conditional_missingness`: RMSE `11.290547`, CRPS `9.179177`, Coverage ``, Cost `9.430964`
- `gp_plus_joint_generative_jvi_training`: RMSE `9.468499`, CRPS `6.185333`, Coverage ``, Cost `9.949378`
- `gp_plus_conformal_reliability`: RMSE `11.292330`, CRPS `9.179677`, Coverage `0.813679`, Cost `10.487021`
- `myopic_policy_baseline`: RMSE `10.835401`, CRPS `8.886242`, Coverage `0.875786`, Cost `10.487021`
- `full_model`: RMSE `9.765540`, CRPS `8.034808`, Coverage `0.834906`, Cost `9.973104`
- `base_gp_only`: RMSE `11.250409`, CRPS `9.140000`, Coverage ``, Cost `10.487021`
- `gp_plus_dynamic_silence`: RMSE `11.270738`, CRPS `9.157365`, Coverage ``, Cost `10.487021`
- `gp_plus_sensor_conditional_missingness`: RMSE `11.286420`, CRPS `9.174506`, Coverage ``, Cost `9.511143`
- `gp_plus_joint_generative_jvi_training`: RMSE `10.549338`, CRPS `8.524007`, Coverage ``, Cost `10.487021`
- `gp_plus_conformal_reliability`: RMSE `11.235981`, CRPS `9.121474`, Coverage `0.813679`, Cost `10.487021`
- `myopic_policy_baseline`: RMSE `7.727931`, CRPS `5.729094`, Coverage `0.880503`, Cost `10.487021`
- `full_model`: RMSE `8.834408`, CRPS `6.913050`, Coverage `0.869497`, Cost `10.487021`
- `base_gp_only`: RMSE `11.284975`, CRPS `9.171829`, Coverage ``, Cost `10.487021`
- `gp_plus_dynamic_silence`: RMSE `11.265248`, CRPS `9.155031`, Coverage ``, Cost `10.487021`
- `gp_plus_sensor_conditional_missingness`: RMSE `11.302624`, CRPS `9.189945`, Coverage ``, Cost `9.766155`
- `gp_plus_joint_generative_jvi_training`: RMSE `9.732662`, CRPS `7.821117`, Coverage ``, Cost `10.487021`
- `gp_plus_conformal_reliability`: RMSE `11.256622`, CRPS `9.141748`, Coverage `0.813679`, Cost `10.487021`
- `myopic_policy_baseline`: RMSE `7.444597`, CRPS `5.515756`, Coverage `0.876572`, Cost `10.487021`
- `full_model`: RMSE `8.522129`, CRPS `6.569809`, Coverage `0.812107`, Cost `9.973104`
- `base_gp_only`: RMSE `11.263283`, CRPS `9.151057`, Coverage ``, Cost `10.487021`
- `gp_plus_dynamic_silence`: RMSE `11.272009`, CRPS `9.162828`, Coverage ``, Cost `10.487021`
- `gp_plus_sensor_conditional_missingness`: RMSE `11.248575`, CRPS `9.134597`, Coverage ``, Cost `9.782473`
- `gp_plus_joint_generative_jvi_training`: RMSE `11.008101`, CRPS `9.122947`, Coverage ``, Cost `10.487021`
- `gp_plus_conformal_reliability`: RMSE `11.266590`, CRPS `9.155327`, Coverage `0.813679`, Cost `10.487021`
- `myopic_policy_baseline`: RMSE `6.571212`, CRPS `4.809812`, Coverage `0.883648`, Cost `10.487021`
- `full_model`: RMSE `11.036248`, CRPS `9.110056`, Coverage `0.802673`, Cost `9.973104`
- `base_gp_only`: RMSE `11.287009`, CRPS `9.174297`, Coverage ``, Cost `10.487021`
- `gp_plus_dynamic_silence`: RMSE `11.256392`, CRPS `9.146762`, Coverage ``, Cost `10.487021`
- `gp_plus_sensor_conditional_missingness`: RMSE `11.288091`, CRPS `9.177169`, Coverage ``, Cost `9.511143`
- `gp_plus_joint_generative_jvi_training`: RMSE `9.667006`, CRPS `7.698256`, Coverage ``, Cost `10.487021`
- `gp_plus_conformal_reliability`: RMSE `11.239961`, CRPS `9.129460`, Coverage `0.813679`, Cost `10.487021`
- `myopic_policy_baseline`: RMSE `8.503911`, CRPS `5.303910`, Coverage `0.963836`, Cost `10.487021`
- `full_model`: RMSE `8.799706`, CRPS `6.746591`, Coverage `0.813679`, Cost `9.973104`

## Policy Comparison
- Best cost-normalized gain: `variance_policy_baseline` with `0.320336`
