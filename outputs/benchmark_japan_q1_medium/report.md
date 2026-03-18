# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\noaa_isd_japan_q1_2025\framework_isd_q1.json`
- Data path: `C:\Programming Prctice\data\noaa_isd_japan_q1_2025\noaa_isd_framework.csv`
- Target: `air_temperature`
- Rows: train=7645, calibration=1696, evaluation=1728
- Stations: 16, stride=3h, inducing=24
- Repeat seeds: 5 (7, 11, 19, 23, 29)

## Predictive MNAR Benchmark
- Best CRPS: `full_joint_jvi_training` under `value_dependent_mnar` @ `0.300000` with CRPS `1.624523` and gap CRPS `1.738488`

## Reliability Under Shift
- Best target-coverage error: `adaptive_conformal` with coverage `0.877315` and shift coverage `0.969551`

## Region Holdout

## Fault Diagnosis
- Region holdout was not enabled for this benchmark config.
- Best F1: `pointwise_threshold_baseline` on `random_dropout` with F1 `0.748441`, AUROC `0.910744`, FAR `0.061801`

## Ablation Snapshot
- `base_gp_only`: RMSE `5.108225`, CRPS `4.048275`, Coverage ``, Cost `13.796070`
- `gp_plus_dynamic_silence`: RMSE `5.125135`, CRPS `4.065396`, Coverage ``, Cost `13.796070`
- `gp_plus_sensor_conditional_missingness`: RMSE `5.108288`, CRPS `4.047991`, Coverage ``, Cost `11.490646`
- `gp_plus_joint_generative_jvi_training`: RMSE `5.054405`, CRPS `3.559622`, Coverage ``, Cost `13.796070`
- `gp_plus_conformal_reliability`: RMSE `5.111902`, CRPS `4.050833`, Coverage `0.895833`, Cost `13.796070`
- `myopic_policy_baseline`: RMSE `2.928024`, CRPS `1.836494`, Coverage `0.880208`, Cost `13.244774`
- `full_model`: RMSE `5.185722`, CRPS `4.326187`, Coverage `0.880787`, Cost `11.713454`
- `base_gp_only`: RMSE `5.118057`, CRPS `4.055895`, Coverage ``, Cost `13.796070`
- `gp_plus_dynamic_silence`: RMSE `5.129132`, CRPS `4.072201`, Coverage ``, Cost `13.796070`
- `gp_plus_sensor_conditional_missingness`: RMSE `5.127991`, CRPS `4.069720`, Coverage ``, Cost `11.901304`
- `gp_plus_joint_generative_jvi_training`: RMSE `3.084117`, CRPS `2.084549`, Coverage ``, Cost `13.796070`
- `gp_plus_conformal_reliability`: RMSE `5.116445`, CRPS `4.057799`, Coverage `0.895833`, Cost `13.796070`
- `myopic_policy_baseline`: RMSE `4.396619`, CRPS `3.310157`, Coverage `0.876736`, Cost `13.244774`
- `full_model`: RMSE `4.132274`, CRPS `2.961962`, Coverage `0.876157`, Cost `12.501831`
- `base_gp_only`: RMSE `5.126360`, CRPS `4.065907`, Coverage ``, Cost `13.796070`
- `gp_plus_dynamic_silence`: RMSE `5.126945`, CRPS `4.066675`, Coverage ``, Cost `13.796070`
- `gp_plus_sensor_conditional_missingness`: RMSE `5.121864`, CRPS `4.063661`, Coverage ``, Cost `13.421179`
- `gp_plus_joint_generative_jvi_training`: RMSE `5.154034`, CRPS `4.306828`, Coverage ``, Cost `13.204985`
- `gp_plus_conformal_reliability`: RMSE `5.125392`, CRPS `4.065861`, Coverage `0.895833`, Cost `13.796070`
- `myopic_policy_baseline`: RMSE `3.426672`, CRPS `2.133584`, Coverage `0.878472`, Cost `13.244774`
- `full_model`: RMSE `3.549989`, CRPS `2.378319`, Coverage `0.887153`, Cost `12.501831`
- `base_gp_only`: RMSE `5.123250`, CRPS `4.061628`, Coverage ``, Cost `13.796070`
- `gp_plus_dynamic_silence`: RMSE `5.122859`, CRPS `4.063037`, Coverage ``, Cost `13.796070`
- `gp_plus_sensor_conditional_missingness`: RMSE `5.126141`, CRPS `4.065131`, Coverage ``, Cost `13.796070`
- `gp_plus_joint_generative_jvi_training`: RMSE `4.568480`, CRPS `3.411458`, Coverage ``, Cost `13.796070`
- `gp_plus_conformal_reliability`: RMSE `5.111160`, CRPS `4.050736`, Coverage `0.895833`, Cost `13.796070`
- `myopic_policy_baseline`: RMSE `4.262643`, CRPS `3.130061`, Coverage `0.880787`, Cost `13.244774`
- `full_model`: RMSE `5.138704`, CRPS `4.042417`, Coverage `0.880208`, Cost `12.501831`
- `base_gp_only`: RMSE `5.118402`, CRPS `4.057596`, Coverage ``, Cost `13.796070`
- `gp_plus_dynamic_silence`: RMSE `5.108163`, CRPS `4.046509`, Coverage ``, Cost `13.796070`
- `gp_plus_sensor_conditional_missingness`: RMSE `5.109531`, CRPS `4.048877`, Coverage ``, Cost `13.204985`
- `gp_plus_joint_generative_jvi_training`: RMSE `5.078766`, CRPS `3.980907`, Coverage ``, Cost `13.796070`
- `gp_plus_conformal_reliability`: RMSE `5.116661`, CRPS `4.056044`, Coverage `0.895833`, Cost `13.796070`
- `myopic_policy_baseline`: RMSE `5.125139`, CRPS `3.990921`, Coverage `0.870370`, Cost `13.244774`
- `full_model`: RMSE `4.678522`, CRPS `3.496026`, Coverage `0.880787`, Cost `12.501831`

## Policy Comparison
- Best cost-normalized gain: `variance_policy_baseline` with `0.360407`
