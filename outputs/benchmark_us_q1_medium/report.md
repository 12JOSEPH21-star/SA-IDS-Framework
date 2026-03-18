# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\noaa_isd_us_q1_2025\framework_isd_q1.json`
- Data path: `C:\Programming Prctice\data\noaa_isd_us_q1_2025\noaa_isd_framework.csv`
- Target: `air_temperature`
- Rows: train=5419, calibration=1130, evaluation=1124
- Stations: 16, stride=3h, inducing=24
- Repeat seeds: 5 (7, 11, 19, 23, 29)

## Predictive MNAR Benchmark
- Best CRPS: `joint_generative_missingness` under `state_dependent_mnar` @ `0.700000` with CRPS `6.814765` and gap CRPS `7.068513`

## Reliability Under Shift
- Best target-coverage error: `relational_adaptive` with coverage `0.871429` and shift coverage `0.912833`

## Region Holdout

## Fault Diagnosis
- Region holdout was not enabled for this benchmark config.
- Best F1: `dbn_lite` on `spike_burst` with F1 `0.795918`, AUROC `0.991965`, FAR `0.018433`

## Ablation Snapshot
- `base_gp_only`: RMSE `18.526169`, CRPS `16.557280`, Coverage ``, Cost `74.879486`
- `gp_plus_dynamic_silence`: RMSE `18.533680`, CRPS `16.561209`, Coverage ``, Cost `74.879486`
- `gp_plus_sensor_conditional_missingness`: RMSE `18.527601`, CRPS `16.557676`, Coverage ``, Cost `73.090271`
- `gp_plus_joint_generative_jvi_training`: RMSE `15.769935`, CRPS `13.359431`, Coverage ``, Cost `74.620667`
- `gp_plus_conformal_reliability`: RMSE `18.522547`, CRPS `16.553080`, Coverage `0.878571`, Cost `74.879486`
- `myopic_policy_baseline`: RMSE `15.642000`, CRPS `13.853723`, Coverage `0.893750`, Cost `5.004000`
- `full_model`: RMSE `17.834221`, CRPS `16.188263`, Coverage `0.861607`, Cost `21.249090`
- `base_gp_only`: RMSE `18.527113`, CRPS `16.556326`, Coverage ``, Cost `74.879486`
- `gp_plus_dynamic_silence`: RMSE `18.528477`, CRPS `16.557871`, Coverage ``, Cost `74.879486`
- `gp_plus_sensor_conditional_missingness`: RMSE `18.532104`, CRPS `16.561499`, Coverage ``, Cost `74.620667`
- `gp_plus_joint_generative_jvi_training`: RMSE `15.266973`, CRPS `13.127736`, Coverage ``, Cost `74.620667`
- `gp_plus_conformal_reliability`: RMSE `18.528025`, CRPS `16.557030`, Coverage `0.878571`, Cost `74.879486`
- `myopic_policy_baseline`: RMSE `16.596594`, CRPS `14.951921`, Coverage `0.852679`, Cost `5.004000`
- `full_model`: RMSE `16.372807`, CRPS `14.108650`, Coverage `0.863393`, Cost `21.249090`
- `base_gp_only`: RMSE `18.524622`, CRPS `16.553783`, Coverage ``, Cost `74.879486`
- `gp_plus_dynamic_silence`: RMSE `18.534328`, CRPS `16.563030`, Coverage ``, Cost `74.879486`
- `gp_plus_sensor_conditional_missingness`: RMSE `18.529278`, CRPS `16.559099`, Coverage ``, Cost `75.103409`
- `gp_plus_joint_generative_jvi_training`: RMSE `17.194128`, CRPS `15.649187`, Coverage ``, Cost `74.620667`
- `gp_plus_conformal_reliability`: RMSE `18.533720`, CRPS `16.562498`, Coverage `0.878571`, Cost `74.879486`
- `myopic_policy_baseline`: RMSE `15.647849`, CRPS `13.457753`, Coverage `0.875893`, Cost `5.004000`
- `full_model`: RMSE `13.658011`, CRPS `11.811138`, Coverage `0.861607`, Cost `21.249090`
- `base_gp_only`: RMSE `18.530001`, CRPS `16.559614`, Coverage ``, Cost `74.879486`
- `gp_plus_dynamic_silence`: RMSE `18.526131`, CRPS `16.555367`, Coverage ``, Cost `74.879486`
- `gp_plus_sensor_conditional_missingness`: RMSE `18.527748`, CRPS `16.557825`, Coverage ``, Cost `74.620667`
- `gp_plus_joint_generative_jvi_training`: RMSE `16.366880`, CRPS `14.175958`, Coverage ``, Cost `74.620667`
- `gp_plus_conformal_reliability`: RMSE `18.531864`, CRPS `16.560837`, Coverage `0.879464`, Cost `74.879486`
- `myopic_policy_baseline`: RMSE `16.346880`, CRPS `14.356853`, Coverage `0.885714`, Cost `5.004000`
- `full_model`: RMSE `16.393528`, CRPS `14.124181`, Coverage `0.894643`, Cost `21.249090`
- `base_gp_only`: RMSE `18.529053`, CRPS `16.560717`, Coverage ``, Cost `74.879486`
- `gp_plus_dynamic_silence`: RMSE `18.525171`, CRPS `16.557405`, Coverage ``, Cost `74.879486`
- `gp_plus_sensor_conditional_missingness`: RMSE `18.529757`, CRPS `16.558455`, Coverage ``, Cost `74.620667`
- `gp_plus_joint_generative_jvi_training`: RMSE `14.933599`, CRPS `12.801756`, Coverage ``, Cost `74.620667`
- `gp_plus_conformal_reliability`: RMSE `18.528883`, CRPS `16.559799`, Coverage `0.877679`, Cost `74.879486`
- `myopic_policy_baseline`: RMSE `17.952257`, CRPS `16.301891`, Coverage `0.867857`, Cost `5.004000`
- `full_model`: RMSE `16.024355`, CRPS `14.574674`, Coverage `0.859821`, Cost `21.249090`

## Policy Comparison
- Best cost-normalized gain: `myopic_policy_baseline` with `0.339562`
