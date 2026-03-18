# Silence-Aware IDS Benchmark Report

## Canonical Setup
- Framework config: `C:\Programming Prctice\data\noaa_isd_taiwan_q1_2025\framework_isd_q1.json`
- Data path: `C:\Programming Prctice\data\noaa_isd_taiwan_q1_2025\noaa_isd_framework.csv`
- Target: `air_temperature`
- Rows: train=2729, calibration=587, evaluation=595
- Stations: 6, stride=3h, inducing=24
- Repeat seeds: 5 (7, 11, 19, 23, 29)

## Predictive MNAR Benchmark
- Best CRPS: `joint_generative_missingness` under `value_dependent_mnar` @ `0.500000` with CRPS `5.944893` and gap CRPS `6.109420`

## Reliability Under Shift
- Best target-coverage error: `graph_corel` with coverage `0.608403` and shift coverage `0.631336`

## Region Holdout

## Fault Diagnosis
- Region holdout was not enabled for this benchmark config.
- Best F1: `pi_ssd_only` on `spike_burst` with F1 `0.765957`, AUROC `0.995764`, FAR `0.019064`

## Ablation Snapshot
- `base_gp_only`: RMSE `19.720287`, CRPS `18.598227`, Coverage ``, Cost `8.569140`
- `gp_plus_dynamic_silence`: RMSE `19.720165`, CRPS `18.597979`, Coverage ``, Cost `8.569140`
- `gp_plus_sensor_conditional_missingness`: RMSE `19.718592`, CRPS `18.596842`, Coverage ``, Cost `8.569140`
- `gp_plus_joint_generative_jvi_training`: RMSE `12.914417`, CRPS `10.581114`, Coverage ``, Cost `8.781652`
- `gp_plus_conformal_reliability`: RMSE `19.715418`, CRPS `18.595068`, Coverage `0.900840`, Cost `8.569140`
- `myopic_policy_baseline`: RMSE `12.006884`, CRPS `9.688448`, Coverage `0.862185`, Cost `8.588649`
- `full_model`: RMSE `12.397338`, CRPS `10.512735`, Coverage `0.868908`, Cost `8.588649`
- `base_gp_only`: RMSE `19.718962`, CRPS `18.597027`, Coverage ``, Cost `8.569140`
- `gp_plus_dynamic_silence`: RMSE `19.717697`, CRPS `18.595911`, Coverage ``, Cost `8.569140`
- `gp_plus_sensor_conditional_missingness`: RMSE `19.718464`, CRPS `18.596497`, Coverage ``, Cost `8.569140`
- `gp_plus_joint_generative_jvi_training`: RMSE `10.350804`, CRPS `6.637396`, Coverage ``, Cost `8.781652`
- `gp_plus_conformal_reliability`: RMSE `19.718021`, CRPS `18.595615`, Coverage `0.902521`, Cost `8.569140`
- `myopic_policy_baseline`: RMSE `14.020349`, CRPS `10.807364`, Coverage `0.858824`, Cost `8.527013`
- `full_model`: RMSE `12.005708`, CRPS `10.045438`, Coverage `0.863866`, Cost `8.588649`
- `base_gp_only`: RMSE `19.718880`, CRPS `18.597010`, Coverage ``, Cost `8.569140`
- `gp_plus_dynamic_silence`: RMSE `19.718765`, CRPS `18.597498`, Coverage ``, Cost `8.569140`
- `gp_plus_sensor_conditional_missingness`: RMSE `19.717443`, CRPS `18.596117`, Coverage ``, Cost `8.569140`
- `gp_plus_joint_generative_jvi_training`: RMSE `15.021185`, CRPS `11.059279`, Coverage ``, Cost `8.781652`
- `gp_plus_conformal_reliability`: RMSE `19.716515`, CRPS `18.596062`, Coverage `0.900840`, Cost `8.569140`
- `myopic_policy_baseline`: RMSE `12.084611`, CRPS `8.700293`, Coverage `0.870588`, Cost `8.527013`
- `full_model`: RMSE `10.795730`, CRPS `8.762964`, Coverage `0.863866`, Cost `8.588649`
- `base_gp_only`: RMSE `19.718855`, CRPS `18.597775`, Coverage ``, Cost `8.569140`
- `gp_plus_dynamic_silence`: RMSE `19.719526`, CRPS `18.597315`, Coverage ``, Cost `8.569140`
- `gp_plus_sensor_conditional_missingness`: RMSE `19.719149`, CRPS `18.596920`, Coverage ``, Cost `8.569140`
- `gp_plus_joint_generative_jvi_training`: RMSE `13.449378`, CRPS `11.517153`, Coverage ``, Cost `8.781652`
- `gp_plus_conformal_reliability`: RMSE `19.720308`, CRPS `18.598196`, Coverage `0.902521`, Cost `8.569140`
- `myopic_policy_baseline`: RMSE `11.965003`, CRPS `9.884639`, Coverage `0.870588`, Cost `8.527013`
- `full_model`: RMSE `12.587328`, CRPS `10.884000`, Coverage `0.877311`, Cost `8.588649`
- `base_gp_only`: RMSE `19.719698`, CRPS `18.598238`, Coverage ``, Cost `8.569140`
- `gp_plus_dynamic_silence`: RMSE `19.718164`, CRPS `18.596403`, Coverage ``, Cost `8.569140`
- `gp_plus_sensor_conditional_missingness`: RMSE `19.719286`, CRPS `18.597368`, Coverage ``, Cost `8.569140`
- `gp_plus_joint_generative_jvi_training`: RMSE `11.000791`, CRPS `8.865929`, Coverage ``, Cost `8.781652`
- `gp_plus_conformal_reliability`: RMSE `19.718575`, CRPS `18.596779`, Coverage `0.902521`, Cost `8.569140`
- `myopic_policy_baseline`: RMSE `9.387270`, CRPS `7.370912`, Coverage `0.878992`, Cost `8.527013`
- `full_model`: RMSE `13.135927`, CRPS `11.362321`, Coverage `0.875630`, Cost `8.588649`

## Policy Comparison
- Best cost-normalized gain: `variance_policy_baseline` with `0.395043`
