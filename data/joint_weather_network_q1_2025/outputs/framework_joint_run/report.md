# Silence-Aware IDS Framework Report

## Runtime
- Python executable: C:\Programming Prctice\.venv311\Scripts\python.exe
- Python version: 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
- Torch: 2.10.0+cpu
- GPyTorch: 1.15.2

## Reproducibility
- Seed: 7
- Deterministic algorithms: False
- Torch threads: 4
- Matmul precision: high

## Dataset
- Train rows: 131072
- Calibration rows: 32768
- Evaluation rows: 32768
- Input dimension: 7
- Context dimension: 16
- Metadata cardinalities: {"sensor_type": 3, "sensor_group": 1, "sensor_modality": 2, "installation_environment": 3, "maintenance_state": 1}

## Base Metrics
- RMSE: 10.584022
- MAE: 5.299886
- CRPS: 4.390200
- Log-Score: 4.113341
- Coverage: 0.865784
- Interval Width: 26.554470
- Observation-link steps: 25

## Ablations
| Variant | Assumption | Mode | Inference | State Train | Policy | Planner | RMSE | CRPS | Log-Score | Coverage |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| base_gp_only | disabled | selection | joint_generative | sequential | mi_proxy | lazy_greedy | 14.972314 | 8.915380 | 6.230020 |  |
| gp_plus_dynamic_silence | disabled | selection | joint_generative | sequential | mi_proxy | lazy_greedy | 15.018577 | 8.968994 | 6.244637 |  |
| gp_plus_homogeneous_missingness | homogeneous | selection | plug_in | sequential | mi_proxy | lazy_greedy | 14.984511 | 8.925834 | 6.221289 |  |
| gp_plus_sensor_conditional_missingness | sensor_conditional | selection | plug_in | sequential | mi_proxy | lazy_greedy | 14.987001 | 8.933031 | 6.236806 |  |
| gp_plus_joint_variational_missingness | sensor_conditional | selection | joint_variational | sequential | mi_proxy | lazy_greedy | 9.595656 | 3.626455 | 8.513081 |  |
| gp_plus_joint_jvi_training | sensor_conditional | selection | joint_variational | joint_variational | mi_proxy | lazy_greedy | 7.220633 | 1.947899 | 3.577593 |  |
| gp_plus_joint_generative_missingness | sensor_conditional | selection | joint_generative | sequential | mi_proxy | lazy_greedy | 10.299322 | 4.226414 | 13.060030 |  |
| gp_plus_joint_generative_jvi_training | sensor_conditional | selection | joint_generative | joint_generative | mi_proxy | lazy_greedy | 8.506250 | 2.568295 | 16.177700 |  |
| gp_plus_pattern_mixture_missingness | sensor_conditional | pattern_mixture | plug_in | sequential | mi_proxy | lazy_greedy | 14.917412 | 8.850182 | 6.207430 |  |
| gp_plus_conformal_reliability | disabled | selection | joint_generative | sequential | mi_proxy | lazy_greedy | 14.996703 | 8.942399 | 6.233453 | 0.913574 |
| relational_reliability_baseline | disabled | selection | joint_generative | sequential | mi_proxy | lazy_greedy | 15.021276 | 8.974325 | 6.253019 | 0.896576 |
| myopic_policy_baseline | sensor_conditional | selection | joint_generative | joint_generative | mi_proxy | lazy_greedy | 8.038019 | 2.581950 | 11.409929 | 0.867920 |
| ppo_warmstart_baseline | sensor_conditional | selection | joint_generative | joint_generative | mi_proxy | ppo_warmstart | 8.396911 | 3.928145 | 3.575854 | 0.874512 |
| rollout_policy_baseline | sensor_conditional | selection | joint_generative | joint_generative | mi_proxy | non_myopic_rollout | 10.996949 | 4.217385 | 4.682009 | 0.873372 |
| variance_policy_baseline | sensor_conditional | selection | joint_generative | joint_generative | variance | lazy_greedy | 8.710786 | 2.898466 | 17.147051 | 0.871094 |
| full_model | sensor_conditional | selection | joint_generative | joint_generative | mi_proxy | ppo_online | 5.570463 | 1.843512 | 47.755199 | 0.867676 |

## Missingness Sensitivity
| Setting | RMSE | CRPS | Log-Score | Mean Missingness |
| --- | ---: | ---: | ---: | ---: |
| logit_scale_0.500 | 10.377397 | 4.358962 | 4.068381 | 0.006317 |
| logit_scale_1.000 | 10.373059 | 4.352098 | 4.066350 | 0.000111 |
| logit_scale_1.500 | 10.373015 | 4.352092 | 4.066421 | 0.000100 |
| logit_scale_2.000 | 10.373011 | 4.352092 | 4.066429 | 0.000100 |

## Benchmark
- Expansion factor: 4
- Benchmark rows: 65536
- Prediction batch size: 512
- Predict seconds: 0.514019
- Missingness seconds: 0.714886
- Selection seconds: 3.407353
- Input tensor bytes: 11796480
- Output tensor bytes: 786432
- Mean predictive variance: 3.048299
- Mean missingness probability: 0.000195

## Selection
- base_policy:base_policy rank 1 -> index 6413 (utility=3.715111, total_cost=3.000000)
- base_policy:base_policy rank 2 -> index 32508 (utility=3.065953, total_cost=3.000000)
- base_policy:base_policy rank 3 -> index 252 (utility=3.064676, total_cost=3.000000)
- ablation:base_gp_only rank 1 -> index 0 (utility=1.837657, total_cost=3.000000)
- ablation:base_gp_only rank 2 -> index 32508 (utility=1.837657, total_cost=3.000000)
- ablation:base_gp_only rank 3 -> index 376 (utility=1.836890, total_cost=3.000000)
- ablation:gp_plus_dynamic_silence rank 1 -> index 0 (utility=1.821416, total_cost=3.000000)
- ablation:gp_plus_dynamic_silence rank 2 -> index 32508 (utility=1.821416, total_cost=3.000000)
- ablation:gp_plus_dynamic_silence rank 3 -> index 376 (utility=1.820656, total_cost=3.000000)
- ablation:gp_plus_homogeneous_missingness rank 1 -> index 0 (utility=1.835694, total_cost=3.000000)
- ablation:gp_plus_homogeneous_missingness rank 2 -> index 32508 (utility=1.835694, total_cost=3.000000)
- ablation:gp_plus_homogeneous_missingness rank 3 -> index 376 (utility=1.834920, total_cost=3.000000)
