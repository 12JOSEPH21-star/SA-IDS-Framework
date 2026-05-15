# NMI Benchmark Comparison

This table consolidates the completed main and external pilot benchmarks.

| Dataset | Rows (train/cal/eval) | Stations | Predictive Best | JVI vs Plugin CRPS @ state-MNAR 0.7 | Reliability Best | Fault Best | Policy Best | Full Model Snapshot |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| Joint Korea Large | 32768/8192/8192 | 96 | full_joint_jvi_training (state_dependent_mnar @ 0.300000, CRPS 3.1603) | plugin 15.4558, joint 7.9689, delta -7.4869 | graph_corel (cov 0.7828, err 0.1172) | pi_ssd_only / random_dropout (F1 0.7373, FAR 0.0644) | random (gain 1.5919) | ablation CRPS 4.2420, cov 0.7386, policy 0.8488 |
| Korea NOAA Medium | 5904/1286/1273 | 16 | joint_variational_missingness (mar @ 0.300000, CRPS 4.5399) | plugin 8.7480, joint 8.1177, delta -0.6303 | graph_corel (cov 0.8970, err 0.0030) | pointwise_threshold_baseline / random_dropout (F1 0.6548, FAR 0.0660) | variance_policy_baseline (gain 0.3203) | ablation CRPS 7.4749, cov 0.8266, policy 0.1587 |
| Japan Medium | 7645/1696/1728 | 16 | full_joint_jvi_training (value_dependent_mnar @ 0.300000, CRPS 1.6245) | plugin 3.1786, joint 3.4134, delta 0.2348 | adaptive_conformal (cov 0.8773, err 0.0227) | pointwise_threshold_baseline / random_dropout (F1 0.7484, FAR 0.0618) | variance_policy_baseline (gain 0.3604) | ablation CRPS 3.4410, cov 0.8810, policy 0.0894 |
| China Medium | 7793/1665/1655 | 16 | full_joint_jvi_training (value_dependent_mnar @ 0.300000, CRPS 4.2653) | plugin 6.4390, joint 5.3824, delta -1.0566 | graph_corel (cov 0.9045, err 0.0045) | pointwise_threshold_baseline / random_dropout (F1 0.7251, FAR 0.0439) | variance_policy_baseline (gain 0.2794) | ablation CRPS 4.8844, cov 0.8816, policy 0.1657 |
| US Medium | 5419/1130/1124 | 16 | joint_generative_missingness (state_dependent_mnar @ 0.700000, CRPS 6.8148) | plugin 17.1363, joint 12.6267, delta -4.5096 | relational_adaptive (cov 0.8714, err 0.0286) | dbn_lite / spike_burst (F1 0.7959, FAR 0.0184) | myopic_policy_baseline (gain 0.3396) | ablation CRPS 14.1614, cov 0.8682, policy 0.0788 |

## Notes

- `joint_jvi_minus_plugin_crps < 0` means the joint JVI path improved over the sensor-conditional plug-in baseline.
- The large benchmark includes region-holdout rows; medium pilots do not.
- `Full Model Snapshot` reports the mean ablation CRPS/coverage and the mean policy gain for `full_model`.
