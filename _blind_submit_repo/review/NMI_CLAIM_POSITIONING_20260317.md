# NMI Claim Positioning (2026-03-17)

This document converts the current benchmark evidence into manuscript-facing
claim guidance. It is intentionally conservative. The goal is to align the
paper with what is actually supported by the completed benchmark artifacts.

## Authoritative evidence sources

- Main benchmark:
  - `outputs/benchmark_joint_q1_large/summary.json`
  - `outputs/benchmark_joint_q1_large/paper_tables.json`
  - `outputs/benchmark_joint_q1_large/report.md`
- External pilots:
  - `outputs/benchmark_korea_noaa_q1_medium/report.md`
  - `outputs/benchmark_japan_q1_medium/report.md`
  - `outputs/benchmark_china_q1_medium/report.md`
  - `outputs/benchmark_us_q1_medium/report.md`
- Cross-benchmark comparison:
  - `review/NMI_BENCHMARK_COMPARISON_20260317.md`

## Recommended main claims

### 1. Informative missingness modeling improves predictive performance under MNAR

This is the strongest claim and should be the paper's primary result.

Supported wording:

- "Joint informative-missingness modeling improves predictive performance under
  heterogeneous missingness, especially under state-dependent MNAR settings."
- "Across the large joint benchmark, the joint JVI-style path materially
  improves over the sensor-conditional plug-in baseline under state-dependent
  MNAR."

Why this is safe:

- Large benchmark:
  - `joint_jvi_minus_plugin_crps = -7.4869` at state-MNAR 0.7
- China pilot:
  - `joint_jvi_minus_plugin_crps = -1.0566`
- US pilot:
  - `joint_jvi_minus_plugin_crps = -4.5096`

Qualification:

- Do not claim that the same proposed variant is uniformly best across every
  dataset and every mechanism. Japan already breaks that pattern.

### 2. Large-joint ablation supports the importance of M3/JVI-style missingness modeling

Supported wording:

- "Large-joint ablation indicates that informative missingness modeling is a
  major contributor to the overall performance gains."
- "The strongest ablation improvements come from the missingness-aware and
  joint-training variants rather than from the base GP alone."

Why this is safe:

- In `benchmark_joint_q1_large/report.md`, the strongest ablation CRPS is tied
  to `gp_plus_joint_generative_jvi_training`, not to `base_gp_only`.

Qualification:

- Keep this as an ablation contribution, not as a theoretical optimality claim.

### 3. Region/external generalization evidence is positive but should be framed as pilot validation

Supported wording:

- "External pilot benchmarks in Japan, China, and the US suggest that the
  informative-missingness improvements transfer beyond the main Korean joint
  benchmark."

Why this is safe:

- Joint/JVI-style variants remain competitive or best in Japan, China, and the
  US pilots.

Qualification:

- Use "pilot" or "external pilot" explicitly.
- Do not claim full cross-climate generalization. These are still NOAA-only
  medium-scale pilots.

## Recommended secondary claims

### 4. Reliability is promising, but the best method is dataset-dependent

Supported wording:

- "Reliability control is promising but dataset-dependent; different adaptive
  conformal variants perform best on different benchmarks."
- "The reliability module family improves coverage behavior, but the best
  conformal strategy is not yet stable enough to justify a single universal
  recommendation."

Why this is safe:

- Large: `graph_corel` is best by target-coverage error, but coverage remains
  only `0.7828`.
- Japan: `adaptive_conformal` is best.
- China: `graph_corel` is best.
- US: `relational_adaptive` is best.

What not to say:

- Do not claim that `graph_corel` is the clear best reliability method overall.
- Do not claim uniformly well-calibrated coverage across all datasets.

### 5. Diagnosis is useful supporting evidence, not a main scientific pillar

Supported wording:

- "Fault-diagnosis modules provide useful supporting evidence and perform
  competitively under several corruption scenarios."

Why this is safe:

- Large: `pi_ssd_only` reaches F1 `0.7373`
- US: `dbn_lite / spike_burst` reaches F1 `0.7959`

Qualification:

- Diagnosis winners differ by dataset and scenario.
- Keep diagnosis in a supporting section or supplemental emphasis, not as a
  headline contribution on par with JVI.

## Claims to downgrade

### 6. Policy / active sensing should be downgraded from headline to exploratory or secondary

Current evidence:

- Large benchmark best cost-normalized gain is `random`, not `full_model`.
- External pilots also do not consistently favor the proposed policy family.

Recommended wording:

- "The active-sensing component is included as an operational module and
  exploratory benchmark, but current evidence does not yet support it as a
  headline empirical advantage."

What to avoid:

- Do not claim the policy is superior to standard baselines across benchmarks.
- Do not claim the full model dominates in cost-normalized gain.

### 7. The full model should not be framed as uniformly best

Current evidence:

- In several tables, `full_model` is not the best predictive, reliability, or
  policy variant.
- The strongest evidence is modular rather than monolithic.

Recommended wording:

- "The full pipeline is operationally complete, but the strongest empirical
  evidence is currently modular: informative missingness modeling is the clearest
  gain, while reliability and policy remain more configuration-sensitive."

## Claims to remove unless new data arrives

### 8. Remove or soften any 'real NWP anchor' claim

Current data status:

- Historical Q1 `LDAPS/RDAPS` overlap is not present in the benchmarked Korean
  joint data path.
- The current practical anchor is still the ERA5/context fallback.

Allowed wording:

- "context-anchored"
- "ERA5-anchored"
- "reanalysis/context-assisted"

Avoid:

- "real forecast/NWP anchored" as a central validated claim

### 9. Remove any strong maintenance/outage-ground-truth claim

Current data status:

- QC/maintenance inputs are still proxy sidecars rather than official
  maintenance logs.

Avoid:

- "validated against official maintenance history"
- "explicitly grounded in operational outage records"

## Manuscript positioning recommendation

### Main claims

1. Joint informative missingness modeling improves prediction under MNAR.
2. Large-joint ablation confirms missingness-aware modeling is a major source of
   gain.
3. External pilots provide supporting evidence that the predictive advantage is
   not limited to a single dataset.

### Secondary claims

1. Adaptive reliability is promising but method selection is dataset-dependent.
2. Diagnosis provides useful supporting analysis under synthetic fault regimes.

### Claims to downgrade or remove

1. Policy / active sensing as a headline empirical win.
2. A single reliability method as universally best.
3. Real historical NWP-anchor validation.
4. Maintenance/outage-ground-truth validation.
5. Any statement that the full model is uniformly best end-to-end.

## Wording replacements

Prefer:

- "improves under MNAR"
- "large-joint benchmark"
- "supporting pilot evidence"
- "dataset-dependent reliability behavior"
- "exploratory active-sensing evidence"

Avoid:

- "consistently outperforms all baselines"
- "robust across all regimes"
- "universally calibrated"
- "operationally validated with real maintenance records"
- "real NWP anchored" without new data

## Immediate paper-edit implication

If the paper is being edited now, the safest structure is:

- Title / abstract / introduction:
  emphasize informative missingness and large-joint predictive gains
- Main results:
  lead with large-joint prediction + ablation
- Reliability:
  keep as an adaptive uncertainty-control study with mixed but positive evidence
- Diagnosis:
  move to supporting analysis
- Policy:
  demote to exploratory or supplemental unless re-tuned evidence improves
