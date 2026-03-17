# NMI Materials Audit

- Generated at: `2026-03-17T11:29:38+00:00`
- Commit hash: `1a2a07695c53adf3fbb798818d9935e58e70517c`
- Canonical aggregation rule: `multi_seed_mean_with_percentile_bootstrap_ci`

## What Changed

- Paper-facing numbers now come from raw benchmark `summary.json` seed-level rows only.
- Existing `report.md` files remain diagnostic quick summaries and are not the canonical paper source.
- Existing `NMI_BENCHMARK_COMPARISON_20260317.*` files are superseded because they mix single-row bests and grouped means.

## Root Cause

- Old predictive significance expected `base_gp_only`, but raw predictive rows use `variant_name=gp_only`.
- Old reliability significance expected `gp_plus_conformal_reliability`, but raw reliability rows use `variant_name=split_conformal`.

## Canonical Outputs

- prediction_large_joint_table.csv
- prediction_large_joint_claim_table.csv
- reliability_large_joint_table.csv
- ablation_large_joint_table.csv
- region_holdout_large_joint_table.csv
- external_pilots_table.csv
- significance_summary_table.csv

## Remaining Gaps

- Full `framework-run` is still incomplete and should not be treated as primary evidence.
- Historical NWP anchor evidence remains unavailable; current pipeline is effectively ERA5/context-anchored.
- QC / maintenance / outage ground truth remains proxy-level only.