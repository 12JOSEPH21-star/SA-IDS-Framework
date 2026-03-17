"""Regenerate canonical paper-facing benchmark tables from raw seed-level outputs."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\Programming Prctice")
DATE_STAMP = "20260317"
OUTPUT_DIR = ROOT / "review" / f"canonical_paper_materials_{DATE_STAMP}"


@dataclass(frozen=True)
class BenchmarkSpec:
    """Benchmark location and manuscript role."""

    key: str
    label: str
    summary_path: Path
    role: str
    preset_name: str


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        key="large_joint",
        label="Joint Korea Large",
        summary_path=ROOT / "outputs" / "benchmark_joint_q1_large" / "summary.json",
        role="main",
        preset_name="large",
    ),
    BenchmarkSpec(
        key="korea_noaa_medium",
        label="Korea NOAA Medium",
        summary_path=ROOT / "outputs" / "benchmark_korea_noaa_q1_medium" / "summary.json",
        role="external_pilot",
        preset_name="medium",
    ),
    BenchmarkSpec(
        key="taiwan_pilot",
        label="Taiwan Q1 Pilot",
        summary_path=ROOT / "outputs" / "benchmark_taiwan_q1_pilot" / "summary.json",
        role="external_pilot",
        preset_name="pilot",
    ),
    BenchmarkSpec(
        key="japan_medium",
        label="Japan Medium",
        summary_path=ROOT / "outputs" / "benchmark_japan_q1_medium" / "summary.json",
        role="external_pilot",
        preset_name="medium",
    ),
    BenchmarkSpec(
        key="china_medium",
        label="China Medium",
        summary_path=ROOT / "outputs" / "benchmark_china_q1_medium" / "summary.json",
        role="external_pilot",
        preset_name="medium",
    ),
    BenchmarkSpec(
        key="us_medium",
        label="US Medium",
        summary_path=ROOT / "outputs" / "benchmark_us_q1_medium" / "summary.json",
        role="external_pilot",
        preset_name="medium",
    ),
    BenchmarkSpec(
        key="korea_noaa_q2_safe",
        label="Korea NOAA Q2 Safe",
        summary_path=ROOT / "outputs" / "benchmark_korea_noaa_q2_safe" / "summary.json",
        role="seasonal_pilot",
        preset_name="safe",
    ),
    BenchmarkSpec(
        key="korea_noaa_q3_safe",
        label="Korea NOAA Q3 Safe",
        summary_path=ROOT / "outputs" / "benchmark_korea_noaa_q3_safe" / "summary.json",
        role="seasonal_pilot",
        preset_name="safe",
    ),
)

AGGREGATION_MODE = "multi_seed_mean_with_percentile_bootstrap_ci"
CI_LEVEL = 0.95
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SAMPLES = 4096
SIGNIFICANCE_ALPHA = 0.05
BOOTSTRAP_SEED = 20260317

LOWER_IS_BETTER = {
    "rmse": True,
    "mae": True,
    "crps": True,
    "log_score": True,
    "gaussian_coverage": False,
    "calibration_error": True,
    "gap_rmse": True,
    "gap_crps": True,
    "gap_fraction": False,
    "mean_synthetic_missingness_probability": False,
    "mean_predicted_missingness_probability": False,
    "coverage": False,
    "target_coverage_error": True,
    "interval_width": True,
    "stable_coverage": False,
    "shift_coverage": False,
    "stable_interval_width": True,
    "shift_interval_width": True,
    "recovery_speed_hours": True,
    "final_adaptive_epsilon": True,
    "mean_graph_quantile": True,
    "f1": False,
    "precision": False,
    "recall": False,
    "auroc": False,
    "detection_delay": True,
    "false_alarm_rate": True,
    "fault_fraction": False,
    "station_count": False,
    "evaluation_rows": False,
    "cumulative_information_gain": False,
    "uncertainty_reduction": False,
    "cumulative_energy_cost": True,
    "routing_cost": True,
    "total_operational_cost": True,
    "cost_normalized_gain": False,
    "num_selected": False,
    "mean_missingness_proba": False,
    "total_cost": True,
}

PREDICTIVE_METRICS = (
    "rmse",
    "mae",
    "crps",
    "log_score",
    "gaussian_coverage",
    "calibration_error",
    "gap_rmse",
    "gap_crps",
    "gap_fraction",
    "mean_synthetic_missingness_probability",
    "mean_predicted_missingness_probability",
)
RELIABILITY_METRICS = (
    "coverage",
    "target_coverage_error",
    "interval_width",
    "stable_coverage",
    "shift_coverage",
    "stable_interval_width",
    "shift_interval_width",
    "crps",
    "log_score",
    "recovery_speed_hours",
    "final_adaptive_epsilon",
    "mean_graph_quantile",
)
ABLATION_METRICS = (
    "rmse",
    "mae",
    "crps",
    "log_score",
    "coverage",
    "interval_width",
    "mean_missingness_proba",
    "total_cost",
    "num_selected",
)
REGION_HOLDOUT_METRICS = (
    "station_count",
    "evaluation_rows",
    "rmse",
    "mae",
    "crps",
    "coverage",
    "interval_width",
)
FAULT_METRICS = (
    "f1",
    "precision",
    "recall",
    "auroc",
    "detection_delay",
    "false_alarm_rate",
    "fault_fraction",
)
POLICY_METRICS = (
    "cumulative_information_gain",
    "uncertainty_reduction",
    "cumulative_energy_cost",
    "routing_cost",
    "total_operational_cost",
    "cost_normalized_gain",
    "num_selected",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty values.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_interval(values: list[float], *, samples: int, alpha: float, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap empty values.")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    return (
        _percentile(estimates, alpha / 2.0),
        _percentile(estimates, 1.0 - alpha / 2.0),
    )


def _best_single_seed(values: list[float], metric_key: str) -> float | None:
    if not values:
        return None
    lower_is_better = LOWER_IS_BETTER.get(metric_key, True)
    return min(values) if lower_is_better else max(values)


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: BenchmarkSpec,
    group_keys: tuple[str, ...],
    numeric_keys: tuple[str, ...],
    commit_hash: str | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group = tuple(row.get(key) for key in group_keys)
        grouped.setdefault(group, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        seed_list = sorted(
            {
                int(seed)
                for row in group_rows
                for seed in [row.get("repeat_seed")]
                if seed not in (None, "")
            }
        )
        output: dict[str, Any] = {
            key: value
            for key, value in zip(group_keys, group)
        }
        output.update(
            {
                "benchmark_key": benchmark.key,
                "benchmark_label": benchmark.label,
                "preset_name": benchmark.preset_name,
                "source_summary_path": str(benchmark.summary_path),
                "aggregation_mode": AGGREGATION_MODE,
                "ci_level": CI_LEVEL,
                "n_seeds": len(seed_list),
                "seed_list": ",".join(str(seed) for seed in seed_list),
                "commit_hash": commit_hash or "",
                "generated_at": _now_iso(),
            }
        )
        for numeric_key in numeric_keys:
            values = [
                numeric
                for row in group_rows
                for numeric in [_to_float(row.get(numeric_key))]
                if numeric is not None
            ]
            if not values:
                output[f"{numeric_key}_mean"] = ""
                output[f"{numeric_key}_std"] = ""
                output[f"{numeric_key}_ci_low"] = ""
                output[f"{numeric_key}_ci_high"] = ""
                output[f"{numeric_key}_best_single_seed"] = ""
                continue
            mean_value = sum(values) / len(values)
            std_value = statistics.stdev(values) if len(values) > 1 else 0.0
            ci_low, ci_high = _bootstrap_interval(
                values,
                samples=BOOTSTRAP_SAMPLES,
                alpha=BOOTSTRAP_ALPHA,
                seed=BOOTSTRAP_SEED + abs(hash((benchmark.key, group, numeric_key))) % 1_000_000,
            )
            output[f"{numeric_key}_mean"] = _format_float(mean_value)
            output[f"{numeric_key}_std"] = _format_float(std_value)
            output[f"{numeric_key}_ci_low"] = _format_float(ci_low)
            output[f"{numeric_key}_ci_high"] = _format_float(ci_high)
            output[f"{numeric_key}_best_single_seed"] = _format_float(_best_single_seed(values, numeric_key))
        aggregated.append(output)
    return aggregated


def _binomial_tail(successes: int, n: int) -> float:
    total = 0.0
    for k in range(0, successes + 1):
        total += math.comb(n, k)
    return total / (2**n)


def _paired_significance(
    rows: list[dict[str, Any]],
    *,
    benchmark: BenchmarkSpec,
    metric_key: str,
    baseline_variant: str,
    compare_variants: list[str] | None,
    group_keys: tuple[str, ...],
    lower_is_better: bool,
    commit_hash: str | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group = tuple(row.get(key) for key in group_keys)
        grouped.setdefault(group, []).append(row)

    outputs: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        by_variant_seed: dict[str, dict[int, float]] = {}
        for row in group_rows:
            variant_name = str(row.get("variant_name"))
            seed = row.get("repeat_seed")
            value = _to_float(row.get(metric_key))
            if value is None or seed in (None, ""):
                continue
            by_variant_seed.setdefault(variant_name, {})[int(seed)] = value

        baseline = by_variant_seed.get(baseline_variant)
        if not baseline:
            continue
        candidates = (
            [variant for variant in compare_variants if variant in by_variant_seed]
            if compare_variants is not None
            else [variant for variant in by_variant_seed if variant != baseline_variant]
        )
        for variant_name in candidates:
            variant = by_variant_seed.get(variant_name)
            if not variant:
                continue
            shared_seeds = sorted(set(baseline.keys()) & set(variant.keys()))
            deltas = []
            for seed in shared_seeds:
                delta = variant[seed] - baseline[seed]
                deltas.append(delta if lower_is_better else -delta)
            if not deltas:
                continue
            effect_mean = sum(deltas) / len(deltas)
            effect_ci_low, effect_ci_high = _bootstrap_interval(
                deltas,
                samples=BOOTSTRAP_SAMPLES,
                alpha=BOOTSTRAP_ALPHA,
                seed=BOOTSTRAP_SEED + abs(hash((benchmark.key, group, metric_key, variant_name))) % 1_000_000,
            )
            non_zero = [delta for delta in deltas if abs(delta) > 1e-12]
            positives = sum(1 for delta in non_zero if delta > 0.0)
            negatives = sum(1 for delta in non_zero if delta < 0.0)
            sign_count = positives + negatives
            if sign_count == 0:
                p_value = 1.0
            else:
                tail = _binomial_tail(min(positives, negatives), sign_count)
                p_value = min(1.0, 2.0 * tail)
            output = {
                key: value
                for key, value in zip(group_keys, group)
            }
            output.update(
                {
                    "benchmark_key": benchmark.key,
                    "benchmark_label": benchmark.label,
                    "source_summary_path": str(benchmark.summary_path),
                    "aggregation_mode": AGGREGATION_MODE,
                    "metric": metric_key,
                    "baseline_variant": baseline_variant,
                    "variant_name": variant_name,
                    "paired_sample_count": len(shared_seeds),
                    "seed_list": ",".join(str(seed) for seed in shared_seeds),
                    "effect_direction": "improvement" if lower_is_better else "gain",
                    "mean_delta": _format_float(effect_mean),
                    "ci_low": _format_float(effect_ci_low),
                    "ci_high": _format_float(effect_ci_high),
                    "p_value": _format_float(p_value),
                    "significant": p_value <= SIGNIFICANCE_ALPHA,
                    "alpha": SIGNIFICANCE_ALPHA,
                    "commit_hash": commit_hash or "",
                    "generated_at": _now_iso(),
                }
            )
            outputs.append(output)
    return outputs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _strongest_row(
    aggregated_rows: list[dict[str, Any]],
    *,
    metric_key: str,
    allowed_variants: set[str] | None = None,
) -> dict[str, Any] | None:
    valid = []
    for row in aggregated_rows:
        if allowed_variants is not None and row.get("variant_name") not in allowed_variants:
            continue
        value = _to_float(row.get(f"{metric_key}_mean"))
        if value is None:
            continue
        valid.append((value, row))
    if not valid:
        return None
    lower_is_better = LOWER_IS_BETTER.get(metric_key, True)
    valid.sort(key=lambda item: item[0], reverse=not lower_is_better)
    return valid[0][1]


def _predictive_family(variant_name: str) -> str:
    if variant_name in {"joint_variational_missingness", "joint_generative_missingness", "full_joint_jvi_training"}:
        return "joint_informative_missingness"
    if variant_name in {"homogeneous_missingness", "sensor_conditional_plugin_missingness"}:
        return "plugin_missingness"
    if variant_name == "gp_only":
        return "gp_only"
    return variant_name


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    commit_hash = _git_commit_hash()
    benchmark_payloads = {spec.key: _load_json(spec.summary_path) for spec in BENCHMARKS}

    sources = {
        "raw_benchmark_outputs": {spec.key: str(spec.summary_path) for spec in BENCHMARKS},
        "existing_summary_artifacts": {
            spec.key: {
                "paper_tables": str(spec.summary_path.with_name("paper_tables.json")),
                "significance_summary": str(spec.summary_path.with_name("significance_summary.json")),
                "report_markdown": str(spec.summary_path.with_name("report.md")),
            }
            for spec in BENCHMARKS
        },
    }
    _write_json(OUTPUT_DIR / "authoritative_sources.json", sources)

    aggregation_rule = {
        "aggregation_mode": AGGREGATION_MODE,
        "ci_level": CI_LEVEL,
        "bootstrap_alpha": BOOTSTRAP_ALPHA,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paper_primary_statistic": "multi-seed mean",
        "paper_uncertainty_statistic": "percentile bootstrap confidence interval over seed-level rows",
        "best_single_seed_rule": "reported for audit only; not used as the paper primary statistic",
        "significance_test": "paired exact sign test on per-seed deltas with bootstrap CI on the mean delta",
        "significance_alpha": SIGNIFICANCE_ALPHA,
        "prediction_primary_metric": "crps",
        "prediction_baseline_variant": "gp_only",
        "reliability_primary_metric": "target_coverage_error",
        "reliability_baseline_variant": "split_conformal",
        "generated_at": _now_iso(),
        "commit_hash": commit_hash or "",
    }
    _write_json(OUTPUT_DIR / "canonical_aggregation_rule.json", aggregation_rule)

    large_spec = BENCHMARKS[0]
    large_summary = benchmark_payloads[large_spec.key]
    large_predictive = _aggregate_rows(
        large_summary["predictive_rows"],
        benchmark=large_spec,
        group_keys=("variant_name", "pipeline_variant", "mechanism", "missing_intensity"),
        numeric_keys=PREDICTIVE_METRICS,
        commit_hash=commit_hash,
    )
    large_reliability = _aggregate_rows(
        large_summary["reliability_rows"],
        benchmark=large_spec,
        group_keys=("variant_name",),
        numeric_keys=RELIABILITY_METRICS,
        commit_hash=commit_hash,
    )
    large_ablation = _aggregate_rows(
        large_summary["ablation_rows"],
        benchmark=large_spec,
        group_keys=("variant_name",),
        numeric_keys=ABLATION_METRICS,
        commit_hash=commit_hash,
    )
    large_region_holdout = _aggregate_rows(
        large_summary["region_holdout_rows"],
        benchmark=large_spec,
        group_keys=("variant_name", "holdout_region"),
        numeric_keys=REGION_HOLDOUT_METRICS,
        commit_hash=commit_hash,
    )

    prediction_significance = _paired_significance(
        large_summary["predictive_rows"],
        benchmark=large_spec,
        metric_key="crps",
        baseline_variant="gp_only",
        compare_variants=[
            "homogeneous_missingness",
            "sensor_conditional_plugin_missingness",
            "joint_variational_missingness",
            "joint_generative_missingness",
            "full_joint_jvi_training",
        ],
        group_keys=("mechanism", "missing_intensity"),
        lower_is_better=True,
        commit_hash=commit_hash,
    )
    reliability_significance = _paired_significance(
        large_summary["reliability_rows"],
        benchmark=large_spec,
        metric_key="target_coverage_error",
        baseline_variant="split_conformal",
        compare_variants=["adaptive_conformal", "relational_adaptive", "graph_corel", "full_model"],
        group_keys=(),
        lower_is_better=True,
        commit_hash=commit_hash,
    ) + _paired_significance(
        large_summary["reliability_rows"],
        benchmark=large_spec,
        metric_key="interval_width",
        baseline_variant="split_conformal",
        compare_variants=["adaptive_conformal", "relational_adaptive", "graph_corel", "full_model"],
        group_keys=(),
        lower_is_better=True,
        commit_hash=commit_hash,
    )

    main_slice = [
        row
        for row in large_predictive
        if row.get("mechanism") == "state_dependent_mnar" and row.get("missing_intensity") == "0.700000"
    ]
    strongest_baseline = _strongest_row(
        main_slice,
        metric_key="crps",
        allowed_variants={"gp_only", "homogeneous_missingness", "sensor_conditional_plugin_missingness"},
    )
    strongest_proposed = _strongest_row(
        main_slice,
        metric_key="crps",
        allowed_variants={"joint_variational_missingness", "joint_generative_missingness", "full_joint_jvi_training"},
    )
    main_prediction_claim: list[dict[str, Any]] = []
    if strongest_baseline and strongest_proposed:
        baseline_crps = _to_float(strongest_baseline["crps_mean"])
        proposed_crps = _to_float(strongest_proposed["crps_mean"])
        delta = baseline_crps - proposed_crps if baseline_crps is not None and proposed_crps is not None else None
        main_prediction_claim.append(
            {
                "benchmark_key": large_spec.key,
                "benchmark_label": large_spec.label,
                "mechanism": "state_dependent_mnar",
                "missing_intensity": "0.700000",
                "aggregation_mode": AGGREGATION_MODE,
                "baseline_variant": strongest_baseline["variant_name"],
                "baseline_crps_mean": strongest_baseline["crps_mean"],
                "baseline_crps_ci_low": strongest_baseline["crps_ci_low"],
                "baseline_crps_ci_high": strongest_baseline["crps_ci_high"],
                "proposed_variant": strongest_proposed["variant_name"],
                "proposed_crps_mean": strongest_proposed["crps_mean"],
                "proposed_crps_ci_low": strongest_proposed["crps_ci_low"],
                "proposed_crps_ci_high": strongest_proposed["crps_ci_high"],
                "delta_crps_mean": _format_float(delta),
                "n_seeds": strongest_proposed["n_seeds"],
                "seed_list": strongest_proposed["seed_list"],
                "commit_hash": commit_hash or "",
                "generated_at": _now_iso(),
            }
        )

    external_rows: list[dict[str, Any]] = []
    for spec in BENCHMARKS[1:]:
        summary = benchmark_payloads[spec.key]
        predictive = _aggregate_rows(
            summary["predictive_rows"],
            benchmark=spec,
            group_keys=("variant_name", "pipeline_variant", "mechanism", "missing_intensity"),
            numeric_keys=PREDICTIVE_METRICS,
            commit_hash=commit_hash,
        )
        reliability = _aggregate_rows(
            summary["reliability_rows"],
            benchmark=spec,
            group_keys=("variant_name",),
            numeric_keys=RELIABILITY_METRICS,
            commit_hash=commit_hash,
        )
        faults = _aggregate_rows(
            summary["fault_rows"],
            benchmark=spec,
            group_keys=("variant_name", "fault_scenario"),
            numeric_keys=FAULT_METRICS,
            commit_hash=commit_hash,
        )
        policy = _aggregate_rows(
            summary["policy_rows"],
            benchmark=spec,
            group_keys=("variant_name",),
            numeric_keys=POLICY_METRICS,
            commit_hash=commit_hash,
        )
        best_predictive = _strongest_row(predictive, metric_key="crps")
        best_reliability = _strongest_row(reliability, metric_key="target_coverage_error")
        best_fault = _strongest_row(faults, metric_key="f1")
        best_policy = _strongest_row(policy, metric_key="cost_normalized_gain")
        external_rows.append(
            {
                "benchmark_key": spec.key,
                "benchmark_label": spec.label,
                "preset_name": spec.preset_name,
                "station_count": summary["canonical_setup"]["station_count"],
                "train_rows": summary["canonical_setup"]["train_rows"],
                "calibration_rows": summary["canonical_setup"]["calibration_rows"],
                "evaluation_rows": summary["canonical_setup"]["evaluation_rows"],
                "n_seeds": len(summary["canonical_setup"].get("repeat_seeds", [])),
                "seed_list": ",".join(str(seed) for seed in summary["canonical_setup"].get("repeat_seeds", [])),
                "best_predictive_variant": best_predictive["variant_name"] if best_predictive else "",
                "best_predictive_family": _predictive_family(str(best_predictive["variant_name"])) if best_predictive else "",
                "best_predictive_crps_mean": best_predictive["crps_mean"] if best_predictive else "",
                "best_predictive_crps_ci_low": best_predictive["crps_ci_low"] if best_predictive else "",
                "best_predictive_crps_ci_high": best_predictive["crps_ci_high"] if best_predictive else "",
                "best_reliability_variant": best_reliability["variant_name"] if best_reliability else "",
                "best_reliability_coverage_mean": best_reliability["coverage_mean"] if best_reliability else "",
                "best_reliability_target_coverage_error_mean": best_reliability["target_coverage_error_mean"] if best_reliability else "",
                "best_fault_variant": best_fault["variant_name"] if best_fault else "",
                "best_fault_scenario": best_fault["fault_scenario"] if best_fault else "",
                "best_fault_f1_mean": best_fault["f1_mean"] if best_fault else "",
                "best_policy_variant": best_policy["variant_name"] if best_policy else "",
                "best_policy_cost_normalized_gain_mean": best_policy["cost_normalized_gain_mean"] if best_policy else "",
                "source_summary_path": str(spec.summary_path),
                "aggregation_mode": AGGREGATION_MODE,
                "commit_hash": commit_hash or "",
                "generated_at": _now_iso(),
            }
        )

    significance_summary: list[dict[str, Any]] = []
    for row in prediction_significance + reliability_significance:
        significance_summary.append(
            {
                "benchmark_key": row["benchmark_key"],
                "benchmark_label": row["benchmark_label"],
                "comparison_family": "predictive_vs_base_gp" if "mechanism" in row else "reliability_vs_conformal",
                "group_label": f"{row['mechanism']} @ {row['missing_intensity']}" if "mechanism" in row else "all_seeds",
                "metric": row["metric"],
                "baseline_variant": row["baseline_variant"],
                "variant_name": row["variant_name"],
                "paired_sample_count": row["paired_sample_count"],
                "mean_delta": row["mean_delta"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "p_value": row["p_value"],
                "significant": row["significant"],
                "aggregation_mode": AGGREGATION_MODE,
                "commit_hash": commit_hash or "",
                "generated_at": _now_iso(),
            }
        )

    inconsistency_notes = []
    old_comparison_path = ROOT / "review" / "NMI_BENCHMARK_COMPARISON_20260317.json"
    if old_comparison_path.exists():
        inconsistency_notes.append(
            {
                "artifact": str(old_comparison_path),
                "issue": "Existing comparison file mixes best single rows and grouped means. Superseded by canonical tables.",
            }
        )
    old_significance_path = ROOT / "outputs" / "benchmark_joint_q1_large" / "significance_summary.json"
    if old_significance_path.exists():
        old_significance = _load_json(old_significance_path)
        if not old_significance.get("predictive_vs_base_gp") or not old_significance.get("reliability_vs_conformal"):
            inconsistency_notes.append(
                {
                    "artifact": str(old_significance_path),
                    "issue": "Existing predictive/reliability significance outputs are empty because baseline names did not match raw summary variants.",
                    "predictive_baseline_expected": "gp_only",
                    "reliability_baseline_expected": "split_conformal",
                }
            )

    _write_json(OUTPUT_DIR / "prediction_large_joint_table.json", large_predictive)
    _write_csv(OUTPUT_DIR / "prediction_large_joint_table.csv", large_predictive)
    _write_json(OUTPUT_DIR / "prediction_large_joint_claim_table.json", main_prediction_claim)
    _write_csv(OUTPUT_DIR / "prediction_large_joint_claim_table.csv", main_prediction_claim)
    _write_json(OUTPUT_DIR / "reliability_large_joint_table.json", large_reliability)
    _write_csv(OUTPUT_DIR / "reliability_large_joint_table.csv", large_reliability)
    _write_json(OUTPUT_DIR / "ablation_large_joint_table.json", large_ablation)
    _write_csv(OUTPUT_DIR / "ablation_large_joint_table.csv", large_ablation)
    _write_json(OUTPUT_DIR / "region_holdout_large_joint_table.json", large_region_holdout)
    _write_csv(OUTPUT_DIR / "region_holdout_large_joint_table.csv", large_region_holdout)
    _write_json(OUTPUT_DIR / "external_pilots_table.json", external_rows)
    _write_csv(OUTPUT_DIR / "external_pilots_table.csv", external_rows)
    _write_json(OUTPUT_DIR / "significance_prediction_vs_base_gp_large.json", prediction_significance)
    _write_csv(OUTPUT_DIR / "significance_prediction_vs_base_gp_large.csv", prediction_significance)
    _write_json(OUTPUT_DIR / "significance_reliability_vs_conformal_large.json", reliability_significance)
    _write_csv(OUTPUT_DIR / "significance_reliability_vs_conformal_large.csv", reliability_significance)
    _write_json(OUTPUT_DIR / "significance_summary_table.json", significance_summary)
    _write_csv(OUTPUT_DIR / "significance_summary_table.csv", significance_summary)
    _write_json(OUTPUT_DIR / "figure_prediction_improvement_large.json", large_predictive)
    _write_csv(OUTPUT_DIR / "figure_prediction_improvement_large.csv", large_predictive)
    _write_json(OUTPUT_DIR / "figure_region_holdout_large.json", large_region_holdout)
    _write_csv(OUTPUT_DIR / "figure_region_holdout_large.csv", large_region_holdout)
    _write_json(OUTPUT_DIR / "figure_reliability_pareto_large.json", large_reliability)
    _write_csv(OUTPUT_DIR / "figure_reliability_pareto_large.csv", large_reliability)
    _write_json(OUTPUT_DIR / "figure_external_pilot_comparison.json", external_rows)
    _write_csv(OUTPUT_DIR / "figure_external_pilot_comparison.csv", external_rows)
    _write_json(OUTPUT_DIR / "inconsistency_notes.json", inconsistency_notes)

    summary_payload = {
        "generated_at": _now_iso(),
        "commit_hash": commit_hash or "",
        "aggregation_rule_path": str(OUTPUT_DIR / "canonical_aggregation_rule.json"),
        "files": [
            "prediction_large_joint_table.csv",
            "prediction_large_joint_claim_table.csv",
            "reliability_large_joint_table.csv",
            "ablation_large_joint_table.csv",
            "region_holdout_large_joint_table.csv",
            "external_pilots_table.csv",
            "significance_summary_table.csv",
        ],
    }
    _write_json(OUTPUT_DIR / "canonical_materials_summary.json", summary_payload)

    audit_lines = [
        "# NMI Materials Audit",
        "",
        f"- Generated at: `{summary_payload['generated_at']}`",
        f"- Commit hash: `{commit_hash or 'unknown'}`",
        f"- Canonical aggregation rule: `{AGGREGATION_MODE}`",
        "",
        "## What Changed",
        "",
        "- Paper-facing numbers now come from raw benchmark `summary.json` seed-level rows only.",
        "- Existing `report.md` files remain diagnostic quick summaries and are not the canonical paper source.",
        "- Existing `NMI_BENCHMARK_COMPARISON_20260317.*` files are superseded because they mix single-row bests and grouped means.",
        "",
        "## Root Cause",
        "",
        "- Old predictive significance expected `base_gp_only`, but raw predictive rows use `variant_name=gp_only`.",
        "- Old reliability significance expected `gp_plus_conformal_reliability`, but raw reliability rows use `variant_name=split_conformal`.",
        "",
        "## Canonical Outputs",
        "",
        "- prediction_large_joint_table.csv",
        "- prediction_large_joint_claim_table.csv",
        "- reliability_large_joint_table.csv",
        "- ablation_large_joint_table.csv",
        "- region_holdout_large_joint_table.csv",
        "- external_pilots_table.csv",
        "- significance_summary_table.csv",
        "",
        "## Remaining Gaps",
        "",
        "- Full `framework-run` is still incomplete and should not be treated as primary evidence.",
        "- Historical NWP anchor evidence remains unavailable; current pipeline is effectively ERA5/context-anchored.",
        "- QC / maintenance / outage ground truth remains proxy-level only.",
    ]
    (OUTPUT_DIR / "nmi_materials_audit_20260317.md").write_text("\n".join(audit_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
