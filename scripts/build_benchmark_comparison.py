"""Build manuscript-facing benchmark comparison tables.

This script reads completed benchmark ``summary.json`` files and emits a
dataset-level comparison table in CSV/JSON/Markdown form for manuscript review.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\Programming Prctice")
REVIEW_DIR = ROOT / "review"

BENCHMARKS: list[tuple[str, Path]] = [
    ("Joint Korea Large", ROOT / "outputs" / "benchmark_joint_q1_large" / "summary.json"),
    (
        "Korea NOAA Medium",
        ROOT / "outputs" / "benchmark_korea_noaa_q1_medium" / "summary.json",
    ),
    ("Japan Medium", ROOT / "outputs" / "benchmark_japan_q1_medium" / "summary.json"),
    ("China Medium", ROOT / "outputs" / "benchmark_china_q1_medium" / "summary.json"),
    ("US Medium", ROOT / "outputs" / "benchmark_us_q1_medium" / "summary.json"),
]


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _best_row(rows: list[dict[str, Any]], key: str, reverse: bool = False) -> dict[str, Any] | None:
    valid = [row for row in rows if _to_float(row.get(key)) is not None]
    if not valid:
        return None
    return sorted(valid, key=lambda row: _to_float(row[key]), reverse=reverse)[0]


def _group_mean(rows: list[dict[str, Any]], group_key: str, metric_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metric = _to_float(row.get(metric_key))
        group = row.get(group_key)
        if metric is None or group in (None, ""):
            continue
        grouped[str(group)].append(metric)
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }


def _group_mean_with_filter(
    rows: list[dict[str, Any]],
    group_key: str,
    metric_key: str,
    **filters: Any,
) -> dict[str, float]:
    filtered = [
        row
        for row in rows
        if all(row.get(filter_key) == filter_value for filter_key, filter_value in filters.items())
    ]
    return _group_mean(filtered, group_key, metric_key)


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_record(dataset_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    canonical = summary["canonical_setup"]
    predictive_rows = summary.get("predictive_rows", [])
    fault_rows = summary.get("fault_rows", [])
    reliability_rows = summary.get("reliability_rows", [])
    ablation_rows = summary.get("ablation_rows", [])
    policy_rows = summary.get("policy_rows", [])
    region_holdout_rows = summary.get("region_holdout_rows", [])

    best_predictive = _best_row(predictive_rows, "crps")
    best_reliability = _best_row(reliability_rows, "target_coverage_error")
    best_fault = _best_row(fault_rows, "f1", reverse=True)
    best_policy = _best_row(policy_rows, "cost_normalized_gain", reverse=True)
    best_region_holdout = _best_row(region_holdout_rows, "crps")

    ablation_mean_crps = _group_mean(ablation_rows, "variant_name", "crps")
    best_ablation_variant = None
    best_ablation_crps = None
    if ablation_mean_crps:
        best_ablation_variant, best_ablation_crps = min(
            ablation_mean_crps.items(),
            key=lambda item: item[1],
        )

    full_model_ablation_rows = [row for row in ablation_rows if row.get("variant_name") == "full_model"]
    full_model_ablation_crps = _group_mean(full_model_ablation_rows, "variant_name", "crps").get("full_model")
    full_model_ablation_coverage = _group_mean(
        full_model_ablation_rows,
        "variant_name",
        "coverage",
    ).get("full_model")
    full_model_ablation_cost = _group_mean(
        full_model_ablation_rows,
        "variant_name",
        "total_cost",
    ).get("full_model")

    full_model_reliability_rows = [
        row for row in reliability_rows if row.get("variant_name") == "full_model"
    ]
    full_model_reliability_coverage = _group_mean(
        full_model_reliability_rows,
        "variant_name",
        "coverage",
    ).get("full_model")
    full_model_reliability_error = _group_mean(
        full_model_reliability_rows,
        "variant_name",
        "target_coverage_error",
    ).get("full_model")

    full_model_policy_rows = [row for row in policy_rows if row.get("variant_name") == "full_model"]
    full_model_policy_gain = _group_mean(
        full_model_policy_rows,
        "variant_name",
        "cost_normalized_gain",
    ).get("full_model")

    plugin_state_dep_07 = _group_mean_with_filter(
        predictive_rows,
        "variant_name",
        "crps",
        mechanism="state_dependent_mnar",
        missing_intensity="0.700000",
    ).get("sensor_conditional_plugin_missingness")
    joint_jvi_state_dep_07 = _group_mean_with_filter(
        predictive_rows,
        "variant_name",
        "crps",
        mechanism="state_dependent_mnar",
        missing_intensity="0.700000",
    ).get("full_joint_jvi_training")

    return {
        "dataset": dataset_name,
        "target": canonical["target_column"],
        "train_rows": canonical["train_rows"],
        "calibration_rows": canonical["calibration_rows"],
        "evaluation_rows": canonical["evaluation_rows"],
        "station_count": canonical["station_count"],
        "temporal_stride_hours": canonical["temporal_stride_hours"],
        "best_predictive_variant": best_predictive.get("variant_name") if best_predictive else "-",
        "best_predictive_mechanism": best_predictive.get("mechanism") if best_predictive else "-",
        "best_predictive_intensity": best_predictive.get("missing_intensity") if best_predictive else "-",
        "best_predictive_crps": _to_float(best_predictive.get("crps")) if best_predictive else None,
        "best_predictive_gap_crps": _to_float(best_predictive.get("gap_crps")) if best_predictive else None,
        "plugin_state_dep_mnar_0p7_crps": plugin_state_dep_07,
        "joint_jvi_state_dep_mnar_0p7_crps": joint_jvi_state_dep_07,
        "joint_jvi_minus_plugin_crps": (
            joint_jvi_state_dep_07 - plugin_state_dep_07
            if plugin_state_dep_07 is not None and joint_jvi_state_dep_07 is not None
            else None
        ),
        "best_reliability_variant": best_reliability.get("variant_name") if best_reliability else "-",
        "best_reliability_coverage": _to_float(best_reliability.get("coverage")) if best_reliability else None,
        "best_reliability_target_coverage_error": (
            _to_float(best_reliability.get("target_coverage_error")) if best_reliability else None
        ),
        "best_reliability_shift_coverage": (
            _to_float(best_reliability.get("shift_coverage")) if best_reliability else None
        ),
        "full_model_reliability_coverage": full_model_reliability_coverage,
        "full_model_reliability_error": full_model_reliability_error,
        "best_fault_variant": best_fault.get("variant_name") if best_fault else "-",
        "best_fault_scenario": best_fault.get("fault_scenario") if best_fault else "-",
        "best_fault_f1": _to_float(best_fault.get("f1")) if best_fault else None,
        "best_fault_auroc": _to_float(best_fault.get("auroc")) if best_fault else None,
        "best_fault_far": _to_float(best_fault.get("false_alarm_rate")) if best_fault else None,
        "best_policy_variant": best_policy.get("variant_name") if best_policy else "-",
        "best_policy_cost_normalized_gain": (
            _to_float(best_policy.get("cost_normalized_gain")) if best_policy else None
        ),
        "full_model_policy_cost_normalized_gain": full_model_policy_gain,
        "best_ablation_variant": best_ablation_variant or "-",
        "best_ablation_crps": best_ablation_crps,
        "full_model_ablation_crps": full_model_ablation_crps,
        "full_model_ablation_coverage": full_model_ablation_coverage,
        "full_model_ablation_cost": full_model_ablation_cost,
        "best_region_holdout_variant": (
            best_region_holdout.get("variant_name") if best_region_holdout else "-"
        ),
        "best_region_holdout_region": (
            best_region_holdout.get("holdout_region") if best_region_holdout else "-"
        ),
        "best_region_holdout_crps": (
            _to_float(best_region_holdout.get("crps")) if best_region_holdout else None
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# NMI Benchmark Comparison",
        "",
        "This table consolidates the completed main and external pilot benchmarks.",
        "",
        "| Dataset | Rows (train/cal/eval) | Stations | Predictive Best | JVI vs Plugin CRPS @ state-MNAR 0.7 | Reliability Best | Fault Best | Policy Best | Full Model Snapshot |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['dataset']} | "
            + f"{row['train_rows']}/{row['calibration_rows']}/{row['evaluation_rows']} | "
            + f"{row['station_count']} | "
            + f"{row['best_predictive_variant']} ({row['best_predictive_mechanism']} @ {row['best_predictive_intensity']}, CRPS {_format_float(row['best_predictive_crps'])}) | "
            + (
                f"plugin {_format_float(row['plugin_state_dep_mnar_0p7_crps'])}, "
                f"joint {_format_float(row['joint_jvi_state_dep_mnar_0p7_crps'])}, "
                f"delta {_format_float(row['joint_jvi_minus_plugin_crps'])}"
            )
            + " | "
            + (
                f"{row['best_reliability_variant']} "
                f"(cov {_format_float(row['best_reliability_coverage'])}, "
                f"err {_format_float(row['best_reliability_target_coverage_error'])})"
            )
            + " | "
            + (
                f"{row['best_fault_variant']} / {row['best_fault_scenario']} "
                f"(F1 {_format_float(row['best_fault_f1'])}, FAR {_format_float(row['best_fault_far'])})"
            )
            + " | "
            + (
                f"{row['best_policy_variant']} "
                f"(gain {_format_float(row['best_policy_cost_normalized_gain'])})"
            )
            + " | "
            + (
                f"ablation CRPS {_format_float(row['full_model_ablation_crps'])}, "
                f"cov {_format_float(row['full_model_ablation_coverage'])}, "
                f"policy {_format_float(row['full_model_policy_cost_normalized_gain'])}"
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `joint_jvi_minus_plugin_crps < 0` means the joint JVI path improved over the sensor-conditional plug-in baseline.",
            "- The large benchmark includes region-holdout rows; medium pilots do not.",
            "- `Full Model Snapshot` reports the mean ablation CRPS/coverage and the mean policy gain for `full_model`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_build_record(name, _load_summary(path)) for name, path in BENCHMARKS]
    csv_path = REVIEW_DIR / "NMI_BENCHMARK_COMPARISON_20260317.csv"
    json_path = REVIEW_DIR / "NMI_BENCHMARK_COMPARISON_20260317.json"
    md_path = REVIEW_DIR / "NMI_BENCHMARK_COMPARISON_20260317.md"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)
    _write_markdown(md_path, rows)
    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
