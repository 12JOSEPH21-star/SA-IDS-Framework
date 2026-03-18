from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "review" / "canonical_paper_materials_20260317"
OUTPUT_DIR = ROOT / "reports" / "nmi_figures_20260318"

PAPER_COLORS = {
    "navy": "#1d3557",
    "blue": "#2563eb",
    "sky": "#60a5fa",
    "orange": "#d97706",
    "gold": "#f59e0b",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "gray": "#6b7280",
    "slate": "#334155",
    "mint": "#0f766e",
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.patch.set_facecolor("white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#475569",
            "axes.linewidth": 0.9,
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PAPER_COLORS["slate"],
    )


def render_figure_1() -> dict[str, str]:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.06),
            0.96,
            0.84,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=0.8,
            edgecolor="#dbe4f0",
            facecolor="#f8fbff",
        )
    )

    def add_box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.8,
            edgecolor=color,
            facecolor=color,
            alpha=0.12,
        )
        box.set_zorder(1)
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h * 0.74,
            title,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color=PAPER_COLORS["slate"],
            zorder=6,
        )
        ax.text(
            x + w / 2,
            y + h * 0.38,
            body,
            ha="center",
            va="center",
            fontsize=10,
            color="#334155",
            zorder=6,
        )

    def add_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        label: str | None = None,
        *,
        connectionstyle: str = "arc3,rad=0.0",
        label_xy: tuple[float, float] | None = None,
    ) -> None:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=PAPER_COLORS["slate"],
            connectionstyle=connectionstyle,
            zorder=2,
        )
        ax.add_patch(arrow)
        if label:
            if label_xy is None:
                mx = (start[0] + end[0]) / 2
                my = (start[1] + end[1]) / 2
                label_xy = (mx, my + 0.03)
            ax.text(
                label_xy[0],
                label_xy[1],
                label,
                ha="center",
                va="center",
                fontsize=8.8,
                color=PAPER_COLORS["slate"],
                zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
            )

    add_box(0.05, 0.56, 0.21, 0.22, "M1  Joint JVI / Sparse GP", "Latent state zₜ\nhealth state hₜ", "#8ecae6")
    add_box(0.33, 0.61, 0.21, 0.15, "Shared Silence-Aware Latent", "Forecast + observation\nprocess representation", "#bde0fe")
    add_box(0.59, 0.68, 0.18, 0.16, "M2 / M3 Diagnosis", "DBN + PI-SSD\nfault / silence cues", "#ffd6a5")
    add_box(0.60, 0.38, 0.20, 0.18, "M4 Active Sensing", "Policy / acquisition\nbudgeted selection", "#caffbf")
    add_box(0.81, 0.56, 0.15, 0.22, "M5 Reliability", "Adaptive / relational\nconformal intervals", "#ffcad4")
    add_box(0.08, 0.18, 0.25, 0.16, "Inputs", "Sensor streams\nmetadata\ncontext anchors", "#e9ecef")
    add_box(0.40, 0.11, 0.22, 0.18, "Outputs", "Predictions\nfault evidence\nselected sensors", "#e9ecef")

    add_arrow((0.24, 0.34), (0.17, 0.57), "X_t, y_t, c_t", label_xy=(0.18, 0.47))
    add_arrow((0.26, 0.67), (0.33, 0.70), connectionstyle="arc3,rad=0.02")
    add_arrow((0.54, 0.71), (0.59, 0.77), connectionstyle="arc3,rad=0.06")
    add_arrow((0.54, 0.63), (0.61, 0.49), connectionstyle="arc3,rad=-0.12")
    add_arrow((0.78, 0.76), (0.83, 0.70), connectionstyle="arc3,rad=0.02")
    add_arrow((0.78, 0.47), (0.84, 0.64), connectionstyle="arc3,rad=0.10")
    add_arrow((0.86, 0.59), (0.60, 0.21), "calibrated intervals", connectionstyle="arc3,rad=-0.10", label_xy=(0.77, 0.39))
    add_arrow((0.64, 0.40), (0.51, 0.30), "actions", connectionstyle="arc3,rad=0.02", label_xy=(0.59, 0.35))
    add_arrow((0.69, 0.68), (0.54, 0.31), "diagnosis", connectionstyle="arc3,rad=0.12", label_xy=(0.58, 0.47))
    add_arrow((0.50, 0.30), (0.22, 0.29), "feedback / labels", connectionstyle="arc3,rad=0.0", label_xy=(0.37, 0.32))

    ax.text(
        0.5,
        0.95,
        "SA-IDS Framework Overview",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=PAPER_COLORS["navy"],
        zorder=7,
    )
    ax.text(
        0.5,
        0.90,
        "A shared latent representation conditions prediction, diagnosis, sensing, and calibrated reliability.",
        ha="center",
        va="center",
        fontsize=10,
        color="#495057",
        zorder=7,
    )
    return _save_figure(fig, "fig_01_saids_framework")


def render_figure_2() -> dict[str, str]:
    _apply_paper_style()
    rows = _load_csv(CANONICAL_DIR / "figure_prediction_improvement_large.csv")
    mechanism_order = ["mar", "state_dependent_mnar", "value_dependent_mnar"]
    mechanism_label = {
        "mar": "MAR",
        "state_dependent_mnar": "State-Dependent MNAR",
        "value_dependent_mnar": "Value-Dependent MNAR",
    }
    variant_order = [
        "sensor_conditional_plugin_missingness",
        "joint_variational_missingness",
        "full_joint_jvi_training",
    ]
    variant_label = {
        "sensor_conditional_plugin_missingness": "Plug-in Baseline",
        "joint_variational_missingness": "Joint Variational",
        "full_joint_jvi_training": "Full Joint JVI",
    }
    colors = {
        "sensor_conditional_plugin_missingness": PAPER_COLORS["gray"],
        "joint_variational_missingness": PAPER_COLORS["blue"],
        "full_joint_jvi_training": PAPER_COLORS["orange"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for panel_label, ax, mechanism in zip(["A", "B", "C"], axes, mechanism_order):
        _add_panel_label(ax, panel_label)
        mech_rows = [row for row in rows if row["mechanism"] == mechanism and row["variant_name"] in variant_order]
        for variant in variant_order:
            variant_rows = sorted(
                (row for row in mech_rows if row["variant_name"] == variant),
                key=lambda row: float(row["missing_intensity"]),
            )
            x = [round(float(row["missing_intensity"]) * 100) for row in variant_rows]
            y = [_to_float(row["crps_mean"]) for row in variant_rows]
            yerr = [_to_float(row["crps_std"]) for row in variant_rows]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=2,
                capsize=4,
                label=variant_label[variant],
                color=colors[variant],
                markersize=5.5,
            )
        ax.set_title(mechanism_label[mechanism], fontsize=12, fontweight="bold")
        ax.set_xlabel("Missing Intensity (%)")
        ax.set_xticks([30, 50, 70])
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("CRPS")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=3)
    fig.suptitle("Fig. 2  CRPS across Missingness Mechanisms and Intensities", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return _save_figure(fig, "fig_02_crps_mechanism_intensity")


def render_figure_3() -> dict[str, str]:
    _apply_paper_style()
    rows = _load_csv(CANONICAL_DIR / "figure_region_holdout_large.csv")
    region_order = ["northeast", "northwest", "southeast", "southwest"]
    region_label = {
        "northeast": "Northeast",
        "northwest": "Northwest",
        "southeast": "Southeast",
        "southwest": "Southwest",
    }
    variant_order = ["base_gp_only", "full_model"]
    variant_label = {"base_gp_only": "Base GP", "full_model": "Full SA-IDS"}
    colors = {"base_gp_only": "#9ca3af", "full_model": "#2563eb"}

    fig, ax = plt.subplots(figsize=(10, 5))
    _add_panel_label(ax, "A")
    x_positions = list(range(len(region_order)))
    bar_width = 0.34
    full_means: list[float] = []
    base_means: list[float] = []
    for offset_index, variant in enumerate(variant_order):
        means = []
        stds = []
        for region in region_order:
            row = next(row for row in rows if row["variant_name"] == variant and row["holdout_region"] == region)
            means.append(float(row["crps_mean"]))
            stds.append(float(row["crps_std"]))
        if variant == "base_gp_only":
            base_means = means
        else:
            full_means = means
        shifted = [x + (offset_index - 0.5) * bar_width for x in x_positions]
        ax.bar(shifted, means, width=bar_width, color=colors[variant], label=variant_label[variant], alpha=0.9)
        ax.errorbar(shifted, means, yerr=stds, fmt="none", ecolor="#1f2937", capsize=4, linewidth=1.2)
    for xpos, base_val, full_val in zip(x_positions, base_means, full_means):
        ax.text(xpos, max(base_val, full_val) + 0.6, f"-{base_val - full_val:.1f}", ha="center", va="bottom", fontsize=9, color=PAPER_COLORS["navy"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels([region_label[name] for name in region_order])
    ax.set_ylabel("Held-out CRPS")
    ax.set_title("Fig. 3  Region-Holdout CRPS: Base GP vs Full SA-IDS", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, "fig_03_region_holdout_crps")


def render_figure_4() -> dict[str, str]:
    _apply_paper_style()
    benchmark_sources = [
        ("96-station", ROOT / "outputs" / "benchmark_joint_q1_large" / "paper_tables.json"),
        ("Korea", ROOT / "outputs" / "benchmark_korea_noaa_q1_medium" / "paper_tables.json"),
        ("Japan", ROOT / "outputs" / "benchmark_japan_q1_medium" / "paper_tables.json"),
        ("China", ROOT / "outputs" / "benchmark_china_q1_medium" / "paper_tables.json"),
        ("US", ROOT / "outputs" / "benchmark_us_q1_medium" / "paper_tables.json"),
    ]
    variant_order = [
        "no_conformal",
        "split_conformal",
        "adaptive_conformal",
        "relational_adaptive",
        "graph_corel",
    ]
    variant_label = {
        "no_conformal": "No Conformal",
        "split_conformal": "Split CP",
        "adaptive_conformal": "Adaptive CP",
        "relational_adaptive": "Relational ACI",
        "graph_corel": "Graph CoRel",
    }
    colors = {
        "no_conformal": PAPER_COLORS["gray"],
        "split_conformal": PAPER_COLORS["red"],
        "adaptive_conformal": PAPER_COLORS["gold"],
        "relational_adaptive": PAPER_COLORS["blue"],
        "graph_corel": PAPER_COLORS["purple"],
    }

    figure_rows: dict[str, dict[str, dict[str, float]]] = {}
    for benchmark_label, path in benchmark_sources:
        payload = _load_json(path)
        rows = payload.get("table_4_reliability_shift", [])
        figure_rows[benchmark_label] = {}
        for variant in variant_order:
            row = next((item for item in rows if item["variant_name"] == variant), None)
            if row is None:
                continue
            figure_rows[benchmark_label][variant] = {
                "coverage_mean": _to_float(row.get("coverage_mean")),
                "interval_width_mean": _to_float(row.get("interval_width_mean")),
            }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    x_positions = list(range(len(benchmark_sources)))
    x_labels = [label for label, _ in benchmark_sources]
    markers = ["o", "s", "^", "D", "P"]

    for panel_label, ax in zip(["A", "B"], axes):
        _add_panel_label(ax, panel_label)

    for idx, variant in enumerate(variant_order):
        coverage_values = []
        width_values = []
        for benchmark_label in x_labels:
            row = figure_rows[benchmark_label][variant]
            coverage_values.append(row["coverage_mean"])
            width_values.append(row["interval_width_mean"])
        axes[0].plot(
            x_positions,
            coverage_values,
            marker=markers[idx],
            linewidth=2,
            color=colors[variant],
            label=variant_label[variant],
            markersize=5.5,
        )
        axes[1].plot(
            x_positions,
            width_values,
            marker=markers[idx],
            linewidth=2,
            color=colors[variant],
            label=variant_label[variant],
            markersize=5.5,
        )

    axes[0].axhline(0.90, linestyle="--", color="#111827", linewidth=1.2, label="Target 0.90")
    axes[0].set_ylabel("Coverage")
    axes[1].set_ylabel("Mean Interval Width")
    for ax in axes:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_title("Coverage")
    axes[1].set_title("Interval Width")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Fig. 4  Reliability Calibration under Chronological Shift", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.9, 0.94))
    return _save_figure(fig, "fig_04_reliability_shift")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "figure_1": render_figure_1(),
        "figure_2": render_figure_2(),
        "figure_3": render_figure_3(),
        "figure_4": render_figure_4(),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_dirs": [
            str(CANONICAL_DIR),
            str(ROOT / "outputs" / "benchmark_joint_q1_large"),
            str(ROOT / "outputs" / "benchmark_korea_noaa_q1_medium"),
            str(ROOT / "outputs" / "benchmark_japan_q1_medium"),
            str(ROOT / "outputs" / "benchmark_china_q1_medium"),
            str(ROOT / "outputs" / "benchmark_us_q1_medium"),
        ],
        "figures": outputs,
    }
    (OUTPUT_DIR / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
