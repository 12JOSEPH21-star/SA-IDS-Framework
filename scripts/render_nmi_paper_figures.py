from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "review" / "canonical_paper_materials_20260317"
OUTPUT_DIR = ROOT / "reports" / "nmi_figures_20260318"

PAPER_COLORS = {
    "navy": "#16324f",
    "blue": "#2563eb",
    "sky": "#60a5fa",
    "teal": "#0f766e",
    "orange": "#ea580c",
    "gold": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "gray": "#6b7280",
    "slate": "#334155",
    "light": "#e2e8f0",
    "panel": "#f8fafc",
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
            "font.size": 10,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#475569",
            "axes.linewidth": 0.9,
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PAPER_COLORS["slate"],
    )


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    edge: str,
    face_alpha: float = 0.12,
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=edge,
        alpha=face_alpha,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=11.3,
        fontweight="bold",
        color=PAPER_COLORS["slate"],
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=9.4,
        color=PAPER_COLORS["slate"],
        zorder=4,
    )
    return (x, y, w, h)


def _anchor(box: tuple[float, float, float, float], side: str) -> tuple[float, float]:
    x, y, w, h = box
    points = {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
    }
    return points[side]


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    *,
    curve: float = 0.0,
    label_dx: float = 0.0,
    label_dy: float = 0.02,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color=PAPER_COLORS["slate"],
        connectionstyle=f"arc3,rad={curve}",
        zorder=1,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2 + label_dx
        my = (start[1] + end[1]) / 2 + label_dy
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=8.7,
            color=PAPER_COLORS["slate"],
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 0.18},
        )


def render_figure_1() -> dict[str, str]:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(13, 7.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.06),
            0.96,
            0.86,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            linewidth=0.8,
            edgecolor="#dbe4f0",
            facecolor="#f8fbff",
            zorder=0,
        )
    )

    inputs_box = _box(
        ax,
        0.06,
        0.36,
        0.17,
        0.22,
        "Inputs",
        "AWS / ASOS / NOAA\ncontext anchors\nmasks and gaps",
        "#cbd5e1",
    )
    m1_box = _box(
        ax,
        0.29,
        0.60,
        0.28,
        0.18,
        "M1  Sparse GP + Joint Observation Model",
        "Shared spatiotemporal state\nand informative missingness",
        "#8ecae6",
    )
    latent_box = _box(
        ax,
        0.31,
        0.28,
        0.24,
        0.18,
        "Shared Latent Representation",
        "state z_t, health h_t,\nsilence-aware features",
        "#bde0fe",
    )
    diag_box = _box(
        ax,
        0.66,
        0.69,
        0.24,
        0.13,
        "M2 / M3 Diagnosis",
        "DBN + PI-SSD fault evidence",
        "#ffd6a5",
    )
    sense_box = _box(
        ax,
        0.66,
        0.47,
        0.24,
        0.13,
        "M4 Active Sensing",
        "budgeted sensor selection",
        "#caffbf",
    )
    rel_box = _box(
        ax,
        0.66,
        0.25,
        0.24,
        0.13,
        "M5 Reliability",
        "adaptive and relational intervals",
        "#ffcad4",
    )
    outputs_box = _box(
        ax,
        0.30,
        0.08,
        0.26,
        0.11,
        "Outputs",
        "predictions, diagnostics,\nselected sensors, intervals",
        "#e2e8f0",
    )

    _arrow(ax, _anchor(inputs_box, "right"), _anchor(m1_box, "left"), "observations")
    _arrow(ax, _anchor(m1_box, "bottom"), _anchor(latent_box, "top"), "shared latent")
    _arrow(ax, _anchor(latent_box, "right"), _anchor(diag_box, "left"), "diagnosis cues", label_dy=0.03)
    _arrow(ax, _anchor(latent_box, "right"), _anchor(sense_box, "left"), "acquisition utility", label_dy=0.03)
    _arrow(ax, _anchor(latent_box, "right"), _anchor(rel_box, "left"), "uncertainty context", label_dy=0.03)
    _arrow(ax, _anchor(diag_box, "bottom"), _anchor(outputs_box, "right"), "fault evidence", curve=0.12, label_dx=0.03)
    _arrow(ax, _anchor(sense_box, "left"), _anchor(outputs_box, "right"), "sensor actions", curve=0.03, label_dx=0.03)
    _arrow(ax, _anchor(rel_box, "left"), _anchor(outputs_box, "right"), "calibrated intervals", curve=-0.10, label_dx=0.05)

    ax.text(
        0.5,
        0.95,
        "Fig. 1  SA-IDS architecture",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=PAPER_COLORS["navy"],
    )
    ax.text(
        0.5,
        0.905,
        "A shared observation-process representation conditions forecasting, diagnosis, sensing, and reliability control.",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )
    return _save_figure(fig, "fig_01_saids_framework")


def render_figure_2() -> dict[str, str]:
    _apply_paper_style()
    rows = _load_csv(CANONICAL_DIR / "prediction_large_joint_table.csv")
    mechanism_order = ["mar", "state_dependent_mnar", "value_dependent_mnar"]
    mechanism_label = {
        "mar": "MAR",
        "state_dependent_mnar": "State-dependent MNAR",
        "value_dependent_mnar": "Value-dependent MNAR",
    }
    variant_order = [
        "gp_only",
        "homogeneous_missingness",
        "sensor_conditional_plugin_missingness",
        "joint_variational_missingness",
        "joint_generative_missingness",
        "full_joint_jvi_training",
    ]
    variant_label = {
        "gp_only": "Base GP",
        "homogeneous_missingness": "Homogeneous",
        "sensor_conditional_plugin_missingness": "Sensor-conditional plugin",
        "joint_variational_missingness": "Joint variational",
        "joint_generative_missingness": "Joint generative",
        "full_joint_jvi_training": "Full joint JVI",
    }
    colors = {
        "gp_only": PAPER_COLORS["gray"],
        "homogeneous_missingness": PAPER_COLORS["slate"],
        "sensor_conditional_plugin_missingness": PAPER_COLORS["red"],
        "joint_variational_missingness": PAPER_COLORS["blue"],
        "joint_generative_missingness": PAPER_COLORS["teal"],
        "full_joint_jvi_training": PAPER_COLORS["orange"],
    }
    markers = {
        "gp_only": "o",
        "homogeneous_missingness": "s",
        "sensor_conditional_plugin_missingness": "D",
        "joint_variational_missingness": "^",
        "joint_generative_missingness": "P",
        "full_joint_jvi_training": "X",
    }
    linestyles = {
        "gp_only": "--",
        "homogeneous_missingness": "-.",
        "sensor_conditional_plugin_missingness": ":",
        "joint_variational_missingness": "-",
        "joint_generative_missingness": "-",
        "full_joint_jvi_training": "-",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), sharey=True)
    for panel_label, ax, mechanism in zip(["A", "B", "C"], axes, mechanism_order):
        _add_panel_label(ax, panel_label)
        for variant in variant_order:
            variant_rows = sorted(
                (
                    row
                    for row in rows
                    if row["benchmark_key"] == "large_joint"
                    and row["mechanism"] == mechanism
                    and row["variant_name"] == variant
                ),
                key=lambda row: float(row["missing_intensity"]),
            )
            x = [round(float(row["missing_intensity"]) * 100) for row in variant_rows]
            y = [float(row["crps_mean"]) for row in variant_rows]
            yerr_low = [float(row["crps_mean"]) - float(row["crps_ci_low"]) for row in variant_rows]
            yerr_high = [float(row["crps_ci_high"]) - float(row["crps_mean"]) for row in variant_rows]
            ax.errorbar(
                x,
                y,
                yerr=[yerr_low, yerr_high],
                marker=markers[variant],
                linestyle=linestyles[variant],
                linewidth=2.0 if variant in {"joint_variational_missingness", "full_joint_jvi_training"} else 1.5,
                capsize=3,
                label=variant_label[variant],
                color=colors[variant],
                markersize=5.5,
                alpha=0.95,
            )
        ax.set_title(mechanism_label[mechanism], fontsize=12, fontweight="bold")
        ax.set_xlabel("Missing intensity (%)")
        ax.set_xticks([30, 50, 70])
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    highlight_rows = {
        row["variant_name"]: row
        for row in rows
        if row["benchmark_key"] == "large_joint"
        and row["mechanism"] == "state_dependent_mnar"
        and row["missing_intensity"] == "0.700000"
        and row["variant_name"] in {"homogeneous_missingness", "full_joint_jvi_training"}
    }
    if "homogeneous_missingness" in highlight_rows and "full_joint_jvi_training" in highlight_rows:
        base_crps = float(highlight_rows["homogeneous_missingness"]["crps_mean"])
        jvi_crps = float(highlight_rows["full_joint_jvi_training"]["crps_mean"])
        axes[1].annotate(
            f"Delta CRPS = {base_crps - jvi_crps:.2f}",
            xy=(70, jvi_crps),
            xytext=(54, jvi_crps + 4.0),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": PAPER_COLORS["orange"]},
            fontsize=9,
            color=PAPER_COLORS["orange"],
            bbox={"facecolor": "white", "edgecolor": PAPER_COLORS["light"], "pad": 0.25},
        )

    axes[0].set_ylabel("CRPS")
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.suptitle("Fig. 2  Predictive CRPS across missingness mechanisms", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return _save_figure(fig, "fig_02_crps_mechanism_intensity")


def render_figure_3() -> dict[str, str]:
    _apply_paper_style()
    rows = _load_csv(CANONICAL_DIR / "region_holdout_large_joint_table.csv")
    region_order = ["northeast", "northwest", "southeast", "southwest"]
    region_label = {
        "northeast": "Northeast",
        "northwest": "Northwest",
        "southeast": "Southeast",
        "southwest": "Southwest",
    }
    region_colors = {
        "northeast": PAPER_COLORS["blue"],
        "northwest": PAPER_COLORS["teal"],
        "southeast": PAPER_COLORS["orange"],
        "southwest": PAPER_COLORS["purple"],
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    _add_panel_label(ax, "A")
    x_left, x_right = 0.0, 1.0

    for region in region_order:
        base_row = next(row for row in rows if row["variant_name"] == "base_gp_only" and row["holdout_region"] == region)
        full_row = next(row for row in rows if row["variant_name"] == "full_model" and row["holdout_region"] == region)
        base_mean = float(base_row["crps_mean"])
        full_mean = float(full_row["crps_mean"])
        base_std = float(base_row["crps_std"])
        full_std = float(full_row["crps_std"])
        stations = int(round(float(base_row["station_count_mean"])))
        color = region_colors[region]

        ax.plot([x_left, x_right], [base_mean, full_mean], color=color, linewidth=2.4, alpha=0.9)
        ax.scatter([x_left, x_right], [base_mean, full_mean], color=color, s=44, zorder=3)
        ax.errorbar([x_left], [base_mean], yerr=[[base_std], [base_std]], fmt="none", ecolor=color, capsize=3, linewidth=1.0)
        ax.errorbar([x_right], [full_mean], yerr=[[full_std], [full_std]], fmt="none", ecolor=color, capsize=3, linewidth=1.0)
        ax.text(x_left - 0.05, base_mean, f"{region_label[region]} (n={stations})", ha="right", va="center", fontsize=9.2, color=color)
        ax.text(x_right + 0.05, full_mean, f"{full_mean:.2f}", ha="left", va="center", fontsize=9.2, color=color)
        ax.text(0.5, (base_mean + full_mean) / 2 + 0.35, f"-{base_mean - full_mean:.2f}", ha="center", va="center", fontsize=8.9, color=color)

    ax.text(x_left, ax.get_ylim()[1] if ax.get_ylim() else 0, "")
    ax.set_xlim(-0.28, 1.28)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["Base GP", "Full model"])
    ax.set_ylabel("Held-out CRPS")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title("Fig. 3  Region holdout improvement across all four subnetworks", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, "fig_03_region_holdout_crps")


def _reliability_rows_from_paper_tables(path: Path) -> dict[str, dict]:
    payload = _load_json(path)
    return {row["variant_name"]: row for row in payload.get("table_4_reliability_shift", [])}


def render_figure_4() -> dict[str, str]:
    _apply_paper_style()
    large_rows = _load_csv(CANONICAL_DIR / "reliability_large_joint_table.csv")
    pilot_rows = _load_csv(CANONICAL_DIR / "external_pilots_table.csv")
    pilot_path_lookup = {
        "korea_noaa_medium": ROOT / "outputs" / "benchmark_korea_noaa_q1_medium" / "paper_tables.json",
        "taiwan_pilot": ROOT / "outputs" / "benchmark_taiwan_q1_pilot" / "paper_tables.json",
        "japan_medium": ROOT / "outputs" / "benchmark_japan_q1_medium" / "paper_tables.json",
        "china_medium": ROOT / "outputs" / "benchmark_china_q1_medium" / "paper_tables.json",
        "us_medium": ROOT / "outputs" / "benchmark_us_q1_medium" / "paper_tables.json",
    }
    pilot_display = {
        "korea_noaa_medium": "Korea",
        "taiwan_pilot": "Taiwan",
        "japan_medium": "Japan",
        "china_medium": "China",
        "us_medium": "US",
    }
    method_colors = {
        "no_conformal": PAPER_COLORS["gray"],
        "split_conformal": PAPER_COLORS["red"],
        "adaptive_conformal": PAPER_COLORS["gold"],
        "relational_adaptive": PAPER_COLORS["blue"],
        "graph_corel": PAPER_COLORS["purple"],
        "full_model": PAPER_COLORS["orange"],
    }
    method_labels = {
        "no_conformal": "No conformal",
        "split_conformal": "Split CP",
        "adaptive_conformal": "Adaptive CP",
        "relational_adaptive": "Relational adaptive",
        "graph_corel": "Graph CoRel",
        "full_model": "Full model",
    }
    pilot_colors = {
        "Korea": PAPER_COLORS["blue"],
        "Japan": PAPER_COLORS["orange"],
        "China": PAPER_COLORS["teal"],
        "US": PAPER_COLORS["purple"],
        "Taiwan": PAPER_COLORS["gray"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.6), sharey=True)
    for panel_label, ax in zip(["A", "B"], axes):
        _add_panel_label(ax, panel_label)
        ax.axhline(0.90, linestyle="--", linewidth=1.0, color="#111827")
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xlabel("Interval width")
        ax.set_ylabel("Coverage")
        ax.set_ylim(0.0, 1.0)

    large_variant_order = [
        "no_conformal",
        "split_conformal",
        "adaptive_conformal",
        "relational_adaptive",
        "graph_corel",
        "full_model",
    ]
    for variant in large_variant_order:
        row = next(row for row in large_rows if row["variant_name"] == variant)
        width = float(row["interval_width_mean"])
        coverage = float(row["coverage_mean"])
        axes[0].scatter(width, coverage, s=74, color=method_colors[variant], edgecolor="white", linewidth=0.8, zorder=3)
        axes[0].text(width + 0.45, coverage + 0.01, method_labels[variant], fontsize=8.6, color=method_colors[variant])
    axes[0].set_title("96-station benchmark methods", fontsize=12, fontweight="bold")
    axes[0].set_xlim(0, 25)

    pilot_points: list[tuple[str, float, float, str]] = []
    for row in pilot_rows:
        benchmark_key = row["benchmark_key"]
        if benchmark_key not in pilot_path_lookup:
            continue
        best_variant = row["best_reliability_variant"]
        paper_rows = _reliability_rows_from_paper_tables(pilot_path_lookup[benchmark_key])
        if best_variant not in paper_rows:
            continue
        best_row = paper_rows[best_variant]
        pilot_points.append(
            (
                pilot_display[benchmark_key],
                float(best_row["interval_width_mean"]),
                float(best_row["coverage_mean"]),
                best_variant,
            )
        )
    for label, width, coverage, variant in pilot_points:
        axes[1].scatter(width, coverage, s=74, color=pilot_colors[label], edgecolor="white", linewidth=0.8, zorder=3)
        axes[1].text(
            width + 1.3,
            coverage + 0.01,
            f"{label}\n{method_labels.get(variant, variant)}",
            fontsize=8.5,
            color=pilot_colors[label],
            va="center",
        )
    axes[1].set_title("Best reliability setting per Q1 pilot", fontsize=12, fontweight="bold")
    axes[1].set_xlim(0, 82)

    fig.suptitle("Fig. 4  Reliability trade-off: large-scale limit versus country pilots", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
            str(ROOT / "outputs" / "benchmark_taiwan_q1_pilot"),
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
