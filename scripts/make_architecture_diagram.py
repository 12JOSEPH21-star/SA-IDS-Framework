"""
SA-IDS framework architecture diagram (Fig. 1 for EMS submission).

Usage:
    python scripts/make_architecture_diagram.py [--outdir paper_figures/ems_patch_v010]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


C = {
    "data": "#eceff1",
    "data_bd": "#607d8b",
    "m1": "#e3f2fd",
    "m1_bd": "#1565c0",
    "m2": "#fce4ec",
    "m2_bd": "#ad1457",
    "m3": "#e8f5e9",
    "m3_bd": "#2e7d32",
    "m4": "#fff8e1",
    "m4_bd": "#f57f17",
    "m5": "#f3e5f5",
    "m5_bd": "#6a1b9a",
    "out": "#e8eaf6",
    "out_bd": "#283593",
    "arr": "#424242",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.20,
    }
)

TITLE_KW = dict(fontsize=10.7, fontweight="bold", va="center", ha="center")
BODY_KW = dict(fontsize=8.7, va="center", ha="center", color="#263238", linespacing=1.45)
SUB_KW = dict(fontsize=7.8, va="center", ha="center", color="#546e7a", style="italic")
BADGE_KW = dict(fontsize=9.0, fontweight="bold", va="center", ha="center", color="white")


def _box(ax, xy, wh, fc, ec, lw=1.8, radius=0.018):
    x, y = xy
    w, h = wh
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(rect)
    return rect


def _badge(ax, cx, cy, label, fc, r=0.024):
    circle = plt.Circle((cx, cy), r, color=fc, zorder=5, clip_on=False)
    ax.add_patch(circle)
    ax.text(cx, cy, label, zorder=6, clip_on=False, **BADGE_KW)


def _arrow(ax, x0, y0, x1, y1, color=C["arr"], style="->", lw=1.6, connectionstyle="arc3,rad=0.0"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, connectionstyle=connectionstyle),
        zorder=4,
    )


def _dashed_arrow(ax, x0, y0, x1, y1, color=C["arr"], lw=1.2):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            linestyle="dashed",
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=4,
    )


def make_diagram(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.4, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "SA-IDS Framework: Silence-Aware Intelligent Data Selection",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#212121",
    )
    ax.text(
        0.5,
        0.915,
        "Sparse weather-network data are modelled jointly with silence, missingness, sensing, and reliability.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#455a64",
    )

    def block(x, y, w, h, title, body, footer, fill, edge, badge=None):
        _box(ax, (x, y), (w, h), fill, edge)
        if badge:
            _badge(ax, x + 0.025, y + h - 0.025, badge, edge)
        ax.text(x + w / 2, y + h * 0.76, title, **TITLE_KW, color=edge)
        ax.text(x + w / 2, y + h * 0.47, body, **BODY_KW)
        ax.text(x + w / 2, y + h * 0.15, footer, **SUB_KW)
        return (x, y, w, h)

    input_box = block(
        0.035,
        0.36,
        0.135,
        0.25,
        "Sensor Network\nData",
        "905 stations\nAWS / ASOS / ISD\ncontext + ERA5",
        "Observed values + masks",
        C["data"],
        C["data_bd"],
    )
    modules = [
        block(
            0.215,
            0.26,
            0.135,
            0.32,
            "Sparse\nVariational GP",
            "SVGP state model\n24 inducing points\nspatiotemporal kernel",
            "mean(x), var(x)",
            C["m1"],
            C["m1_bd"],
            "M1",
        ),
        block(
            0.385,
            0.26,
            0.135,
            0.32,
            "Dynamic Silence\nDiagnosis",
            "PI-SSD embeddings\nDBN-lite smoothing\ncorruption curriculum",
            "silence score s(t)",
            C["m2"],
            C["m2_bd"],
            "M2",
        ),
        block(
            0.555,
            0.26,
            0.135,
            0.32,
            "MNAR\nMissingness",
            "Joint generative VAE\nsensor-health latent\nensemble CRPS scoring",
            "p(R=1 | x,z)",
            C["m3"],
            C["m3_bd"],
            "M3",
        ),
        block(
            0.725,
            0.26,
            0.135,
            0.32,
            "Active Sensing\nPolicy",
            "MI-proxy utility\nlazy greedy / rollout\nPPO surrogate",
            "budgeted sensor set S*",
            C["m4"],
            C["m4_bd"],
            "M4",
        ),
    ]
    output_box = block(
        0.895,
        0.28,
        0.085,
        0.28,
        "Outputs",
        "forecasts\nintervals\ndiagnostics\nsensor choices",
        "review-ready artifacts",
        C["out"],
        C["out_bd"],
    )
    m5_box = block(
        0.415,
        0.675,
        0.405,
        0.17,
        "Conformal Reliability",
        "Adaptive and graph-aware calibration\ncoverage-width reporting at 90% target",
        "calibrated interval [l(x), u(x)]",
        C["m5"],
        C["m5_bd"],
        "M5",
    )

    def right(box):
        x, y, w, h = box
        return x + w, y + h / 2

    def left(box):
        x, y, w, h = box
        return x, y + h / 2

    def top(box):
        x, y, w, h = box
        return x + w / 2, y + h

    chain = [input_box, *modules, output_box]
    for start, end in zip(chain[:-1], chain[1:]):
        _arrow(ax, *right(start), *left(end))

    _dashed_arrow(ax, *top(modules[0]), 0.45, 0.675, color=C["m1_bd"])
    _dashed_arrow(ax, *top(modules[1]), 0.56, 0.675, color=C["m2_bd"])
    _dashed_arrow(ax, *top(modules[2]), 0.67, 0.675, color=C["m3_bd"])
    _arrow(ax, *right(m5_box), 0.895, 0.46, color=C["m5_bd"], lw=1.8)

    ax.text(0.30, 0.635, "posterior samples", fontsize=8.5, color=C["m1_bd"], style="italic")
    ax.text(0.455, 0.635, "silence evidence", fontsize=8.5, color=C["m2_bd"], style="italic")
    ax.text(0.595, 0.635, "missingness context", fontsize=8.5, color=C["m3_bd"], style="italic")

    ax.add_patch(
        FancyBboxPatch(
            (0.08, 0.08),
            0.84,
            0.095,
            boxstyle="round,pad=0.01,rounding_size=0.018",
            facecolor="#fafafa",
            edgecolor="#cfd8dc",
            linewidth=1.0,
            zorder=1,
        )
    )
    ax.text(
        0.50,
        0.128,
        "Configuration-driven workflow: framework-run produces summary.json, tables, diagnostics, and paper figures.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#37474f",
    )

    fig.tight_layout(rect=[0, 0, 1, 1])
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = outdir / f"fig_01_architecture.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.20)
        print(f"  [fig]   {p.name}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="paper_figures/ems_patch_v010")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    print(f"Output: {outdir}\n")
    make_diagram(outdir)
    print("\nDone.")


if __name__ == "__main__":
    main()
