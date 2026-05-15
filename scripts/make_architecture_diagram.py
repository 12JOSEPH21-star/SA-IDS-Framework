"""
SA-IDS Framework architecture diagram (Fig. 1 for EMS submission).

Usage:
    python scripts/make_architecture_diagram.py [--outdir paper_figures/ems_patch_v010]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ─────────────────────────────────────────────────────────────
C = {
    "data":    "#eceff1",
    "data_bd": "#607d8b",
    "m1":      "#e3f2fd",
    "m1_bd":   "#1565c0",
    "m2":      "#fce4ec",
    "m2_bd":   "#ad1457",
    "m3":      "#e8f5e9",
    "m3_bd":   "#2e7d32",
    "m4":      "#fff8e1",
    "m4_bd":   "#f57f17",
    "m5":      "#f3e5f5",
    "m5_bd":   "#6a1b9a",
    "out":     "#e8eaf6",
    "out_bd":  "#283593",
    "arr":     "#424242",
    "sub":     "#78909c",
}

TITLE_KW  = dict(fontsize=9,  fontweight="bold", va="center", ha="center")
BODY_KW   = dict(fontsize=7.5, va="center",       ha="center", color="#37474f",
                 linespacing=1.5)
SUB_KW    = dict(fontsize=6.5, va="center",       ha="center", color=C["sub"],
                 style="italic")
BADGE_KW  = dict(fontsize=8,   fontweight="bold", va="center", ha="center",
                 color="white")


def _box(ax, xy, wh, fc, ec, lw=1.4, radius=0.02):
    x, y = xy
    w, h = wh
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(rect)
    return rect


def _badge(ax, cx, cy, label, fc, r=0.028):
    circle = plt.Circle((cx, cy), r, color=fc, zorder=5, clip_on=False)
    ax.add_patch(circle)
    ax.text(cx, cy, label, zorder=6, clip_on=False, **BADGE_KW)


def _arrow(ax, x0, y0, x1, y1, color=C["arr"], style="->", lw=1.2,
           connectionstyle="arc3,rad=0.0"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        connectionstyle=connectionstyle),
        zorder=4,
    )


def _dashed_arrow(ax, x0, y0, x1, y1, color=C["arr"], lw=1.0):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        linestyle="dashed",
                        connectionstyle="arc3,rad=0.0"),
        zorder=4,
    )


def make_diagram(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── layout constants ───────────────────────────────────────────────────────
    #   columns  x:  0.02  0.18  0.38  0.58  0.78  0.92
    #   rows     y:  top=0.82  mid=0.48  bot=0.14

    COL  = [0.02, 0.175, 0.375, 0.575, 0.775, 0.905]
    BW   = 0.155   # box width
    BH   = 0.30    # box height for module boxes
    BH_D = 0.12    # data / output box height

    # ── INPUT DATA box ─────────────────────────────────────────────────────────
    ix, iy = 0.02, 0.44
    _box(ax, (ix, iy), (BW, BH_D * 1.6), C["data"], C["data_bd"])
    ax.text(ix + BW/2, iy + BH_D * 1.6 * 0.72,
            "Sensor Network\nData", **TITLE_KW, color=C["data_bd"])
    ax.text(ix + BW/2, iy + BH_D * 1.6 * 0.32,
            "905 stations\n(AWS · ASOS · ISD)\nwith context + ERA5",
            **{**BODY_KW, "fontsize": 6.8})

    # sensor dots (stylised)
    dots_x = np.linspace(ix + 0.018, ix + BW - 0.018, 6)
    for i, dx in enumerate(dots_x):
        alpha = 0.35 if i in (1, 4) else 0.85
        col   = "#ef9a9a" if i in (1, 4) else C["data_bd"]
        ax.plot(dx, iy - 0.038, "o", ms=5, color=col, alpha=alpha, zorder=4)
        ax.plot([dx, dx], [iy - 0.038, iy - 0.006],
                color=col, lw=0.8, alpha=alpha, zorder=3)
    ax.text(ix + BW/2, iy - 0.058,
            "missing / silenced", fontsize=6, color="#e53935",
            ha="center", va="center", style="italic")

    # ── arrow: data → M1 ──────────────────────────────────────────────────────
    _arrow(ax, ix + BW, iy + BH_D * 0.8, COL[1], iy + BH_D * 0.8)

    # ── M1: Sparse Variational GP ──────────────────────────────────────────────
    mx1 = COL[1]
    _box(ax, (mx1, 0.10), (BW, BH), C["m1"], C["m1_bd"])
    _badge(ax, mx1 + 0.018, 0.10 + BH - 0.018, "M1", C["m1_bd"])
    ax.text(mx1 + BW/2, 0.10 + BH * 0.80,
            "Sparse Variational GP", **TITLE_KW, color=C["m1_bd"])
    ax.text(mx1 + BW/2, 0.10 + BH * 0.48,
            "SVGP with 24 inducing pts\nSpatiotemporal kernel\n(RBF spatial + periodic\ntemporal)",
            **BODY_KW)
    ax.text(mx1 + BW/2, 0.10 + BH * 0.13,
            "Output: μ(x), σ²(x)", **SUB_KW)

    # ── M2: Dynamic Silence Diagnosis ─────────────────────────────────────────
    mx2 = COL[2]
    _box(ax, (mx2, 0.10), (BW, BH), C["m2"], C["m2_bd"])
    _badge(ax, mx2 + 0.018, 0.10 + BH - 0.018, "M2", C["m2_bd"])
    ax.text(mx2 + BW/2, 0.10 + BH * 0.80,
            "Dynamic Silence\nDiagnosis", **TITLE_KW, color=C["m2_bd"])
    ax.text(mx2 + BW/2, 0.10 + BH * 0.48,
            "PI-SSD embedding\n+ DBN temporal model\nLinear corruption\ncurriculum (5%→20%)",
            **BODY_KW)
    ax.text(mx2 + BW/2, 0.10 + BH * 0.13,
            "Output: silence score s(t)", **SUB_KW)

    # ── M3: MNAR Missingness ───────────────────────────────────────────────────
    mx3 = COL[3]
    _box(ax, (mx3, 0.10), (BW, BH), C["m3"], C["m3_bd"])
    _badge(ax, mx3 + 0.018, 0.10 + BH - 0.018, "M3", C["m3_bd"])
    ax.text(mx3 + BW/2, 0.10 + BH * 0.80,
            "MNAR Missingness\nModel", **TITLE_KW, color=C["m3_bd"])
    ax.text(mx3 + BW/2, 0.10 + BH * 0.48,
            "Sensor-conditional VAE\nJoint generative decoder\nHealth latent + KL reg\nEnsemble CRPS scoring",
            **BODY_KW)
    ax.text(mx3 + BW/2, 0.10 + BH * 0.13,
            "Output: p(R=1|x,z)", **SUB_KW)

    # ── M4: Active Sensing Policy ──────────────────────────────────────────────
    mx4 = COL[4]
    _box(ax, (mx4, 0.10), (BW, BH), C["m4"], C["m4_bd"])
    _badge(ax, mx4 + 0.018, 0.10 + BH - 0.018, "M4", C["m4_bd"])
    ax.text(mx4 + BW/2, 0.10 + BH * 0.80,
            "Active Sensing\nPolicy", **TITLE_KW, color=C["m4_bd"])
    ax.text(mx4 + BW/2, 0.10 + BH * 0.48,
            "MI-proxy utility\nLazy greedy planning\n(budget-constrained)\nPPO actor-critic",
            **BODY_KW)
    ax.text(mx4 + BW/2, 0.10 + BH * 0.13,
            "Output: S* ⊆ stations", **SUB_KW)

    # ── M5: Conformal Reliability ──────────────────────────────────────────────
    # spans top row, centred between M2 and M4
    mx5_cx = (COL[2] + COL[4] + BW) / 2
    mx5_w  = BW * 1.55
    mx5_x  = mx5_cx - mx5_w / 2
    _box(ax, (mx5_x, 0.62), (mx5_w, BH * 0.78), C["m5"], C["m5_bd"])
    _badge(ax, mx5_x + 0.018, 0.62 + BH * 0.78 - 0.018, "M5", C["m5_bd"])
    ax.text(mx5_cx, 0.62 + BH * 0.78 * 0.78,
            "Conformal Reliability",
            **{**TITLE_KW, "fontsize": 9.5, "color": C["m5_bd"]})
    ax.text(mx5_cx, 0.62 + BH * 0.78 * 0.42,
            "Relational-adaptive conformal prediction  ·  Graph-based "
            "neighbour calibration\nAdaptive epsilon annealing  ·  "
            "Post-hoc interval width control  ·  Target: 90% coverage",
            **{**BODY_KW, "fontsize": 7})
    ax.text(mx5_cx, 0.62 + BH * 0.78 * 0.10,
            "Output: [l(x), u(x)] with empirical coverage >= 1-alpha", **SUB_KW)

    # ── OUTPUT box ─────────────────────────────────────────────────────────────
    ox = 0.905
    _box(ax, (ox, 0.10), (BW - 0.005, BH), C["out"], C["out_bd"])
    ax.text(ox + (BW-0.005)/2, 0.10 + BH * 0.82,
            "Outputs", **TITLE_KW, color=C["out_bd"])
    items = [
        ("pred",   0.65, "Calibrated μ, σ²"),
        ("intv",   0.50, "Prediction interval\n[l, u] @ 90%"),
        ("mprob",  0.33, "Missingness prob\np(R=1|x,z)"),
        ("sel",    0.16, "Optimal sensor set\nS* (budget B)"),
    ]
    icons = ["≈", "↔", "?", "★"]
    for icon, frac, txt in zip(icons, [v[1] for v in items], [v[2] for v in items]):
        cy = 0.10 + BH * frac
        ax.text(ox + 0.012, cy, icon, fontsize=8, va="center", color=C["out_bd"])
        ax.text(ox + 0.028, cy, txt, fontsize=6.5, va="center",
                color="#37474f", linespacing=1.4)

    # ── MODULE ARROWS ──────────────────────────────────────────────────────────
    mid_y = 0.10 + BH / 2

    # M1 → M2
    _arrow(ax, mx1 + BW, mid_y, mx2, mid_y)
    # M2 → M3
    _arrow(ax, mx2 + BW, mid_y, mx3, mid_y)
    # M3 → M4
    _arrow(ax, mx3 + BW, mid_y, mx4, mid_y)
    # M4 → Output
    _arrow(ax, mx4 + BW, mid_y, ox, mid_y)

    # M1 residuals → M2 (diagonal up from M1 top)
    _dashed_arrow(ax, mx1 + BW/2, 0.10 + BH,
                  mx2 + BW/2, 0.62,
                  color=C["m2_bd"], lw=0.9)
    ax.text((mx1 + BW/2 + mx2 + BW/2)/2 - 0.02, 0.10 + BH + 0.045,
            "residuals", fontsize=5.8, color=C["m2_bd"], style="italic")

    # M2 silence → M3 (through M5 or direct)
    _dashed_arrow(ax, mx2 + BW/2, 0.62,
                  mx3 + BW/2, 0.62,
                  color=C["m5_bd"], lw=0.9)

    # M3 latent → M4
    _dashed_arrow(ax, mx3 + BW/2, 0.62,
                  mx4 + BW/2, 0.62,
                  color=C["m5_bd"], lw=0.9)

    # M5 → Output (calibrated intervals)
    _arrow(ax, mx5_x + mx5_w, 0.62 + BH * 0.78 * 0.5,
           ox, 0.10 + BH * 0.58,
           color=C["m5_bd"], lw=1.1,
           connectionstyle="arc3,rad=-0.25")

    # M1 → M5 (posterior)
    _dashed_arrow(ax, mx1 + BW/2, 0.10 + BH,
                  mx5_x + 0.04, 0.62,
                  color=C["m1_bd"], lw=0.9)
    ax.text(mx1 + BW/2 - 0.055, 0.10 + BH + 0.055,
            "posterior\nsamples", fontsize=5.8, color=C["m1_bd"],
            style="italic", ha="center")

    # M1 → Output (predictions)
    _arrow(ax, mx1 + BW/2, 0.10,
           ox + (BW-0.005)/2, 0.10 + BH * 0.70,
           color=C["m1_bd"], lw=0.9,
           connectionstyle="arc3,rad=0.35")

    # ── TRAINING / INFERENCE labels ────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.018, 0.017), 0.96, 0.063,
                                boxstyle="round,pad=0,rounding_size=0.01",
                                facecolor="#f5f5f5", edgecolor="#bdbdbd",
                                linewidth=0.8, zorder=1))
    stages = [
        (COL[1] + BW/2, "Joint GP\ntraining"),
        (COL[2] + BW/2, "DBN\nfitting"),
        (COL[3] + BW/2, "VAE ELBO\noptimisation"),
        (COL[4] + BW/2, "PPO policy\nlearning"),
        (mx5_cx,        "Conformal\ncalibration"),
    ]
    colors_tr = [C["m1_bd"], C["m2_bd"], C["m3_bd"], C["m4_bd"], C["m5_bd"]]
    for (cx, lbl), col in zip(stages, colors_tr):
        ax.text(cx, 0.049, lbl, fontsize=6.2, ha="center", va="center",
                color=col, linespacing=1.3)
        ax.plot([cx - 0.04, cx + 0.04], [0.072, 0.072],
                color=col, lw=1.5, solid_capstyle="round")
    ax.text(0.023, 0.049, "Training\nstages:", fontsize=6.5,
            ha="left", va="center", color="#616161", fontweight="bold")

    # ── TITLE ──────────────────────────────────────────────────────────────────
    ax.set_title(
        "SA-IDS Framework: Silence-Aware Intelligent Data Selection\n"
        "for Weather Sensor Networks",
        fontsize=11, fontweight="bold", pad=8, color="#212121",
    )

    fig.tight_layout(rect=[0, 0, 1, 1])
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = outdir / f"fig_01_architecture.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else 150)
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
