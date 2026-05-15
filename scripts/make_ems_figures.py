"""
Generate EMS-submission tables and figures from ems_patch_v010 benchmark results.

Usage:
    python scripts/make_ems_figures.py [--summary outputs/ems_patch_v010/summary.json]
                                       [--outdir outputs/ems_patch_v010/figures]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── display names ────────────────────────────────────────────────────────────
VARIANT_LABELS: dict[str, str] = {
    "persistence_24h_baseline":              "Persistence (24 h)",
    "climatology_hourly_baseline":           "Climatology (hourly)",
    "base_gp_only":                          "GP only (M1)",
    "gp_plus_joint_generative_missingness":  "GP + JG-missingness (M1+M2+M3)",
    "gp_plus_joint_generative_jvi_training": "GP + JG-JVI training (M1+M2+M3†)",
    "gp_plus_conformal_reliability":         "GP + Conformal (M1+M2+M5)",
    "full_model":                            "Full SA-IDS (M1–M5)",
}

ABLATION_ORDER = list(VARIANT_LABELS.keys())

# colours
COL_BASELINE = "#9e9e9e"
COL_GP       = "#1976d2"
COL_M3       = "#388e3c"
COL_JVI      = "#f57c00"
COL_M5       = "#7b1fa2"
COL_FULL     = "#c62828"

VARIANT_COLOURS: dict[str, str] = {
    "persistence_24h_baseline":              COL_BASELINE,
    "climatology_hourly_baseline":           COL_BASELINE,
    "base_gp_only":                          COL_GP,
    "gp_plus_joint_generative_missingness":  COL_M3,
    "gp_plus_joint_generative_jvi_training": COL_JVI,
    "gp_plus_conformal_reliability":         COL_M5,
    "full_model":                            COL_FULL,
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def _latex_bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def _best_in_col(values: list[float | None]) -> int:
    """Return index of minimum non-None value."""
    best_i, best_v = -1, float("inf")
    for i, v in enumerate(values):
        if v is not None and v < best_v:
            best_v = v
            best_i = i
    return best_i


# ─── Table 1: ablation metrics ────────────────────────────────────────────────

def make_table_ablation(ablations: dict, outdir: Path) -> None:
    metrics = ["rmse", "mae", "crps"]
    rows = []
    for key in ABLATION_ORDER:
        if key not in ablations:
            continue
        m = ablations[key].get("metrics", {})
        rows.append({
            "variant": key,
            "label":   VARIANT_LABELS[key],
            "rmse":    m.get("rmse"),
            "mae":     m.get("mae"),
            "crps":    m.get("crps"),
        })

    # CSV
    csv_path = outdir / "table_ablation.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("variant,label,RMSE,MAE,CRPS\n")
        for r in rows:
            f.write(f"{r['variant']},{r['label']},{_fmt(r['rmse'])},{_fmt(r['mae'])},{_fmt(r['crps'])}\n")

    # LaTeX
    best_idx = {m: _best_in_col([r[m] for r in rows]) for m in metrics}
    tex_path = outdir / "table_ablation.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[ht]
\centering
\caption{Ablation study: predictive performance on the held-out evaluation set
         (n\,=\,4{,}096 rows). Bold: best in column.}
\label{tab:ablation}
\begin{tabular}{lrrr}
\toprule
Model variant & RMSE & MAE & CRPS \\
\midrule
""")
        for i, r in enumerate(rows):
            cells = []
            for m in metrics:
                s = _fmt(r[m])
                cells.append(_latex_bold(s) if i == best_idx[m] else s)
            # add midrule after baselines
            if r["variant"] == "climatology_hourly_baseline":
                prefix = ""
                suffix = r" \\" + "\n" + r"\midrule" + "\n"
            else:
                prefix = ""
                suffix = r" \\"
            f.write(f"{prefix}{r['label']} & {' & '.join(cells)}{suffix}\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

    print(f"  [table] {csv_path.name}  {tex_path.name}")


# ─── Table 2: baseline comparison ─────────────────────────────────────────────

def make_table_baseline_comparison(ablations: dict, base_metrics: dict, outdir: Path) -> None:
    best_crps = base_metrics.get("crps")
    best_rmse = base_metrics.get("rmse")
    best_mae  = base_metrics.get("mae")

    rows_src = [
        ("persistence_24h_baseline",    "Persistence (24 h)"),
        ("climatology_hourly_baseline",  "Climatology (hourly)"),
        ("base_gp_only",                 "GP only (M1)"),
    ]

    csv_path = outdir / "table_baseline_comparison.csv"
    tex_path = outdir / "table_baseline_comparison.tex"

    with csv_path.open("w", encoding="utf-8") as fc, tex_path.open("w", encoding="utf-8") as ft:
        fc.write("baseline,RMSE_baseline,MAE_baseline,CRPS_baseline,"
                 "RMSE_saids,MAE_saids,CRPS_saids,"
                 "delta_RMSE_pct,delta_MAE_pct,delta_CRPS_pct\n")
        ft.write(r"""\begin{table}[ht]
\centering
\caption{SA-IDS (full model: base metrics) vs.\ statistical and GP baselines.
         $\Delta$\% = relative improvement (lower is better for all metrics).}
\label{tab:baseline_comparison}
\begin{tabular}{lrrr|rrr|rrr}
\toprule
Baseline & \multicolumn{3}{c|}{Baseline} & \multicolumn{3}{c|}{SA-IDS} & \multicolumn{3}{c}{$\Delta$\% $\downarrow$} \\
 & RMSE & MAE & CRPS & RMSE & MAE & CRPS & RMSE & MAE & CRPS \\
\midrule
""")
        for key, label in rows_src:
            if key not in ablations:
                continue
            m = ablations[key].get("metrics", {})
            br, bm, bc = m.get("rmse"), m.get("mae"), m.get("crps")

            def dpct(b, s):
                if b and s:
                    return (b - s) / b * 100.0
                return None

            dr = dpct(br, best_rmse)
            dm = dpct(bm, best_mae)
            dc = dpct(bc, best_crps)

            fc.write(f"{key},{_fmt(br)},{_fmt(bm)},{_fmt(bc)},"
                     f"{_fmt(best_rmse)},{_fmt(best_mae)},{_fmt(best_crps)},"
                     f"{_fmt(dr, 1)},{_fmt(dm, 1)},{_fmt(dc, 1)}\n")

            ft.write(f"{label} & {_fmt(br)} & {_fmt(bm)} & {_fmt(bc)} & "
                     f"{_fmt(best_rmse)} & {_fmt(best_mae)} & {_fmt(best_crps)} & "
                     f"{_fmt(dr, 1)}\\% & {_fmt(dm, 1)}\\% & {_fmt(dc, 1)}\\% \\\\\n")

        ft.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

    print(f"  [table] {csv_path.name}  {tex_path.name}")


# ─── Table 3: coverage / interval width ───────────────────────────────────────

def make_table_coverage(ablations: dict, base_metrics: dict, outdir: Path) -> None:
    cov_variants = [
        ("base_metrics",                 "Full SA-IDS (base metrics)",       base_metrics),
        ("gp_plus_conformal_reliability", "GP + Conformal (M1+M2+M5)",       ablations.get("gp_plus_conformal_reliability", {}).get("metrics", {})),
        ("full_model",                   "Full SA-IDS ablation",              ablations.get("full_model", {}).get("metrics", {})),
    ]
    csv_path = outdir / "table_coverage.csv"
    tex_path = outdir / "table_coverage.tex"
    with csv_path.open("w", encoding="utf-8") as fc, tex_path.open("w", encoding="utf-8") as ft:
        fc.write("variant,coverage,interval_width,target\n")
        ft.write(r"""\begin{table}[ht]
\centering
\caption{Coverage and interval width for variants with conformal calibration.
         Target coverage: 90\%.}
\label{tab:coverage}
\begin{tabular}{lrr}
\toprule
Variant & Coverage & Interval width \\
\midrule
""")
        for _, label, m in cov_variants:
            cov = m.get("coverage")
            iw  = m.get("interval_width")
            if cov is None:
                continue
            fc.write(f"{label},{_fmt(cov, 4)},{_fmt(iw, 3)},0.90\n")
            ft.write(f"{label} & {_fmt(cov, 3)} & {_fmt(iw, 2)} \\\\\n")
        ft.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"  [table] {csv_path.name}  {tex_path.name}")


# ─── Fig 1: CRPS comparison (horizontal bar) ──────────────────────────────────

def fig_crps_comparison(ablations: dict, outdir: Path) -> None:
    keys = [k for k in ABLATION_ORDER if k in ablations]
    labels = [VARIANT_LABELS[k] for k in keys]
    crps   = [ablations[k]["metrics"].get("crps") for k in keys]
    colours = [VARIANT_COLOURS[k] for k in keys]

    valid = [(l, v, c) for l, v, c in zip(labels, crps, colours) if v is not None]
    labels_v, crps_v, colours_v = zip(*valid)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(labels_v))
    bars = ax.barh(y, crps_v, color=colours_v, height=0.6, edgecolor="white", linewidth=0.5)

    # reference lines: persistence and climatology
    persist_crps = ablations.get("persistence_24h_baseline", {}).get("metrics", {}).get("crps")
    clim_crps    = ablations.get("climatology_hourly_baseline", {}).get("metrics", {}).get("crps")
    if persist_crps:
        ax.axvline(persist_crps, color=COL_BASELINE, lw=1.2, ls="--", alpha=0.6, label="Persistence")
    if clim_crps:
        ax.axvline(clim_crps,    color=COL_BASELINE, lw=1.2, ls=":",  alpha=0.6, label="Climatology")

    # value labels
    for bar, val in zip(bars, crps_v):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_v)
    ax.set_xlabel("CRPS (↓ better)")
    ax.set_title("CRPS by model variant — EMS patch v0.1.0")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="x", which="major", lw=0.4, alpha=0.5)
    ax.grid(axis="x", which="minor", lw=0.2, alpha=0.3)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()

    path = outdir / "fig_crps_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig]   {path.name}")


# ─── Fig 2: RMSE / MAE / CRPS grouped bars ────────────────────────────────────

def fig_rmse_mae_crps(ablations: dict, outdir: Path) -> None:
    keys = [k for k in ABLATION_ORDER if k in ablations
            and ablations[k]["metrics"].get("crps") is not None]
    labels  = [VARIANT_LABELS[k] for k in keys]
    rmse_v  = [ablations[k]["metrics"]["rmse"] for k in keys]
    mae_v   = [ablations[k]["metrics"]["mae"]  for k in keys]
    crps_v  = [ablations[k]["metrics"]["crps"] for k in keys]

    n = len(keys)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w, rmse_v, width=w, label="RMSE", color="#1565c0", alpha=0.85)
    ax.bar(x,     mae_v,  width=w, label="MAE",  color="#2e7d32", alpha=0.85)
    ax.bar(x + w, crps_v, width=w, label="CRPS", color="#c62828", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=7.5)
    ax.set_ylabel("Score (↓ better, °C)")
    ax.set_title("RMSE / MAE / CRPS by model variant — EMS patch v0.1.0")
    ax.legend(framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", lw=0.4, alpha=0.5)
    ax.grid(axis="y", which="minor", lw=0.2, alpha=0.3)
    fig.tight_layout()

    path = outdir / "fig_rmse_mae_crps.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig]   {path.name}")


# ─── Fig 3: training-loss curves ──────────────────────────────────────────────

def fig_training_loss(fit_summary: dict, outdir: Path) -> None:
    obs_h  = fit_summary.get("observation_history", {})
    mis_h  = fit_summary.get("missingness_history", {})
    sta_h  = fit_summary.get("state_history", {})

    obs_loss  = obs_h.get("loss", [])
    obs_ss    = obs_h.get("self_supervised_loss", [])
    obs_recon = obs_h.get("reconstruction_loss", [])
    mis_loss  = mis_h.get("loss", [])
    sta_loss  = sta_h.get("loss", [])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    # panel 1: observation model
    ax = axes[0]
    if obs_loss:
        steps = np.arange(1, len(obs_loss) + 1)
        ax.plot(steps, obs_loss,  label="Total",       color="#1565c0", lw=1.4)
        if obs_ss:
            ax.plot(steps, obs_ss,  label="Self-sup.",  color="#42a5f5", lw=1.0, ls="--")
        if obs_recon:
            ax.plot(steps[:len(obs_recon)], obs_recon,
                    label="Recon.",    color="#ef9a9a", lw=1.0, ls=":")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Observation model")
    ax.legend(fontsize=7)
    ax.grid(lw=0.35, alpha=0.5)
    ax.set_yscale("log")

    # panel 2: missingness model (single epoch → single point or short list)
    ax = axes[1]
    if mis_loss:
        ax.bar(np.arange(1, len(mis_loss) + 1), mis_loss, color="#388e3c", alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ELBO loss")
        ax.set_title("Missingness model (ELBO)")
    ax.grid(axis="y", lw=0.35, alpha=0.5)

    # panel 3: state model
    ax = axes[2]
    if sta_loss:
        ax.bar(np.arange(1, len(sta_loss) + 1), sta_loss, color="#7b1fa2", alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Negative ELBO")
        ax.set_title("State model (−ELBO)")
    ax.grid(axis="y", lw=0.35, alpha=0.5)

    fig.suptitle("Training loss curves — EMS patch v0.1.0", y=1.01)
    fig.tight_layout()

    path = outdir / "fig_training_loss.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig]   {path.name}")


# ─── Fig 4: coverage / interval-width ─────────────────────────────────────────

def fig_coverage_width(ablations: dict, base_metrics: dict, outdir: Path) -> None:
    entries = [
        ("Full SA-IDS\n(base)",           base_metrics),
        ("GP + Conformal\n(M1+M2+M5)",    ablations.get("gp_plus_conformal_reliability", {}).get("metrics", {})),
        ("Full SA-IDS\n(ablation)",        ablations.get("full_model", {}).get("metrics", {})),
    ]
    labels, covs, iws = [], [], []
    for lbl, m in entries:
        c = m.get("coverage")
        w = m.get("interval_width")
        if c is not None and w is not None:
            labels.append(lbl)
            covs.append(c * 100.0)
            iws.append(w)

    if not labels:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
    x = np.arange(len(labels))

    ax1.bar(x, covs, color=["#c62828", "#7b1fa2", "#c62828"], alpha=0.82, width=0.5)
    ax1.axhline(90.0, color="k", lw=1.2, ls="--", label="Target 90%")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Coverage (%)")
    ax1.set_title("Empirical coverage")
    ax1.set_ylim(85, 97)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", lw=0.35, alpha=0.5)

    ax2.bar(x, iws, color=["#c62828", "#7b1fa2", "#c62828"], alpha=0.82, width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Interval width (°C)")
    ax2.set_title("Prediction interval width")
    ax2.grid(axis="y", lw=0.35, alpha=0.5)

    fig.suptitle("Conformal calibration — EMS patch v0.1.0", y=1.01)
    fig.tight_layout()

    path = outdir / "fig_coverage_width.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig]   {path.name}")


# ─── Fig 5: curriculum mask probability ───────────────────────────────────────

def fig_curriculum(fit_summary: dict, outdir: Path) -> None:
    probs = fit_summary.get("observation_history", {}).get("curriculum_mask_probability", [])
    if not probs:
        return

    fig, ax = plt.subplots(figsize=(5, 2.8))
    steps = np.arange(1, len(probs) + 1)
    ax.plot(steps, [p * 100 for p in probs], color="#f57c00", lw=1.6)
    ax.fill_between(steps, [p * 100 for p in probs], alpha=0.15, color="#f57c00")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mask probability (%)")
    ax.set_title("Corruption curriculum schedule — EMS patch v0.1.0")
    ax.grid(lw=0.35, alpha=0.5)
    fig.tight_layout()

    path = outdir / "fig_curriculum.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig]   {path.name}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EMS figures from summary.json")
    parser.add_argument("--summary", default="outputs/ems_patch_v010/summary.json")
    parser.add_argument("--outdir",  default="outputs/ems_patch_v010/figures")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with summary_path.open(encoding="utf-8") as f:
        data = json.load(f)

    ablations   = data.get("ablations", {})
    base_metrics = data.get("base_metrics", {})
    fit_summary  = data.get("fit_summary", {})

    print(f"\nLoaded: {summary_path}")
    print(f"Output: {outdir}\n")

    make_table_ablation(ablations, outdir)
    make_table_baseline_comparison(ablations, base_metrics, outdir)
    make_table_coverage(ablations, base_metrics, outdir)
    fig_crps_comparison(ablations, outdir)
    fig_rmse_mae_crps(ablations, outdir)
    fig_training_loss(fit_summary, outdir)
    fig_coverage_width(ablations, base_metrics, outdir)
    fig_curriculum(fit_summary, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
