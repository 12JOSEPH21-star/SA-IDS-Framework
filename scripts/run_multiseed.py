"""
Multi-seed runner for EMS patch v0.1.0 ablation.

Runs the framework for seeds [7, 11, 19, 23, 29] and aggregates
mean +/- std across seeds for each ablation variant.

Seed 7 result is reused from outputs/ems_patch_v010/summary.json if present.
The prepared-data cache (CSV-encoding result) is reused from seed-7 run.

Usage:
    python scripts/run_multiseed.py [--base-config framework_ems_patch_v010.json]
                                    [--seeds 7 11 19 23 29]
                                    [--outdir outputs/multiseed_v010]
                                    [--paper-dir paper_figures/ems_patch_v010]
                                    [--skip-existing]
                                    [--aggregate-only]
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

METRICS = ["rmse", "mae", "crps"]
SEEDS_DEFAULT = [7, 11, 19, 23, 29]

VARIANT_ORDER = [
    "persistence_24h_baseline",
    "climatology_hourly_baseline",
    "base_gp_only",
    "gp_plus_joint_generative_missingness",
    "gp_plus_joint_generative_jvi_training",
    "gp_plus_conformal_reliability",
    "full_model",
]

VARIANT_LABELS = {
    "persistence_24h_baseline":              "Persistence (24 h)",
    "climatology_hourly_baseline":           "Climatology (hourly)",
    "base_gp_only":                          "GP only (M1)",
    "gp_plus_joint_generative_missingness":  "GP + JG-missingness (M1+M2+M3)",
    "gp_plus_joint_generative_jvi_training": "GP + JG-JVI training (M1+M2+M3*)",
    "gp_plus_conformal_reliability":         "GP + Conformal (M1+M2+M5)",
    "full_model":                            "Full SA-IDS (M1-M5)",
}

PLOT_LABELS = {
    "persistence_24h_baseline": "Persistence\n(24 h)",
    "climatology_hourly_baseline": "Hourly\nclimatology",
    "base_gp_only": "GP only\n(M1)",
    "gp_plus_joint_generative_missingness": "GP + JG\nmissingness\n(M1-M3)",
    "gp_plus_joint_generative_jvi_training": "GP + JG-JVI\ntraining\n(M1-M3*)",
    "gp_plus_conformal_reliability": "GP +\nconformal\n(M1+M2+M5)",
    "full_model": "Full SA-IDS\n(M1-M5)",
}

# Location of seed-7 cache (prepared data, expensive to recompute)
SEED7_CACHE_SRC = Path("outputs/ems_patch_v010/cache")


def _seed_output_dir(outdir: Path, seed: int) -> Path:
    return outdir / f"seed_{seed:02d}"


def _copy_cache(dst_output_dir: Path) -> None:
    """Copy seed-7 prepared-data cache into a new seed output dir."""
    if not SEED7_CACHE_SRC.exists():
        return
    dst_cache = dst_output_dir / "cache"
    dst_cache.mkdir(parents=True, exist_ok=True)
    for f in SEED7_CACHE_SRC.glob("prepared_*.pt"):
        dst = dst_cache / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            print(f"    cached {f.name} -> {dst}")


def run_seed(base_cfg: dict, seed: int, outdir: Path, skip_existing: bool) -> Path:
    seed_dir     = _seed_output_dir(outdir, seed)
    summary_path = seed_dir / "summary.json"

    if skip_existing and summary_path.exists():
        print(f"  [seed {seed}] found existing result, skipping.")
        return summary_path

    seed_dir.mkdir(parents=True, exist_ok=True)

    # Reuse expensive prepared-data cache from seed-7
    _copy_cache(seed_dir)

    # Build per-seed config — use absolute paths so the config can live
    # anywhere and still resolve data_path correctly.
    project_root = Path(__file__).parent.parent.resolve()
    cfg = copy.deepcopy(base_cfg)
    cfg["run"]["seed"] = seed
    # Make data_path absolute (resolved from project root)
    raw_data_path = cfg.get("data", {}).get("data_path", "")
    cfg["data"]["data_path"] = str((project_root / raw_data_path).resolve())
    cfg["output_path"] = str(summary_path.resolve())

    # Write config next to the project root so base_dir = project root
    cfg_path = project_root / f".multiseed_config_seed_{seed:02d}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"\n  [seed {seed}] starting run -> {summary_path}")
    result = subprocess.run(
        [sys.executable, "-m", "task_cli", "framework-run",
         "--config", str(cfg_path)],
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        print(f"  [seed {seed}] FAILED (exit {result.returncode})")
    else:
        print(f"  [seed {seed}] done.")
    return summary_path


# ── aggregation ────────────────────────────────────────────────────────────────

def aggregate(seed_paths: list[Path], outdir: Path) -> dict:
    summaries = []
    for p in seed_paths:
        if not p.exists():
            print(f"  WARNING: missing {p}, skipping")
            continue
        with p.open(encoding="utf-8") as f:
            summaries.append(json.load(f))

    if not summaries:
        raise RuntimeError("No valid summaries to aggregate.")

    agg: dict = {"n_seeds": len(summaries), "variants": {}}

    for variant in VARIANT_ORDER:
        rows = []
        for s in summaries:
            m = s.get("ablations", {}).get(variant, {}).get("metrics", {})
            if not m:
                continue
            row = {k: m.get(k) for k in METRICS}
            rows.append(row)

        if not rows:
            continue

        variant_agg: dict = {"n": len(rows), "metrics": {}}
        for metric in METRICS:
            vals = [r[metric] for r in rows if r.get(metric) is not None]
            if vals:
                variant_agg["metrics"][metric] = {
                    "mean":   float(np.mean(vals)),
                    "std":    float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                    "min":    float(np.min(vals)),
                    "max":    float(np.max(vals)),
                    "values": [float(v) for v in vals],
                }
        agg["variants"][variant] = variant_agg

    # base_metrics aggregation
    bm_keys = ["rmse", "mae", "crps", "coverage", "interval_width"]
    base_rows: dict[str, list] = {k: [] for k in bm_keys}
    for s in summaries:
        bm = s.get("base_metrics", {})
        for k in bm_keys:
            if bm.get(k) is not None:
                base_rows[k].append(bm[k])
    agg["base_metrics"] = {}
    for k, vals in base_rows.items():
        if vals:
            agg["base_metrics"][k] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            }

    out_path = outdir / "aggregate.json"
    out_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n  [aggregate] saved to {out_path}")
    return agg


# ── table writers ──────────────────────────────────────────────────────────────

def _fmt(v: float | None, d: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def write_tables(agg: dict, paper_dir: Path) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    n = agg["n_seeds"]

    csv_path = paper_dir / "table_ablation_multiseed.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("variant,label,RMSE_mean,RMSE_std,MAE_mean,MAE_std,"
                "CRPS_mean,CRPS_std\n")
        for v in VARIANT_ORDER:
            if v not in agg["variants"]:
                continue
            m   = agg["variants"][v]["metrics"]
            lbl = VARIANT_LABELS[v]
            f.write(
                f"{v},{lbl},"
                f"{_fmt(m['rmse']['mean'])},{_fmt(m['rmse']['std'])},"
                f"{_fmt(m['mae']['mean'])},{_fmt(m['mae']['std'])},"
                f"{_fmt(m['crps']['mean'])},{_fmt(m['crps']['std'])}\n"
            )

    tex_path = paper_dir / "table_ablation_multiseed.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(
            r"\begin{table}[ht]" + "\n"
            r"\centering" + "\n"
            r"\caption{Ablation study over " + str(n)
            + r" random seeds (mean\,$\pm$\,std, n="
            + str(n) + r" $\times$ 4{,}096 evaluation rows). "
            r"Best mean per column in bold.}" + "\n"
            r"\label{tab:ablation_multiseed}" + "\n"
            r"\begin{tabular}{lcccccc}" + "\n"
            r"\toprule" + "\n"
            r"Model variant & \multicolumn{2}{c}{RMSE} "
            r"& \multicolumn{2}{c}{MAE} "
            r"& \multicolumn{2}{c}{CRPS} \\" + "\n"
            r" & mean & std & mean & std & mean & std \\" + "\n"
            r"\midrule" + "\n"
        )
        best: dict[str, float] = {}
        for metric in METRICS:
            vals = [agg["variants"][v]["metrics"].get(metric, {}).get("mean")
                    for v in VARIANT_ORDER if v in agg["variants"]]
            vals = [x for x in vals if x is not None]
            best[metric] = min(vals) if vals else float("inf")

        for i, v in enumerate(VARIANT_ORDER):
            if v not in agg["variants"]:
                continue
            m   = agg["variants"][v]["metrics"]
            lbl = VARIANT_LABELS[v]
            cells = []
            for metric in METRICS:
                mu  = m.get(metric, {}).get("mean")
                std = m.get(metric, {}).get("std")
                s_mu = _fmt(mu)
                if mu is not None and abs(mu - best[metric]) < 1e-6:
                    s_mu = r"\textbf{" + s_mu + "}"
                cells.append(f"{s_mu} & {_fmt(std)}")
            suffix = r" \\" + ("\n" + r"\midrule" if i == 1 else "")
            f.write(f"{lbl} & {' & '.join(cells)}{suffix}\n")

        f.write(r"\bottomrule" + "\n"
                + r"\end{tabular}" + "\n"
                + r"\end{table}" + "\n")

    print(f"  [table] {csv_path.name}  {tex_path.name}")


# ── figure ─────────────────────────────────────────────────────────────────────

def write_figure(agg: dict, paper_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 12.5, "axes.labelsize": 10.5,
        "xtick.labelsize": 9.2, "ytick.labelsize": 9.5,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#263238",
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "grid.color": "#b0bec5", "grid.alpha": 0.45,
    })

    variants = [v for v in VARIANT_ORDER if v in agg["variants"]]
    labels   = [PLOT_LABELS.get(v, VARIANT_LABELS[v]) for v in variants]
    COLS     = {"rmse": "#1565c0", "mae": "#2e7d32", "crps": "#c62828"}
    y        = np.arange(len(variants))

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.4), sharey=True)

    for ax, metric, col in zip(axes, METRICS, COLS.values()):
        means = [agg["variants"][v]["metrics"].get(metric, {}).get("mean")
                 for v in variants]
        stds  = [agg["variants"][v]["metrics"].get(metric, {}).get("std", 0.0)
                 for v in variants]
        valid_pairs = [(m, s) for m, s in zip(means, stds) if m is not None]

        bars = ax.barh(y, means, xerr=stds, color=col, alpha=0.88, height=0.62,
                       capsize=4.5,
                       error_kw=dict(lw=1.5, capthick=1.5, ecolor="#263238"),
                       edgecolor="white", linewidth=0.8)
        label_offset = (max(m + s for m, s in valid_pairs) if valid_pairs else 1.0) * 0.018
        for bar, mu, sd in zip(bars, means, stds):
            if mu is None:
                continue
            ax.text(bar.get_width() + sd + label_offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{mu:.2f} +/- {sd:.2f}",
                    ha="left", va="center", fontsize=8.5,
                    color="#263238")
        ax.set_xlabel(f"{metric.upper()} (lower is better, deg C)")
        ax.set_title(metric.upper(), fontweight="bold")
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="x", which="major", lw=0.7)
        ax.grid(axis="x", which="minor", lw=0.35, alpha=0.25)
        top = max(m + s for m, s in valid_pairs) * 1.20
        ax.set_xlim(0, top)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9.3, linespacing=1.1)
    axes[0].invert_yaxis()
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    n_seeds = agg["n_seeds"]
    fig.suptitle(
        f"Multi-seed ablation (n={n_seeds} seeds: 7, 11, 19, 23, 29) -- mean +/- std",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = paper_dir / "fig_multiseed_ablation.pdf"
    fig.savefig(path)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  [fig]   {path.name}  {png_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config",    default="framework_ems_patch_v010.json")
    parser.add_argument("--seeds",          nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--outdir",         default="outputs/multiseed_v010")
    parser.add_argument("--paper-dir",      default="paper_figures/ems_patch_v010")
    parser.add_argument("--skip-existing",  action="store_true", default=True)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip runs, only aggregate existing results.")
    args = parser.parse_args()

    base_cfg_path = Path(args.base_config)
    outdir        = Path(args.outdir)
    paper_dir     = Path(args.paper_dir)

    with base_cfg_path.open(encoding="utf-8") as f:
        base_cfg = json.load(f)

    EXISTING_SEED7 = Path("outputs/ems_patch_v010/summary.json")

    seed_paths: list[Path] = []
    for seed in args.seeds:
        if seed == 7 and EXISTING_SEED7.exists() and args.skip_existing:
            print(f"  [seed 7] reusing existing {EXISTING_SEED7}")
            seed_paths.append(EXISTING_SEED7)
        elif args.aggregate_only:
            seed_paths.append(_seed_output_dir(outdir, seed) / "summary.json")
        else:
            seed_paths.append(run_seed(base_cfg, seed, outdir, args.skip_existing))

    print(f"\nAggregating {len(seed_paths)} seed result(s) ...")
    agg = aggregate(seed_paths, outdir)

    print("\nWriting tables and figure ...")
    write_tables(agg, paper_dir)
    write_figure(agg, paper_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
