# Changelog

All notable changes to SA-IDS Framework are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `standardize_inputs` toggle in `TabularDataConfig` (z-score normalization for lat/lon/elapsed-hours and extra columns).
- `ensemble_crps` method in `ConformalScorer`; used automatically when `inference_strategy="joint_generative"`.
- Persistence and monthly-climatology baselines computed alongside ablations.
- `--log-level` CLI flag; `logging.basicConfig` wired in `main()`.
- `pyproject.toml` enabling `pip install -e .` and `sa-ids` entry-point.
- `LICENSE` (MIT) and `CITATION.cff` for Zenodo DOI and GitHub citation sidebar.
- `data/MANIFEST.json` with SHA-256 hashes and download instructions for all source data.
- `.github/workflows/test.yml` GitHub Actions CI (Linux, Python 3.11).
- `Dockerfile` for CPU-only reproducible container.

---

## [0.1.0] — 2026-05-15

Initial public release accompanying the EMS submission.

Commit: `1a2a076` — *Initial SA-IDS framework implementation*

### Modules
- **M1** Sparse variational spatiotemporal GP (`models.py`, `SpatiotemporalSparseGP`).
- **M2** Sequence-aware dynamic silence diagnosis with PI-SSD embeddings and DBN-lite state posteriors.
- **M3** Heterogeneous MNAR missingness modelling (`plug_in`, `joint_variational`, `joint_generative`, pattern-mixture).
- **M4** Myopic lazy-greedy, non-myopic rollout, and PPO-warm-start active-sensing policies.
- **M5** Normalized conformal reliability with `split`, `adaptive`, `relational_adaptive`, and `graph_corel` modes.

### Dataset
- Joint Q1 2025 Korean weather network: 905 stations, 7.6 M hourly rows.
- Sources: KMA AWS, KMA ASOS, NOAA ISD-Lite, ERA5 reanalysis.

### Key results (framework_joint_run)
- Base GP CRPS: 4.39 | RMSE: 10.58 | MAE: 5.30
- Full model: see `data/joint_weather_network_q1_2025/outputs/framework_joint_run/summary.json`

[Unreleased]: https://github.com/12JOSEPH21-star/SA-IDS-Framework/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/12JOSEPH21-star/SA-IDS-Framework/releases/tag/v0.1.0
