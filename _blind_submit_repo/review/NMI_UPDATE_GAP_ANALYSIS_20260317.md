# NMI Update Gap Analysis (2026-03-17)

## Sources reviewed

- `review/nmi_section_revision_instructions_extracted.txt`
- `review/pre_submission_feedback_analysis_extracted.txt`
- `review/saids_nmi_rewritten_from_feedback_extracted.txt`

## Current state already aligned with the documents

- The repository now supports a true large-benchmark preset:
  - `benchmark_joint_q1_large.json`
  - `station_limit=96`
  - `repeat_seeds=[7,11,19,23,29]`
- Benchmark export now supports repeated-seed summaries with:
  - mean/std
  - bootstrap CI
  - paired significance summaries
- Multi-source scale evidence is no longer limited to Korea-only bookkeeping:
  - Korea NOAA Q1
  - Japan NOAA Q1
  - Taiwan NOAA Q1
- Station-level registry sidecar now exists for benchmark scale and geography reporting:
  - `data/joint_weather_network_q1_2025/station_registry_q1.csv`
  - `data/joint_weather_network_q1_2025/station_registry_summary.json`
- Coarse region-holdout support now exists in `benchmark_suite.py`.
- Framework-run resume/debugging is now safer:
  - detached launcher
  - file heartbeat
  - prepared-data cache

## Highest-priority remaining updates

### 1. Run the large benchmark and publish the main table from it

The documents are explicit that the 8-station benchmark must be demoted to a controlled ablation bed and that the main evidence must come from a 50-100+ station benchmark.

Current gap:

- `benchmark_joint_q1_large.json` exists, but the large benchmark has not yet been run to completion.
- `small` and `medium` are populated; `large` is not.

Needed output:

- large-scale `summary.json`
- large-scale `paper_tables.json`
- large-scale `report.md`
- large-scale region-holdout table

### 2. Replace provisional wording with results backed by 5-seed evidence

The documents require that terms like `repeatable`, `robust`, and `general` be used only after repeated runs and significance reporting.

Current gap:

- The pipeline can now export repeated-seed CI/significance summaries.
- However, the manuscript-facing reported numbers are still mostly tied to prior small/medium outputs and not yet to the new repeated-seed large benchmark.

Needed output:

- final benchmark numbers for all main manuscript tables
- mean/std or CI in manuscript-facing tables
- significance markers for main claims

### 3. Add explicit reliability figures beyond one summary row

The section instructions require:

- coverage-width Pareto
- alpha sweep
- calibration/reliability plot
- stable vs shifted decomposition

Current gap:

- `coverage_over_time.csv` already exists
- reliability tables already exist
- but there is no dedicated Pareto / alpha-sweep figure export yet

Needed code/output:

- reliability alpha-sweep runner
- coverage-width Pareto CSV
- calibration-bin / reliability-diagram CSV

### 4. Run and report region-holdout / external pilot evidence

The documents explicitly call for station-holdout or region-holdout and suggest external-validity evidence.

Current state:

- coarse region-holdout code exists
- NOAA external pilots now exist:
  - Korea
  - Japan
  - Taiwan

Current gap:

- no completed external-region benchmark artifacts yet
- no manuscript-ready cross-region summary table yet

Needed output:

- Korea-internal region-holdout results
- Korea vs Japan pilot or Korea-trained / Japan-tested pilot
- short external-validity table in supplement or discussion

### 5. Decide diagnosis positioning and rewrite around the actual evidence

All three documents are consistent: diagnosis is still the weakest axis and should either be strengthened materially or moved to supporting analysis.

Current state:

- diagnosis code is stronger than before
- medium benchmark showed stronger cases
- but evidence is still scenario-dependent and not uniformly dominant

Recommended update:

- keep diagnosis in the paper
- demote it from central claim to supporting/qualified contribution
- add event-type breakdown and confusion-style figure only if the new retuned benchmark stays stable under repeated seeds

### 6. Finish the full framework-run or explicitly demote it to scalability evidence

Current state:

- `framework-run` now resumes safely and logs heartbeat
- latest hang is localized to `relational_adaptive` interval computation during `evaluating_base_pipeline`

Current gap:

- there is still no completed full-scale `framework-run` final artifact set

Recommended update:

- do not block the manuscript on full framework-run completion
- use benchmark suite as the main scientific evidence
- use full framework-run only as operational/scalability evidence after the hang is fixed

## Data/package updates still needed

### Data availability / code availability

The documents require final-form statements, not planning text.

Current gap:

- local manifests and package zips exist
- but there is no archival DOI, versioned release note, or frozen manuscript release manifest yet

Needed output:

- one frozen release manifest with:
  - commit hash
  - benchmark configs used in the paper
  - artifact paths
  - dataset manifests
- one manuscript-facing data availability statement
- one manuscript-facing code availability statement

### Reproducibility supplement bundle

The documents explicitly require:

- split manifest
- station list
- seed list
- preprocessing
- hyperparameters
- runtime/hardware
- failure cases

Current gap:

- pieces exist across config JSON, benchmark outputs, README, and review notes
- but there is no single supplement-ready reproducibility bundle document

Needed output:

- one consolidated reproducibility appendix markdown/CSV package

## Figures/tables not yet fully supported

The documents ask for:

- Figure 1: framework overview
- Figure 2: station map / climate zones / timeline
- Figure 3: large-scale result + compact ablation separation
- Figure 4: coverage-width Pareto / alpha sweep / calibration
- Figure 5: policy or diagnosis breakdown

Current status:

- Figure 2 support is partially ready via station registry
- Figure 3 support is partially ready via benchmark tables
- Figure 5 support is partially ready via existing policy/fault CSVs
- Figure 4 is not ready enough yet
- Figure 1 is manuscript/diagram work, not code

## Recommended execution order

1. Run `benchmark_joint_q1_large.json`
2. Export main repeated-seed tables from the large run
3. Run region-holdout / external NOAA pilot benchmark
4. Add reliability alpha-sweep / Pareto exports
5. Freeze diagnosis positioning
6. Write one reproducibility + availability bundle

## Bottom line

The repository is much closer to the NMI-oriented document set than before. The main remaining gaps are no longer architecture gaps; they are evidence-packaging gaps:

- complete the large benchmark
- convert repeated-seed outputs into manuscript tables/figures
- add reliability figure support
- add external/region generalization evidence
- freeze availability/reproducibility materials
