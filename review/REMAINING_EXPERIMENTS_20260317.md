# Remaining Experiments

## Started Now

- `Taiwan NOAA Q1 pilot`
  - Config: [benchmark_taiwan_q1_pilot.json](/C:/Programming%20Prctice/benchmark_taiwan_q1_pilot.json)
  - Goal: smallest external pilot completion for low-station regime.
- `Korea NOAA Q2 medium`
  - Config: [benchmark_korea_noaa_q2_medium.json](/C:/Programming%20Prctice/benchmark_korea_noaa_q2_medium.json)
  - Goal: seasonal robustness check beyond Q1.
- `Korea NOAA Q3 medium`
  - Config: [benchmark_korea_noaa_q3_medium.json](/C:/Programming%20Prctice/benchmark_korea_noaa_q3_medium.json)
  - Goal: second seasonal robustness check.

## Still Blocked

- `Historical NWP anchor benchmark`
  - Blocker: Q1 historical LDAPS/RDAPS archive not available through current KMA API Hub path.
  - Current status: main pipeline remains effectively ERA5/context-anchored.
- `Real QC / maintenance / outage ground-truth benchmark`
  - Blocker: only proxy QC sidecar is available.
  - Current status: health/diagnosis claims should stay secondary.
- `Station-matched extreme-event truth refinement`
  - Blocker: current warning/info sidecar exists, but event type/severity is not yet rich enough for strong fault-vs-extreme attribution.

## Optional Next

- `Cross-region transfer benchmark`
  - Korea train -> Japan/China test
  - Requires a transfer-specific runner rather than the current single-dataset benchmark preset path.
- `Full framework-run completion`
  - Still incomplete and not paper-primary.
