# 2nd Experiment Plan (2nd_exp_plan)

## Goal

Build the second-round experiment plan around spectral probing with clearer quantitative trends, deeper feature-level interpretation, attention-head analysis, and multi-task validation on commonly used music/audio datasets.

---

## 1) Aggregated Overall Genres: Common Trend via Regression Slope

### What to do
- Aggregate genre-level probe performance curves into an overall view.
- For each curve, fit linear regression over data points and report slope.
- Compare slopes across:
  - full-spectrum vs band-only vs cumulative-band settings,
  - FFT vs DCT (if both are available),
  - CPU/GPU backend variants (when relevant).

### Why
- We currently inspect many per-genre plots, but we need one global trend metric.
- Slope gives a compact indicator of whether adding higher-frequency bands helps or hurts performance.

### Metrics to report
- `slope`, `intercept`, `R^2`, and confidence interval (or bootstrap interval) for each trend line.
- Mean slope across genres + std.
- Outlier genres with opposite slope sign.

### Deliverables
- `results/2nd_exp/overall_genre_trends.json`
- `results/2nd_exp/overall_genre_trends.png`
- Short table in report: top positive-slope / negative-slope genres.

---

## 2) Feature-Dimension Analysis: Add Std Over Features on Weight Plot

### What to do
- Extend current spectral weight visualization:
  - keep mean weight-per-frequency curve,
  - add standard deviation across feature dimensions as uncertainty band/error bar.
- Beyond 3D inspection, analyze per-feature contribution statistics:
  - mean, std, and optionally percentile band (P25/P75) over features.

### Why
- Current 3D visual checks are not enough to explain "which feature groups are stable vs noisy".
- Std-over-features can reveal whether peaks are universal or driven by a small subset of features.

### Metrics to report
- For each frequency bin:
  - `mean_weight_over_features`,
  - `std_weight_over_features`,
  - optional `median`, `iqr`.
- Rank frequencies by signal-to-variation ratio: `abs(mean) / (std + eps)`.

### Deliverables
- `results/2nd_exp/feature_weight_stats_<task>.json`
- `results/2nd_exp/feature_weight_mean_std_<task>.png`
- One comparison figure: old weight curve vs new mean±std curve.

---

## 3) Attention-Head Spectral Difference Study

### What to do
- Treat attention heads as separate analysis units and compare band preference:
  - extract head-level representations (or head-specific outputs if available),
  - run same spectral probing pipeline per head.
- Test whether different heads focus on different frequency bands.
- Add a practical check: whether selecting/combining head-specific spectral features improves classification consistently.

### Why
- Hypothesis: attention heads specialize on different spectral patterns.
- If true, head-aware feature selection may improve robustness and accuracy.

### Metrics to report
- Per-head:
  - best band accuracy,
  - cumulative curve slope,
  - spectral weight profile.
- Across heads:
  - pairwise profile similarity/correlation,
  - diversity score (variance across heads),
  - improvement from head-aware ensemble vs baseline.

### Success criterion
- Classification improvement is consistent across seeds/splits (not only one run).
- At least one head-aware strategy beats baseline by a meaningful margin (define threshold per task, e.g. +1~2% absolute).

### Deliverables
- `results/2nd_exp/attention_head_profiles/`
- `results/2nd_exp/attention_head_comparison.md`

---

## 4) Expand to More Tasks and Datasets

### Candidate tasks
- Genre classification
- Chord recognition/classification
- Music/audio tagging (multi-label detection)
- Beat/downbeat related task
- Key detection/classification

### Dataset strategy (very important)
- Prioritize frequently used, reusable benchmarks with broad adoption.
- Prefer datasets with multiple labels per track (genre + tags + key/chord/beat annotations) to avoid repeated feature extraction.
- Build one shared embedding cache, many task-specific label views.

### Proposed dataset shortlist (to confirm by availability/licensing)
- FMA (genre, metadata-driven tasks)
- GTZAN (genre baseline)
- MagnaTagATune / MTG-Jamendo (audio tagging, multi-label)
- GiantSteps / key-oriented sets (key)
- Chord-related public sets (for chord tasks)
- Beat-oriented sets (for beat/downbeat)

### Reuse protocol
- One canonical embedding artifact per audio collection.
- Separate label mapping files per task.
- Task runners should load same embeddings and switch only target labels + evaluation protocol.

### Deliverables
- `data_artifacts/task_label_maps/` (task-specific labels referencing shared track ids)
- `results/2nd_exp/multitask_summary.csv`

---

## 5) Visualize More vs Run More Experiments (Balanced Plan)

### Visualization upgrades
- Aggregated slope dashboard across genres/tasks.
- Mean±std feature-weight curves.
- Attention-head band preference heatmap.
- Task-wise comparison matrix (task x band strategy).

### Additional experiments
- Seed stability (>=3 seeds).
- Split robustness (if multiple official splits exist).
- Backend consistency check (CPU vs GPU/ROCm path where relevant).

### Decision rule
- If a trend is not stable across seeds/splits, prioritize more experiments.
- If trend is stable but not interpretable, prioritize visualization.

---

## 6) Execution Plan and Milestones

### Milestone A: Aggregated trend + slope
- Implement aggregation scripts.
- Export trend JSON/plots.
- Implementation method:
  - Input: reuse per-genre metrics JSON under `results/fft_genre_specific_analysis/` (and optional task folders if available).
  - Build one unified table with columns: `task`, `genre`, `method` (`full|band|cumulative|auto`), `band_idx`, `accuracy`.
  - For each `(task, genre, method)` curve, fit linear regression on `(x=band_idx, y=accuracy)`, store `slope`, `intercept`, `R2`, `n_points`.
  - Aggregate by task and globally: mean/std slope, positive-slope ratio, and outlier genres with opposite sign.
  - Save:
    - `results/2nd_exp/overall_genre_trends.json` (all per-curve regression stats)
    - `results/2nd_exp/overall_genre_trends.csv` (flat table for quick filtering)
    - `results/2nd_exp/overall_genre_trends.png` (slope distribution + task-level summary)
  - Script entrypoint:
    - `python scripts/aggregate_spectral_trends.py --metrics_glob "results/**/*_spectral_summary_*_metrics.json" --output_dir results/2nd_exp --min_points 3`
  - Validation:
    - check monotonic `band_idx`,
    - drop invalid curves with `<3` points,
    - print data coverage (`#genres`, `#curves`) in log/markdown summary.

### Milestone B: Feature std on weight plots
- Add std-over-feature stats and visualization.
- Integrate into current report figures.
- Implementation method:
  - Input: probe weights from full-spectrum model (`coef_`) or saved weight arrays from existing spectral profile outputs.
  - Reshape weights to `(n_features, n_coeffs)` consistently with current pipeline.
  - Compute per-frequency statistics across features:
    - `mean_weight_over_features`,
    - `std_weight_over_features`,
    - optional `median`, `q25`, `q75`.
  - Visualization update:
    - plot mean curve + shaded `mean ± std`,
    - optionally overlay percentile band for robustness.
  - Save:
    - `results/2nd_exp/feature_weight_stats_<task>.json`
    - `results/2nd_exp/feature_weight_stats_<task>.csv`
    - `results/2nd_exp/feature_weight_mean_std_<task>.png`
  - Script entrypoint:
    - rerun probing with `python scripts/run_generic_fft_band_probe.py ...` (now auto-exports `feature_weight_stats_*.json/.csv` and `feature_weight_mean_std_*.png` in `results_dir`)
  - Integration:
    - add one comparison panel in report: old weight-only vs new mean±std;
    - keep same normalization rule as prior plots and document it in caption.
  - Validation:
    - ensure no feature-axis mismatch after reshape,
    - assert `n_coeffs` matches transform setting (`rfft(T)` or configured DCT coeff count).

### Milestone C: Attention-head analysis
- Build head-level extraction/adapter path.
- Run probing per head + head-aware fusion.

### Milestone D: Multi-task benchmark expansion
- Finalize dataset list based on data availability.
- Reuse one embedding cache across tasks.
- Generate multitask result table.

### Milestone E: FMA-medium 8-shard streaming (large-scale, imbalanced genres)
- Goal:
  - Avoid OOM / GPU-memory bottlenecks on very large shard NPZs.
  - Run genre spectral probing in shard-streaming mode without merging all shards into one giant NPZ.
  - Handle long-tail class imbalance explicitly.
- Why streaming:
  - `passt_fma_medium_shard0..7.npz` is too large for "load-all + fit" flow.
  - Streaming processes shard-by-shard with incremental scaler/classifier updates.
  - This mode is CPU-friendly and does not require fitting all tensors in GPU memory.
- Imbalance handling policy:
  - Use one-vs-rest incremental logistic (`SGDClassifier`) with `class_weight="balanced"` (already implemented in streaming script).
  - Keep a minimum support threshold (`MIN_SAMPLES_PER_CLASS=10`) to skip unreliable micro-classes.
  - Report both macro-level behavior (class-wise profile) and class counts in summary JSON.
- Implementation method:
  - Use `scripts/run_passt_spectral_profiles.py --npz_paths ... --streaming` to iterate through shards.
  - For each eligible genre:
    - stream FFT transform per shard,
    - update scaler via `partial_fit`,
    - update classifier via `partial_fit`,
    - compute online test accuracy and spectral profile.
  - Save per-genre plots + global summary JSON.
- Execution steps (recommended):
  - Step E1 (smoke test on 2 shards):
    - `python scripts/run_passt_spectral_profiles.py --npz_paths data_artifacts/passt_fma_medium_shard0.npz data_artifacts/passt_fma_medium_shard1.npz --label_key genres --results_dir results/passt_fma_medium_streaming_smoke --prefix passt_fma_medium_stream --streaming`
  - Step E2 (full 8-shard run):
    - `python scripts/run_passt_spectral_profiles.py --npz_paths data_artifacts/passt_fma_medium_shard0.npz data_artifacts/passt_fma_medium_shard1.npz data_artifacts/passt_fma_medium_shard2.npz data_artifacts/passt_fma_medium_shard3.npz data_artifacts/passt_fma_medium_shard4.npz data_artifacts/passt_fma_medium_shard5.npz data_artifacts/passt_fma_medium_shard6.npz data_artifacts/passt_fma_medium_shard7.npz --label_key genres --results_dir results/passt_fma_medium_streaming --prefix passt_fma_medium_stream --streaming`
  - Step E3 (aggregate trend from produced JSON metrics where applicable):
    - keep using `scripts/aggregate_spectral_trends.py` for files that follow `*_spectral_summary_*_metrics.json`;
    - for streaming summary JSON, add a lightweight adapter (next subtask) if direct schema mismatch appears.
- Validation checklist:
  - Confirm summary JSON exists: `results/passt_fma_medium_streaming/passt_fma_medium_stream_spectral_summary_genre.json` (or actual label suffix).
  - Confirm every major genre has a PNG profile and non-zero sample count.
  - Compare streaming accuracy vs small non-streaming baseline on overlapping genres to estimate approximation gap.
- Optional enhancement (if memory still tight on single shard load):
  - Add true chunk-level NPZ iteration (batch/chunk reads) to replace per-shard full load in `_streaming_spectral_profiles`.
  - Keep same `partial_fit` interface so experiment protocol stays unchanged.

---

## 7) Risks and Mitigation

- Risk: task labels are noisy/incomplete.
  - Mitigation: track label coverage and report effective sample count per task.
- Risk: multi-task comparisons become unfair due to different splits/metrics.
  - Mitigation: standardize evaluation templates and log exact split protocol.
- Risk: head-level extraction cost is high.
  - Mitigation: start with subset, then scale after confirming signal.

---

## 8) Immediate Next Actions

1. Implement overall genre trend aggregation with regression slope reporting.
2. Upgrade spectral weight plot to include feature-dimension std.
3. Prototype attention-head differential analysis on one dataset first.
4. Finalize multi-task dataset list with "extract once, reuse many labels" priority.
5. Run Milestone E streaming smoke test on shard0-1, then launch full 8-shard streaming run.

