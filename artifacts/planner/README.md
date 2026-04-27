# Planner Artifacts

This directory is reserved for deterministic planner outputs that are safe to reuse across notebooks and future sessions.

Expected layout:

```text
artifacts/planner/<floorplan_name>/<config_fingerprint>/
  manifest.json
  01_candidates.npz
  02_visibility.npz
  03_sparse_scores.npz
  04_solution_K*.json
  05_metrics.csv
```

Caching rules:

- Persist only deterministic outputs derived from the locked tri-state model.
- The cache fingerprint should vary when floorplan identity, `grid_cell_size_m`, camera resolution, field of view, orientation step, `K`, or DORI thresholds change.
- Do not treat cached artifacts as ground truth if the floorplan, configuration, or scoring rules changed.
