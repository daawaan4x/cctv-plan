# cctv-plan

Resolution-aware CCTV placement optimization using a tri-state floor-plan grid and DORI-based coverage scoring.

## Common Commands

### Install dependencies

```powershell
uv sync
```

### Run tests

```powershell
uv run python -m unittest discover -s tests -p "test_*.py"
```

### Launch Jupyter

```powershell
uv run jupyter lab
```

### Run linting

```powershell
uv run ruff check .
```

### Run type checking

```powershell
uv run pyright src tests
```

### Run the end-to-end planner

One floor plan, first 3 configured `k` values:

```powershell
uv run python -m src.planner.runner --floorplan ground-back --first-k-values 3
```

One floor plan, all configured `k` values:

```powershell
uv run python -m src.planner.runner --floorplan ground-back --all-k-values
```

All floor plans, first 2 configured `k` values:

```powershell
uv run python -m src.planner.runner --all-floorplans --first-k-values 2
```

All floor plans, first 2 configured `k` values, with explicit parallel worker count:

```powershell
uv run python -m src.planner.runner --all-floorplans --first-k-values 2 --workers 4
```

All floor plans, all configured `k` values, forcing fresh recomputation:

```powershell
uv run python -m src.planner.runner --all-floorplans --all-k-values --force
```

One floor plan, explicit `k` values, with a JSONL completion log:

```powershell
uv run python -m src.planner.runner --floorplan second-front --k-values 10 12 14 --status-log artifacts/planner/run-status.jsonl
```

## Notes

- The optimization input is a tri-state occupancy grid:
  - `-1 = null / out-of-bounds`
  - `0 = open`
  - `1 = solid`
- Each traced PNG under `static/floor-plan/traced` must have a sibling JSON file with `grid_cell_size_m`.
- Traced PNG assets under `static/floor-plan/traced` are interpreted as:
  - transparent = null
  - black = solid
  - white = open
- Common floor-plan code and the loader check notebook live under `src/common`.
- The optimization package, phase notebooks, and presentation notebook live under `src/planner`.
- The loader check notebook is `src/common/check.ipynb`.
- The planner presentation notebook is `src/planner/main.ipynb`.
- Deterministic reusable planner artifacts should be written under `artifacts/planner`.
