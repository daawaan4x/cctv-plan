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
