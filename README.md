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
- Traced PNG assets under `static/floor-plan/traced` are interpreted as:
  - transparent = null
  - black = solid
  - white = open
- Common floor-plan code lives under `src/common`.
- The notebook entrypoint for interactive loader checks is `src/planner/main.ipynb`.
