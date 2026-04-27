"""Small path-resolution helpers shared by planner scripts and notebooks."""

from __future__ import annotations

from pathlib import Path


# Repository path discovery
def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until the repository root containing `pyproject.toml` is found."""

    current = (start or Path.cwd()).resolve()
    # The repo is identified by both `pyproject.toml` and `src/` so that an
    # unrelated ancestor directory with only one of those names does not get
    # misidentified as the planner workspace root.
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from the current path.")


# Static asset resolution
def get_traced_floorplan_path(
    floorplan_name: str,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Return the traced PNG path for one named floor plan in the repo assets."""

    resolved_repo_root = repo_root or find_repo_root()
    return (
        resolved_repo_root
        / "static"
        / "floor-plan"
        / "traced"
        / f"{floorplan_name}.png"
    )


__all__ = [
    "find_repo_root",
    "get_traced_floorplan_path",
]
