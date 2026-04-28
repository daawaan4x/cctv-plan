"""Small path-resolution helpers shared by planner scripts and notebooks."""

from __future__ import annotations

from collections.abc import Sequence
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


def list_traced_floorplan_names(
    *,
    repo_root: Path | None = None,
) -> tuple[str, ...]:
    """Return the traced floor-plan names available under the repo assets."""

    resolved_repo_root = repo_root or find_repo_root()
    traced_dir = resolved_repo_root / "static" / "floor-plan" / "traced"
    floorplan_names: list[str] = []
    for traced_png_path in sorted(traced_dir.glob("*.png")):
        metadata_path = traced_png_path.with_suffix(".json")
        if metadata_path.exists():
            floorplan_names.append(traced_png_path.stem)
    return tuple(floorplan_names)


__all__ = [
    "find_repo_root",
    "get_traced_floorplan_path",
    "list_traced_floorplan_names",
]
