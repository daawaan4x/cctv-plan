from __future__ import annotations

from pathlib import Path
import sys


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from the current path.")


def ensure_repo_on_sys_path(start: Path | None = None) -> Path:
    repo_root = find_repo_root(start)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def get_traced_floorplan_path(
    floorplan_name: str,
    *,
    repo_root: Path | None = None,
) -> Path:
    resolved_repo_root = repo_root or find_repo_root()
    return (
        resolved_repo_root
        / "static"
        / "floor-plan"
        / "traced"
        / f"{floorplan_name}.png"
    )


__all__ = [
    "ensure_repo_on_sys_path",
    "find_repo_root",
    "get_traced_floorplan_path",
]
