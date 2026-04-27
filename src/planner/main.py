from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.floorplan import FloorPlanInput
from src.common.floorplan_loader import load_traced_floorplan

from ._shared.bootstrap import find_repo_root, get_traced_floorplan_path
from ._shared.cache import ensure_artifact_dir, write_manifest
from ._shared.config import PlannerConfig


@dataclass(frozen=True, slots=True)
class PlannerWorkspace:
    repo_root: Path
    floorplan: FloorPlanInput
    config: PlannerConfig
    artifact_dir: Path


def load_workspace(config: PlannerConfig | None = None) -> PlannerWorkspace:
    resolved_config = config or PlannerConfig()
    repo_root = find_repo_root()
    floorplan_path = get_traced_floorplan_path(
        resolved_config.floorplan_name,
        repo_root=repo_root,
    )
    floorplan = load_traced_floorplan(floorplan_path)
    artifact_dir = ensure_artifact_dir(
        floorplan,
        resolved_config,
        repo_root=repo_root,
    )
    return PlannerWorkspace(
        repo_root=repo_root,
        floorplan=floorplan,
        config=resolved_config,
        artifact_dir=artifact_dir,
    )


def write_workspace_manifest(workspace: PlannerWorkspace) -> Path:
    return write_manifest(
        workspace.floorplan,
        workspace.config,
        repo_root=workspace.repo_root,
    )


__all__ = [
    "PlannerWorkspace",
    "load_workspace",
    "write_workspace_manifest",
]
