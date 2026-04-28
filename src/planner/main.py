"""Workspace-loading helpers for the planner notebooks and scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.floorplan import FloorPlanInput
from src.common.floorplan_loader import load_traced_floorplan

from ._shared.bootstrap import find_repo_root, get_traced_floorplan_path
from ._shared.cache import ensure_shared_artifact_dir, write_manifest
from ._shared.config import PlannerConfig


# Workspace bundle
@dataclass(frozen=True, slots=True)
class PlannerWorkspace:
    """Bundle the resolved repo root, floor plan, config, and artifact directory."""

    repo_root: Path
    floorplan: FloorPlanInput
    config: PlannerConfig
    artifact_dir: Path


# Workspace loading
def load_workspace(config: PlannerConfig | None = None) -> PlannerWorkspace:
    """Resolve the default planner workspace for the configured traced floor plan."""

    resolved_config = config or PlannerConfig()
    # The workspace loader is the single choke point where repo layout, floor-plan
    # metadata, and artifact cache setup come together. Later phase notebooks can
    # assume these invariants instead of rediscovering paths in each notebook.
    repo_root = find_repo_root()
    floorplan_path = get_traced_floorplan_path(
        resolved_config.floorplan_name,
        repo_root=repo_root,
    )
    floorplan = load_traced_floorplan(floorplan_path)
    artifact_dir = ensure_shared_artifact_dir(
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


# Workspace persistence
def write_workspace_manifest(workspace: PlannerWorkspace) -> Path:
    """Write the standard artifact manifest for an already loaded workspace."""

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
