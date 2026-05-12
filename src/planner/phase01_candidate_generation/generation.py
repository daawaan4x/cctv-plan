from __future__ import annotations

from pathlib import Path

from src.common.floorplan import FloorPlanInput
from src.planner._shared.cache import get_shared_artifact_dir
from src.planner._shared.config import PlannerConfig

from .artifacts import CandidateGenerationArtifacts
from .constants import PHASE_ARTIFACT_STEM, PHASE_NAME
from .geometry import (
    _build_directional_masks,
    _build_eligible_candidate_arrays,
    _build_open_cell_indices,
    _flat_indices_to_coords,
    _thin_candidate_set,
)
from .io import load_candidate_generation_artifacts, save_candidate_generation_artifacts
from .validation import validate_candidate_generation_artifacts


def generate_candidate_generation_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig | None = None,
) -> CandidateGenerationArtifacts:
    """Build eligible and thinned candidate-camera arrays for phase 01."""

    resolved_config = config or PlannerConfig(floorplan_name=floorplan.name)
    grid = floorplan.grid
    _, width = floorplan.shape

    open_cell_indices = _build_open_cell_indices(grid)
    directional_masks = _build_directional_masks(grid)
    (
        eligible_candidate_cell_indices,
        eligible_candidate_boundary_flags,
        eligible_candidate_solid_flags,
    ) = _build_eligible_candidate_arrays(directional_masks)
    (
        candidate_cell_indices,
        candidate_boundary_flags,
        candidate_exception_flags,
    ) = _thin_candidate_set(
        floorplan,
        resolved_config,
        eligible_candidate_cell_indices,
        eligible_candidate_boundary_flags,
        eligible_candidate_solid_flags,
        directional_masks,
    )

    artifacts = CandidateGenerationArtifacts(
        grid_shape=floorplan.shape,
        open_cell_indices=open_cell_indices,
        open_cell_coords_rc=_flat_indices_to_coords(open_cell_indices, width),
        eligible_candidate_cell_indices=eligible_candidate_cell_indices,
        eligible_candidate_cell_coords_rc=_flat_indices_to_coords(
            eligible_candidate_cell_indices,
            width,
        ),
        eligible_candidate_boundary_flags=eligible_candidate_boundary_flags,
        candidate_cell_indices=candidate_cell_indices,
        candidate_cell_coords_rc=_flat_indices_to_coords(candidate_cell_indices, width),
        candidate_boundary_flags=candidate_boundary_flags,
        candidate_exception_flags=candidate_exception_flags,
    )
    validate_candidate_generation_artifacts(
        floorplan,
        artifacts,
        config=resolved_config,
    )
    return artifacts

def resolve_candidate_generation_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    force: bool = False,
) -> CandidateGenerationArtifacts:
    """Load, validate, or rebuild the canonical cached phase-01 artifact."""

    artifact_path = (
        get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
        / f"{PHASE_ARTIFACT_STEM}.npz"
    )
    if not force and artifact_path.exists():
        try:
            artifacts = load_candidate_generation_artifacts(artifact_path)
            validate_candidate_generation_artifacts(
                floorplan,
                artifacts,
                config=config,
            )
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    artifacts = generate_candidate_generation_artifacts(floorplan, config)
    save_candidate_generation_artifacts(artifact_path, artifacts)
    return artifacts
