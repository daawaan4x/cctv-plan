from __future__ import annotations

from pathlib import Path

from src.common.floorplan import FloorPlanInput
from src.planner._shared.cache import get_shared_artifact_dir
from src.planner._shared.config import PlannerConfig
from src.planner._shared.sparse import build_offsets_from_counts as _build_offsets_from_counts
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)

from .artifacts import VisibilityArtifacts
from .constants import PHASE_ARTIFACT_STEM, PHASE_NAME
from .io import load_visibility_artifacts, save_visibility_artifacts
from .line_of_sight import _count_visibility_pairs, _fill_visibility_pairs
from .validation import _validate_phase01_compatibility, validate_visibility_artifacts


def generate_visibility_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
) -> VisibilityArtifacts:
    """Build deterministic sparse LOS artifacts for all candidate-open pairs."""

    _validate_phase01_compatibility(floorplan, phase01_artifacts)

    grid = floorplan.grid
    candidate_coords = phase01_artifacts.candidate_cell_coords_rc
    open_coords = phase01_artifacts.open_cell_coords_rc

    visible_counts, diagonal_counts = _count_visibility_pairs(
        grid,
        candidate_coords,
        open_coords,
    )
    los_candidate_offsets = _build_offsets_from_counts(visible_counts)
    diagonal_candidate_offsets = _build_offsets_from_counts(diagonal_counts)
    los_target_ordinals, diagonal_target_ordinals = _fill_visibility_pairs(
        grid,
        candidate_coords,
        open_coords,
        los_candidate_offsets,
        diagonal_candidate_offsets,
    )

    artifacts = VisibilityArtifacts(
        grid_shape=floorplan.shape,
        candidate_count=len(candidate_coords),
        open_cell_count=len(open_coords),
        los_candidate_offsets=los_candidate_offsets,
        los_target_ordinals=los_target_ordinals,
        diagonal_candidate_offsets=diagonal_candidate_offsets,
        diagonal_target_ordinals=diagonal_target_ordinals,
    )
    validate_visibility_artifacts(floorplan, phase01_artifacts, artifacts)
    return artifacts

def resolve_visibility_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    force: bool = False,
) -> VisibilityArtifacts:
    """Load, validate, or rebuild the canonical cached phase-02 artifact."""

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    artifact_path = (
        get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
        / f"{PHASE_ARTIFACT_STEM}.npz"
    )
    if not force and artifact_path.exists():
        try:
            artifacts = load_visibility_artifacts(artifact_path)
            validate_visibility_artifacts(floorplan, phase01_artifacts, artifacts)
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)
    save_visibility_artifacts(artifact_path, artifacts)
    return artifacts
