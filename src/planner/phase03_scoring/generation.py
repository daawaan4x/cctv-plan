"""Phase-03 sparse score generation and cache resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.floorplan import FloorPlanInput
from src.planner._shared.cache import get_shared_artifact_dir
from src.planner._shared.config import PlannerConfig
from src.planner._shared.sparse import build_offsets_from_counts as _build_offsets_from_counts
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)
from src.planner.phase02_visibility import (
    VisibilityArtifacts,
    resolve_visibility_artifacts,
)

from .artifacts import SparseScoreArtifacts
from .io import load_sparse_score_artifacts, save_sparse_score_artifacts
from .scoring import (
    _build_candidate_base_score_arrays,
    _build_configuration_index_arrays,
    _build_inside_fov_mask,
    _build_orientation_angles_array,
    _build_scoring_constants,
    _require_grid_cell_size_m,
)
from .validation import _validate_phase_dependencies, validate_sparse_score_artifacts

PHASE_NAME = "scoring"
PHASE_ARTIFACT_STEM = "03_sparse_scores"


def generate_sparse_score_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase02_artifacts: VisibilityArtifacts,
) -> SparseScoreArtifacts:
    """Build deterministic sparse score artifacts for all candidate orientations."""

    grid_cell_size_m = _require_grid_cell_size_m(floorplan)
    orientation_angles_deg = _build_orientation_angles_array(config)
    _validate_phase_dependencies(
        floorplan,
        config,
        phase01_artifacts,
        phase02_artifacts,
        orientation_angles_deg,
    )

    candidate_count = len(phase01_artifacts.candidate_cell_indices)
    open_coords_rc = phase01_artifacts.open_cell_coords_rc
    candidate_coords_rc = phase01_artifacts.candidate_cell_coords_rc
    orientation_count = len(orientation_angles_deg)
    configuration_candidate_ordinals, configuration_angle_ordinals = (
        _build_configuration_index_arrays(candidate_count, orientation_count)
    )

    scoring_constants = _build_scoring_constants(config, grid_cell_size_m)
    score_counts = np.zeros(candidate_count * orientation_count, dtype=np.int32)

    for candidate_ordinal in range(candidate_count):
        base_target_ordinals, base_target_angles_deg, _ = (
            _build_candidate_base_score_arrays(
                candidate_ordinal,
                candidate_coords_rc,
                open_coords_rc,
                phase02_artifacts,
                scoring_constants,
            )
        )
        if len(base_target_ordinals) == 0:
            continue

        configuration_base = candidate_ordinal * orientation_count
        for angle_ordinal, orientation_deg in enumerate(orientation_angles_deg):
            inside_fov_mask = _build_inside_fov_mask(
                base_target_angles_deg,
                float(orientation_deg),
                scoring_constants.half_fov_deg,
            )
            score_counts[configuration_base + angle_ordinal] = np.int32(
                np.count_nonzero(inside_fov_mask)
            )

    score_configuration_offsets = _build_offsets_from_counts(score_counts)
    score_target_ordinals = np.empty(
        int(score_configuration_offsets[-1]),
        dtype=np.int32,
    )
    score_values = np.empty(int(score_configuration_offsets[-1]), dtype=np.int8)

    for candidate_ordinal in range(candidate_count):
        base_target_ordinals, base_target_angles_deg, base_target_scores = (
            _build_candidate_base_score_arrays(
                candidate_ordinal,
                candidate_coords_rc,
                open_coords_rc,
                phase02_artifacts,
                scoring_constants,
            )
        )
        configuration_base = candidate_ordinal * orientation_count
        if len(base_target_ordinals) == 0:
            continue

        for angle_ordinal, orientation_deg in enumerate(orientation_angles_deg):
            configuration_ordinal = configuration_base + angle_ordinal
            write_start = int(score_configuration_offsets[configuration_ordinal])
            write_stop = int(score_configuration_offsets[configuration_ordinal + 1])
            if write_start == write_stop:
                continue

            inside_fov_mask = _build_inside_fov_mask(
                base_target_angles_deg,
                float(orientation_deg),
                scoring_constants.half_fov_deg,
            )
            configuration_target_ordinals = base_target_ordinals[inside_fov_mask]
            configuration_scores = base_target_scores[inside_fov_mask]
            score_target_ordinals[write_start:write_stop] = configuration_target_ordinals
            score_values[write_start:write_stop] = configuration_scores

    artifacts = SparseScoreArtifacts(
        grid_shape=floorplan.shape,
        candidate_count=candidate_count,
        open_cell_count=len(phase01_artifacts.open_cell_indices),
        orientation_angles_deg=orientation_angles_deg,
        configuration_candidate_ordinals=configuration_candidate_ordinals,
        configuration_angle_ordinals=configuration_angle_ordinals,
        score_configuration_offsets=score_configuration_offsets,
        score_target_ordinals=score_target_ordinals,
        score_values=score_values,
    )
    validate_sparse_score_artifacts(
        floorplan,
        config,
        phase01_artifacts,
        phase02_artifacts,
        artifacts,
    )
    return artifacts


def resolve_sparse_score_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    force: bool = False,
) -> SparseScoreArtifacts:
    """Load, validate, or rebuild the canonical cached phase-03 artifact."""

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    phase02_artifacts = resolve_visibility_artifacts(
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
            artifacts = load_sparse_score_artifacts(artifact_path)
            validate_sparse_score_artifacts(
                floorplan,
                config,
                phase01_artifacts,
                phase02_artifacts,
                artifacts,
            )
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    artifacts = generate_sparse_score_artifacts(
        floorplan,
        config,
        phase01_artifacts,
        phase02_artifacts,
    )
    save_sparse_score_artifacts(artifact_path, artifacts)
    return artifacts


__all__ = [
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
    "generate_sparse_score_artifacts",
    "resolve_sparse_score_artifacts",
]
