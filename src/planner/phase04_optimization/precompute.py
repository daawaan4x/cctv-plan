from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.floorplan import FloorPlanInput
from src.planner._shared.cache import get_shared_artifact_dir
from src.planner._shared.config import PlannerConfig
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)
from src.planner.phase02_visibility import resolve_visibility_artifacts
from src.planner.phase03_scoring import SparseScoreArtifacts, resolve_sparse_score_artifacts

from .artifacts import OptimizationPrecomputeArtifacts
from .constants import PHASE_PRECOMPUTE_ARTIFACT_STEM
from .io import (
    load_optimization_precompute_artifacts,
    save_optimization_precompute_artifacts,
)
from .threshold_index import _build_threshold_cover_index
from .validation import (
    _validate_phase_dependencies,
    validate_optimization_precompute_artifacts,
)


def build_optimization_precompute_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
) -> OptimizationPrecomputeArtifacts:
    """Build the persisted reusable threshold-index artifacts for phase 04."""

    _validate_phase_dependencies(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )
    threshold_cover_index = _build_threshold_cover_index(phase03_artifacts)
    candidate_configuration_offsets = _build_candidate_configuration_offsets(
        phase03_artifacts.configuration_candidate_ordinals,
        candidate_count=len(phase01_artifacts.candidate_cell_indices),
    )
    artifacts = OptimizationPrecomputeArtifacts(
        grid_shape=floorplan.shape,
        open_cell_count=phase03_artifacts.open_cell_count,
        candidate_count=phase03_artifacts.candidate_count,
        configuration_count=len(phase03_artifacts.configuration_candidate_ordinals),
        candidate_configuration_offsets=candidate_configuration_offsets,
        level_offsets=threshold_cover_index.level_offsets,
        level_configuration_ordinals=threshold_cover_index.level_configuration_ordinals,
    )
    validate_optimization_precompute_artifacts(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
        artifacts,
    )
    return artifacts

def ensure_optimization_precompute_artifacts(
    artifact_path: Path,
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
) -> OptimizationPrecomputeArtifacts:
    """Load a persisted precompute artifact when present, otherwise build and save it."""

    if artifact_path.exists():
        artifacts = load_optimization_precompute_artifacts(artifact_path)
        validate_optimization_precompute_artifacts(
            floorplan,
            phase01_artifacts,
            phase03_artifacts,
            artifacts,
        )
        return artifacts

    artifacts = build_optimization_precompute_artifacts(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )
    save_optimization_precompute_artifacts(artifact_path, artifacts)
    return artifacts

def resolve_optimization_precompute_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    force: bool = False,
) -> OptimizationPrecomputeArtifacts:
    """Load, validate, or rebuild the canonical cached phase-04 precompute artifact."""

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    _ = resolve_visibility_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    phase03_artifacts = resolve_sparse_score_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    artifact_path = (
        get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
        / f"{PHASE_PRECOMPUTE_ARTIFACT_STEM}.npz"
    )
    if not force and artifact_path.exists():
        try:
            artifacts = load_optimization_precompute_artifacts(artifact_path)
            validate_optimization_precompute_artifacts(
                floorplan,
                phase01_artifacts,
                phase03_artifacts,
                artifacts,
            )
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    artifacts = build_optimization_precompute_artifacts(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )
    save_optimization_precompute_artifacts(artifact_path, artifacts)
    return artifacts

def _build_candidate_configuration_offsets(
    configuration_candidate_ordinals: NDArray[np.int32],
    *,
    candidate_count: int,
) -> NDArray[np.int32]:
    """Build contiguous candidate-to-configuration offsets from candidate-major order."""

    counts = np.bincount(
        configuration_candidate_ordinals.astype(np.int64, copy=False),
        minlength=candidate_count,
    ).astype(np.int32, copy=False)
    offsets = np.zeros(candidate_count + 1, dtype=np.int32)
    np.cumsum(counts, dtype=np.int32, out=offsets[1:])
    return offsets
