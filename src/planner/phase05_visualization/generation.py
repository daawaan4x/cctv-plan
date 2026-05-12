from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pulp

from src.common.floorplan import FloorPlanInput
from src.planner._shared.cache import get_shared_artifact_dir
from src.planner._shared.config import PlannerConfig
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)
from src.planner.phase04_optimization import (
    OptimizationArtifacts,
    resolve_optimization_artifact,
    resolve_optimization_artifacts_for_k_values,
)

from .artifacts import VisualizationArtifacts
from .constants import PHASE_ARTIFACT_STEM
from .io import (
    load_visualization_artifacts,
    save_visualization_artifacts,
    save_visualization_summary,
)
from .metrics import compute_coverage_metrics
from .transforms import _decode_selected_candidate_coords, _reconstruct_final_score_grid
from .validation import _validate_phase_dependencies, validate_visualization_artifacts


def build_visualization_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase04_artifacts: OptimizationArtifacts,
) -> VisualizationArtifacts:
    """Build the deterministic phase-05 artifact from phase 01 and phase 04 outputs."""

    _validate_phase_dependencies(floorplan, phase01_artifacts, phase04_artifacts)

    # Phase 04 stores all target-facing vectors in the canonical row-major open-cell
    # ordinal order from phase 01. Phase 05's main job is to scatter those vectors
    # back onto the full floor-plan grid without losing the null/open/solid semantics.
    final_score_grid = _reconstruct_final_score_grid(
        floorplan.shape,
        phase01_artifacts.open_cell_coords_rc,
        phase04_artifacts.final_open_cell_scores,
    )
    blind_spot_mask = floorplan.open_mask & (final_score_grid == 0)
    selected_candidate_coords_rc = _decode_selected_candidate_coords(
        phase01_artifacts.candidate_cell_coords_rc,
        phase04_artifacts.selected_candidate_ordinals,
    )
    metrics = compute_coverage_metrics(phase04_artifacts.final_open_cell_scores)

    artifacts = VisualizationArtifacts(
        grid_shape=floorplan.shape,
        open_cell_count=phase04_artifacts.open_cell_count,
        candidate_count=phase04_artifacts.candidate_count,
        configuration_count=phase04_artifacts.configuration_count,
        solved_k=phase04_artifacts.solved_k,
        solver_name=phase04_artifacts.solver_name,
        solver_status=phase04_artifacts.solver_status,
        selected_camera_count=int(
            len(phase04_artifacts.selected_configuration_ordinals)
        ),
        metrics=metrics,
        final_open_cell_scores=phase04_artifacts.final_open_cell_scores.astype(
            np.int8, copy=False
        ),
        final_score_grid=final_score_grid,
        blind_spot_mask=blind_spot_mask.astype(np.bool_, copy=False),
        selected_configuration_ordinals=phase04_artifacts.selected_configuration_ordinals.astype(
            np.int32, copy=False
        ),
        selected_candidate_ordinals=phase04_artifacts.selected_candidate_ordinals.astype(
            np.int32, copy=False
        ),
        selected_candidate_coords_rc=selected_candidate_coords_rc,
        selected_angle_ordinals=phase04_artifacts.selected_angle_ordinals.astype(
            np.int16, copy=False
        ),
        selected_angles_deg=phase04_artifacts.selected_angles_deg.astype(
            np.float32, copy=False
        ),
        best_configuration_ordinals=phase04_artifacts.best_configuration_ordinals.astype(
            np.int32, copy=False
        ),
    )
    validate_visualization_artifacts(
        floorplan,
        phase01_artifacts,
        phase04_artifacts,
        artifacts,
    )
    return artifacts

def ensure_visualization_artifacts(
    artifact_path: Path,
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase04_artifacts: OptimizationArtifacts,
) -> VisualizationArtifacts:
    """Load a persisted phase-05 artifact when present, otherwise build and save it."""

    if artifact_path.exists():
        artifacts = load_visualization_artifacts(artifact_path)
        validate_visualization_artifacts(
            floorplan,
            phase01_artifacts,
            phase04_artifacts,
            artifacts,
        )
        return artifacts

    artifacts = build_visualization_artifacts(
        floorplan,
        phase01_artifacts,
        phase04_artifacts,
    )
    save_visualization_artifacts(artifact_path, artifacts)
    return artifacts

def resolve_visualization_artifact(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    k: int,
    force: bool = False,
    solver: pulp.LpSolver | None = None,
) -> VisualizationArtifacts:
    """Load, validate, or build one canonical cached per-`K` phase-05 artifact."""

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    phase04_artifacts = resolve_optimization_artifact(
        floorplan,
        config,
        repo_root=repo_root,
        k=k,
        force=force,
        solver=solver,
    )
    artifact_dir = get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
    artifact_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{k}.npz"
    summary_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{k}_summary.json"

    if not force and artifact_path.exists():
        try:
            artifacts = load_visualization_artifacts(artifact_path)
            validate_visualization_artifacts(
                floorplan,
                phase01_artifacts,
                phase04_artifacts,
                artifacts,
            )
            if not summary_path.exists():
                save_visualization_summary(summary_path, artifacts)
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    artifacts = build_visualization_artifacts(
        floorplan,
        phase01_artifacts,
        phase04_artifacts,
    )
    save_visualization_artifacts(artifact_path, artifacts)
    save_visualization_summary(summary_path, artifacts)
    return artifacts

def resolve_visualization_artifacts_for_k_values(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    k_values: Sequence[int] | None = None,
    force: bool = False,
    solver: pulp.LpSolver | None = None,
) -> tuple[VisualizationArtifacts, ...]:
    """Resolve phase-05 artifacts for one or more `K` values in request order."""

    requested_k_values = tuple(config.k_values if k_values is None else k_values)
    if not requested_k_values:
        return ()

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    phase04_artifacts_by_k = resolve_optimization_artifacts_for_k_values(
        floorplan,
        config,
        repo_root=repo_root,
        k_values=requested_k_values,
        force=force,
        solver=solver,
    )
    artifact_dir = get_shared_artifact_dir(floorplan, config, repo_root=repo_root)

    resolved_by_k: dict[int, VisualizationArtifacts] = {}
    for phase04_artifacts in phase04_artifacts_by_k:
        solved_k = int(phase04_artifacts.solved_k)
        artifact_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{solved_k}.npz"
        summary_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{solved_k}_summary.json"

        if not force and artifact_path.exists():
            try:
                artifacts = load_visualization_artifacts(artifact_path)
                validate_visualization_artifacts(
                    floorplan,
                    phase01_artifacts,
                    phase04_artifacts,
                    artifacts,
                )
                if not summary_path.exists():
                    save_visualization_summary(summary_path, artifacts)
                resolved_by_k[solved_k] = artifacts
                continue
            except (KeyError, TypeError, ValueError):
                pass

        artifacts = build_visualization_artifacts(
            floorplan,
            phase01_artifacts,
            phase04_artifacts,
        )
        save_visualization_artifacts(artifact_path, artifacts)
        save_visualization_summary(summary_path, artifacts)
        resolved_by_k[solved_k] = artifacts

    return tuple(resolved_by_k[int(requested_k)] for requested_k in requested_k_values)

