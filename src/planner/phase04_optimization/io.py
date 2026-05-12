from __future__ import annotations

from pathlib import Path

import numpy as np

from src.planner._shared.cache import write_json, write_npz

from .artifacts import OptimizationArtifacts, OptimizationPrecomputeArtifacts
from .constants import PHASE_NAME, _DORI_LEVELS
from .coverage import _compute_coverage_metrics
from .validation import (
    _validate_final_score_arrays,
    _validate_offsets,
    _validate_selected_configuration_ordinals,
)


def save_optimization_precompute_artifacts(
    artifact_path: Path,
    artifacts: OptimizationPrecomputeArtifacts,
) -> Path:
    """Persist reusable threshold-index artifacts for future repeated solves."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        open_cell_count=np.asarray(artifacts.open_cell_count, dtype=np.int32),
        candidate_count=np.asarray(artifacts.candidate_count, dtype=np.int32),
        configuration_count=np.asarray(artifacts.configuration_count, dtype=np.int32),
        candidate_configuration_offsets=artifacts.candidate_configuration_offsets,
        level1_offsets=artifacts.level_offsets[0],
        level1_configuration_ordinals=artifacts.level_configuration_ordinals[0],
        level2_offsets=artifacts.level_offsets[1],
        level2_configuration_ordinals=artifacts.level_configuration_ordinals[1],
        level3_offsets=artifacts.level_offsets[2],
        level3_configuration_ordinals=artifacts.level_configuration_ordinals[2],
        level4_offsets=artifacts.level_offsets[3],
        level4_configuration_ordinals=artifacts.level_configuration_ordinals[3],
    )

def load_optimization_precompute_artifacts(
    artifact_path: Path,
) -> OptimizationPrecomputeArtifacts:
    """Load persisted reusable threshold-index artifacts from disk."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = OptimizationPrecomputeArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            open_cell_count=int(payload["open_cell_count"].item()),
            candidate_count=int(payload["candidate_count"].item()),
            configuration_count=int(payload["configuration_count"].item()),
            candidate_configuration_offsets=payload[
                "candidate_configuration_offsets"
            ].astype(np.int32, copy=False),
            level_offsets=(
                payload["level1_offsets"].astype(np.int32, copy=False),
                payload["level2_offsets"].astype(np.int32, copy=False),
                payload["level3_offsets"].astype(np.int32, copy=False),
                payload["level4_offsets"].astype(np.int32, copy=False),
            ),
            level_configuration_ordinals=(
                payload["level1_configuration_ordinals"].astype(np.int32, copy=False),
                payload["level2_configuration_ordinals"].astype(np.int32, copy=False),
                payload["level3_configuration_ordinals"].astype(np.int32, copy=False),
                payload["level4_configuration_ordinals"].astype(np.int32, copy=False),
            ),
        )

    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")
    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.configuration_count < 0:
        raise ValueError("configuration_count must be non-negative.")

    _validate_offsets(
        artifacts.candidate_configuration_offsets,
        expected_entry_count=artifacts.candidate_count,
        expected_total=artifacts.configuration_count,
    )
    for level_index in range(len(_DORI_LEVELS)):
        _validate_offsets(
            artifacts.level_offsets[level_index],
            expected_entry_count=artifacts.open_cell_count,
            expected_total=len(artifacts.level_configuration_ordinals[level_index]),
        )
    return artifacts

def save_optimization_artifacts(
    artifact_path: Path,
    artifacts: OptimizationArtifacts,
) -> Path:
    """Persist phase-04 artifacts to a deterministic `04_solution_k<K>.npz` schema."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        open_cell_count=np.asarray(artifacts.open_cell_count, dtype=np.int32),
        candidate_count=np.asarray(artifacts.candidate_count, dtype=np.int32),
        configuration_count=np.asarray(artifacts.configuration_count, dtype=np.int32),
        solved_k=np.asarray(artifacts.solved_k, dtype=np.int32),
        solver_name=np.asarray(artifacts.solver_name),
        solver_status=np.asarray(artifacts.solver_status),
        objective_value=np.asarray(artifacts.objective_value, dtype=np.float64),
        selected_configuration_ordinals=artifacts.selected_configuration_ordinals,
        selected_candidate_ordinals=artifacts.selected_candidate_ordinals,
        selected_angle_ordinals=artifacts.selected_angle_ordinals,
        selected_angles_deg=artifacts.selected_angles_deg,
        final_open_cell_scores=artifacts.final_open_cell_scores,
        best_configuration_ordinals=artifacts.best_configuration_ordinals,
    )

def load_optimization_artifacts(
    artifact_path: Path,
) -> OptimizationArtifacts:
    """Load phase-04 artifacts from disk and perform structural validation."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = OptimizationArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            open_cell_count=int(payload["open_cell_count"].item()),
            candidate_count=int(payload["candidate_count"].item()),
            configuration_count=int(payload["configuration_count"].item()),
            solved_k=int(payload["solved_k"].item()),
            solver_name=str(payload["solver_name"].item()),
            solver_status=str(payload["solver_status"].item()),
            objective_value=float(payload["objective_value"].item()),
            selected_configuration_ordinals=payload[
                "selected_configuration_ordinals"
            ].astype(np.int32, copy=False),
            selected_candidate_ordinals=payload["selected_candidate_ordinals"].astype(
                np.int32,
                copy=False,
            ),
            selected_angle_ordinals=payload["selected_angle_ordinals"].astype(
                np.int16,
                copy=False,
            ),
            selected_angles_deg=payload["selected_angles_deg"].astype(
                np.float32,
                copy=False,
            ),
            final_open_cell_scores=payload["final_open_cell_scores"].astype(
                np.int8,
                copy=False,
            ),
            best_configuration_ordinals=payload["best_configuration_ordinals"].astype(
                np.int32,
                copy=False,
            ),
        )

    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")
    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.configuration_count < 0:
        raise ValueError("configuration_count must be non-negative.")
    if artifacts.solved_k <= 0:
        raise ValueError("solved_k must be positive.")

    _validate_selected_configuration_ordinals(
        artifacts.selected_configuration_ordinals,
        configuration_count=artifacts.configuration_count,
        solved_k=artifacts.solved_k,
    )
    _validate_final_score_arrays(artifacts)
    return artifacts

def save_optimization_summary(
    summary_path: Path,
    artifacts: OptimizationArtifacts,
) -> Path:
    """Persist the human-readable per-`K` optimization summary JSON."""

    metrics = _compute_coverage_metrics(artifacts.final_open_cell_scores)
    summary_payload = {
        "phase_name": PHASE_NAME,
        "solved_k": artifacts.solved_k,
        "solver_name": artifacts.solver_name,
        "solver_status": artifacts.solver_status,
        "objective_value": artifacts.objective_value,
        "selected_camera_count": int(len(artifacts.selected_configuration_ordinals)),
        "open_cell_count": artifacts.open_cell_count,
        "total_dori_score": metrics.total_dori_score,
        "coverage_detection_plus_pct": metrics.detection_plus_pct,
        "coverage_observation_plus_pct": metrics.observation_plus_pct,
        "coverage_recognition_plus_pct": metrics.recognition_plus_pct,
        "coverage_identification_pct": metrics.identification_pct,
        "blind_spot_pct": metrics.blind_spot_pct,
    }
    return write_json(summary_path, summary_payload)

