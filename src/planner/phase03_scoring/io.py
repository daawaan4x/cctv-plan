"""Persistence helpers for phase-03 sparse score artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.planner._shared.cache import write_npz

from .artifacts import SparseScoreArtifacts
from .validation import (
    _validate_configuration_index_arrays,
    _validate_offsets,
    _validate_orientation_angles_array,
    _validate_score_values,
    _validate_target_ordinals,
)


def save_sparse_score_artifacts(
    artifact_path: Path,
    artifacts: SparseScoreArtifacts,
) -> Path:
    """Persist phase-03 artifacts to the deterministic `03_sparse_scores.npz` schema."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        candidate_count=np.asarray(artifacts.candidate_count, dtype=np.int32),
        open_cell_count=np.asarray(artifacts.open_cell_count, dtype=np.int32),
        orientation_angles_deg=artifacts.orientation_angles_deg,
        configuration_candidate_ordinals=artifacts.configuration_candidate_ordinals,
        configuration_angle_ordinals=artifacts.configuration_angle_ordinals,
        score_configuration_offsets=artifacts.score_configuration_offsets,
        score_target_ordinals=artifacts.score_target_ordinals,
        score_values=artifacts.score_values,
    )


def load_sparse_score_artifacts(
    artifact_path: Path,
) -> SparseScoreArtifacts:
    """Load phase-03 artifacts from disk and perform structural validation."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = SparseScoreArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            candidate_count=int(payload["candidate_count"].item()),
            open_cell_count=int(payload["open_cell_count"].item()),
            orientation_angles_deg=payload["orientation_angles_deg"].astype(
                np.float32,
                copy=False,
            ),
            configuration_candidate_ordinals=payload[
                "configuration_candidate_ordinals"
            ].astype(np.int32, copy=False),
            configuration_angle_ordinals=payload["configuration_angle_ordinals"].astype(
                np.int16,
                copy=False,
            ),
            score_configuration_offsets=payload["score_configuration_offsets"].astype(
                np.int32,
                copy=False,
            ),
            score_target_ordinals=payload["score_target_ordinals"].astype(
                np.int32,
                copy=False,
            ),
            score_values=payload["score_values"].astype(np.int8, copy=False),
        )

    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")

    _validate_orientation_angles_array(
        artifacts.orientation_angles_deg,
        artifacts.orientation_angles_deg,
    )
    _validate_configuration_index_arrays(
        artifacts.configuration_candidate_ordinals,
        artifacts.configuration_angle_ordinals,
        candidate_count=artifacts.candidate_count,
        orientation_count=len(artifacts.orientation_angles_deg),
    )
    _validate_offsets(
        artifacts.score_configuration_offsets,
        expected_configuration_count=len(artifacts.configuration_candidate_ordinals),
        expected_total=len(artifacts.score_target_ordinals),
    )
    _validate_target_ordinals(
        artifacts.score_target_ordinals,
        open_cell_count=artifacts.open_cell_count,
    )
    _validate_score_values(
        artifacts.score_values,
        expected_total=len(artifacts.score_target_ordinals),
    )
    return artifacts


__all__ = [
    "load_sparse_score_artifacts",
    "save_sparse_score_artifacts",
]
