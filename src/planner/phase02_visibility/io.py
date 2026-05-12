from __future__ import annotations

from pathlib import Path

import numpy as np

from src.planner._shared.cache import write_npz

from .artifacts import VisibilityArtifacts
from .validation import _validate_offsets, _validate_target_ordinals


def save_visibility_artifacts(
    artifact_path: Path,
    artifacts: VisibilityArtifacts,
) -> Path:
    """Persist phase-02 LOS artifacts to the deterministic `02_visibility.npz` schema."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        candidate_count=np.asarray(artifacts.candidate_count, dtype=np.int32),
        open_cell_count=np.asarray(artifacts.open_cell_count, dtype=np.int32),
        los_candidate_offsets=artifacts.los_candidate_offsets,
        los_target_ordinals=artifacts.los_target_ordinals,
        diagonal_candidate_offsets=artifacts.diagonal_candidate_offsets,
        diagonal_target_ordinals=artifacts.diagonal_target_ordinals,
    )

def load_visibility_artifacts(
    artifact_path: Path,
) -> VisibilityArtifacts:
    """Load phase-02 LOS artifacts from disk and perform structural validation."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = VisibilityArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            candidate_count=int(payload["candidate_count"].item()),
            open_cell_count=int(payload["open_cell_count"].item()),
            los_candidate_offsets=payload["los_candidate_offsets"].astype(
                np.int32,
                copy=False,
            ),
            los_target_ordinals=payload["los_target_ordinals"].astype(
                np.int32,
                copy=False,
            ),
            diagonal_candidate_offsets=payload["diagonal_candidate_offsets"].astype(
                np.int32,
                copy=False,
            ),
            diagonal_target_ordinals=payload["diagonal_target_ordinals"].astype(
                np.int32,
                copy=False,
            ),
        )

    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")

    _validate_offsets(
        "los_candidate_offsets",
        artifacts.los_candidate_offsets,
        expected_candidate_count=artifacts.candidate_count,
        expected_total=len(artifacts.los_target_ordinals),
    )
    _validate_offsets(
        "diagonal_candidate_offsets",
        artifacts.diagonal_candidate_offsets,
        expected_candidate_count=artifacts.candidate_count,
        expected_total=len(artifacts.diagonal_target_ordinals),
    )
    _validate_target_ordinals(
        "los_target_ordinals",
        artifacts.los_target_ordinals,
        open_cell_count=artifacts.open_cell_count,
    )
    _validate_target_ordinals(
        "diagonal_target_ordinals",
        artifacts.diagonal_target_ordinals,
        open_cell_count=artifacts.open_cell_count,
    )

    return artifacts
