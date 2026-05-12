from __future__ import annotations

from pathlib import Path

import numpy as np

from src.planner._shared.cache import write_npz

from .artifacts import CandidateGenerationArtifacts
from .validation import (
    _validate_coordinate_array,
    _validate_flat_index_array,
    _validate_uint8_array,
)


def save_candidate_generation_artifacts(
    artifact_path: Path,
    artifacts: CandidateGenerationArtifacts,
) -> Path:
    """Persist phase-01 artifacts to the deterministic `01_candidates.npz` schema."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        open_cell_indices=artifacts.open_cell_indices,
        open_cell_coords_rc=artifacts.open_cell_coords_rc,
        eligible_candidate_cell_indices=artifacts.eligible_candidate_cell_indices,
        eligible_candidate_cell_coords_rc=artifacts.eligible_candidate_cell_coords_rc,
        eligible_candidate_boundary_flags=artifacts.eligible_candidate_boundary_flags,
        candidate_cell_indices=artifacts.candidate_cell_indices,
        candidate_cell_coords_rc=artifacts.candidate_cell_coords_rc,
        candidate_boundary_flags=artifacts.candidate_boundary_flags,
        candidate_exception_flags=artifacts.candidate_exception_flags,
    )

def load_candidate_generation_artifacts(
    artifact_path: Path,
) -> CandidateGenerationArtifacts:
    """Load phase-01 artifacts from disk and perform structural validation."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = CandidateGenerationArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            open_cell_indices=payload["open_cell_indices"].astype(np.int32, copy=False),
            open_cell_coords_rc=payload["open_cell_coords_rc"].astype(
                np.int32,
                copy=False,
            ),
            eligible_candidate_cell_indices=payload[
                "eligible_candidate_cell_indices"
            ].astype(np.int32, copy=False),
            eligible_candidate_cell_coords_rc=payload[
                "eligible_candidate_cell_coords_rc"
            ].astype(np.int32, copy=False),
            eligible_candidate_boundary_flags=payload[
                "eligible_candidate_boundary_flags"
            ].astype(np.uint8, copy=False),
            candidate_cell_indices=payload["candidate_cell_indices"].astype(
                np.int32,
                copy=False,
            ),
            candidate_cell_coords_rc=payload["candidate_cell_coords_rc"].astype(
                np.int32,
                copy=False,
            ),
            candidate_boundary_flags=payload["candidate_boundary_flags"].astype(
                np.uint8,
                copy=False,
            ),
            candidate_exception_flags=payload["candidate_exception_flags"].astype(
                np.uint8,
                copy=False,
            ),
        )

    height, width = artifacts.grid_shape
    grid_size = height * width
    _validate_flat_index_array(
        "open_cell_indices",
        artifacts.open_cell_indices,
        size=grid_size,
    )
    _validate_flat_index_array(
        "eligible_candidate_cell_indices",
        artifacts.eligible_candidate_cell_indices,
        size=grid_size,
    )
    _validate_flat_index_array(
        "candidate_cell_indices",
        artifacts.candidate_cell_indices,
        size=grid_size,
    )
    _validate_coordinate_array(
        "open_cell_coords_rc",
        artifacts.open_cell_coords_rc,
        expected_length=len(artifacts.open_cell_indices),
    )
    _validate_coordinate_array(
        "eligible_candidate_cell_coords_rc",
        artifacts.eligible_candidate_cell_coords_rc,
        expected_length=len(artifacts.eligible_candidate_cell_indices),
    )
    _validate_coordinate_array(
        "candidate_cell_coords_rc",
        artifacts.candidate_cell_coords_rc,
        expected_length=len(artifacts.candidate_cell_indices),
    )
    _validate_uint8_array(
        "eligible_candidate_boundary_flags",
        artifacts.eligible_candidate_boundary_flags,
        expected_length=len(artifacts.eligible_candidate_cell_indices),
    )
    _validate_uint8_array(
        "candidate_boundary_flags",
        artifacts.candidate_boundary_flags,
        expected_length=len(artifacts.candidate_cell_indices),
    )
    _validate_uint8_array(
        "candidate_exception_flags",
        artifacts.candidate_exception_flags,
        expected_length=len(artifacts.candidate_cell_indices),
    )
    return artifacts
