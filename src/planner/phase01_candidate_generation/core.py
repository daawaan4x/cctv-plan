from __future__ import annotations

"""Phase 01 candidate-generation helpers for the CCTV planner."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput, OPEN_CELL
from src.planner._shared.cache import write_npz

PHASE_NAME = "candidate_generation"
PHASE_ARTIFACT_STEM = "01_candidates"

_BOUNDARY_FLAG_NORTH = np.uint8(1)
_BOUNDARY_FLAG_EAST = np.uint8(2)
_BOUNDARY_FLAG_SOUTH = np.uint8(4)
_BOUNDARY_FLAG_WEST = np.uint8(8)


@dataclass(frozen=True, slots=True)
class CandidateGenerationArtifacts:
    """Deterministic phase-01 outputs derived from a tri-state floor-plan grid."""

    grid_shape: tuple[int, int]
    open_cell_indices: NDArray[np.int32]
    open_cell_coords_rc: NDArray[np.int32]
    candidate_cell_indices: NDArray[np.int32]
    candidate_cell_coords_rc: NDArray[np.int32]
    candidate_boundary_flags: NDArray[np.uint8]


# Artifact generation
def generate_candidate_generation_artifacts(
    floorplan: FloorPlanInput,
) -> CandidateGenerationArtifacts:
    """Build the canonical open-target and candidate-camera arrays for phase 01."""

    grid = floorplan.grid
    height, width = floorplan.shape

    open_cell_indices = _build_open_cell_indices(grid)
    candidate_cell_indices, candidate_boundary_flags = _build_candidate_boundary_flags(
        grid
    )
    artifacts = CandidateGenerationArtifacts(
        grid_shape=(height, width),
        open_cell_indices=open_cell_indices,
        open_cell_coords_rc=_flat_indices_to_coords(open_cell_indices, width),
        candidate_cell_indices=candidate_cell_indices,
        candidate_cell_coords_rc=_flat_indices_to_coords(candidate_cell_indices, width),
        candidate_boundary_flags=candidate_boundary_flags,
    )
    validate_candidate_generation_artifacts(floorplan, artifacts)
    return artifacts


def validate_candidate_generation_artifacts(
    floorplan: FloorPlanInput,
    artifacts: CandidateGenerationArtifacts,
) -> None:
    """Validate structural and semantic consistency for phase-01 artifacts."""

    height, width = floorplan.shape

    if artifacts.grid_shape != (height, width):
        raise ValueError("Candidate-generation grid_shape does not match floorplan.shape.")

    _validate_flat_index_array(
        "open_cell_indices",
        artifacts.open_cell_indices,
        size=height * width,
    )
    _validate_flat_index_array(
        "candidate_cell_indices",
        artifacts.candidate_cell_indices,
        size=height * width,
    )
    _validate_coordinate_array(
        "open_cell_coords_rc",
        artifacts.open_cell_coords_rc,
        expected_length=len(artifacts.open_cell_indices),
    )
    _validate_coordinate_array(
        "candidate_cell_coords_rc",
        artifacts.candidate_cell_coords_rc,
        expected_length=len(artifacts.candidate_cell_indices),
    )

    if artifacts.candidate_boundary_flags.dtype != np.uint8:
        raise TypeError("candidate_boundary_flags must use dtype np.uint8.")
    if artifacts.candidate_boundary_flags.ndim != 1:
        raise ValueError("candidate_boundary_flags must be a 1D array.")
    if len(artifacts.candidate_boundary_flags) != len(artifacts.candidate_cell_indices):
        raise ValueError(
            "candidate_boundary_flags length must match candidate_cell_indices length."
        )
    if len(artifacts.open_cell_indices) != floorplan.open_cell_count:
        raise ValueError(
            "open_cell_indices length does not match floorplan.open_cell_count."
        )
    if not np.isin(
        artifacts.candidate_cell_indices,
        artifacts.open_cell_indices,
        assume_unique=True,
    ).all():
        raise ValueError("candidate_cell_indices must be a subset of open_cell_indices.")

    expected_open_coords = _flat_indices_to_coords(artifacts.open_cell_indices, width)
    expected_candidate_coords = _flat_indices_to_coords(
        artifacts.candidate_cell_indices,
        width,
    )
    if not np.array_equal(artifacts.open_cell_coords_rc, expected_open_coords):
        raise ValueError("open_cell_coords_rc does not match open_cell_indices.")
    if not np.array_equal(artifacts.candidate_cell_coords_rc, expected_candidate_coords):
        raise ValueError(
            "candidate_cell_coords_rc does not match candidate_cell_indices."
        )

    _validate_open_cells_match_grid(floorplan.grid, artifacts.open_cell_coords_rc)
    _validate_candidate_cells_match_grid(
        floorplan.grid,
        artifacts.candidate_cell_coords_rc,
        artifacts.candidate_boundary_flags,
    )


# Artifact persistence
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
        candidate_cell_indices=artifacts.candidate_cell_indices,
        candidate_cell_coords_rc=artifacts.candidate_cell_coords_rc,
        candidate_boundary_flags=artifacts.candidate_boundary_flags,
    )


def load_candidate_generation_artifacts(
    artifact_path: Path,
) -> CandidateGenerationArtifacts:
    """Load phase-01 artifacts from disk and perform structural validation."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        grid_shape = (int(raw_grid_shape[0]), int(raw_grid_shape[1]))
        artifacts = CandidateGenerationArtifacts(
            grid_shape=grid_shape,
            open_cell_indices=payload["open_cell_indices"].astype(
                np.int32, copy=False
            ),
            open_cell_coords_rc=payload["open_cell_coords_rc"].astype(
                np.int32, copy=False
            ),
            candidate_cell_indices=payload["candidate_cell_indices"].astype(
                np.int32, copy=False
            ),
            candidate_cell_coords_rc=payload["candidate_cell_coords_rc"].astype(
                np.int32, copy=False
            ),
            candidate_boundary_flags=payload["candidate_boundary_flags"].astype(
                np.uint8, copy=False
            ),
        )

    # The on-disk artifact is intentionally self-describing, so the load path checks
    # basic array structure even before the caller provides a `FloorPlanInput`. The
    # semantic grid-based checks still live in `validate_candidate_generation_artifacts`
    # because only that function has the source grid needed to re-derive the rule.
    height, width = artifacts.grid_shape
    _validate_flat_index_array(
        "open_cell_indices",
        artifacts.open_cell_indices,
        size=height * width,
    )
    _validate_flat_index_array(
        "candidate_cell_indices",
        artifacts.candidate_cell_indices,
        size=height * width,
    )
    _validate_coordinate_array(
        "open_cell_coords_rc",
        artifacts.open_cell_coords_rc,
        expected_length=len(artifacts.open_cell_indices),
    )
    _validate_coordinate_array(
        "candidate_cell_coords_rc",
        artifacts.candidate_cell_coords_rc,
        expected_length=len(artifacts.candidate_cell_indices),
    )
    if artifacts.candidate_boundary_flags.dtype != np.uint8:
        raise TypeError("candidate_boundary_flags must use dtype np.uint8.")
    if artifacts.candidate_boundary_flags.ndim != 1:
        raise ValueError("candidate_boundary_flags must be a 1D array.")
    if len(artifacts.candidate_boundary_flags) != len(artifacts.candidate_cell_indices):
        raise ValueError(
            "candidate_boundary_flags length must match candidate_cell_indices length."
        )

    return artifacts


# Grid traversal helpers
def _build_open_cell_indices(grid: NDArray[np.int8]) -> NDArray[np.int32]:
    """Return all open-cell flat indices in deterministic row-major order."""

    open_mask = grid == OPEN_CELL
    return np.flatnonzero(open_mask).astype(np.int32, copy=False)


def _build_candidate_boundary_flags(
    grid: NDArray[np.int8],
) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
    """Return candidate flat indices plus a direction bitmask for each candidate."""

    open_mask = grid == OPEN_CELL
    padded_open = np.pad(open_mask, pad_width=1, mode="constant", constant_values=False)

    # The padded boolean view is the core trick in this phase. By padding with
    # `False`, every out-of-bounds neighbor automatically behaves like a non-open
    # cell, which matches the locked rule that floor-plan boundaries count as valid
    # candidate-defining neighbors. The four slices below then answer a single
    # question for every original grid cell: "is the neighbor in this direction open?"
    north_open = padded_open[0:-2, 1:-1]
    east_open = padded_open[1:-1, 2:]
    south_open = padded_open[2:, 1:-1]
    west_open = padded_open[1:-1, 0:-2]

    non_open_north = ~north_open
    non_open_east = ~east_open
    non_open_south = ~south_open
    non_open_west = ~west_open

    candidate_mask = open_mask & (
        non_open_north | non_open_east | non_open_south | non_open_west
    )
    boundary_flags_full = (
        (non_open_north.astype(np.uint8) * _BOUNDARY_FLAG_NORTH)
        | (non_open_east.astype(np.uint8) * _BOUNDARY_FLAG_EAST)
        | (non_open_south.astype(np.uint8) * _BOUNDARY_FLAG_SOUTH)
        | (non_open_west.astype(np.uint8) * _BOUNDARY_FLAG_WEST)
    )

    candidate_cell_indices = np.flatnonzero(candidate_mask).astype(np.int32, copy=False)
    candidate_boundary_flags = boundary_flags_full[candidate_mask].astype(
        np.uint8, copy=False
    )
    return candidate_cell_indices, candidate_boundary_flags


def _flat_indices_to_coords(
    indices: NDArray[np.int32],
    width: int,
) -> NDArray[np.int32]:
    """Decode row-major flat indices into `(row, col)` coordinate pairs."""

    rows = indices // width
    cols = indices % width
    return np.column_stack((rows, cols)).astype(np.int32, copy=False)


# Validation helpers
def _validate_flat_index_array(
    name: str,
    indices: NDArray[np.int32],
    *,
    size: int,
) -> None:
    """Validate dtype, shape, ordering, uniqueness, and bounds for flat indices."""

    if indices.dtype != np.int32:
        raise TypeError(f"{name} must use dtype np.int32.")
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if len(indices) > 1 and not np.all(indices[:-1] <= indices[1:]):
        raise ValueError(f"{name} must be sorted in row-major ascending order.")
    if len(indices) > 1 and np.any(indices[:-1] == indices[1:]):
        raise ValueError(f"{name} must not contain duplicate flat indices.")
    if indices.size and (indices[0] < 0 or indices[-1] >= size):
        raise ValueError(f"{name} contains an out-of-range flat index.")


def _validate_coordinate_array(
    name: str,
    coords: NDArray[np.int32],
    *,
    expected_length: int,
) -> None:
    """Validate dtype and shape for a `(row, col)` coordinate array."""

    if coords.dtype != np.int32:
        raise TypeError(f"{name} must use dtype np.int32.")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2).")
    if coords.shape[0] != expected_length:
        raise ValueError(f"{name} length must match its paired flat-index array.")


def _validate_open_cells_match_grid(
    grid: NDArray[np.int8],
    open_cell_coords_rc: NDArray[np.int32],
) -> None:
    """Confirm that every stored open-cell coordinate still maps to an open cell."""

    for row, col in open_cell_coords_rc:
        if grid[row, col] != OPEN_CELL:
            raise ValueError("open_cell_coords_rc includes a non-open grid cell.")


def _validate_candidate_cells_match_grid(
    grid: NDArray[np.int8],
    candidate_cell_coords_rc: NDArray[np.int32],
    candidate_boundary_flags: NDArray[np.uint8],
) -> None:
    """Confirm that every candidate is open and satisfies the boundary rule."""

    for (row, col), boundary_flag in zip(
        candidate_cell_coords_rc,
        candidate_boundary_flags,
        strict=True,
    ):
        if grid[row, col] != OPEN_CELL:
            raise ValueError("candidate_cell_coords_rc includes a non-open grid cell.")
        if boundary_flag == 0:
            raise ValueError(
                "candidate_boundary_flags must be non-zero for every candidate cell."
            )

        # This loop is intentionally written as a direct semantic re-check instead of
        # reusing the vectorized mask logic above. The validation path is slower, but
        # it is easier to audit because it mirrors the written rule exactly:
        # a candidate must be open and have at least one non-open 4-neighbor, where
        # out-of-bounds counts as non-open. That makes the validator a useful guard
        # against future refactors that might accidentally change the traversal logic.
        has_non_open_neighbor = False
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            neighbor_row = row + dr
            neighbor_col = col + dc
            if (
                neighbor_row < 0
                or neighbor_row >= grid.shape[0]
                or neighbor_col < 0
                or neighbor_col >= grid.shape[1]
                or grid[neighbor_row, neighbor_col] != OPEN_CELL
            ):
                has_non_open_neighbor = True
                break

        if not has_non_open_neighbor:
            raise ValueError(
                "candidate_cell_coords_rc includes an open cell with no non-open "
                "4-neighbor."
            )


__all__ = [
    "CandidateGenerationArtifacts",
    "generate_candidate_generation_artifacts",
    "load_candidate_generation_artifacts",
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
    "save_candidate_generation_artifacts",
    "validate_candidate_generation_artifacts",
]
