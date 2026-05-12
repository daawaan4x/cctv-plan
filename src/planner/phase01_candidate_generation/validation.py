from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput, OPEN_CELL, SOLID_CELL
from src.planner._shared.config import PlannerConfig

from .artifacts import CandidateGenerationArtifacts, _DirectionalMasks
from .geometry import (
    _build_directional_masks,
    _build_eligible_candidate_arrays,
    _build_run_anchor_positions,
    _build_wall_runs,
    _flat_indices_to_coords,
)


def validate_candidate_generation_artifacts(
    floorplan: FloorPlanInput,
    artifacts: CandidateGenerationArtifacts,
    *,
    config: PlannerConfig | None = None,
) -> None:
    """Validate structural and semantic consistency for phase-01 artifacts."""

    resolved_config = config or PlannerConfig(floorplan_name=floorplan.name)
    height, width = floorplan.shape
    grid_size = height * width

    if artifacts.grid_shape != floorplan.shape:
        raise ValueError(
            "Candidate-generation grid_shape does not match floorplan.shape."
        )

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

    if len(artifacts.open_cell_indices) != floorplan.open_cell_count:
        raise ValueError(
            "open_cell_indices length does not match floorplan.open_cell_count."
        )

    expected_open_coords = _flat_indices_to_coords(artifacts.open_cell_indices, width)
    expected_eligible_coords = _flat_indices_to_coords(
        artifacts.eligible_candidate_cell_indices,
        width,
    )
    expected_candidate_coords = _flat_indices_to_coords(
        artifacts.candidate_cell_indices,
        width,
    )
    if not np.array_equal(artifacts.open_cell_coords_rc, expected_open_coords):
        raise ValueError("open_cell_coords_rc does not match open_cell_indices.")
    if not np.array_equal(
        artifacts.eligible_candidate_cell_coords_rc,
        expected_eligible_coords,
    ):
        raise ValueError(
            "eligible_candidate_cell_coords_rc does not match "
            "eligible_candidate_cell_indices."
        )
    if not np.array_equal(
        artifacts.candidate_cell_coords_rc, expected_candidate_coords
    ):
        raise ValueError(
            "candidate_cell_coords_rc does not match candidate_cell_indices."
        )

    if not np.isin(
        artifacts.eligible_candidate_cell_indices,
        artifacts.open_cell_indices,
        assume_unique=True,
    ).all():
        raise ValueError(
            "eligible_candidate_cell_indices must be a subset of open_cell_indices."
        )
    if not np.isin(
        artifacts.candidate_cell_indices,
        artifacts.eligible_candidate_cell_indices,
        assume_unique=True,
    ).all():
        raise ValueError(
            "candidate_cell_indices must be a subset of eligible_candidate_cell_indices."
        )
    if artifacts.eligible_candidate_boundary_flags.size and np.any(
        artifacts.eligible_candidate_boundary_flags == 0
    ):
        raise ValueError(
            "eligible_candidate_boundary_flags must be non-zero for every eligible "
            "candidate cell."
        )
    if artifacts.candidate_boundary_flags.size and np.any(
        artifacts.candidate_boundary_flags == 0
    ):
        raise ValueError(
            "candidate_boundary_flags must be non-zero for every candidate cell."
        )

    directional_masks = _build_directional_masks(floorplan.grid)
    (
        expected_eligible_indices,
        expected_eligible_boundary_flags,
        expected_eligible_solid_flags,
    ) = _build_eligible_candidate_arrays(directional_masks)
    if not np.array_equal(
        artifacts.eligible_candidate_cell_indices,
        expected_eligible_indices,
    ):
        raise ValueError(
            "eligible_candidate_cell_indices does not match the locked solid-adjacency rule."
        )
    if not np.array_equal(
        artifacts.eligible_candidate_boundary_flags,
        expected_eligible_boundary_flags,
    ):
        raise ValueError(
            "eligible_candidate_boundary_flags does not match a re-derived solid "
            "boundary bitmask."
        )

    _validate_open_cells_match_grid(floorplan.grid, artifacts.open_cell_coords_rc)
    _validate_candidate_cells_match_grid(
        floorplan.grid,
        artifacts.eligible_candidate_cell_coords_rc,
        artifacts.eligible_candidate_boundary_flags,
    )
    _validate_candidate_cells_match_grid(
        floorplan.grid,
        artifacts.candidate_cell_coords_rc,
        artifacts.candidate_boundary_flags,
    )
    _validate_candidate_spacing_rules(
        floorplan,
        resolved_config,
        artifacts,
        directional_masks,
        expected_eligible_solid_flags,
    )

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
    if len(indices) > 1 and not np.all(indices[:-1] < indices[1:]):
        raise ValueError(f"{name} must be strictly increasing and duplicate-free.")
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

def _validate_uint8_array(
    name: str,
    values: NDArray[np.uint8],
    *,
    expected_length: int,
) -> None:
    """Validate dtype and length for compact uint8 metadata arrays."""

    if values.dtype != np.uint8:
        raise TypeError(f"{name} must use dtype np.uint8.")
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if len(values) != expected_length:
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
    """Confirm that every candidate is open and satisfies the solid-adjacency rule."""

    for (row, col), boundary_flag in zip(
        candidate_cell_coords_rc,
        candidate_boundary_flags,
        strict=True,
    ):
        if grid[row, col] != OPEN_CELL:
            raise ValueError(
                "candidate coordinate array includes a non-open grid cell."
            )
        if boundary_flag == 0:
            raise ValueError("candidate boundary-flag arrays must be non-zero.")

        has_solid_neighbor = False
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            neighbor_row = row + dr
            neighbor_col = col + dc
            if (
                0 <= neighbor_row < grid.shape[0]
                and 0 <= neighbor_col < grid.shape[1]
                and grid[neighbor_row, neighbor_col] == SOLID_CELL
            ):
                has_solid_neighbor = True
                break

        if not has_solid_neighbor:
            raise ValueError(
                "candidate coordinate array includes an open cell with no solid "
                "4-neighbor."
            )

def _validate_candidate_spacing_rules(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    artifacts: CandidateGenerationArtifacts,
    directional_masks: _DirectionalMasks,
    eligible_candidate_solid_flags: NDArray[np.uint8],
) -> None:
    """Validate anchor retention and the exception-first spacing rule per wall run."""

    eligible_solid_lookup = {
        int(flat_index): int(solid_flag)
        for flat_index, solid_flag in zip(
            artifacts.eligible_candidate_cell_indices,
            eligible_candidate_solid_flags,
            strict=True,
        )
    }
    selected_lookup = {
        int(flat_index): int(exception_flag)
        for flat_index, exception_flag in zip(
            artifacts.candidate_cell_indices,
            artifacts.candidate_exception_flags,
            strict=True,
        )
    }

    for run in _build_wall_runs(floorplan.shape[1], directional_masks):
        anchor_flags_by_flat: dict[int, int] = {}
        anchor_positions = _build_run_anchor_positions(
            run,
            eligible_solid_lookup,
            anchor_flags_by_flat,
        )
        for anchor_position in anchor_positions:
            anchor_flat = int(run[anchor_position])
            if anchor_flat not in selected_lookup:
                raise ValueError(
                    "Exception-anchor candidate was not retained in the final thinned "
                    "candidate set."
                )

        selected_positions = [
            position
            for position, flat_index in enumerate(run)
            if int(flat_index) in selected_lookup
        ]
        for left_position, right_position in zip(
            selected_positions,
            selected_positions[1:],
        ):
            if (right_position - left_position) >= config.candidate_spacing_cells:
                continue

            left_exception_flag = selected_lookup[int(run[left_position])]
            right_exception_flag = selected_lookup[int(run[right_position])]
            if left_exception_flag == 0 and right_exception_flag == 0:
                raise ValueError(
                    "Final thinned candidates contain a sub-minimum wall-run spacing "
                    "gap that does not involve an exception anchor."
                )
