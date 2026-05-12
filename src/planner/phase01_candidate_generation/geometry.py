from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput, OPEN_CELL, SOLID_CELL
from src.planner._shared.config import PlannerConfig

from .artifacts import _DirectionalMasks
from .constants import (
    _BOUNDARY_FLAG_EAST,
    _BOUNDARY_FLAG_NORTH,
    _BOUNDARY_FLAG_SOUTH,
    _BOUNDARY_FLAG_WEST,
    _EXCEPTION_FLAG_CORNER,
    _EXCEPTION_FLAG_ENDPOINT,
    _EXCEPTION_FLAG_JUNCTION,
    _EXCEPTION_FLAG_MIDPOINT,
)


def _build_open_cell_indices(grid: NDArray[np.int8]) -> NDArray[np.int32]:
    """Return all open-cell flat indices in deterministic row-major order."""

    return np.flatnonzero(grid == OPEN_CELL).astype(np.int32, copy=False)

def _build_directional_masks(grid: NDArray[np.int8]) -> _DirectionalMasks:
    """Build reusable directional neighbor masks for solid adjacency."""

    open_mask = grid == OPEN_CELL
    solid_mask = grid == SOLID_CELL
    padded_solid = np.pad(
        solid_mask,
        pad_width=1,
        mode="constant",
        constant_values=False,
    )

    return _DirectionalMasks(
        open_mask=open_mask,
        north_solid=padded_solid[0:-2, 1:-1],
        east_solid=padded_solid[1:-1, 2:],
        south_solid=padded_solid[2:, 1:-1],
        west_solid=padded_solid[1:-1, 0:-2],
    )

def _build_eligible_candidate_arrays(
    directional_masks: _DirectionalMasks,
) -> tuple[NDArray[np.int32], NDArray[np.uint8], NDArray[np.uint8]]:
    """Return eligible solid-adjacent candidate flat indices plus direction bitmasks."""

    eligible_mask = directional_masks.open_mask & (
        directional_masks.north_solid
        | directional_masks.east_solid
        | directional_masks.south_solid
        | directional_masks.west_solid
    )
    solid_flags_full = (
        (directional_masks.north_solid.astype(np.uint8) * _BOUNDARY_FLAG_NORTH)
        | (directional_masks.east_solid.astype(np.uint8) * _BOUNDARY_FLAG_EAST)
        | (directional_masks.south_solid.astype(np.uint8) * _BOUNDARY_FLAG_SOUTH)
        | (directional_masks.west_solid.astype(np.uint8) * _BOUNDARY_FLAG_WEST)
    )

    eligible_candidate_cell_indices = np.flatnonzero(eligible_mask).astype(
        np.int32,
        copy=False,
    )
    eligible_candidate_boundary_flags = solid_flags_full[eligible_mask].astype(
        np.uint8,
        copy=False,
    )
    eligible_candidate_solid_flags = solid_flags_full[eligible_mask].astype(
        np.uint8,
        copy=False,
    )
    return (
        eligible_candidate_cell_indices,
        eligible_candidate_boundary_flags,
        eligible_candidate_solid_flags,
    )

def _thin_candidate_set(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    eligible_candidate_cell_indices: NDArray[np.int32],
    eligible_candidate_boundary_flags: NDArray[np.uint8],
    eligible_candidate_solid_flags: NDArray[np.uint8],
    directional_masks: _DirectionalMasks,
) -> tuple[NDArray[np.int32], NDArray[np.uint8], NDArray[np.uint8]]:
    """Apply deterministic exception-first wall-run spacing to eligible candidates."""

    if len(eligible_candidate_cell_indices) == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint8),
        )

    eligible_boundary_lookup = {
        int(flat_index): int(boundary_flag)
        for flat_index, boundary_flag in zip(
            eligible_candidate_cell_indices,
            eligible_candidate_boundary_flags,
            strict=True,
        )
    }
    eligible_solid_lookup = {
        int(flat_index): int(solid_flag)
        for flat_index, solid_flag in zip(
            eligible_candidate_cell_indices,
            eligible_candidate_solid_flags,
            strict=True,
        )
    }
    exception_flags_by_flat: dict[int, int] = {}
    selected_candidate_flats: set[int] = set()

    wall_runs = _build_wall_runs(floorplan.shape[1], directional_masks)

    for run in wall_runs:
        run_anchor_positions = _build_run_anchor_positions(
            run,
            eligible_solid_lookup,
            exception_flags_by_flat,
        )
        run_selected_positions = set(run_anchor_positions)

        if run_anchor_positions:
            if run_anchor_positions[0] > 0:
                run_selected_positions.update(
                    _build_segment_candidate_positions(
                        0,
                        run_anchor_positions[0] - 1,
                        config.candidate_spacing_cells,
                    )
                )
            for left_anchor, right_anchor in zip(
                run_anchor_positions,
                run_anchor_positions[1:],
            ):
                run_selected_positions.update(
                    _build_anchor_gap_candidate_positions(
                        left_anchor,
                        right_anchor,
                        config.candidate_spacing_cells,
                    )
                )
            if run_anchor_positions[-1] < len(run) - 1:
                run_selected_positions.update(
                    _build_segment_candidate_positions(
                        run_anchor_positions[-1] + 1,
                        len(run) - 1,
                        config.candidate_spacing_cells,
                    )
                )
        else:
            run_selected_positions.update(
                _build_segment_candidate_positions(
                    0,
                    len(run) - 1,
                    config.candidate_spacing_cells,
                )
            )

        for position in run_selected_positions:
            selected_candidate_flats.add(int(run[position]))

    candidate_cell_indices = np.asarray(
        sorted(selected_candidate_flats),
        dtype=np.int32,
    )
    candidate_boundary_flags = np.asarray(
        [
            eligible_boundary_lookup[int(flat_index)]
            for flat_index in candidate_cell_indices
        ],
        dtype=np.uint8,
    )
    candidate_exception_flags = np.asarray(
        [
            exception_flags_by_flat.get(int(flat_index), 0)
            for flat_index in candidate_cell_indices
        ],
        dtype=np.uint8,
    )
    return candidate_cell_indices, candidate_boundary_flags, candidate_exception_flags

def _build_wall_runs(
    width: int,
    directional_masks: _DirectionalMasks,
) -> list[NDArray[np.int32]]:
    """Build deterministic solid-wall runs in row-major or axis-major order."""

    return (
        _collect_horizontal_runs(
            directional_masks.open_mask & directional_masks.north_solid, width
        )
        + _collect_horizontal_runs(
            directional_masks.open_mask & directional_masks.south_solid, width
        )
        + _collect_vertical_runs(
            directional_masks.open_mask & directional_masks.east_solid, width
        )
        + _collect_vertical_runs(
            directional_masks.open_mask & directional_masks.west_solid, width
        )
    )

def _collect_horizontal_runs(
    mask: NDArray[np.bool_],
    width: int,
) -> list[NDArray[np.int32]]:
    """Collect contiguous horizontal run slices from one directional mask."""

    runs: list[NDArray[np.int32]] = []
    for row in range(mask.shape[0]):
        col = 0
        while col < mask.shape[1]:
            if not bool(mask[row, col]):
                col += 1
                continue
            start_col = col
            while col < mask.shape[1] and bool(mask[row, col]):
                col += 1
            run_cols = np.arange(start_col, col, dtype=np.int32)
            run_rows = np.full(run_cols.shape, row, dtype=np.int32)
            runs.append((run_rows * width + run_cols).astype(np.int32, copy=False))
    return runs

def _collect_vertical_runs(
    mask: NDArray[np.bool_],
    width: int,
) -> list[NDArray[np.int32]]:
    """Collect contiguous vertical run slices from one directional mask."""

    runs: list[NDArray[np.int32]] = []
    for col in range(mask.shape[1]):
        row = 0
        while row < mask.shape[0]:
            if not bool(mask[row, col]):
                row += 1
                continue
            start_row = row
            while row < mask.shape[0] and bool(mask[row, col]):
                row += 1
            run_rows = np.arange(start_row, row, dtype=np.int32)
            run_cols = np.full(run_rows.shape, col, dtype=np.int32)
            runs.append((run_rows * width + run_cols).astype(np.int32, copy=False))
    return runs

def _build_run_anchor_positions(
    run: NDArray[np.int32],
    eligible_solid_lookup: dict[int, int],
    exception_flags_by_flat: dict[int, int],
) -> list[int]:
    """Return the sorted run-order anchor positions and record exception bits."""

    anchor_positions: set[int] = set()
    run_length = len(run)
    if run_length == 0:
        return []

    # The performance revision locked these exception classes on permanently, so
    # phase 01 now bakes them into the thinning rule instead of carrying redundant
    # booleans through `PlannerConfig`.
    anchor_positions.add(0)
    anchor_positions.add(run_length - 1)
    exception_flags_by_flat[int(run[0])] = exception_flags_by_flat.get(
        int(run[0]), 0
    ) | int(_EXCEPTION_FLAG_ENDPOINT)
    exception_flags_by_flat[int(run[-1])] = exception_flags_by_flat.get(
        int(run[-1]), 0
    ) | int(_EXCEPTION_FLAG_ENDPOINT)

    midpoint_position = (run_length - 1) // 2
    anchor_positions.add(midpoint_position)
    midpoint_flat = int(run[midpoint_position])
    exception_flags_by_flat[midpoint_flat] = exception_flags_by_flat.get(
        midpoint_flat, 0
    ) | int(_EXCEPTION_FLAG_MIDPOINT)

    for position, flat_index in enumerate(run):
        flat_value = int(flat_index)
        solid_flags = eligible_solid_lookup.get(flat_value, 0)
        solid_dir_count = _count_direction_bits(solid_flags)
        has_vertical_solid = (
            solid_flags & int(_BOUNDARY_FLAG_NORTH | _BOUNDARY_FLAG_SOUTH)
        ) != 0
        has_horizontal_solid = (
            solid_flags & int(_BOUNDARY_FLAG_EAST | _BOUNDARY_FLAG_WEST)
        ) != 0

        if solid_dir_count == 2 and has_vertical_solid and has_horizontal_solid:
            anchor_positions.add(position)
            exception_flags_by_flat[flat_value] = exception_flags_by_flat.get(
                flat_value, 0
            ) | int(_EXCEPTION_FLAG_CORNER)

        if solid_dir_count >= 3:
            anchor_positions.add(position)
            exception_flags_by_flat[flat_value] = exception_flags_by_flat.get(
                flat_value, 0
            ) | int(_EXCEPTION_FLAG_JUNCTION)

    return sorted(anchor_positions)

def _count_direction_bits(boundary_flags: int) -> int:
    """Count the active directional bits in one compact boundary mask."""

    count = 0
    if boundary_flags & int(_BOUNDARY_FLAG_NORTH):
        count += 1
    if boundary_flags & int(_BOUNDARY_FLAG_EAST):
        count += 1
    if boundary_flags & int(_BOUNDARY_FLAG_SOUTH):
        count += 1
    if boundary_flags & int(_BOUNDARY_FLAG_WEST):
        count += 1
    return count

def _build_segment_candidate_positions(
    start: int,
    end: int,
    spacing: int,
) -> set[int]:
    """Distribute the maximum feasible non-exception candidates across one segment."""

    if start > end:
        return set()

    length = (end - start) + 1
    candidate_count = 1 + ((length - 1) // spacing)
    if candidate_count <= 0:
        return set()
    if candidate_count == 1:
        return {start + ((length - 1) // 2)}

    required_span = spacing * (candidate_count - 1)
    slack = (length - 1) - required_span
    positions = {
        start
        + (position_ordinal * spacing)
        + ((position_ordinal * slack) // (candidate_count - 1))
        for position_ordinal in range(candidate_count)
    }
    return positions

def _build_anchor_gap_candidate_positions(
    left_anchor: int,
    right_anchor: int,
    spacing: int,
) -> set[int]:
    """Distribute interior candidates across one anchor-bounded wall-run gap.

    The current thinning rule allows sub-minimum gaps only when one neighbor is
    an exception anchor. To avoid filling a gap by snapping candidates directly
    onto the cells beside both anchors, derive the interior count from the full
    anchor span first, then repair the rounded positions so every
    non-exception-to-non-exception gap still respects `spacing`.
    """

    interior_start = left_anchor + 1
    interior_end = right_anchor - 1
    if interior_start > interior_end:
        return set()

    interior_length = (interior_end - interior_start) + 1
    candidate_count = interior_length // spacing
    if candidate_count <= 0:
        return set()

    full_span = right_anchor - left_anchor
    positions = [
        int(np.floor(left_anchor + ((ordinal + 1) * full_span / (candidate_count + 1))))
        for ordinal in range(candidate_count)
    ]

    min_positions = [
        interior_start + (ordinal * spacing) for ordinal in range(candidate_count)
    ]
    max_positions = [
        interior_end - ((candidate_count - ordinal - 1) * spacing)
        for ordinal in range(candidate_count)
    ]

    for ordinal in range(candidate_count):
        positions[ordinal] = max(positions[ordinal], min_positions[ordinal])
        if ordinal > 0:
            positions[ordinal] = max(
                positions[ordinal], positions[ordinal - 1] + spacing
            )

    for ordinal in range(candidate_count - 1, -1, -1):
        positions[ordinal] = min(positions[ordinal], max_positions[ordinal])
        if ordinal < candidate_count - 1:
            positions[ordinal] = min(
                positions[ordinal], positions[ordinal + 1] - spacing
            )

    return set(positions)

def _flat_indices_to_coords(
    indices: NDArray[np.int32],
    width: int,
) -> NDArray[np.int32]:
    """Decode row-major flat indices into `(row, col)` coordinate pairs."""

    rows = indices // width
    cols = indices % width
    return np.column_stack((rows, cols)).astype(np.int32, copy=False)
