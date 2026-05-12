"""Line-of-sight classification and sparse pair builders for phase 02."""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.common.floorplan import OPEN_CELL

_VISIBILITY_SKIPPED_SELF = np.int8(0)
_VISIBILITY_VISIBLE = np.int8(1)
_VISIBILITY_BLOCKED_DIRECT = np.int8(2)
_VISIBILITY_BLOCKED_DIAGONAL = np.int8(3)

@njit(cache=True)
def _count_visibility_pairs(
    grid: NDArray[np.int8],
    candidate_coords_rc: NDArray[np.int32],
    open_coords_rc: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Count visible and diagonal-blocked target pairs for each candidate ordinal."""

    candidate_count = candidate_coords_rc.shape[0]
    open_cell_count = open_coords_rc.shape[0]
    visible_counts = np.zeros(candidate_count, dtype=np.int32)
    diagonal_counts = np.zeros(candidate_count, dtype=np.int32)

    for candidate_ordinal in range(candidate_count):
        start_row = int(candidate_coords_rc[candidate_ordinal, 0])
        start_col = int(candidate_coords_rc[candidate_ordinal, 1])
        visible_count = 0
        diagonal_count = 0

        for target_ordinal in range(open_cell_count):
            target_row = int(open_coords_rc[target_ordinal, 0])
            target_col = int(open_coords_rc[target_ordinal, 1])
            classification = _classify_visibility_pair(
                grid,
                start_row,
                start_col,
                target_row,
                target_col,
            )
            if classification == _VISIBILITY_VISIBLE:
                visible_count += 1
            elif classification == _VISIBILITY_BLOCKED_DIAGONAL:
                diagonal_count += 1

        visible_counts[candidate_ordinal] = visible_count
        diagonal_counts[candidate_ordinal] = diagonal_count

    return visible_counts, diagonal_counts


@njit(cache=True)
def _fill_visibility_pairs(
    grid: NDArray[np.int8],
    candidate_coords_rc: NDArray[np.int32],
    open_coords_rc: NDArray[np.int32],
    los_candidate_offsets: NDArray[np.int32],
    diagonal_candidate_offsets: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Fill sparse LOS and diagonal-block arrays from precomputed offsets."""

    candidate_count = candidate_coords_rc.shape[0]
    open_cell_count = open_coords_rc.shape[0]
    los_target_ordinals = np.empty(int(los_candidate_offsets[-1]), dtype=np.int32)
    diagonal_target_ordinals = np.empty(
        int(diagonal_candidate_offsets[-1]),
        dtype=np.int32,
    )
    los_write_positions = los_candidate_offsets[:-1].copy()
    diagonal_write_positions = diagonal_candidate_offsets[:-1].copy()

    for candidate_ordinal in range(candidate_count):
        start_row = int(candidate_coords_rc[candidate_ordinal, 0])
        start_col = int(candidate_coords_rc[candidate_ordinal, 1])
        los_write = int(los_write_positions[candidate_ordinal])
        diagonal_write = int(diagonal_write_positions[candidate_ordinal])

        for target_ordinal in range(open_cell_count):
            target_row = int(open_coords_rc[target_ordinal, 0])
            target_col = int(open_coords_rc[target_ordinal, 1])
            classification = _classify_visibility_pair(
                grid,
                start_row,
                start_col,
                target_row,
                target_col,
            )
            if classification == _VISIBILITY_VISIBLE:
                los_target_ordinals[los_write] = np.int32(target_ordinal)
                los_write += 1
            elif classification == _VISIBILITY_BLOCKED_DIAGONAL:
                diagonal_target_ordinals[diagonal_write] = np.int32(target_ordinal)
                diagonal_write += 1

    return los_target_ordinals, diagonal_target_ordinals


@njit(cache=True)
def _classify_visibility_pair(
    grid: NDArray[np.int8],
    start_row: int,
    start_col: int,
    target_row: int,
    target_col: int,
) -> np.int8:
    """Classify one candidate-target pair under the locked supercover LOS model."""

    if start_row == target_row and start_col == target_col:
        return _VISIBILITY_SKIPPED_SELF

    delta_row = target_row - start_row
    delta_col = target_col - start_col
    step_row = _sign(delta_row)
    step_col = _sign(delta_col)
    row_progress = 0
    col_progress = 0
    row_steps = abs(delta_row)
    col_steps = abs(delta_col)
    current_row = start_row
    current_col = start_col

    while row_progress < row_steps or col_progress < col_steps:
        decision = ((1 + 2 * col_progress) * row_steps) - (
            (1 + 2 * row_progress) * col_steps
        )
        previous_row = current_row
        previous_col = current_col
        moved_diagonally = False

        if decision == 0:
            current_row += step_row
            current_col += step_col
            row_progress += 1
            col_progress += 1
            moved_diagonally = True
        elif decision < 0:
            current_col += step_col
            col_progress += 1
        else:
            current_row += step_row
            row_progress += 1

        if not _cell_is_open(grid, current_row, current_col):
            return _VISIBILITY_BLOCKED_DIRECT
        if moved_diagonally and _is_diagonal_corner_blocked(
            grid,
            previous_row,
            previous_col,
            current_row,
            current_col,
        ):
            return _VISIBILITY_BLOCKED_DIAGONAL

    return _VISIBILITY_VISIBLE


@njit(cache=True)
def _is_diagonal_corner_blocked(
    grid: NDArray[np.int8],
    previous_row: int,
    previous_col: int,
    current_row: int,
    current_col: int,
) -> bool:
    """Apply the locked conservative blocking rule for diagonal supercover steps."""

    delta_row = current_row - previous_row
    delta_col = current_col - previous_col
    if delta_row == 0 or delta_col == 0:
        return False

    side_row_open = _cell_is_open(grid, previous_row + delta_row, previous_col)
    side_col_open = _cell_is_open(grid, previous_row, previous_col + delta_col)
    return not side_row_open and not side_col_open


@njit(cache=True)
def _cell_is_open(grid: NDArray[np.int8], row: int, col: int) -> bool:
    """Return whether one grid cell is in-bounds and open."""

    if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
        return False
    return grid[row, col] == OPEN_CELL


@njit(cache=True)
def _sign(value: int) -> int:
    """Return the integer sign of one delta value."""

    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


__all__ = [
    "_VISIBILITY_BLOCKED_DIAGONAL",
    "_VISIBILITY_VISIBLE",
    "_classify_visibility_pair",
    "_count_visibility_pairs",
    "_fill_visibility_pairs",
]
