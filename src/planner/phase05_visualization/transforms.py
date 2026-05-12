from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _reconstruct_final_score_grid(
    grid_shape: tuple[int, int],
    open_cell_coords_rc: NDArray[np.int32],
    final_open_cell_scores: NDArray[np.int8],
) -> NDArray[np.int8]:
    """Scatter the 1D open-cell score vector back onto the full floor-plan grid."""

    final_score_grid = np.full(grid_shape, -1, dtype=np.int8)
    if len(open_cell_coords_rc) == 0:
        return final_score_grid

    rows = open_cell_coords_rc[:, 0].astype(np.int64, copy=False)
    cols = open_cell_coords_rc[:, 1].astype(np.int64, copy=False)
    final_score_grid[rows, cols] = final_open_cell_scores
    return final_score_grid

def _decode_selected_candidate_coords(
    candidate_cell_coords_rc: NDArray[np.int32],
    selected_candidate_ordinals: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Decode selected candidate ordinals into persisted `(row, col)` coordinates."""

    if len(selected_candidate_ordinals) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return candidate_cell_coords_rc[
        selected_candidate_ordinals.astype(np.int64, copy=False)
    ].astype(np.int32, copy=False)

def _build_dori_score_histogram(
    final_open_cell_scores: NDArray[np.int8],
) -> dict[str, int]:
    """Return the deterministic score histogram used by summaries and notebook checks."""

    counts = np.bincount(
        final_open_cell_scores.astype(np.int64, copy=False),
        minlength=5,
    )
    return {str(score): int(counts[score]) for score in range(5)}
