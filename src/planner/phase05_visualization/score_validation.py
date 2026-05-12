from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import _VALID_DORI_SCORES


def _validate_final_open_cell_scores(
    final_open_cell_scores: NDArray[np.int8],
    *,
    open_cell_count: int | None,
) -> None:
    """Validate dtype, shape, length, and admissible DORI scores for open cells."""

    if final_open_cell_scores.dtype != np.int8:
        raise TypeError("final_open_cell_scores must use dtype np.int8.")
    if final_open_cell_scores.ndim != 1:
        raise ValueError("final_open_cell_scores must be a 1D array.")
    if open_cell_count is not None and len(final_open_cell_scores) != open_cell_count:
        raise ValueError("final_open_cell_scores length must equal open_cell_count.")
    if final_open_cell_scores.size and not np.isin(
        final_open_cell_scores,
        _VALID_DORI_SCORES,
    ).all():
        raise ValueError("final_open_cell_scores must contain only the scores 0..4.")
