from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class _CoverageMetrics:
    """Coverage metrics computed over open cells only for one solution artifact."""

    total_dori_score: float
    detection_plus_pct: float
    observation_plus_pct: float
    recognition_plus_pct: float
    identification_pct: float
    blind_spot_pct: float

def _compute_coverage_metrics(
    final_open_cell_scores: NDArray[np.int8],
) -> _CoverageMetrics:
    """Compute the planned summary metrics over open cells only."""

    open_cell_count = len(final_open_cell_scores)
    if open_cell_count == 0:
        return _CoverageMetrics(
            total_dori_score=0.0,
            detection_plus_pct=0.0,
            observation_plus_pct=0.0,
            recognition_plus_pct=0.0,
            identification_pct=0.0,
            blind_spot_pct=0.0,
        )

    total_dori_score = float(np.sum(final_open_cell_scores, dtype=np.int64))
    return _CoverageMetrics(
        total_dori_score=total_dori_score,
        detection_plus_pct=(
            100.0 * float(np.count_nonzero(final_open_cell_scores >= 1))
        )
        / open_cell_count,
        observation_plus_pct=(
            100.0 * float(np.count_nonzero(final_open_cell_scores >= 2))
        )
        / open_cell_count,
        recognition_plus_pct=(
            100.0 * float(np.count_nonzero(final_open_cell_scores >= 3))
        )
        / open_cell_count,
        identification_pct=(
            100.0 * float(np.count_nonzero(final_open_cell_scores >= 4))
        )
        / open_cell_count,
        blind_spot_pct=(100.0 * float(np.count_nonzero(final_open_cell_scores == 0)))
        / open_cell_count,
    )
