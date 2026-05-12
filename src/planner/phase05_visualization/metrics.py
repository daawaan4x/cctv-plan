from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .artifacts import CoverageMetrics
from .score_validation import _validate_final_open_cell_scores


def compute_coverage_metrics(
    final_open_cell_scores: NDArray[np.int8],
) -> CoverageMetrics:
    """Compute coverage metrics over open cells only from the final score vector."""

    _validate_final_open_cell_scores(final_open_cell_scores, open_cell_count=None)
    open_cell_count = len(final_open_cell_scores)
    if open_cell_count == 0:
        return CoverageMetrics(
            total_dori_score=0.0,
            detection_plus_pct=0.0,
            observation_plus_pct=0.0,
            recognition_plus_pct=0.0,
            identification_pct=0.0,
            blind_spot_pct=0.0,
        )

    total_dori_score = float(np.sum(final_open_cell_scores, dtype=np.int64))
    return CoverageMetrics(
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

def _metrics_equal(left: CoverageMetrics, right: CoverageMetrics) -> bool:
    """Return whether two metric bundles match within float tolerance."""

    return bool(
        np.isclose(left.total_dori_score, right.total_dori_score)
        and np.isclose(left.detection_plus_pct, right.detection_plus_pct)
        and np.isclose(left.observation_plus_pct, right.observation_plus_pct)
        and np.isclose(left.recognition_plus_pct, right.recognition_plus_pct)
        and np.isclose(left.identification_pct, right.identification_pct)
        and np.isclose(left.blind_spot_pct, right.blind_spot_pct)
    )
