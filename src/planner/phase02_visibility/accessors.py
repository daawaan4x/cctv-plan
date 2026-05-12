from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .artifacts import VisibilityArtifacts


def get_visible_target_ordinals(
    artifacts: VisibilityArtifacts,
    candidate_ordinal: int,
) -> NDArray[np.int32]:
    """Return the LOS-positive target-ordinal slice for one candidate ordinal."""

    _validate_candidate_ordinal(candidate_ordinal, artifacts.candidate_count)
    start = int(artifacts.los_candidate_offsets[candidate_ordinal])
    stop = int(artifacts.los_candidate_offsets[candidate_ordinal + 1])
    return artifacts.los_target_ordinals[start:stop]

def get_diagonal_blocked_target_ordinals(
    artifacts: VisibilityArtifacts,
    candidate_ordinal: int,
) -> NDArray[np.int32]:
    """Return the corner-blocked target-ordinal slice for one candidate ordinal."""

    _validate_candidate_ordinal(candidate_ordinal, artifacts.candidate_count)
    start = int(artifacts.diagonal_candidate_offsets[candidate_ordinal])
    stop = int(artifacts.diagonal_candidate_offsets[candidate_ordinal + 1])
    return artifacts.diagonal_target_ordinals[start:stop]

def _validate_candidate_ordinal(candidate_ordinal: int, candidate_count: int) -> None:
    """Ensure a public query request references an existing candidate ordinal."""

    if candidate_ordinal < 0 or candidate_ordinal >= candidate_count:
        raise IndexError("candidate_ordinal is out of range for the visibility artifact.")
