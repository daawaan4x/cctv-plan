"""Public and private accessors for phase-03 sparse score artifacts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .artifacts import SparseScoreArtifacts


def get_configuration_target_ordinals(
    artifacts: SparseScoreArtifacts,
    configuration_ordinal: int,
) -> NDArray[np.int32]:
    """Return the sparse target-ordinal slice for one configuration ordinal."""

    _validate_configuration_ordinal(
        configuration_ordinal,
        configuration_count=len(artifacts.configuration_candidate_ordinals),
    )
    start = int(artifacts.score_configuration_offsets[configuration_ordinal])
    stop = int(artifacts.score_configuration_offsets[configuration_ordinal + 1])
    return artifacts.score_target_ordinals[start:stop]


def get_configuration_dori_scores(
    artifacts: SparseScoreArtifacts,
    configuration_ordinal: int,
) -> NDArray[np.int8]:
    """Return the sparse DORI-score slice for one configuration ordinal."""

    _validate_configuration_ordinal(
        configuration_ordinal,
        configuration_count=len(artifacts.configuration_candidate_ordinals),
    )
    start = int(artifacts.score_configuration_offsets[configuration_ordinal])
    stop = int(artifacts.score_configuration_offsets[configuration_ordinal + 1])
    return artifacts.score_values[start:stop]


def decode_configuration_ordinal(
    artifacts: SparseScoreArtifacts,
    configuration_ordinal: int,
) -> tuple[int, int, float]:
    """Decode one configuration ordinal into candidate ordinal, angle ordinal, and angle."""

    _validate_configuration_ordinal(
        configuration_ordinal,
        configuration_count=len(artifacts.configuration_candidate_ordinals),
    )
    candidate_ordinal = int(
        artifacts.configuration_candidate_ordinals[configuration_ordinal]
    )
    angle_ordinal = int(artifacts.configuration_angle_ordinals[configuration_ordinal])
    angle_deg = float(artifacts.orientation_angles_deg[angle_ordinal])
    return candidate_ordinal, angle_ordinal, angle_deg


def _validate_configuration_ordinal(
    configuration_ordinal: int,
    *,
    configuration_count: int,
) -> None:
    """Ensure one public query request references an existing configuration ordinal."""

    if configuration_ordinal < 0 or configuration_ordinal >= configuration_count:
        raise IndexError(
            "configuration_ordinal is out of range for the sparse-score artifact."
        )


__all__ = [
    "decode_configuration_ordinal",
    "get_configuration_dori_scores",
    "get_configuration_target_ordinals",
]
