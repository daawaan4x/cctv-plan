from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.planner._shared.sparse import build_offsets_from_counts as _build_offsets_from_counts
from src.planner.phase03_scoring import SparseScoreArtifacts

from .constants import _DORI_LEVELS, _LARGE_THRESHOLD_MEMBERSHIP_WARNING


@dataclass(frozen=True, slots=True)
class _ThresholdCoverIndex:
    """Target-major threshold cover arrays for score levels one through four."""

    level_offsets: tuple[
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
    ]
    level_configuration_ordinals: tuple[
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
    ]

def _build_threshold_cover_index(
    phase03_artifacts: SparseScoreArtifacts,
) -> _ThresholdCoverIndex:
    """Invert phase-03 configuration-major scores into target-major threshold lists."""

    counts_by_level = _count_threshold_cover_memberships(
        phase03_artifacts.score_configuration_offsets,
        phase03_artifacts.score_target_ordinals,
        phase03_artifacts.score_values,
        phase03_artifacts.open_cell_count,
    )
    level_offsets = tuple(
        _build_offsets_from_counts(counts_by_level[level_index])
        for level_index in range(len(_DORI_LEVELS))
    )
    total_memberships = sum(int(offsets[-1]) for offsets in level_offsets)
    if total_memberships >= _LARGE_THRESHOLD_MEMBERSHIP_WARNING:
        warnings.warn(
            "The exact phase-04 threshold index is large and may require substantial "
            "memory or solver build time for this workspace.",
            stacklevel=2,
        )

    level_configuration_ordinals = tuple(
        np.empty(int(offsets[-1]), dtype=np.int32) for offsets in level_offsets
    )
    _fill_threshold_cover_arrays(
        phase03_artifacts.score_configuration_offsets,
        phase03_artifacts.score_target_ordinals,
        phase03_artifacts.score_values,
        level_offsets[0],
        level_configuration_ordinals[0],
        level_offsets[1],
        level_configuration_ordinals[1],
        level_offsets[2],
        level_configuration_ordinals[2],
        level_offsets[3],
        level_configuration_ordinals[3],
    )

    return _ThresholdCoverIndex(
        level_offsets=(
            level_offsets[0],
            level_offsets[1],
            level_offsets[2],
            level_offsets[3],
        ),
        level_configuration_ordinals=(
            level_configuration_ordinals[0],
            level_configuration_ordinals[1],
            level_configuration_ordinals[2],
            level_configuration_ordinals[3],
        ),
    )

@njit(cache=True)
def _count_threshold_cover_memberships(
    score_configuration_offsets: NDArray[np.int32],
    score_target_ordinals: NDArray[np.int32],
    score_values: NDArray[np.int8],
    open_cell_count: int,
) -> NDArray[np.int32]:
    """Count threshold-cover memberships for each target and score level."""

    counts = np.zeros((4, open_cell_count), dtype=np.int32)
    configuration_count = len(score_configuration_offsets) - 1
    for configuration_ordinal in range(configuration_count):
        start = int(score_configuration_offsets[configuration_ordinal])
        stop = int(score_configuration_offsets[configuration_ordinal + 1])
        for flat_index in range(start, stop):
            target_ordinal = int(score_target_ordinals[flat_index])
            score = int(score_values[flat_index])
            if score >= 1:
                counts[0, target_ordinal] += 1
            if score >= 2:
                counts[1, target_ordinal] += 1
            if score >= 3:
                counts[2, target_ordinal] += 1
            if score >= 4:
                counts[3, target_ordinal] += 1
    return counts

@njit(cache=True)
def _fill_threshold_cover_arrays(
    score_configuration_offsets: NDArray[np.int32],
    score_target_ordinals: NDArray[np.int32],
    score_values: NDArray[np.int8],
    level1_offsets: NDArray[np.int32],
    level1_configuration_ordinals: NDArray[np.int32],
    level2_offsets: NDArray[np.int32],
    level2_configuration_ordinals: NDArray[np.int32],
    level3_offsets: NDArray[np.int32],
    level3_configuration_ordinals: NDArray[np.int32],
    level4_offsets: NDArray[np.int32],
    level4_configuration_ordinals: NDArray[np.int32],
) -> None:
    """Fill preallocated target-major threshold-cover arrays in config-order."""

    level1_cursors = level1_offsets[:-1].copy()
    level2_cursors = level2_offsets[:-1].copy()
    level3_cursors = level3_offsets[:-1].copy()
    level4_cursors = level4_offsets[:-1].copy()
    configuration_count = len(score_configuration_offsets) - 1

    for configuration_ordinal in range(configuration_count):
        start = int(score_configuration_offsets[configuration_ordinal])
        stop = int(score_configuration_offsets[configuration_ordinal + 1])
        for flat_index in range(start, stop):
            target_ordinal = int(score_target_ordinals[flat_index])
            score = int(score_values[flat_index])
            if score >= 1:
                write_index = int(level1_cursors[target_ordinal])
                level1_configuration_ordinals[write_index] = np.int32(
                    configuration_ordinal
                )
                level1_cursors[target_ordinal] = np.int32(write_index + 1)
            if score >= 2:
                write_index = int(level2_cursors[target_ordinal])
                level2_configuration_ordinals[write_index] = np.int32(
                    configuration_ordinal
                )
                level2_cursors[target_ordinal] = np.int32(write_index + 1)
            if score >= 3:
                write_index = int(level3_cursors[target_ordinal])
                level3_configuration_ordinals[write_index] = np.int32(
                    configuration_ordinal
                )
                level3_cursors[target_ordinal] = np.int32(write_index + 1)
            if score >= 4:
                write_index = int(level4_cursors[target_ordinal])
                level4_configuration_ordinals[write_index] = np.int32(
                    configuration_ordinal
                )
                level4_cursors[target_ordinal] = np.int32(write_index + 1)
