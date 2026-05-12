"""Solution reconstruction helpers for phase-04 optimization artifacts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.planner.phase03_scoring import (
    SparseScoreArtifacts,
    get_configuration_dori_scores,
    get_configuration_target_ordinals,
)


def _reconstruct_final_scores_from_selection(
    phase03_artifacts: SparseScoreArtifacts,
    selected_configuration_ordinals: NDArray[np.int32],
) -> tuple[NDArray[np.int8], NDArray[np.int32]]:
    """Rebuild final best-per-target scores directly from selected sparse slices."""

    final_scores = np.zeros(phase03_artifacts.open_cell_count, dtype=np.int8)
    best_configuration_ordinals = np.full(
        phase03_artifacts.open_cell_count,
        -1,
        dtype=np.int32,
    )
    for configuration_ordinal in selected_configuration_ordinals:
        configuration_index = int(configuration_ordinal)
        target_ordinals = get_configuration_target_ordinals(
            phase03_artifacts,
            configuration_index,
        )
        configuration_scores = get_configuration_dori_scores(
            phase03_artifacts,
            configuration_index,
        )
        if len(target_ordinals) == 0:
            continue

        current_scores = final_scores[target_ordinals]
        improve_mask = configuration_scores > current_scores
        if not np.any(improve_mask):
            continue

        improved_targets = target_ordinals[improve_mask]
        final_scores[improved_targets] = configuration_scores[improve_mask]
        best_configuration_ordinals[improved_targets] = np.int32(configuration_index)

    return final_scores, best_configuration_ordinals
