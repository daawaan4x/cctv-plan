"""Validation helpers for phase-03 sparse score artifacts."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput
from src.planner._shared.config import PlannerConfig
from src.planner._shared.sparse import choose_sample_indices as _choose_sample_indices
from src.planner.phase01_candidate_generation import CandidateGenerationArtifacts
from src.planner.phase02_visibility import VisibilityArtifacts, get_visible_target_ordinals

from .accessors import (
    decode_configuration_ordinal,
    get_configuration_dori_scores,
    get_configuration_target_ordinals,
)
from .artifacts import SparseScoreArtifacts
from .scoring import (
    _ScoringConstants,
    _build_configuration_index_arrays,
    _build_scoring_constants,
    _build_orientation_angles_array,
    _require_grid_cell_size_m,
    _score_one_candidate_target_orientation,
)

_MAX_SEMANTIC_VALIDATION_PAIRS: Final[int] = 4096


def validate_sparse_score_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase02_artifacts: VisibilityArtifacts,
    artifacts: SparseScoreArtifacts,
) -> None:
    """Validate structural invariants plus sampled score semantics for phase 03."""

    grid_cell_size_m = _require_grid_cell_size_m(floorplan)
    expected_orientation_angles_deg = _build_orientation_angles_array(config)
    _validate_phase_dependencies(
        floorplan,
        config,
        phase01_artifacts,
        phase02_artifacts,
        expected_orientation_angles_deg,
    )

    if artifacts.grid_shape != floorplan.shape:
        raise ValueError("Sparse-score grid_shape does not match floorplan.shape.")
    if artifacts.grid_shape != phase01_artifacts.grid_shape:
        raise ValueError("Sparse-score grid_shape does not match phase-01 grid_shape.")
    if artifacts.grid_shape != phase02_artifacts.grid_shape:
        raise ValueError("Sparse-score grid_shape does not match phase-02 grid_shape.")
    if artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Sparse-score candidate_count does not match the phase-01 candidate count."
        )
    if artifacts.candidate_count != phase02_artifacts.candidate_count:
        raise ValueError(
            "Sparse-score candidate_count does not match the phase-02 candidate count."
        )
    if artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Sparse-score open_cell_count does not match the phase-01 open-cell count."
        )
    if artifacts.open_cell_count != phase02_artifacts.open_cell_count:
        raise ValueError(
            "Sparse-score open_cell_count does not match the phase-02 open-cell count."
        )

    _validate_orientation_angles_array(
        artifacts.orientation_angles_deg,
        expected_orientation_angles_deg,
    )
    _validate_configuration_index_arrays(
        artifacts.configuration_candidate_ordinals,
        artifacts.configuration_angle_ordinals,
        candidate_count=artifacts.candidate_count,
        orientation_count=len(artifacts.orientation_angles_deg),
    )
    _validate_offsets(
        artifacts.score_configuration_offsets,
        expected_configuration_count=len(artifacts.configuration_candidate_ordinals),
        expected_total=len(artifacts.score_target_ordinals),
    )
    _validate_target_ordinals(
        artifacts.score_target_ordinals,
        open_cell_count=artifacts.open_cell_count,
    )
    _validate_score_values(
        artifacts.score_values,
        expected_total=len(artifacts.score_target_ordinals),
    )

    for configuration_ordinal in range(len(artifacts.configuration_candidate_ordinals)):
        target_ordinals = get_configuration_target_ordinals(
            artifacts,
            configuration_ordinal,
        )
        score_values = get_configuration_dori_scores(artifacts, configuration_ordinal)
        _validate_strictly_increasing(
            target_ordinals,
            configuration_ordinal=configuration_ordinal,
        )
        if len(score_values) != len(target_ordinals):
            raise ValueError(
                "Each configuration score slice must match its target-ordinal slice "
                "length."
            )

    scoring_constants = _build_scoring_constants(config, grid_cell_size_m)
    _validate_sampled_pair_semantics(
        phase01_artifacts,
        phase02_artifacts,
        artifacts,
        scoring_constants,
    )


def _validate_phase_dependencies(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase02_artifacts: VisibilityArtifacts,
    orientation_angles_deg: NDArray[np.float32],
) -> None:
    """Confirm that phase-01 and phase-02 artifacts still match the current inputs."""

    _ = config
    if phase01_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-01 artifacts do not match the current floorplan shape.")
    if phase02_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-02 artifacts do not match the current floorplan shape.")
    if phase02_artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Phase-02 candidate_count does not match the phase-01 candidate count."
        )
    if phase02_artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Phase-02 open_cell_count does not match the phase-01 open-cell count."
        )
    if orientation_angles_deg.ndim != 1 or len(orientation_angles_deg) == 0:
        raise ValueError("orientation_angles_deg must contain at least one angle.")


def _validate_orientation_angles_array(
    orientation_angles_deg: NDArray[np.float32],
    expected_orientation_angles_deg: NDArray[np.float32],
) -> None:
    """Validate dtype, shape, ordering, and config alignment for orientations."""

    if orientation_angles_deg.dtype != np.float32:
        raise TypeError("orientation_angles_deg must use dtype np.float32.")
    if orientation_angles_deg.ndim != 1:
        raise ValueError("orientation_angles_deg must be a 1D array.")
    if len(orientation_angles_deg) == 0:
        raise ValueError("orientation_angles_deg must not be empty.")
    if len(orientation_angles_deg) > 1 and not np.all(
        orientation_angles_deg[:-1] < orientation_angles_deg[1:]
    ):
        raise ValueError("orientation_angles_deg must be strictly increasing.")
    if not np.array_equal(orientation_angles_deg, expected_orientation_angles_deg):
        raise ValueError(
            "orientation_angles_deg does not match the planner configuration's "
            "discrete orientation set."
        )


def _validate_configuration_index_arrays(
    configuration_candidate_ordinals: NDArray[np.int32],
    configuration_angle_ordinals: NDArray[np.int16],
    *,
    candidate_count: int,
    orientation_count: int,
) -> None:
    """Validate deterministic configuration-to-candidate and angle mappings."""

    expected_configuration_count = candidate_count * orientation_count
    if configuration_candidate_ordinals.dtype != np.int32:
        raise TypeError("configuration_candidate_ordinals must use dtype np.int32.")
    if configuration_candidate_ordinals.ndim != 1:
        raise ValueError("configuration_candidate_ordinals must be a 1D array.")
    if len(configuration_candidate_ordinals) != expected_configuration_count:
        raise ValueError(
            "configuration_candidate_ordinals length must equal candidate_count * "
            "orientation_count."
        )
    if configuration_candidate_ordinals.size and (
        configuration_candidate_ordinals[0] < 0
        or configuration_candidate_ordinals[-1] >= candidate_count
    ):
        raise ValueError(
            "configuration_candidate_ordinals contains an out-of-range candidate "
            "ordinal."
        )

    if configuration_angle_ordinals.dtype != np.int16:
        raise TypeError("configuration_angle_ordinals must use dtype np.int16.")
    if configuration_angle_ordinals.ndim != 1:
        raise ValueError("configuration_angle_ordinals must be a 1D array.")
    if len(configuration_angle_ordinals) != expected_configuration_count:
        raise ValueError(
            "configuration_angle_ordinals length must equal candidate_count * "
            "orientation_count."
        )
    if configuration_angle_ordinals.size and (
        configuration_angle_ordinals[0] < 0
        or np.max(configuration_angle_ordinals) >= orientation_count
    ):
        raise ValueError(
            "configuration_angle_ordinals contains an out-of-range angle ordinal."
        )

    expected_candidate_ordinals, expected_angle_ordinals = (
        _build_configuration_index_arrays(candidate_count, orientation_count)
    )
    if not np.array_equal(
        configuration_candidate_ordinals,
        expected_candidate_ordinals,
    ):
        raise ValueError(
            "configuration_candidate_ordinals does not match the required "
            "candidate-major deterministic ordering."
        )
    if not np.array_equal(configuration_angle_ordinals, expected_angle_ordinals):
        raise ValueError(
            "configuration_angle_ordinals does not match the required angle-minor "
            "deterministic ordering."
        )


def _validate_offsets(
    score_configuration_offsets: NDArray[np.int32],
    *,
    expected_configuration_count: int,
    expected_total: int,
) -> None:
    """Validate CSR-style score offsets for configuration-major sparse slices."""

    if score_configuration_offsets.dtype != np.int32:
        raise TypeError("score_configuration_offsets must use dtype np.int32.")
    if score_configuration_offsets.ndim != 1:
        raise ValueError("score_configuration_offsets must be a 1D array.")
    if len(score_configuration_offsets) != expected_configuration_count + 1:
        raise ValueError(
            "score_configuration_offsets length must be configuration_count + 1."
        )
    if score_configuration_offsets[0] != 0:
        raise ValueError("score_configuration_offsets must start at 0.")
    if len(score_configuration_offsets) > 1 and np.any(
        score_configuration_offsets[:-1] > score_configuration_offsets[1:]
    ):
        raise ValueError("score_configuration_offsets must be monotonic nondecreasing.")
    if int(score_configuration_offsets[-1]) != expected_total:
        raise ValueError(
            "score_configuration_offsets final offset must match the paired sparse "
            "array lengths."
        )


def _validate_target_ordinals(
    score_target_ordinals: NDArray[np.int32],
    *,
    open_cell_count: int,
) -> None:
    """Validate the flattened sparse target-ordinal array."""

    if score_target_ordinals.dtype != np.int32:
        raise TypeError("score_target_ordinals must use dtype np.int32.")
    if score_target_ordinals.ndim != 1:
        raise ValueError("score_target_ordinals must be a 1D array.")
    if score_target_ordinals.size and (
        score_target_ordinals[0] < 0 or score_target_ordinals[-1] >= open_cell_count
    ):
        raise ValueError("score_target_ordinals contains an out-of-range target ordinal.")


def _validate_score_values(
    score_values: NDArray[np.int8],
    *,
    expected_total: int,
) -> None:
    """Validate dtype, shape, length, and allowed values for sparse DORI scores."""

    if score_values.dtype != np.int8:
        raise TypeError("score_values must use dtype np.int8.")
    if score_values.ndim != 1:
        raise ValueError("score_values must be a 1D array.")
    if len(score_values) != expected_total:
        raise ValueError(
            "score_values length must match the sparse target-ordinal array length."
        )
    if score_values.size and not np.isin(score_values, np.array([1, 2, 3, 4])).all():
        raise ValueError("score_values must contain only the categorical scores 1..4.")


def _validate_strictly_increasing(
    target_ordinals: NDArray[np.int32],
    *,
    configuration_ordinal: int,
) -> None:
    """Confirm that one configuration slice stays sorted and duplicate-free."""

    if len(target_ordinals) > 1 and np.any(target_ordinals[:-1] >= target_ordinals[1:]):
        raise ValueError(
            "score_target_ordinals slice for configuration ordinal "
            f"{configuration_ordinal} must be strictly increasing."
        )


def _validate_sampled_pair_semantics(
    phase01_artifacts: CandidateGenerationArtifacts,
    phase02_artifacts: VisibilityArtifacts,
    artifacts: SparseScoreArtifacts,
    scoring_constants: _ScoringConstants,
) -> None:
    """Recompute a bounded sample of stored pairs to confirm scoring semantics."""

    total_pairs = len(artifacts.score_target_ordinals)
    if total_pairs == 0:
        return

    sample_indices = _choose_sample_indices(
        total_pairs,
        max_samples=min(_MAX_SEMANTIC_VALIDATION_PAIRS, total_pairs),
    )
    open_coords_rc = phase01_artifacts.open_cell_coords_rc
    candidate_coords_rc = phase01_artifacts.candidate_cell_coords_rc

    for flat_index in sample_indices:
        configuration_ordinal = int(
            np.searchsorted(
                artifacts.score_configuration_offsets,
                flat_index,
                side="right",
            )
            - 1
        )
        target_ordinal = int(artifacts.score_target_ordinals[flat_index])
        stored_score = int(artifacts.score_values[flat_index])
        candidate_ordinal, angle_ordinal, orientation_deg = decode_configuration_ordinal(
            artifacts,
            configuration_ordinal,
        )
        visible_target_ordinals = get_visible_target_ordinals(
            phase02_artifacts,
            candidate_ordinal,
        )
        if not _sorted_contains(visible_target_ordinals, target_ordinal):
            raise ValueError(
                "Stored sparse-score pair is not present in the phase-02 LOS-positive "
                "visibility slice for its candidate ordinal."
            )

        candidate_row = int(candidate_coords_rc[candidate_ordinal, 0])
        candidate_col = int(candidate_coords_rc[candidate_ordinal, 1])
        target_row = int(open_coords_rc[target_ordinal, 0])
        target_col = int(open_coords_rc[target_ordinal, 1])
        recomputed_score = _score_one_candidate_target_orientation(
            candidate_row,
            candidate_col,
            target_row,
            target_col,
            orientation_deg,
            scoring_constants,
        )
        if recomputed_score != stored_score:
            raise ValueError(
                "Stored sparse-score value does not match a recomputed score for the "
                "sampled configuration-target pair."
            )
        if angle_ordinal < 0 or angle_ordinal >= len(artifacts.orientation_angles_deg):
            raise ValueError("Decoded angle ordinal fell outside the orientation array.")


def _sorted_contains(values: NDArray[np.int32], needle: int) -> bool:
    """Return whether one sorted ordinal array contains the requested value."""

    position = int(np.searchsorted(values, needle))
    return position < len(values) and int(values[position]) == needle


__all__ = [
    "_validate_configuration_index_arrays",
    "_validate_offsets",
    "_validate_orientation_angles_array",
    "_validate_score_values",
    "_validate_target_ordinals",
    "validate_sparse_score_artifacts",
]
