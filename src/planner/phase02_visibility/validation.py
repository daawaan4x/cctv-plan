from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput
from src.planner._shared.sparse import choose_sample_indices as _choose_sample_indices
from src.planner.phase01_candidate_generation import CandidateGenerationArtifacts

from .accessors import (
    get_diagonal_blocked_target_ordinals,
    get_visible_target_ordinals,
)
from .artifacts import VisibilityArtifacts
from .line_of_sight import (
    _VISIBILITY_BLOCKED_DIAGONAL,
    _VISIBILITY_VISIBLE,
    _classify_visibility_pair,
)

_MAX_SEMANTIC_VALIDATION_PAIRS = 4096


def validate_visibility_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    artifacts: VisibilityArtifacts,
) -> None:
    """Validate structural invariants plus sampled LOS semantics for phase 02."""

    _validate_phase01_compatibility(floorplan, phase01_artifacts)

    candidate_count = len(phase01_artifacts.candidate_cell_indices)
    open_cell_count = len(phase01_artifacts.open_cell_indices)

    if artifacts.grid_shape != floorplan.shape:
        raise ValueError("Visibility grid_shape does not match floorplan.shape.")
    if artifacts.grid_shape != phase01_artifacts.grid_shape:
        raise ValueError("Visibility grid_shape does not match phase-01 grid_shape.")
    if artifacts.candidate_count != candidate_count:
        raise ValueError(
            "Visibility candidate_count does not match the phase-01 candidate count."
        )
    if artifacts.open_cell_count != open_cell_count:
        raise ValueError(
            "Visibility open_cell_count does not match the phase-01 open-cell count."
        )

    _validate_offsets(
        "los_candidate_offsets",
        artifacts.los_candidate_offsets,
        expected_candidate_count=candidate_count,
        expected_total=len(artifacts.los_target_ordinals),
    )
    _validate_offsets(
        "diagonal_candidate_offsets",
        artifacts.diagonal_candidate_offsets,
        expected_candidate_count=candidate_count,
        expected_total=len(artifacts.diagonal_target_ordinals),
    )
    _validate_target_ordinals(
        "los_target_ordinals",
        artifacts.los_target_ordinals,
        open_cell_count=open_cell_count,
    )
    _validate_target_ordinals(
        "diagonal_target_ordinals",
        artifacts.diagonal_target_ordinals,
        open_cell_count=open_cell_count,
    )

    for candidate_ordinal in range(candidate_count):
        los_targets = get_visible_target_ordinals(artifacts, candidate_ordinal)
        diagonal_targets = get_diagonal_blocked_target_ordinals(
            artifacts,
            candidate_ordinal,
        )
        _validate_strictly_increasing(
            "los_target_ordinals",
            candidate_ordinal,
            los_targets,
        )
        _validate_strictly_increasing(
            "diagonal_target_ordinals",
            candidate_ordinal,
            diagonal_targets,
        )
        _validate_disjoint_candidate_slices(
            candidate_ordinal,
            los_targets,
            diagonal_targets,
        )

    _validate_sampled_pair_semantics(floorplan, phase01_artifacts, artifacts)

def _validate_phase01_compatibility(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
) -> None:
    """Confirm that phase-01 ordinals still describe the current floor plan."""

    if phase01_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-01 artifacts do not match the current floorplan shape.")

def _validate_offsets(
    name: str,
    offsets: NDArray[np.int32],
    *,
    expected_candidate_count: int,
    expected_total: int,
) -> None:
    """Validate CSR-style offset arrays for candidate-major sparse slices."""

    if offsets.dtype != np.int32:
        raise TypeError(f"{name} must use dtype np.int32.")
    if offsets.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if len(offsets) != expected_candidate_count + 1:
        raise ValueError(f"{name} length must be candidate_count + 1.")
    if offsets[0] != 0:
        raise ValueError(f"{name} must start at 0.")
    if len(offsets) > 1 and np.any(offsets[:-1] > offsets[1:]):
        raise ValueError(f"{name} must be monotonic nondecreasing.")
    if int(offsets[-1]) != expected_total:
        raise ValueError(f"{name} final offset must match the paired array length.")

def _validate_target_ordinals(
    name: str,
    target_ordinals: NDArray[np.int32],
    *,
    open_cell_count: int,
) -> None:
    """Validate flattened target-ordinal arrays used by the sparse LOS artifacts."""

    if target_ordinals.dtype != np.int32:
        raise TypeError(f"{name} must use dtype np.int32.")
    if target_ordinals.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if target_ordinals.size and (
        target_ordinals[0] < 0 or target_ordinals[-1] >= open_cell_count
    ):
        raise ValueError(f"{name} contains an out-of-range target ordinal.")

def _validate_strictly_increasing(
    name: str,
    candidate_ordinal: int,
    target_ordinals: NDArray[np.int32],
) -> None:
    """Confirm that each candidate slice is sorted and duplicate-free."""

    if len(target_ordinals) > 1 and np.any(target_ordinals[:-1] >= target_ordinals[1:]):
        raise ValueError(
            f"{name} slice for candidate ordinal {candidate_ordinal} must be strictly "
            "increasing."
        )

def _validate_disjoint_candidate_slices(
    candidate_ordinal: int,
    visible_targets: NDArray[np.int32],
    diagonal_targets: NDArray[np.int32],
) -> None:
    """Reject target ordinals that appear in both visible and diagonal slices."""

    left = 0
    right = 0
    while left < len(visible_targets) and right < len(diagonal_targets):
        visible_target = int(visible_targets[left])
        diagonal_target = int(diagonal_targets[right])
        if visible_target == diagonal_target:
            raise ValueError(
                "A target ordinal cannot be both visible and diagonal-blocked for "
                f"candidate ordinal {candidate_ordinal}."
            )
        if visible_target < diagonal_target:
            left += 1
        else:
            right += 1

def _validate_sampled_pair_semantics(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    artifacts: VisibilityArtifacts,
) -> None:
    """Recompute a bounded sample of stored pairs to confirm LOS semantics."""

    total_pairs = len(artifacts.los_target_ordinals) + len(
        artifacts.diagonal_target_ordinals
    )
    if total_pairs == 0:
        return

    if total_pairs <= _MAX_SEMANTIC_VALIDATION_PAIRS:
        los_sample_indices = np.arange(len(artifacts.los_target_ordinals), dtype=np.int64)
        diagonal_sample_indices = np.arange(
            len(artifacts.diagonal_target_ordinals),
            dtype=np.int64,
        )
    else:
        los_share = len(artifacts.los_target_ordinals) / total_pairs
        los_budget = int(round(_MAX_SEMANTIC_VALIDATION_PAIRS * los_share))
        los_budget = min(
            len(artifacts.los_target_ordinals),
            max(1, los_budget) if len(artifacts.los_target_ordinals) else 0,
        )
        diagonal_budget = min(
            len(artifacts.diagonal_target_ordinals),
            _MAX_SEMANTIC_VALIDATION_PAIRS - los_budget,
        )
        if (
            diagonal_budget == 0
            and len(artifacts.diagonal_target_ordinals) > 0
            and los_budget > 1
        ):
            los_budget -= 1
            diagonal_budget = 1

        los_sample_indices = _choose_sample_indices(
            len(artifacts.los_target_ordinals),
            los_budget,
        )
        diagonal_sample_indices = _choose_sample_indices(
            len(artifacts.diagonal_target_ordinals),
            diagonal_budget,
        )

    _validate_pair_collection(
        floorplan.grid,
        phase01_artifacts.candidate_cell_coords_rc,
        phase01_artifacts.open_cell_coords_rc,
        artifacts.los_candidate_offsets,
        artifacts.los_target_ordinals,
        los_sample_indices,
        expected_classification=_VISIBILITY_VISIBLE,
    )
    _validate_pair_collection(
        floorplan.grid,
        phase01_artifacts.candidate_cell_coords_rc,
        phase01_artifacts.open_cell_coords_rc,
        artifacts.diagonal_candidate_offsets,
        artifacts.diagonal_target_ordinals,
        diagonal_sample_indices,
        expected_classification=_VISIBILITY_BLOCKED_DIAGONAL,
    )

def _validate_pair_collection(
    grid: NDArray[np.int8],
    candidate_coords_rc: NDArray[np.int32],
    open_coords_rc: NDArray[np.int32],
    offsets: NDArray[np.int32],
    target_ordinals: NDArray[np.int32],
    sample_indices: NDArray[np.int64],
    *,
    expected_classification: np.int8,
) -> None:
    """Re-run the LOS classifier for sampled stored sparse entries."""

    for flat_index in sample_indices:
        candidate_ordinal = int(np.searchsorted(offsets, flat_index, side="right") - 1)
        target_ordinal = int(target_ordinals[flat_index])
        start_row = int(candidate_coords_rc[candidate_ordinal, 0])
        start_col = int(candidate_coords_rc[candidate_ordinal, 1])
        target_row = int(open_coords_rc[target_ordinal, 0])
        target_col = int(open_coords_rc[target_ordinal, 1])
        classification = _classify_visibility_pair(
            grid,
            start_row,
            start_col,
            target_row,
            target_col,
        )
        if classification != expected_classification:
            raise ValueError(
                "Stored visibility classification does not match a recomputed LOS "
                "result for the sampled candidate-target pair."
            )
