"""Phase 02 visibility helpers for sparse LOS generation and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput, OPEN_CELL
from src.planner._shared.cache import get_shared_artifact_dir, write_npz
from src.planner._shared.config import PlannerConfig
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)

PHASE_NAME = "visibility"
PHASE_ARTIFACT_STEM = "02_visibility"

_VISIBILITY_SKIPPED_SELF = np.int8(0)
_VISIBILITY_VISIBLE = np.int8(1)
_VISIBILITY_BLOCKED_DIRECT = np.int8(2)
_VISIBILITY_BLOCKED_DIAGONAL = np.int8(3)

_MAX_SEMANTIC_VALIDATION_PAIRS = 4096


@dataclass(frozen=True, slots=True)
class VisibilityArtifacts:
    """Sparse candidate-major LOS relationships derived from the phase-01 ordinals."""

    grid_shape: tuple[int, int]
    candidate_count: int
    open_cell_count: int
    los_candidate_offsets: NDArray[np.int32]
    los_target_ordinals: NDArray[np.int32]
    diagonal_candidate_offsets: NDArray[np.int32]
    diagonal_target_ordinals: NDArray[np.int32]


# Artifact generation
def generate_visibility_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
) -> VisibilityArtifacts:
    """Build deterministic sparse LOS artifacts for all candidate-open pairs."""

    _validate_phase01_compatibility(floorplan, phase01_artifacts)

    grid = floorplan.grid
    candidate_coords = phase01_artifacts.candidate_cell_coords_rc
    open_coords = phase01_artifacts.open_cell_coords_rc

    visible_counts, diagonal_counts = _count_visibility_pairs(
        grid,
        candidate_coords,
        open_coords,
    )
    los_candidate_offsets = _build_offsets_from_counts(visible_counts)
    diagonal_candidate_offsets = _build_offsets_from_counts(diagonal_counts)
    los_target_ordinals, diagonal_target_ordinals = _fill_visibility_pairs(
        grid,
        candidate_coords,
        open_coords,
        los_candidate_offsets,
        diagonal_candidate_offsets,
    )

    artifacts = VisibilityArtifacts(
        grid_shape=floorplan.shape,
        candidate_count=len(candidate_coords),
        open_cell_count=len(open_coords),
        los_candidate_offsets=los_candidate_offsets,
        los_target_ordinals=los_target_ordinals,
        diagonal_candidate_offsets=diagonal_candidate_offsets,
        diagonal_target_ordinals=diagonal_target_ordinals,
    )
    validate_visibility_artifacts(floorplan, phase01_artifacts, artifacts)
    return artifacts


def resolve_visibility_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    force: bool = False,
) -> VisibilityArtifacts:
    """Load, validate, or rebuild the canonical cached phase-02 artifact."""

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    artifact_path = (
        get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
        / f"{PHASE_ARTIFACT_STEM}.npz"
    )
    if not force and artifact_path.exists():
        try:
            artifacts = load_visibility_artifacts(artifact_path)
            validate_visibility_artifacts(floorplan, phase01_artifacts, artifacts)
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)
    save_visibility_artifacts(artifact_path, artifacts)
    return artifacts


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


# Artifact persistence
def save_visibility_artifacts(
    artifact_path: Path,
    artifacts: VisibilityArtifacts,
) -> Path:
    """Persist phase-02 LOS artifacts to the deterministic `02_visibility.npz` schema."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        candidate_count=np.asarray(artifacts.candidate_count, dtype=np.int32),
        open_cell_count=np.asarray(artifacts.open_cell_count, dtype=np.int32),
        los_candidate_offsets=artifacts.los_candidate_offsets,
        los_target_ordinals=artifacts.los_target_ordinals,
        diagonal_candidate_offsets=artifacts.diagonal_candidate_offsets,
        diagonal_target_ordinals=artifacts.diagonal_target_ordinals,
    )


def load_visibility_artifacts(
    artifact_path: Path,
) -> VisibilityArtifacts:
    """Load phase-02 LOS artifacts from disk and perform structural validation."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = VisibilityArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            candidate_count=int(payload["candidate_count"].item()),
            open_cell_count=int(payload["open_cell_count"].item()),
            los_candidate_offsets=payload["los_candidate_offsets"].astype(
                np.int32,
                copy=False,
            ),
            los_target_ordinals=payload["los_target_ordinals"].astype(
                np.int32,
                copy=False,
            ),
            diagonal_candidate_offsets=payload["diagonal_candidate_offsets"].astype(
                np.int32,
                copy=False,
            ),
            diagonal_target_ordinals=payload["diagonal_target_ordinals"].astype(
                np.int32,
                copy=False,
            ),
        )

    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")

    _validate_offsets(
        "los_candidate_offsets",
        artifacts.los_candidate_offsets,
        expected_candidate_count=artifacts.candidate_count,
        expected_total=len(artifacts.los_target_ordinals),
    )
    _validate_offsets(
        "diagonal_candidate_offsets",
        artifacts.diagonal_candidate_offsets,
        expected_candidate_count=artifacts.candidate_count,
        expected_total=len(artifacts.diagonal_target_ordinals),
    )
    _validate_target_ordinals(
        "los_target_ordinals",
        artifacts.los_target_ordinals,
        open_cell_count=artifacts.open_cell_count,
    )
    _validate_target_ordinals(
        "diagonal_target_ordinals",
        artifacts.diagonal_target_ordinals,
        open_cell_count=artifacts.open_cell_count,
    )

    return artifacts


# Artifact query helpers
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


# Validation helpers
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


def _validate_candidate_ordinal(candidate_ordinal: int, candidate_count: int) -> None:
    """Ensure a public query request references an existing candidate ordinal."""

    if candidate_ordinal < 0 or candidate_ordinal >= candidate_count:
        raise IndexError("candidate_ordinal is out of range for the visibility artifact.")


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


def _choose_sample_indices(length: int, max_samples: int) -> NDArray[np.int64]:
    """Choose evenly spaced flat-pair indices for semantic revalidation."""

    if length == 0 or max_samples <= 0:
        return np.empty(0, dtype=np.int64)
    if length <= max_samples:
        return np.arange(length, dtype=np.int64)
    sample_indices = np.linspace(0, length - 1, num=max_samples, dtype=np.int64)
    return np.unique(sample_indices)


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


# Sparse-array construction helpers
def _build_offsets_from_counts(counts: NDArray[np.int32]) -> NDArray[np.int32]:
    """Convert per-candidate pair counts into CSR-style candidate offsets."""

    if counts.dtype != np.int32:
        raise TypeError("counts must use dtype np.int32.")

    offsets = np.zeros(len(counts) + 1, dtype=np.int32)
    np.cumsum(counts, dtype=np.int32, out=offsets[1:])
    return offsets


@njit(cache=True)
def _count_visibility_pairs(
    grid: NDArray[np.int8],
    candidate_coords_rc: NDArray[np.int32],
    open_coords_rc: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Count visible and diagonal-blocked target pairs for each candidate ordinal."""

    candidate_count = candidate_coords_rc.shape[0]
    open_cell_count = open_coords_rc.shape[0]
    visible_counts = np.zeros(candidate_count, dtype=np.int32)
    diagonal_counts = np.zeros(candidate_count, dtype=np.int32)

    for candidate_ordinal in range(candidate_count):
        start_row = int(candidate_coords_rc[candidate_ordinal, 0])
        start_col = int(candidate_coords_rc[candidate_ordinal, 1])
        visible_count = 0
        diagonal_count = 0

        for target_ordinal in range(open_cell_count):
            target_row = int(open_coords_rc[target_ordinal, 0])
            target_col = int(open_coords_rc[target_ordinal, 1])
            classification = _classify_visibility_pair(
                grid,
                start_row,
                start_col,
                target_row,
                target_col,
            )
            if classification == _VISIBILITY_VISIBLE:
                visible_count += 1
            elif classification == _VISIBILITY_BLOCKED_DIAGONAL:
                diagonal_count += 1

        visible_counts[candidate_ordinal] = visible_count
        diagonal_counts[candidate_ordinal] = diagonal_count

    return visible_counts, diagonal_counts


@njit(cache=True)
def _fill_visibility_pairs(
    grid: NDArray[np.int8],
    candidate_coords_rc: NDArray[np.int32],
    open_coords_rc: NDArray[np.int32],
    los_candidate_offsets: NDArray[np.int32],
    diagonal_candidate_offsets: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Fill sparse LOS and diagonal-block arrays from precomputed offsets."""

    candidate_count = candidate_coords_rc.shape[0]
    open_cell_count = open_coords_rc.shape[0]
    los_target_ordinals = np.empty(int(los_candidate_offsets[-1]), dtype=np.int32)
    diagonal_target_ordinals = np.empty(
        int(diagonal_candidate_offsets[-1]),
        dtype=np.int32,
    )
    los_write_positions = los_candidate_offsets[:-1].copy()
    diagonal_write_positions = diagonal_candidate_offsets[:-1].copy()

    for candidate_ordinal in range(candidate_count):
        start_row = int(candidate_coords_rc[candidate_ordinal, 0])
        start_col = int(candidate_coords_rc[candidate_ordinal, 1])
        los_write = int(los_write_positions[candidate_ordinal])
        diagonal_write = int(diagonal_write_positions[candidate_ordinal])

        for target_ordinal in range(open_cell_count):
            target_row = int(open_coords_rc[target_ordinal, 0])
            target_col = int(open_coords_rc[target_ordinal, 1])
            classification = _classify_visibility_pair(
                grid,
                start_row,
                start_col,
                target_row,
                target_col,
            )
            if classification == _VISIBILITY_VISIBLE:
                los_target_ordinals[los_write] = np.int32(target_ordinal)
                los_write += 1
            elif classification == _VISIBILITY_BLOCKED_DIAGONAL:
                diagonal_target_ordinals[diagonal_write] = np.int32(target_ordinal)
                diagonal_write += 1

    return los_target_ordinals, diagonal_target_ordinals


# LOS kernel helpers
@njit(cache=True)
def _classify_visibility_pair(
    grid: NDArray[np.int8],
    start_row: int,
    start_col: int,
    target_row: int,
    target_col: int,
) -> np.int8:
    """Classify one candidate-target pair under the locked supercover LOS model."""

    if start_row == target_row and start_col == target_col:
        return _VISIBILITY_SKIPPED_SELF

    delta_row = target_row - start_row
    delta_col = target_col - start_col
    step_row = _sign(delta_row)
    step_col = _sign(delta_col)
    row_progress = 0
    col_progress = 0
    row_steps = abs(delta_row)
    col_steps = abs(delta_col)
    current_row = start_row
    current_col = start_col

    while row_progress < row_steps or col_progress < col_steps:
        decision = ((1 + 2 * col_progress) * row_steps) - (
            (1 + 2 * row_progress) * col_steps
        )
        previous_row = current_row
        previous_col = current_col
        moved_diagonally = False

        if decision == 0:
            current_row += step_row
            current_col += step_col
            row_progress += 1
            col_progress += 1
            moved_diagonally = True
        elif decision < 0:
            current_col += step_col
            col_progress += 1
        else:
            current_row += step_row
            row_progress += 1

        if not _cell_is_open(grid, current_row, current_col):
            return _VISIBILITY_BLOCKED_DIRECT
        if moved_diagonally and _is_diagonal_corner_blocked(
            grid,
            previous_row,
            previous_col,
            current_row,
            current_col,
        ):
            return _VISIBILITY_BLOCKED_DIAGONAL

    return _VISIBILITY_VISIBLE


@njit(cache=True)
def _is_diagonal_corner_blocked(
    grid: NDArray[np.int8],
    previous_row: int,
    previous_col: int,
    current_row: int,
    current_col: int,
) -> bool:
    """Apply the locked conservative blocking rule for diagonal supercover steps."""

    delta_row = current_row - previous_row
    delta_col = current_col - previous_col
    if delta_row == 0 or delta_col == 0:
        return False

    side_row_open = _cell_is_open(grid, previous_row + delta_row, previous_col)
    side_col_open = _cell_is_open(grid, previous_row, previous_col + delta_col)
    return not side_row_open and not side_col_open


@njit(cache=True)
def _cell_is_open(grid: NDArray[np.int8], row: int, col: int) -> bool:
    """Return whether one grid cell is in-bounds and open."""

    if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
        return False
    return grid[row, col] == OPEN_CELL


@njit(cache=True)
def _sign(value: int) -> int:
    """Return the integer sign of one delta value."""

    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


__all__ = [
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
    "VisibilityArtifacts",
    "generate_visibility_artifacts",
    "get_diagonal_blocked_target_ordinals",
    "get_visible_target_ordinals",
    "load_visibility_artifacts",
    "resolve_visibility_artifacts",
    "save_visibility_artifacts",
    "validate_visibility_artifacts",
]
