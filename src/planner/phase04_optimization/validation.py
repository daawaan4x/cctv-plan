from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput
from src.planner._shared.config import PlannerConfig
from src.planner.phase01_candidate_generation import CandidateGenerationArtifacts
from src.planner.phase03_scoring import (
    SparseScoreArtifacts,
    get_configuration_dori_scores,
    get_configuration_target_ordinals,
)

from .artifacts import OptimizationArtifacts, OptimizationPrecomputeArtifacts
from .constants import _DORI_LEVELS
from .solution import _reconstruct_final_scores_from_selection


def validate_optimization_precompute_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
    artifacts: OptimizationPrecomputeArtifacts,
) -> None:
    """Validate persisted reusable threshold-index artifacts against upstream phases."""

    _validate_phase_dependencies(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )

    if artifacts.grid_shape != floorplan.shape:
        raise ValueError("Optimization precompute grid_shape does not match floorplan.")
    if artifacts.grid_shape != phase01_artifacts.grid_shape:
        raise ValueError(
            "Optimization precompute grid_shape does not match phase-01 artifacts."
        )
    if artifacts.grid_shape != phase03_artifacts.grid_shape:
        raise ValueError(
            "Optimization precompute grid_shape does not match phase-03 artifacts."
        )
    if artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Optimization precompute open_cell_count does not match phase-01."
        )
    if artifacts.open_cell_count != phase03_artifacts.open_cell_count:
        raise ValueError(
            "Optimization precompute open_cell_count does not match phase-03."
        )
    if artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Optimization precompute candidate_count does not match phase-01."
        )
    if artifacts.candidate_count != phase03_artifacts.candidate_count:
        raise ValueError(
            "Optimization precompute candidate_count does not match phase-03."
        )
    if artifacts.configuration_count != len(
        phase03_artifacts.configuration_candidate_ordinals
    ):
        raise ValueError(
            "Optimization precompute configuration_count does not match phase-03."
        )

    _validate_offsets(
        artifacts.candidate_configuration_offsets,
        expected_entry_count=artifacts.candidate_count,
        expected_total=artifacts.configuration_count,
    )
    if len(artifacts.level_offsets) != len(_DORI_LEVELS):
        raise ValueError(
            "Optimization precompute must contain four level offset arrays."
        )
    if len(artifacts.level_configuration_ordinals) != len(_DORI_LEVELS):
        raise ValueError(
            "Optimization precompute must contain four level configuration arrays."
        )

    for level_index in range(len(_DORI_LEVELS)):
        _validate_offsets(
            artifacts.level_offsets[level_index],
            expected_entry_count=artifacts.open_cell_count,
            expected_total=len(artifacts.level_configuration_ordinals[level_index]),
        )
        configuration_ordinals = artifacts.level_configuration_ordinals[level_index]
        if configuration_ordinals.dtype != np.int32:
            raise TypeError(
                "Optimization precompute level configuration arrays must use "
                "dtype np.int32."
            )
        if configuration_ordinals.ndim != 1:
            raise ValueError(
                "Optimization precompute level configuration arrays must be 1D."
            )
        if configuration_ordinals.size and (
            configuration_ordinals[0] < 0
            or configuration_ordinals[-1] >= artifacts.configuration_count
        ):
            raise ValueError(
                "Optimization precompute contains an out-of-range configuration "
                "ordinal."
            )

def validate_optimization_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
    artifacts: OptimizationArtifacts,
) -> None:
    """Validate structural invariants and exact score reconstruction semantics."""

    _ = config
    _validate_phase_dependencies(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )

    if artifacts.grid_shape != floorplan.shape:
        raise ValueError("Optimization grid_shape does not match floorplan.shape.")
    if artifacts.grid_shape != phase01_artifacts.grid_shape:
        raise ValueError("Optimization grid_shape does not match phase-01 grid_shape.")
    if artifacts.grid_shape != phase03_artifacts.grid_shape:
        raise ValueError("Optimization grid_shape does not match phase-03 grid_shape.")
    if artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Optimization open_cell_count does not match the phase-01 open-cell count."
        )
    if artifacts.open_cell_count != phase03_artifacts.open_cell_count:
        raise ValueError(
            "Optimization open_cell_count does not match the phase-03 open-cell count."
        )
    if artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Optimization candidate_count does not match the phase-01 candidate count."
        )
    if artifacts.candidate_count != phase03_artifacts.candidate_count:
        raise ValueError(
            "Optimization candidate_count does not match the phase-03 candidate count."
        )

    expected_configuration_count = len(
        phase03_artifacts.configuration_candidate_ordinals
    )
    if artifacts.configuration_count != expected_configuration_count:
        raise ValueError(
            "Optimization configuration_count does not match the phase-03 "
            "configuration count."
        )
    if artifacts.solved_k <= 0:
        raise ValueError("Optimization solved_k must be positive.")
    if artifacts.solver_name == "":
        raise ValueError("Optimization solver_name must not be empty.")
    if artifacts.solver_status == "":
        raise ValueError("Optimization solver_status must not be empty.")

    _validate_selected_configuration_ordinals(
        artifacts.selected_configuration_ordinals,
        configuration_count=artifacts.configuration_count,
        solved_k=artifacts.solved_k,
    )

    expected_selected_candidates = phase03_artifacts.configuration_candidate_ordinals[
        artifacts.selected_configuration_ordinals
    ].astype(np.int32, copy=False)
    expected_selected_angle_ordinals = phase03_artifacts.configuration_angle_ordinals[
        artifacts.selected_configuration_ordinals
    ].astype(np.int16, copy=False)
    expected_selected_angles_deg = phase03_artifacts.orientation_angles_deg[
        expected_selected_angle_ordinals.astype(np.int64, copy=False)
    ].astype(np.float32, copy=False)

    if artifacts.selected_candidate_ordinals.dtype != np.int32:
        raise TypeError("selected_candidate_ordinals must use dtype np.int32.")
    if artifacts.selected_candidate_ordinals.ndim != 1:
        raise ValueError("selected_candidate_ordinals must be a 1D array.")
    if len(artifacts.selected_candidate_ordinals) != len(
        artifacts.selected_configuration_ordinals
    ):
        raise ValueError(
            "selected_candidate_ordinals length must match the selected "
            "configuration count."
        )
    if not np.array_equal(
        artifacts.selected_candidate_ordinals,
        expected_selected_candidates,
    ):
        raise ValueError(
            "selected_candidate_ordinals does not match the phase-03 configuration "
            "decode."
        )
    if len(artifacts.selected_candidate_ordinals) > 1 and np.any(
        artifacts.selected_candidate_ordinals[:-1]
        == artifacts.selected_candidate_ordinals[1:]
    ):
        raise ValueError(
            "No candidate ordinal may appear more than once in the selected set."
        )

    if artifacts.selected_angle_ordinals.dtype != np.int16:
        raise TypeError("selected_angle_ordinals must use dtype np.int16.")
    if artifacts.selected_angle_ordinals.ndim != 1:
        raise ValueError("selected_angle_ordinals must be a 1D array.")
    if len(artifacts.selected_angle_ordinals) != len(
        artifacts.selected_configuration_ordinals
    ):
        raise ValueError(
            "selected_angle_ordinals length must match the selected configuration "
            "count."
        )
    if not np.array_equal(
        artifacts.selected_angle_ordinals,
        expected_selected_angle_ordinals,
    ):
        raise ValueError(
            "selected_angle_ordinals does not match the phase-03 configuration decode."
        )

    if artifacts.selected_angles_deg.dtype != np.float32:
        raise TypeError("selected_angles_deg must use dtype np.float32.")
    if artifacts.selected_angles_deg.ndim != 1:
        raise ValueError("selected_angles_deg must be a 1D array.")
    if len(artifacts.selected_angles_deg) != len(
        artifacts.selected_configuration_ordinals
    ):
        raise ValueError(
            "selected_angles_deg length must match the selected configuration count."
        )
    if not np.array_equal(artifacts.selected_angles_deg, expected_selected_angles_deg):
        raise ValueError(
            "selected_angles_deg does not match the phase-03 configuration decode."
        )

    _validate_final_score_arrays(artifacts)

    (
        recomputed_scores,
        recomputed_best_configuration_ordinals,
    ) = _reconstruct_final_scores_from_selection(
        phase03_artifacts,
        artifacts.selected_configuration_ordinals,
    )
    if not np.array_equal(artifacts.final_open_cell_scores, recomputed_scores):
        raise ValueError(
            "final_open_cell_scores does not match a recomputation from the selected "
            "configuration slices."
        )
    if not np.array_equal(
        artifacts.best_configuration_ordinals,
        recomputed_best_configuration_ordinals,
    ):
        raise ValueError(
            "best_configuration_ordinals does not match a recomputation from the "
            "selected configuration slices."
        )

    reconstructed_total_score = float(
        np.sum(artifacts.final_open_cell_scores, dtype=np.int64)
    )
    if not np.isclose(artifacts.objective_value, reconstructed_total_score):
        raise ValueError(
            "objective_value does not match the reconstructed total DORI score."
        )

def _validate_phase_dependencies(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
) -> None:
    """Confirm that phase-01 and phase-03 artifacts still match the current inputs."""

    if phase01_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-01 artifacts do not match the current floorplan shape.")
    if phase03_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-03 artifacts do not match the current floorplan shape.")
    if phase03_artifacts.candidate_count != len(
        phase01_artifacts.candidate_cell_indices
    ):
        raise ValueError(
            "Phase-03 candidate_count does not match the phase-01 candidate count."
        )
    if phase03_artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Phase-03 open_cell_count does not match the phase-01 open-cell count."
        )
    if phase03_artifacts.configuration_candidate_ordinals.ndim != 1:
        raise ValueError("configuration_candidate_ordinals must be a 1D array.")
    if phase03_artifacts.configuration_angle_ordinals.ndim != 1:
        raise ValueError("configuration_angle_ordinals must be a 1D array.")
    if len(phase03_artifacts.configuration_candidate_ordinals) != len(
        phase03_artifacts.configuration_angle_ordinals
    ):
        raise ValueError(
            "Phase-03 configuration candidate and angle arrays must have equal length."
        )
    if (
        phase03_artifacts.score_values.size
        and not np.isin(
            phase03_artifacts.score_values,
            np.asarray(_DORI_LEVELS, dtype=np.int8),
        ).all()
    ):
        raise ValueError("Phase-03 score_values must contain only the scores 1..4.")

def _validate_requested_k(
    k: int,
    *,
    configuration_count: int,
) -> None:
    """Validate one requested camera budget and emit the large-K warning if needed."""

    if k <= 0:
        raise ValueError("Phase 04 optimization requires k to be positive.")
    if k > configuration_count:
        warnings.warn(
            "Requested k exceeds the configuration count; the model remains valid, "
            "but the budget will not bind above the number of selectable "
            "configurations.",
            stacklevel=2,
        )

def _validate_selected_configuration_ordinals(
    selected_configuration_ordinals: NDArray[np.int32],
    *,
    configuration_count: int,
    solved_k: int,
) -> None:
    """Validate dtype, ordering, uniqueness, bounds, and budget for selection ordinals."""

    if selected_configuration_ordinals.dtype != np.int32:
        raise TypeError("selected_configuration_ordinals must use dtype np.int32.")
    if selected_configuration_ordinals.ndim != 1:
        raise ValueError("selected_configuration_ordinals must be a 1D array.")
    if len(selected_configuration_ordinals) > solved_k:
        raise ValueError(
            "The selected configuration count must not exceed the solved camera budget."
        )
    if len(selected_configuration_ordinals) > 1 and not np.all(
        selected_configuration_ordinals[:-1] < selected_configuration_ordinals[1:]
    ):
        raise ValueError(
            "selected_configuration_ordinals must be strictly increasing and unique."
        )
    if selected_configuration_ordinals.size and (
        selected_configuration_ordinals[0] < 0
        or selected_configuration_ordinals[-1] >= configuration_count
    ):
        raise ValueError(
            "selected_configuration_ordinals contains an out-of-range configuration "
            "ordinal."
        )

def _validate_final_score_arrays(artifacts: OptimizationArtifacts) -> None:
    """Validate final per-target score arrays and best-configuration bookkeeping."""

    if artifacts.final_open_cell_scores.dtype != np.int8:
        raise TypeError("final_open_cell_scores must use dtype np.int8.")
    if artifacts.final_open_cell_scores.ndim != 1:
        raise ValueError("final_open_cell_scores must be a 1D array.")
    if len(artifacts.final_open_cell_scores) != artifacts.open_cell_count:
        raise ValueError("final_open_cell_scores length must equal open_cell_count.")
    if (
        artifacts.final_open_cell_scores.size
        and not np.isin(
            artifacts.final_open_cell_scores,
            np.asarray([0, 1, 2, 3, 4], dtype=np.int8),
        ).all()
    ):
        raise ValueError("final_open_cell_scores must contain only the scores 0..4.")

    if artifacts.best_configuration_ordinals.dtype != np.int32:
        raise TypeError("best_configuration_ordinals must use dtype np.int32.")
    if artifacts.best_configuration_ordinals.ndim != 1:
        raise ValueError("best_configuration_ordinals must be a 1D array.")
    if len(artifacts.best_configuration_ordinals) != artifacts.open_cell_count:
        raise ValueError(
            "best_configuration_ordinals length must equal open_cell_count."
        )

    selected_lookup = set(artifacts.selected_configuration_ordinals.tolist())
    covered_mask = artifacts.final_open_cell_scores > 0
    uncovered_mask = ~covered_mask
    if np.any(artifacts.best_configuration_ordinals[uncovered_mask] != -1):
        raise ValueError(
            "Uncovered targets must use -1 in best_configuration_ordinals."
        )
    for best_configuration_ordinal in artifacts.best_configuration_ordinals[
        covered_mask
    ]:
        if int(best_configuration_ordinal) not in selected_lookup:
            raise ValueError(
                "Covered targets must point only to selected configuration ordinals."
            )

def _validate_offsets(
    offsets: NDArray[np.int32],
    *,
    expected_entry_count: int,
    expected_total: int,
) -> None:
    """Validate one CSR-style offset array against the paired flattened arrays."""

    if offsets.dtype != np.int32:
        raise TypeError("offset arrays must use dtype np.int32.")
    if offsets.ndim != 1:
        raise ValueError("offset arrays must be 1D.")
    if len(offsets) != expected_entry_count + 1:
        raise ValueError(
            "Offset array length must equal the number of entries plus one."
        )
    if offsets[0] != 0:
        raise ValueError("Offset arrays must start at 0.")
    if len(offsets) > 1 and np.any(offsets[:-1] > offsets[1:]):
        raise ValueError("Offset arrays must be monotonic nondecreasing.")
    if int(offsets[-1]) != expected_total:
        raise ValueError("Offset array final value does not match the flattened total.")

