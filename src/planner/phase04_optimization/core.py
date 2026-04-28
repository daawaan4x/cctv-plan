"""Phase-04 optimization helpers for exact sparse DORI-coverage selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final
import warnings

import numpy as np
from numba import njit
from numpy.typing import NDArray
import pulp

from src.common.floorplan import FloorPlanInput
from src.planner._shared.cache import get_shared_artifact_dir, write_json, write_npz
from src.planner._shared.config import PlannerConfig
from src.planner._shared.progress import ProgressWriter
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)
from src.planner.phase02_visibility import resolve_visibility_artifacts
from src.planner.phase03_scoring import (
    SparseScoreArtifacts,
    get_configuration_dori_scores,
    get_configuration_target_ordinals,
    resolve_sparse_score_artifacts,
)

PHASE_NAME = "optimization"
PHASE_ARTIFACT_STEM = "04_solution"
PHASE_PRECOMPUTE_ARTIFACT_STEM = "04_precompute"

_DORI_LEVELS: Final[tuple[int, ...]] = (1, 2, 3, 4)
_LARGE_THRESHOLD_MEMBERSHIP_WARNING: Final[int] = 100_000_000


@dataclass(frozen=True, slots=True)
class OptimizationArtifacts:
    """Selected configurations plus final best-per-target DORI scores."""

    grid_shape: tuple[int, int]
    open_cell_count: int
    candidate_count: int
    configuration_count: int
    solved_k: int
    solver_name: str
    solver_status: str
    objective_value: float
    selected_configuration_ordinals: NDArray[np.int32]
    selected_candidate_ordinals: NDArray[np.int32]
    selected_angle_ordinals: NDArray[np.int16]
    selected_angles_deg: NDArray[np.float32]
    final_open_cell_scores: NDArray[np.int8]
    best_configuration_ordinals: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class OptimizationPrecomputeArtifacts:
    """Persisted reusable threshold-index artifacts for repeated phase-04 solves."""

    grid_shape: tuple[int, int]
    open_cell_count: int
    candidate_count: int
    configuration_count: int
    candidate_configuration_offsets: NDArray[np.int32]
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


@dataclass(slots=True)
class _OptimizationModelState:
    """One in-memory PuLP model reused across repeated `K` solves."""

    precompute_artifacts: OptimizationPrecomputeArtifacts
    problem: pulp.LpProblem
    x_vars: list[pulp.LpVariable]
    z_vars: list[list[pulp.LpVariable]]
    budget_constraint: pulp.LpConstraint


# Public solver entrypoints
def build_optimization_precompute_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
) -> OptimizationPrecomputeArtifacts:
    """Build the persisted reusable threshold-index artifacts for phase 04."""

    _validate_phase_dependencies(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )
    threshold_cover_index = _build_threshold_cover_index(phase03_artifacts)
    candidate_configuration_offsets = _build_candidate_configuration_offsets(
        phase03_artifacts.configuration_candidate_ordinals,
        candidate_count=len(phase01_artifacts.candidate_cell_indices),
    )
    artifacts = OptimizationPrecomputeArtifacts(
        grid_shape=floorplan.shape,
        open_cell_count=phase03_artifacts.open_cell_count,
        candidate_count=phase03_artifacts.candidate_count,
        configuration_count=len(phase03_artifacts.configuration_candidate_ordinals),
        candidate_configuration_offsets=candidate_configuration_offsets,
        level_offsets=threshold_cover_index.level_offsets,
        level_configuration_ordinals=threshold_cover_index.level_configuration_ordinals,
    )
    validate_optimization_precompute_artifacts(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
        artifacts,
    )
    return artifacts


def ensure_optimization_precompute_artifacts(
    artifact_path: Path,
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
) -> OptimizationPrecomputeArtifacts:
    """Load a persisted precompute artifact when present, otherwise build and save it."""

    if artifact_path.exists():
        artifacts = load_optimization_precompute_artifacts(artifact_path)
        validate_optimization_precompute_artifacts(
            floorplan,
            phase01_artifacts,
            phase03_artifacts,
            artifacts,
        )
        return artifacts

    artifacts = build_optimization_precompute_artifacts(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )
    save_optimization_precompute_artifacts(artifact_path, artifacts)
    return artifacts


def resolve_optimization_precompute_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    force: bool = False,
) -> OptimizationPrecomputeArtifacts:
    """Load, validate, or rebuild the canonical cached phase-04 precompute artifact."""

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    _ = resolve_visibility_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    phase03_artifacts = resolve_sparse_score_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    artifact_path = (
        get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
        / f"{PHASE_PRECOMPUTE_ARTIFACT_STEM}.npz"
    )
    if not force and artifact_path.exists():
        try:
            artifacts = load_optimization_precompute_artifacts(artifact_path)
            validate_optimization_precompute_artifacts(
                floorplan,
                phase01_artifacts,
                phase03_artifacts,
                artifacts,
            )
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    artifacts = build_optimization_precompute_artifacts(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )
    save_optimization_precompute_artifacts(artifact_path, artifacts)
    return artifacts


def resolve_optimization_artifact(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    k: int,
    force: bool = False,
    solver: pulp.LpSolver | None = None,
    progress_writer: ProgressWriter | None = None,
) -> OptimizationArtifacts:
    """Load, validate, or solve one canonical cached per-`K` phase-04 artifact."""

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    _ = resolve_visibility_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    phase03_artifacts = resolve_sparse_score_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    precompute_artifacts = resolve_optimization_precompute_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    artifact_dir = get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
    artifact_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{k}.npz"
    summary_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{k}_summary.json"

    if not force and artifact_path.exists():
        try:
            artifacts = load_optimization_artifacts(artifact_path)
            validate_optimization_artifacts(
                floorplan,
                config,
                phase01_artifacts,
                phase03_artifacts,
                artifacts,
            )
            if not summary_path.exists():
                save_optimization_summary(summary_path, artifacts)
            _write_progress_line(
                progress_writer,
                (
                    f"[phase04] reused cached solution for k={k}: "
                    f"objective={artifacts.objective_value:.1f} "
                    f"selected={len(artifacts.selected_configuration_ordinals)}"
                ),
            )
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    solve_k_sequence, warm_start_artifacts_by_k = _plan_missing_k_solves(
        floorplan,
        config,
        phase01_artifacts,
        phase03_artifacts,
        artifact_dir=artifact_dir,
        missing_target_k_values=(int(k),),
        solved_by_k={},
    )
    solved_artifact_by_k: dict[int, OptimizationArtifacts] = {}

    def persist_solved_artifact(artifacts: OptimizationArtifacts) -> None:
        """Persist each solved `K` immediately so interrupted runs still leave artifacts."""

        solved_artifact_path = (
            artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{artifacts.solved_k}.npz"
        )
        solved_summary_path = (
            artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{artifacts.solved_k}_summary.json"
        )
        save_optimization_artifacts(solved_artifact_path, artifacts)
        save_optimization_summary(solved_summary_path, artifacts)
        solved_artifact_by_k[artifacts.solved_k] = artifacts

    solved_artifacts = solve_for_k_values(
        floorplan,
        config,
        phase01_artifacts,
        phase03_artifacts,
        k_values=solve_k_sequence,
        solver=solver,
        precompute_artifacts=precompute_artifacts,
        warm_start_artifacts_by_k=warm_start_artifacts_by_k,
        on_solved_artifact=persist_solved_artifact,
        progress_writer=progress_writer,
    )
    for solved_artifacts_item in solved_artifacts:
        solved_artifact_by_k.setdefault(
            solved_artifacts_item.solved_k,
            solved_artifacts_item,
        )
    return solved_artifact_by_k[int(k)]


def resolve_optimization_artifacts_for_k_values(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    k_values: Sequence[int] | None = None,
    force: bool = False,
    solver: pulp.LpSolver | None = None,
    progress_writer: ProgressWriter | None = None,
) -> list[OptimizationArtifacts]:
    """Resolve per-`K` artifacts, solving only the missing budgets by default."""

    requested_k_values = tuple(config.k_values if k_values is None else k_values)
    if not requested_k_values:
        return []

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    _ = resolve_visibility_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    phase03_artifacts = resolve_sparse_score_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    precompute_artifacts = resolve_optimization_precompute_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    artifact_dir = get_shared_artifact_dir(floorplan, config, repo_root=repo_root)

    solved_by_k: dict[int, OptimizationArtifacts] = {}
    missing_k_values: list[int] = []
    completed_k_count = 0
    total_k_count = len(requested_k_values)

    for requested_k in requested_k_values:
        artifact_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{int(requested_k)}.npz"
        summary_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{int(requested_k)}_summary.json"
        if force or not artifact_path.exists():
            missing_k_values.append(int(requested_k))
            continue

        try:
            artifacts = load_optimization_artifacts(artifact_path)
            validate_optimization_artifacts(
                floorplan,
                config,
                phase01_artifacts,
                phase03_artifacts,
                artifacts,
            )
            if not summary_path.exists():
                save_optimization_summary(summary_path, artifacts)
            solved_by_k[int(requested_k)] = artifacts
            completed_k_count += 1
            _write_progress_line(
                progress_writer,
                (
                    f"[phase04] {_format_progress_bar(completed_k_count, total_k_count)} "
                    f"{completed_k_count}/{total_k_count} reused cached k={int(requested_k)} "
                    f"objective={artifacts.objective_value:.1f}"
                ),
            )
        except (KeyError, TypeError, ValueError):
            missing_k_values.append(int(requested_k))

    if missing_k_values:
        solve_k_sequence, warm_start_artifacts_by_k = _plan_missing_k_solves(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            artifact_dir=artifact_dir,
            missing_target_k_values=missing_k_values,
            solved_by_k=solved_by_k,
        )
        total_k_count = completed_k_count + len(solve_k_sequence)
        _write_progress_line(
            progress_writer,
            (
                f"[phase04] solving {len(solve_k_sequence)} k-values with model reuse "
                f"to satisfy missing targets {missing_k_values}: {solve_k_sequence}"
            ),
        )

        def persist_solved_artifact(artifacts: OptimizationArtifacts) -> None:
            """Persist each solved batch artifact immediately after its solve completes."""

            artifact_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{artifacts.solved_k}.npz"
            summary_path = (
                artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{artifacts.solved_k}_summary.json"
            )
            save_optimization_artifacts(artifact_path, artifacts)
            save_optimization_summary(summary_path, artifacts)
            solved_by_k[artifacts.solved_k] = artifacts

        newly_solved = solve_for_k_values(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            k_values=solve_k_sequence,
            solver=solver,
            precompute_artifacts=precompute_artifacts,
            warm_start_artifacts_by_k=warm_start_artifacts_by_k,
            on_solved_artifact=persist_solved_artifact,
            progress_writer=progress_writer,
            progress_total=total_k_count,
            progress_completed_before=completed_k_count,
        )
        for artifacts in newly_solved:
            solved_by_k.setdefault(artifacts.solved_k, artifacts)

    return [solved_by_k[int(requested_k)] for requested_k in requested_k_values]


def solve_optimization_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
    *,
    k: int,
    solver: pulp.LpSolver | None = None,
    precompute_artifacts: OptimizationPrecomputeArtifacts | None = None,
    warm_start_from: OptimizationArtifacts | None = None,
    progress_writer: ProgressWriter | None = None,
) -> OptimizationArtifacts:
    """Solve the exact threshold-coverage MILP for one camera budget `K`."""

    _validate_phase_dependencies(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )
    _validate_requested_k(
        k,
        configuration_count=len(phase03_artifacts.configuration_candidate_ordinals),
    )

    reusable_artifacts = (
        precompute_artifacts
        or build_optimization_precompute_artifacts(
            floorplan,
            phase01_artifacts,
            phase03_artifacts,
        )
    )
    validate_optimization_precompute_artifacts(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
        reusable_artifacts,
    )

    model_state = _build_optimization_model_state(reusable_artifacts, k=k)
    return _solve_with_model_state(
        floorplan,
        config,
        phase01_artifacts,
        phase03_artifacts,
        reusable_artifacts,
        model_state,
        solver=solver,
        k=k,
        warm_start_from=warm_start_from,
        progress_writer=progress_writer,
        progress_index=None,
        progress_total=None,
    )


def solve_for_k_values(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
    *,
    k_values: Sequence[int] | None = None,
    solver: pulp.LpSolver | None = None,
    precompute_artifacts: OptimizationPrecomputeArtifacts | None = None,
    warm_start_artifacts_by_k: Mapping[int, OptimizationArtifacts] | None = None,
    on_solved_artifact: Callable[[OptimizationArtifacts], None] | None = None,
    progress_writer: ProgressWriter | None = None,
    progress_total: int | None = None,
    progress_completed_before: int = 0,
) -> list[OptimizationArtifacts]:
    """Solve one optimization artifact per requested camera budget with model reuse."""

    requested_k_values = tuple(config.k_values if k_values is None else k_values)
    if not requested_k_values:
        return []

    _validate_phase_dependencies(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
    )
    for requested_k in requested_k_values:
        _validate_requested_k(
            int(requested_k),
            configuration_count=len(phase03_artifacts.configuration_candidate_ordinals),
        )

    reusable_artifacts = (
        precompute_artifacts
        or build_optimization_precompute_artifacts(
            floorplan,
            phase01_artifacts,
            phase03_artifacts,
        )
    )
    validate_optimization_precompute_artifacts(
        floorplan,
        phase01_artifacts,
        phase03_artifacts,
        reusable_artifacts,
    )

    model_state = _build_optimization_model_state(
        reusable_artifacts,
        k=int(requested_k_values[0]),
    )
    solved_artifacts: list[OptimizationArtifacts] = []
    previous_solved_artifact: OptimizationArtifacts | None = None
    resolved_progress_total = (
        len(requested_k_values) if progress_total is None else progress_total
    )

    for solve_index, requested_k in enumerate(requested_k_values, start=1):
        requested_k_int = int(requested_k)
        warm_start_from: OptimizationArtifacts | None = None
        if (
            previous_solved_artifact is not None
            and previous_solved_artifact.solved_k == requested_k_int - 1
        ):
            warm_start_from = previous_solved_artifact
        elif warm_start_artifacts_by_k is not None:
            warm_start_from = warm_start_artifacts_by_k.get(requested_k_int)

        solved_artifact = _solve_with_model_state(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            reusable_artifacts,
            model_state,
            solver=solver,
            k=requested_k_int,
            warm_start_from=warm_start_from,
            progress_writer=progress_writer,
            progress_index=progress_completed_before + solve_index,
            progress_total=resolved_progress_total,
        )
        solved_artifacts.append(solved_artifact)
        previous_solved_artifact = solved_artifact
        if on_solved_artifact is not None:
            on_solved_artifact(solved_artifact)

    return solved_artifacts


def _plan_missing_k_solves(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
    *,
    artifact_dir: Path,
    missing_target_k_values: Sequence[int],
    solved_by_k: Mapping[int, OptimizationArtifacts],
) -> tuple[list[int], dict[int, OptimizationArtifacts]]:
    """Expand missing targets into one contiguous solve plan plus seed warm starts."""

    actual_artifacts_by_k: dict[int, OptimizationArtifacts] = {
        int(prior_k): artifacts for prior_k, artifacts in solved_by_k.items()
    }
    available_k_values = set(actual_artifacts_by_k)
    solve_k_sequence: list[int] = []
    warm_start_artifacts_by_k: dict[int, OptimizationArtifacts] = {}

    for target_k in sorted(dict.fromkeys(int(k) for k in missing_target_k_values)):
        if target_k in available_k_values:
            continue

        prior_k, prior_artifacts = _find_highest_available_prior_artifact(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            artifact_dir=artifact_dir,
            max_k_exclusive=target_k,
            actual_artifacts_by_k=actual_artifacts_by_k,
            available_k_values=available_k_values,
        )
        start_k = target_k if prior_k is None else prior_k + 1
        if prior_artifacts is not None:
            warm_start_artifacts_by_k.setdefault(start_k, prior_artifacts)

        for solve_k in range(start_k, target_k + 1):
            if solve_k in available_k_values:
                continue
            solve_k_sequence.append(solve_k)
            available_k_values.add(solve_k)

    return solve_k_sequence, warm_start_artifacts_by_k


def _find_highest_available_prior_artifact(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
    *,
    artifact_dir: Path,
    max_k_exclusive: int,
    actual_artifacts_by_k: dict[int, OptimizationArtifacts],
    available_k_values: set[int],
) -> tuple[int | None, OptimizationArtifacts | None]:
    """Return the highest lower available `K` plus artifacts when they already exist."""

    if max_k_exclusive <= 1:
        return None, None

    available_prior_k_values = [
        candidate_k for candidate_k in available_k_values if candidate_k < max_k_exclusive
    ]
    if available_prior_k_values:
        prior_k = max(available_prior_k_values)
        return prior_k, actual_artifacts_by_k.get(prior_k)

    for prior_k in range(max_k_exclusive - 1, 0, -1):
        prior_artifacts = _load_cached_optimization_artifact_if_valid(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            artifact_dir=artifact_dir,
            k=prior_k,
        )
        if prior_artifacts is None:
            continue
        actual_artifacts_by_k[prior_k] = prior_artifacts
        available_k_values.add(prior_k)
        return prior_k, prior_artifacts

    return None, None


def _load_cached_optimization_artifact_if_valid(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
    *,
    artifact_dir: Path,
    k: int,
) -> OptimizationArtifacts | None:
    """Load one cached per-`K` artifact only when it still validates fully."""

    artifact_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{int(k)}.npz"
    if not artifact_path.exists():
        return None

    try:
        artifacts = load_optimization_artifacts(artifact_path)
        validate_optimization_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            artifacts,
        )
        _validate_warm_start_artifacts(phase03_artifacts, artifacts)
    except (KeyError, TypeError, ValueError):
        return None
    return artifacts


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


# Artifact persistence
def save_optimization_precompute_artifacts(
    artifact_path: Path,
    artifacts: OptimizationPrecomputeArtifacts,
) -> Path:
    """Persist reusable threshold-index artifacts for future repeated solves."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        open_cell_count=np.asarray(artifacts.open_cell_count, dtype=np.int32),
        candidate_count=np.asarray(artifacts.candidate_count, dtype=np.int32),
        configuration_count=np.asarray(artifacts.configuration_count, dtype=np.int32),
        candidate_configuration_offsets=artifacts.candidate_configuration_offsets,
        level1_offsets=artifacts.level_offsets[0],
        level1_configuration_ordinals=artifacts.level_configuration_ordinals[0],
        level2_offsets=artifacts.level_offsets[1],
        level2_configuration_ordinals=artifacts.level_configuration_ordinals[1],
        level3_offsets=artifacts.level_offsets[2],
        level3_configuration_ordinals=artifacts.level_configuration_ordinals[2],
        level4_offsets=artifacts.level_offsets[3],
        level4_configuration_ordinals=artifacts.level_configuration_ordinals[3],
    )


def load_optimization_precompute_artifacts(
    artifact_path: Path,
) -> OptimizationPrecomputeArtifacts:
    """Load persisted reusable threshold-index artifacts from disk."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = OptimizationPrecomputeArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            open_cell_count=int(payload["open_cell_count"].item()),
            candidate_count=int(payload["candidate_count"].item()),
            configuration_count=int(payload["configuration_count"].item()),
            candidate_configuration_offsets=payload[
                "candidate_configuration_offsets"
            ].astype(np.int32, copy=False),
            level_offsets=(
                payload["level1_offsets"].astype(np.int32, copy=False),
                payload["level2_offsets"].astype(np.int32, copy=False),
                payload["level3_offsets"].astype(np.int32, copy=False),
                payload["level4_offsets"].astype(np.int32, copy=False),
            ),
            level_configuration_ordinals=(
                payload["level1_configuration_ordinals"].astype(np.int32, copy=False),
                payload["level2_configuration_ordinals"].astype(np.int32, copy=False),
                payload["level3_configuration_ordinals"].astype(np.int32, copy=False),
                payload["level4_configuration_ordinals"].astype(np.int32, copy=False),
            ),
        )

    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")
    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.configuration_count < 0:
        raise ValueError("configuration_count must be non-negative.")

    _validate_offsets(
        artifacts.candidate_configuration_offsets,
        expected_entry_count=artifacts.candidate_count,
        expected_total=artifacts.configuration_count,
    )
    for level_index in range(len(_DORI_LEVELS)):
        _validate_offsets(
            artifacts.level_offsets[level_index],
            expected_entry_count=artifacts.open_cell_count,
            expected_total=len(artifacts.level_configuration_ordinals[level_index]),
        )
    return artifacts


def save_optimization_artifacts(
    artifact_path: Path,
    artifacts: OptimizationArtifacts,
) -> Path:
    """Persist phase-04 artifacts to a deterministic `04_solution_k<K>.npz` schema."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        open_cell_count=np.asarray(artifacts.open_cell_count, dtype=np.int32),
        candidate_count=np.asarray(artifacts.candidate_count, dtype=np.int32),
        configuration_count=np.asarray(artifacts.configuration_count, dtype=np.int32),
        solved_k=np.asarray(artifacts.solved_k, dtype=np.int32),
        solver_name=np.asarray(artifacts.solver_name),
        solver_status=np.asarray(artifacts.solver_status),
        objective_value=np.asarray(artifacts.objective_value, dtype=np.float64),
        selected_configuration_ordinals=artifacts.selected_configuration_ordinals,
        selected_candidate_ordinals=artifacts.selected_candidate_ordinals,
        selected_angle_ordinals=artifacts.selected_angle_ordinals,
        selected_angles_deg=artifacts.selected_angles_deg,
        final_open_cell_scores=artifacts.final_open_cell_scores,
        best_configuration_ordinals=artifacts.best_configuration_ordinals,
    )


def load_optimization_artifacts(
    artifact_path: Path,
) -> OptimizationArtifacts:
    """Load phase-04 artifacts from disk and perform structural validation."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = OptimizationArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            open_cell_count=int(payload["open_cell_count"].item()),
            candidate_count=int(payload["candidate_count"].item()),
            configuration_count=int(payload["configuration_count"].item()),
            solved_k=int(payload["solved_k"].item()),
            solver_name=str(payload["solver_name"].item()),
            solver_status=str(payload["solver_status"].item()),
            objective_value=float(payload["objective_value"].item()),
            selected_configuration_ordinals=payload[
                "selected_configuration_ordinals"
            ].astype(np.int32, copy=False),
            selected_candidate_ordinals=payload["selected_candidate_ordinals"].astype(
                np.int32,
                copy=False,
            ),
            selected_angle_ordinals=payload["selected_angle_ordinals"].astype(
                np.int16,
                copy=False,
            ),
            selected_angles_deg=payload["selected_angles_deg"].astype(
                np.float32,
                copy=False,
            ),
            final_open_cell_scores=payload["final_open_cell_scores"].astype(
                np.int8,
                copy=False,
            ),
            best_configuration_ordinals=payload["best_configuration_ordinals"].astype(
                np.int32,
                copy=False,
            ),
        )

    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")
    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.configuration_count < 0:
        raise ValueError("configuration_count must be non-negative.")
    if artifacts.solved_k <= 0:
        raise ValueError("solved_k must be positive.")

    _validate_selected_configuration_ordinals(
        artifacts.selected_configuration_ordinals,
        configuration_count=artifacts.configuration_count,
        solved_k=artifacts.solved_k,
    )
    _validate_final_score_arrays(artifacts)
    return artifacts


def save_optimization_summary(
    summary_path: Path,
    artifacts: OptimizationArtifacts,
) -> Path:
    """Persist the human-readable per-`K` optimization summary JSON."""

    metrics = _compute_coverage_metrics(artifacts.final_open_cell_scores)
    summary_payload = {
        "phase_name": PHASE_NAME,
        "solved_k": artifacts.solved_k,
        "solver_name": artifacts.solver_name,
        "solver_status": artifacts.solver_status,
        "objective_value": artifacts.objective_value,
        "selected_camera_count": int(len(artifacts.selected_configuration_ordinals)),
        "open_cell_count": artifacts.open_cell_count,
        "total_dori_score": metrics.total_dori_score,
        "coverage_detection_plus_pct": metrics.detection_plus_pct,
        "coverage_observation_plus_pct": metrics.observation_plus_pct,
        "coverage_recognition_plus_pct": metrics.recognition_plus_pct,
        "coverage_identification_pct": metrics.identification_pct,
        "blind_spot_pct": metrics.blind_spot_pct,
    }
    return write_json(summary_path, summary_payload)


# Metric helpers
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


# Threshold-cover inversion helpers
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


# Model-build helpers
def _build_optimization_model_state(
    precompute_artifacts: OptimizationPrecomputeArtifacts,
    *,
    k: int,
) -> _OptimizationModelState:
    """Build one reusable in-memory PuLP model from persisted precompute arrays."""

    problem = pulp.LpProblem(f"phase04_optimization_k{k}", pulp.LpMaximize)
    x_vars = [
        pulp.LpVariable(f"x_{configuration_ordinal}", cat=pulp.LpBinary)
        for configuration_ordinal in range(precompute_artifacts.configuration_count)
    ]
    z_vars: list[list[pulp.LpVariable]] = []
    for target_ordinal in range(precompute_artifacts.open_cell_count):
        z_vars.append(
            [
                pulp.LpVariable(
                    f"z_{target_ordinal}_{level}",
                    lowBound=0.0,
                    upBound=1.0,
                    cat=pulp.LpContinuous,
                )
                for level in _DORI_LEVELS
            ]
        )

    budget_constraint = pulp.lpSum(x_vars) <= k
    problem += budget_constraint, "camera_budget"

    for candidate_ordinal in range(precompute_artifacts.candidate_count):
        start = int(
            precompute_artifacts.candidate_configuration_offsets[candidate_ordinal]
        )
        stop = int(
            precompute_artifacts.candidate_configuration_offsets[candidate_ordinal + 1]
        )
        problem += (
            pulp.lpSum(x_vars[start:stop]) <= 1,
            f"one_orientation_candidate_{candidate_ordinal}",
        )

    for target_ordinal in range(precompute_artifacts.open_cell_count):
        for level_index, level in enumerate(_DORI_LEVELS):
            cover_offsets = precompute_artifacts.level_offsets[level_index]
            cover_configurations = precompute_artifacts.level_configuration_ordinals[
                level_index
            ]
            start = int(cover_offsets[target_ordinal])
            stop = int(cover_offsets[target_ordinal + 1])
            if start == stop:
                problem += (
                    z_vars[target_ordinal][level_index] == 0,
                    f"threshold_{level}_target_{target_ordinal}_disabled",
                )
                continue

            problem += (
                z_vars[target_ordinal][level_index]
                <= pulp.lpSum(
                    x_vars[int(configuration_ordinal)]
                    for configuration_ordinal in cover_configurations[start:stop]
                ),
                f"threshold_{level}_target_{target_ordinal}_enabled",
            )

        problem += (
            z_vars[target_ordinal][3] <= z_vars[target_ordinal][2],
            f"threshold_monotonicity_4_3_target_{target_ordinal}",
        )
        problem += (
            z_vars[target_ordinal][2] <= z_vars[target_ordinal][1],
            f"threshold_monotonicity_3_2_target_{target_ordinal}",
        )
        problem += (
            z_vars[target_ordinal][1] <= z_vars[target_ordinal][0],
            f"threshold_monotonicity_2_1_target_{target_ordinal}",
        )

    problem += pulp.lpSum(
        z_vars[target_ordinal][level_index]
        for target_ordinal in range(precompute_artifacts.open_cell_count)
        for level_index in range(len(_DORI_LEVELS))
    )

    return _OptimizationModelState(
        precompute_artifacts=precompute_artifacts,
        problem=problem,
        x_vars=x_vars,
        z_vars=z_vars,
        budget_constraint=budget_constraint,
    )


def _solve_with_model_state(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase03_artifacts: SparseScoreArtifacts,
    precompute_artifacts: OptimizationPrecomputeArtifacts,
    model_state: _OptimizationModelState,
    *,
    solver: pulp.LpSolver | None,
    k: int,
    warm_start_from: OptimizationArtifacts | None,
    progress_writer: ProgressWriter | None,
    progress_index: int | None,
    progress_total: int | None,
) -> OptimizationArtifacts:
    """Solve one `K` value using a reusable model state and optional warm start."""

    _update_budget_constraint(model_state, k=k)
    if warm_start_from is not None:
        _validate_warm_start_artifacts(
            phase03_artifacts,
            warm_start_from,
        )
    _apply_warm_start(model_state, k=k, warm_start_from=warm_start_from)

    active_solver = _prepare_solver_for_run(
        solver,
        enable_warm_start=warm_start_from is not None,
    )
    solver_name = _get_solver_name(active_solver)
    _write_progress_line(
        progress_writer,
        (
            f"[phase04] k={k} solve starting: solver={solver_name} "
            f"configurations={precompute_artifacts.configuration_count} "
            f"candidates={phase03_artifacts.candidate_count} "
            f"open_cells={phase03_artifacts.open_cell_count} "
            f"warm_start={'yes' if warm_start_from is not None else 'no'}"
        ),
    )

    solve_started_at = perf_counter()
    model_state.problem.solve(active_solver)
    solve_elapsed_s = perf_counter() - solve_started_at
    solver_status = pulp.LpStatus[model_state.problem.status]
    if solver_status != "Optimal":
        raise RuntimeError(
            "Phase 04 optimization did not return an optimal solution; solver status "
            f"was {solver_status!r}."
        )

    selected_configuration_ordinals = np.asarray(
        [
            configuration_ordinal
            for configuration_ordinal, variable in enumerate(model_state.x_vars)
            if _is_selected_solution_value(pulp.value(variable))
        ],
        dtype=np.int32,
    )
    (
        final_open_cell_scores,
        best_configuration_ordinals,
    ) = _reconstruct_final_scores_from_selection(
        phase03_artifacts,
        selected_configuration_ordinals,
    )
    selected_candidate_ordinals = phase03_artifacts.configuration_candidate_ordinals[
        selected_configuration_ordinals
    ].astype(np.int32, copy=False)
    selected_angle_ordinals = phase03_artifacts.configuration_angle_ordinals[
        selected_configuration_ordinals
    ].astype(np.int16, copy=False)
    selected_angles_deg = phase03_artifacts.orientation_angles_deg[
        selected_angle_ordinals.astype(np.int64, copy=False)
    ].astype(np.float32, copy=False)
    objective_value = _coerce_objective_value(pulp.value(model_state.problem.objective))

    artifacts = OptimizationArtifacts(
        grid_shape=floorplan.shape,
        open_cell_count=phase03_artifacts.open_cell_count,
        candidate_count=phase03_artifacts.candidate_count,
        configuration_count=precompute_artifacts.configuration_count,
        solved_k=int(k),
        solver_name=solver_name,
        solver_status=solver_status,
        objective_value=objective_value,
        selected_configuration_ordinals=selected_configuration_ordinals,
        selected_candidate_ordinals=selected_candidate_ordinals,
        selected_angle_ordinals=selected_angle_ordinals,
        selected_angles_deg=selected_angles_deg,
        final_open_cell_scores=final_open_cell_scores,
        best_configuration_ordinals=best_configuration_ordinals,
    )
    validate_optimization_artifacts(
        floorplan,
        config,
        phase01_artifacts,
        phase03_artifacts,
        artifacts,
    )
    _write_progress_line(
        progress_writer,
        _format_solve_complete_message(
            artifacts,
            solve_elapsed_s=solve_elapsed_s,
            progress_index=progress_index,
            progress_total=progress_total,
        ),
    )
    return artifacts


def _write_progress_line(
    progress_writer: ProgressWriter | None,
    message: str,
) -> None:
    """Write one live progress line when an explicit status writer is provided."""

    if progress_writer is None:
        return
    progress_writer.write(message.rstrip() + "\n")
    progress_writer.flush()


def _format_progress_bar(
    completed: int,
    total: int,
    *,
    width: int = 16,
) -> str:
    """Return a fixed-width count-based progress bar for multi-`K` runs."""

    if total <= 0:
        raise ValueError("Progress-bar total must be positive.")
    bounded_completed = min(max(completed, 0), total)
    filled_width = int(round((bounded_completed / total) * width))
    return "[" + ("#" * filled_width) + ("-" * (width - filled_width)) + "]"


def _format_solve_complete_message(
    artifacts: OptimizationArtifacts,
    *,
    solve_elapsed_s: float,
    progress_index: int | None,
    progress_total: int | None,
) -> str:
    """Format one completed-solve status line with optional batch progress."""

    selected_count = len(artifacts.selected_configuration_ordinals)
    if progress_index is not None and progress_total is not None:
        return (
            f"[phase04] {_format_progress_bar(progress_index, progress_total)} "
            f"{progress_index}/{progress_total} solved k={artifacts.solved_k} "
            f"status={artifacts.solver_status} objective={artifacts.objective_value:.1f} "
            f"selected={selected_count} elapsed={solve_elapsed_s:.2f}s"
        )
    return (
        f"[phase04] k={artifacts.solved_k} solve finished: "
        f"status={artifacts.solver_status} objective={artifacts.objective_value:.1f} "
        f"selected={selected_count} elapsed={solve_elapsed_s:.2f}s"
    )


def _update_budget_constraint(
    model_state: _OptimizationModelState,
    *,
    k: int,
) -> None:
    """Change the reusable models budget RHS in place for the next solve."""

    model_state.budget_constraint.changeRHS(float(k))
    model_state.problem.name = f"phase04_optimization_k{k}"


def _apply_warm_start(
    model_state: _OptimizationModelState,
    *,
    k: int,
    warm_start_from: OptimizationArtifacts | None,
) -> None:
    """Fill solver start values from a prior solution or reset them to zero."""

    x_start_values = np.zeros(
        model_state.precompute_artifacts.configuration_count, dtype=np.int8
    )
    z_start_values = np.zeros(
        (model_state.precompute_artifacts.open_cell_count, len(_DORI_LEVELS)),
        dtype=np.int8,
    )
    if warm_start_from is not None:
        x_start_values[warm_start_from.selected_configuration_ordinals] = np.int8(1)
        for level_index, level in enumerate(_DORI_LEVELS):
            z_start_values[:, level_index] = (
                warm_start_from.final_open_cell_scores >= level
            ).astype(np.int8, copy=False)

    for configuration_ordinal, variable in enumerate(model_state.x_vars):
        variable.setInitialValue(int(x_start_values[configuration_ordinal]))

    for target_ordinal, per_level_vars in enumerate(model_state.z_vars):
        for level_index, variable in enumerate(per_level_vars):
            variable.setInitialValue(float(z_start_values[target_ordinal, level_index]))

    _ = k


def _validate_warm_start_artifacts(
    phase03_artifacts: SparseScoreArtifacts,
    artifacts: OptimizationArtifacts,
) -> None:
    """Ensure one warm-start solution matches the current phase-03 dimensions."""

    if artifacts.configuration_count != len(
        phase03_artifacts.configuration_candidate_ordinals
    ):
        raise ValueError(
            "Warm-start optimization artifact does not match the current "
            "configuration count."
        )
    if artifacts.open_cell_count != phase03_artifacts.open_cell_count:
        raise ValueError(
            "Warm-start optimization artifact does not match the current "
            "open-cell count."
        )


# Solution-reconstruction helpers
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


# Structural validation helpers
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


def _build_candidate_configuration_offsets(
    configuration_candidate_ordinals: NDArray[np.int32],
    *,
    candidate_count: int,
) -> NDArray[np.int32]:
    """Build contiguous candidate-to-configuration offsets from candidate-major order."""

    counts = np.bincount(
        configuration_candidate_ordinals.astype(np.int64, copy=False),
        minlength=candidate_count,
    ).astype(np.int32, copy=False)
    offsets = np.zeros(candidate_count + 1, dtype=np.int32)
    np.cumsum(counts, dtype=np.int32, out=offsets[1:])
    return offsets


def _build_offsets_from_counts(counts: NDArray[np.int32]) -> NDArray[np.int32]:
    """Convert per-target membership counts into CSR-style offsets."""

    if counts.dtype != np.int32:
        raise TypeError("counts must use dtype np.int32.")
    offsets = np.zeros(len(counts) + 1, dtype=np.int32)
    np.cumsum(counts, dtype=np.int32, out=offsets[1:])
    return offsets


def _prepare_solver_for_run(
    solver: pulp.LpSolver | None,
    *,
    enable_warm_start: bool,
) -> pulp.LpSolver:
    """Clone or build a solver instance for one solve and toggle warm-start support."""

    active_solver = (
        solver.copy()
        if solver is not None and hasattr(solver, "copy")
        else (solver or _build_default_solver())
    )
    if active_solver is None:
        raise RuntimeError("No solver instance is available for the optimization run.")

    if hasattr(active_solver, "optionsDict"):
        options_dict = getattr(active_solver, "optionsDict")
        if isinstance(options_dict, dict):
            options_dict["warmStart"] = bool(enable_warm_start)
    return active_solver


def _build_default_solver() -> pulp.LpSolver:
    """Build the preferred default solver, falling back cleanly across backends."""

    # Prefer the native HiGHS Python API when the workspace has `highspy`
    # installed because that is the backend this repo now targets by default.
    for solver_factory in (
        lambda: pulp.HiGHS(msg=False),
        lambda: pulp.HiGHS_CMD(msg=False),
        lambda: pulp.PULP_CBC_CMD(msg=False),
        lambda: pulp.COIN_CMD(msg=False),
    ):
        try:
            solver = solver_factory()
        except AttributeError:
            continue

        try:
            available = solver.available() if hasattr(solver, "available") else True
        except Exception:
            continue

        if available:
            return solver

    raise RuntimeError(
        "No supported default PuLP solver interface is available for phase 04."
    )


def _get_solver_name(solver: pulp.LpSolver) -> str:
    """Return a stable human-readable solver name for persisted artifacts."""

    name = getattr(solver, "name", None)
    if isinstance(name, str) and name != "":
        return name
    return type(solver).__name__


def _is_selected_solution_value(raw_value: object) -> bool:
    """Return whether one solved binary variable is effectively selected."""

    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        return float(raw_value) > 0.5
    if raw_value is None:
        return False
    raise TypeError(
        "PuLP returned a non-numeric solved variable value while extracting the "
        "selected configuration set."
    )


def _coerce_objective_value(raw_value: object) -> float:
    """Convert the solved objective into a plain persisted float."""

    if raw_value is None:
        return 0.0
    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        return float(raw_value)
    raise TypeError("PuLP returned a non-numeric solved objective value.")


__all__ = [
    "OptimizationArtifacts",
    "OptimizationPrecomputeArtifacts",
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
    "PHASE_PRECOMPUTE_ARTIFACT_STEM",
    "build_optimization_precompute_artifacts",
    "ensure_optimization_precompute_artifacts",
    "load_optimization_artifacts",
    "load_optimization_precompute_artifacts",
    "resolve_optimization_artifact",
    "resolve_optimization_artifacts_for_k_values",
    "resolve_optimization_precompute_artifacts",
    "save_optimization_artifacts",
    "save_optimization_precompute_artifacts",
    "save_optimization_summary",
    "solve_for_k_values",
    "solve_optimization_artifacts",
    "validate_optimization_artifacts",
    "validate_optimization_precompute_artifacts",
]
