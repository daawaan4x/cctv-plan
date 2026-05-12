from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
import pulp

from src.common.floorplan import FloorPlanInput
from src.planner._shared.config import PlannerConfig
from src.planner._shared.progress import ProgressWriter
from src.planner.phase01_candidate_generation import CandidateGenerationArtifacts
from src.planner.phase03_scoring import (
    SparseScoreArtifacts,
    get_configuration_dori_scores,
    get_configuration_target_ordinals,
)

from .artifacts import OptimizationArtifacts, OptimizationPrecomputeArtifacts
from .model import (
    _OptimizationModelState,
    _apply_warm_start,
    _build_optimization_model_state,
    _update_budget_constraint,
)
from .precompute import build_optimization_precompute_artifacts
from .progress import _format_solve_complete_message, _write_progress_line
from .solution import _reconstruct_final_scores_from_selection
from .validation import (
    _validate_phase_dependencies,
    _validate_requested_k,
    validate_optimization_artifacts,
    validate_optimization_precompute_artifacts,
)


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
