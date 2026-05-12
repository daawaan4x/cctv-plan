from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from .artifacts import OptimizationArtifacts, OptimizationPrecomputeArtifacts
from .constants import _DORI_LEVELS


@dataclass(slots=True)
class _OptimizationModelState:
    """One in-memory PuLP model reused across repeated `K` solves."""

    precompute_artifacts: OptimizationPrecomputeArtifacts
    problem: pulp.LpProblem
    x_vars: list[pulp.LpVariable]
    z_vars: list[list[pulp.LpVariable]]
    budget_constraint: pulp.LpConstraint

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
