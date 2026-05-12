from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pulp

from src.common.floorplan import FloorPlanInput
from src.planner._shared.cache import get_shared_artifact_dir
from src.planner._shared.config import PlannerConfig
from src.planner._shared.progress import ProgressWriter
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)
from src.planner.phase02_visibility import resolve_visibility_artifacts
from src.planner.phase03_scoring import SparseScoreArtifacts, resolve_sparse_score_artifacts

from .artifacts import OptimizationArtifacts
from .constants import PHASE_ARTIFACT_STEM
from .io import (
    load_optimization_artifacts,
    save_optimization_artifacts,
    save_optimization_summary,
)
from .precompute import resolve_optimization_precompute_artifacts
from .progress import _format_progress_bar, _write_progress_line
from .solving import _validate_warm_start_artifacts, solve_for_k_values
from .validation import validate_optimization_artifacts


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
    _, summary_path = _optimization_artifact_paths(artifact_dir, k=k)

    if not force:
        artifacts = _load_cached_optimization_artifact_if_valid(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            artifact_dir=artifact_dir,
            k=k,
        )
        if artifacts is not None:
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

        _persist_optimization_artifact(artifact_dir, artifacts)
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
        artifact_path, summary_path = _optimization_artifact_paths(
            artifact_dir,
            k=int(requested_k),
        )
        if force or not artifact_path.exists():
            missing_k_values.append(int(requested_k))
            continue

        artifacts = _load_cached_optimization_artifact_if_valid(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            artifact_dir=artifact_dir,
            k=int(requested_k),
        )
        if artifacts is None:
            missing_k_values.append(int(requested_k))
            continue

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

            _persist_optimization_artifact(artifact_dir, artifacts)
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

    artifact_path, _ = _optimization_artifact_paths(artifact_dir, k=k)
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

def _optimization_artifact_paths(artifact_dir: Path, *, k: int) -> tuple[Path, Path]:
    """Return the solution artifact and summary paths for one camera budget."""

    solved_k = int(k)
    return (
        artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{solved_k}.npz",
        artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{solved_k}_summary.json",
    )

def _persist_optimization_artifact(
    artifact_dir: Path,
    artifacts: OptimizationArtifacts,
) -> None:
    """Persist the binary artifact and summary JSON for one solved budget."""

    artifact_path, summary_path = _optimization_artifact_paths(
        artifact_dir,
        k=artifacts.solved_k,
    )
    save_optimization_artifacts(artifact_path, artifacts)
    save_optimization_summary(summary_path, artifacts)
