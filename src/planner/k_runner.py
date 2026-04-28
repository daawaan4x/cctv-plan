"""K-selection planner entrypoints built on the canonical per-phase resolver APIs."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from threading import Lock
import threading
from typing import Any, cast
import weakref

import pulp
from concurrent.futures import thread as futures_thread

from ._shared.config import PlannerConfig
from ._shared.progress import ProgressWriter
from .main import PlannerWorkspace, load_workspace, write_workspace_manifest
from .phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)
from .phase02_visibility import VisibilityArtifacts, resolve_visibility_artifacts
from .phase03_scoring import SparseScoreArtifacts, resolve_sparse_score_artifacts
from .phase04_optimization import (
    OptimizationArtifacts,
    OptimizationPrecomputeArtifacts,
    resolve_optimization_artifacts_for_k_values,
    resolve_optimization_precompute_artifacts,
)


@dataclass(frozen=True, slots=True)
class PlannerBatchResult:
    """Resolved planner artifacts for one floor plan and one requested `K` batch."""

    workspace: PlannerWorkspace
    requested_k_values: tuple[int, ...]
    phase01_artifacts: CandidateGenerationArtifacts
    phase02_artifacts: VisibilityArtifacts
    phase03_artifacts: SparseScoreArtifacts
    phase04_precompute_artifacts: OptimizationPrecomputeArtifacts
    phase04_artifacts_by_k: tuple[OptimizationArtifacts, ...]


def run_planner_batch(
    config: PlannerConfig | None = None,
    *,
    k_values: Sequence[int] | None = None,
    force: bool = False,
    solver: pulp.LpSolver | None = None,
    status_log_path: Path | None = None,
    progress_writer: ProgressWriter | None = None,
) -> PlannerBatchResult:
    """Resolve a complete planner batch for one floor plan through phase 04."""

    workspace = load_workspace(config)
    resolved_k_values = tuple(workspace.config.k_values if k_values is None else k_values)
    write_workspace_manifest(workspace)
    floorplan_progress_writer = _PrefixedWriter(
        progress_writer,
        prefix=f"[floorplan={workspace.floorplan.name}] ",
    )

    _write_runner_status(
        floorplan_progress_writer,
        (
            f"[planner] starting batch for k_values={list(resolved_k_values)} force={force}"
        ),
    )

    phase01_artifacts = resolve_candidate_generation_artifacts(
        workspace.floorplan,
        workspace.config,
        repo_root=workspace.repo_root,
        force=force,
    )
    phase02_artifacts = resolve_visibility_artifacts(
        workspace.floorplan,
        workspace.config,
        repo_root=workspace.repo_root,
        force=force,
    )
    phase03_artifacts = resolve_sparse_score_artifacts(
        workspace.floorplan,
        workspace.config,
        repo_root=workspace.repo_root,
        force=force,
    )
    phase04_precompute_artifacts = resolve_optimization_precompute_artifacts(
        workspace.floorplan,
        workspace.config,
        repo_root=workspace.repo_root,
        force=force,
    )
    phase04_artifacts_by_k = tuple(
        resolve_optimization_artifacts_for_k_values(
            workspace.floorplan,
            workspace.config,
            repo_root=workspace.repo_root,
            k_values=resolved_k_values,
            force=force,
            solver=solver,
            progress_writer=floorplan_progress_writer,
        )
    )

    result = PlannerBatchResult(
        workspace=workspace,
        requested_k_values=resolved_k_values,
        phase01_artifacts=phase01_artifacts,
        phase02_artifacts=phase02_artifacts,
        phase03_artifacts=phase03_artifacts,
        phase04_precompute_artifacts=phase04_precompute_artifacts,
        phase04_artifacts_by_k=phase04_artifacts_by_k,
    )
    if status_log_path is not None:
        append_batch_status_log(result, status_log_path)

    _write_runner_status(
        floorplan_progress_writer,
        (
            "[planner] completed batch: "
            f"resolved_k_values={[artifacts.solved_k for artifacts in result.phase04_artifacts_by_k]}"
        ),
    )
    return result


def run_planner_batches_for_floorplans(
    floorplan_names: Sequence[str],
    base_config: PlannerConfig | None = None,
    *,
    k_values: Sequence[int] | None = None,
    force: bool = False,
    solver: pulp.LpSolver | None = None,
    status_log_path: Path | None = None,
    progress_writer: ProgressWriter | None = None,
    max_workers: int | None = None,
) -> tuple[PlannerBatchResult, ...]:
    """Resolve the planner batch for each requested floor plan, in parallel when useful."""

    resolved_base_config = base_config or PlannerConfig()
    resolved_floorplan_names = tuple(str(floorplan_name) for floorplan_name in floorplan_names)
    total_floorplans = len(resolved_floorplan_names)
    if total_floorplans == 0:
        return ()

    resolved_max_workers = _resolve_max_workers(
        total_floorplans,
        requested_max_workers=max_workers,
    )
    synchronized_writer = _SynchronizedWriter(progress_writer)

    if resolved_max_workers == 1:
        results: list[PlannerBatchResult] = []
        for floorplan_index, floorplan_name in enumerate(resolved_floorplan_names, start=1):
            floorplan_config = replace(
                resolved_base_config,
                floorplan_name=floorplan_name,
            )
            _write_runner_status(
                synchronized_writer,
                (
                    f"[planner] floorplans {floorplan_index}/{total_floorplans}: "
                    f"starting {floorplan_name}"
                ),
            )
            result = run_planner_batch(
                floorplan_config,
                k_values=k_values,
                force=force,
                solver=solver,
                status_log_path=None,
                progress_writer=synchronized_writer,
            )
            if status_log_path is not None:
                append_batch_status_log(result, status_log_path)
            results.append(result)
        return tuple(results)

    _write_runner_status(
        synchronized_writer,
        (
            f"[planner] running {total_floorplans} floorplans in parallel with "
            f"{resolved_max_workers} workers"
        ),
    )
    if solver is not None and not hasattr(solver, "copy"):
        raise ValueError(
            "Parallel floorplan execution requires a solver with a .copy() method "
            "or solver=None."
        )

    results_by_index: list[PlannerBatchResult | None] = [None] * total_floorplans
    completed_floorplans = 0
    executor = _DaemonThreadPoolExecutor(max_workers=resolved_max_workers)
    try:
        future_by_index: dict[Future[PlannerBatchResult], int] = {}
        for floorplan_index, floorplan_name in enumerate(resolved_floorplan_names, start=1):
            _write_runner_status(
                synchronized_writer,
                (
                    f"[planner] floorplans {floorplan_index}/{total_floorplans}: "
                    f"queued {floorplan_name}"
                ),
            )
            future = executor.submit(
                _run_floorplan_batch_task,
                resolved_base_config,
                floorplan_name,
                k_values,
                force,
                solver,
                synchronized_writer,
            )
            future_by_index[future] = floorplan_index - 1

        for future in as_completed(future_by_index):
            floorplan_result = future.result()
            result_index = future_by_index[future]
            results_by_index[result_index] = floorplan_result
            if status_log_path is not None:
                append_batch_status_log(floorplan_result, status_log_path)
            completed_floorplans += 1
            _write_runner_status(
                synchronized_writer,
                (
                    f"[planner] floorplans completed {completed_floorplans}/{total_floorplans}: "
                    f"{floorplan_result.workspace.floorplan.name}"
                ),
            )
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)
    return tuple(_require_completed_result(result) for result in results_by_index)


def append_batch_status_log(
    result: PlannerBatchResult,
    status_log_path: Path,
) -> Path:
    """Append one JSONL status record for a completed planner batch run."""

    payload = {
        "floorplan_name": result.workspace.floorplan.name,
        "artifact_dir": str(result.workspace.artifact_dir),
        "requested_k_values": list(result.requested_k_values),
        "resolved_k_values": [artifacts.solved_k for artifacts in result.phase04_artifacts_by_k],
        "candidate_count": len(result.phase01_artifacts.candidate_cell_indices),
        "eligible_candidate_count": len(
            result.phase01_artifacts.eligible_candidate_cell_indices
        ),
        "open_cell_count": len(result.phase01_artifacts.open_cell_indices),
    }
    status_log_path.parent.mkdir(parents=True, exist_ok=True)
    with status_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return status_log_path


def _write_runner_status(
    progress_writer: ProgressWriter | None,
    message: str,
) -> None:
    """Write one runner-level status line when live progress reporting is enabled."""

    if progress_writer is None:
        return
    progress_writer.write(message.rstrip() + "\n")
    progress_writer.flush()


def _resolve_max_workers(
    total_floorplans: int,
    *,
    requested_max_workers: int | None,
) -> int:
    """Resolve the floorplan worker count from the request and machine defaults."""

    if total_floorplans <= 1:
        return 1
    if requested_max_workers is not None:
        if requested_max_workers <= 0:
            raise ValueError("max_workers must be positive when provided.")
        return min(total_floorplans, requested_max_workers)
    detected_cpu_count = os.cpu_count() or 1
    return max(1, min(total_floorplans, detected_cpu_count))


def _run_floorplan_batch_task(
    base_config: PlannerConfig,
    floorplan_name: str,
    k_values: Sequence[int] | None,
    force: bool,
    solver: pulp.LpSolver | None,
    progress_writer: ProgressWriter | None,
) -> PlannerBatchResult:
    """Resolve one floorplan batch inside a worker thread."""

    worker_solver = solver.copy() if solver is not None and hasattr(solver, "copy") else solver
    return run_planner_batch(
        replace(base_config, floorplan_name=floorplan_name),
        k_values=k_values,
        force=force,
        solver=worker_solver,
        status_log_path=None,
        progress_writer=progress_writer,
    )


def _require_completed_result(result: PlannerBatchResult | None) -> PlannerBatchResult:
    """Convert one optional worker result slot into a concrete planner batch result."""

    if result is None:
        raise RuntimeError("Parallel floorplan execution finished with a missing result.")
    return result


class _SynchronizedWriter:
    """Serialize multi-threaded progress writes onto one underlying text stream."""

    def __init__(self, writer: ProgressWriter | None) -> None:
        self._writer = writer
        self._lock = Lock()

    def write(self, message: str) -> int:
        if self._writer is None:
            return 0
        with self._lock:
            return self._writer.write(message)

    def flush(self) -> None:
        if self._writer is None:
            return
        with self._lock:
            self._writer.flush()


class _PrefixedWriter:
    """Prefix each emitted line so interleaved multi-floorplan logs stay attributable."""

    def __init__(
        self,
        writer: ProgressWriter | None,
        *,
        prefix: str,
    ) -> None:
        self._writer = writer
        self._prefix = prefix
        self._buffer = ""

    def write(self, message: str) -> int:
        if self._writer is None:
            return 0

        emitted_character_count = 0
        self._buffer += message
        while True:
            newline_index = self._buffer.find("\n")
            if newline_index < 0:
                break
            line = self._buffer[: newline_index + 1]
            self._buffer = self._buffer[newline_index + 1 :]
            emitted_character_count += self._writer.write(self._prefix + line)
        return emitted_character_count

    def flush(self) -> None:
        if self._writer is None:
            return
        if self._buffer != "":
            self._writer.write(self._prefix + self._buffer)
            self._buffer = ""
        self._writer.flush()


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """Thread pool whose workers are daemon threads so they do not outlive the parent."""

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        work_queue = self._work_queue
        threads = self._threads
        initializer = getattr(self, "_initializer", None)
        initargs = getattr(self, "_initargs", ())

        def weakref_cb(_: object) -> None:
            cast(Any, work_queue).put(None)

        num_threads = len(threads)
        if num_threads >= self._max_workers:
            return

        thread_name = f"{self._thread_name_prefix}_{num_threads}"
        worker_thread = threading.Thread(
            name=thread_name,
            target=cast(Any, futures_thread._worker),
            args=(
                weakref.ref(self, weakref_cb),
                work_queue,
                initializer,
                initargs,
            ),
            daemon=True,
        )
        worker_thread.start()
        cast(Any, threads).add(worker_thread)
        cast(Any, futures_thread._threads_queues)[worker_thread] = work_queue


__all__ = [
    "PlannerBatchResult",
    "append_batch_status_log",
    "run_planner_batch",
    "run_planner_batches_for_floorplans",
]
