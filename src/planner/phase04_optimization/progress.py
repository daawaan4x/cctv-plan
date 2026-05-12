from __future__ import annotations

from src.planner._shared.progress import ProgressWriter

from .artifacts import OptimizationArtifacts


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
