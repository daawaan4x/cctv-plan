"""Presentation-oriented helpers for the planner summary notebook."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pulp

from src.common.floorplan import FloorPlanInput
from src.common.floorplan_loader import load_traced_floorplan

from ._shared.bootstrap import (
    find_repo_root,
    get_traced_floorplan_path,
    list_traced_floorplan_names,
)
from ._shared.config import PlannerConfig
from .phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)
from .phase05_visualization import (
    VisualizationArtifacts,
    resolve_visualization_artifacts_for_k_values,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


DEFAULT_PRESENTATION_FLOORPLAN_NAMES = (
    "ground-back",
    "ground-front",
    "second-back",
    "second-front",
)


@dataclass(frozen=True, slots=True)
class PresentationFloorplanResult:
    """Notebook-ready bundle for one floor plan and its resolved result sweep."""

    floorplan: FloorPlanInput
    config: PlannerConfig
    phase01_artifacts: CandidateGenerationArtifacts
    visualization_artifacts_by_k: tuple[VisualizationArtifacts, ...]

    def get_visualization_artifacts(self, k: int) -> VisualizationArtifacts:
        """Return the resolved phase-05 artifacts for one requested `K`."""

        for artifacts in self.visualization_artifacts_by_k:
            if int(artifacts.solved_k) == int(k):
                return artifacts
        raise KeyError(
            f"No visualization artifacts were resolved for floorplan={self.floorplan.name} "
            f"and k={k}."
        )


def resolve_presentation_floorplan_results(
    *,
    floorplan_names: Sequence[str] | None = None,
    base_config: PlannerConfig | None = None,
    repo_root: Path | None = None,
    k_values: Sequence[int] | None = None,
    force: bool = False,
    solver: pulp.LpSolver | None = None,
) -> tuple[PresentationFloorplanResult, ...]:
    """Resolve notebook-ready data for the requested traced floor plans."""

    resolved_repo_root = (repo_root or find_repo_root()).resolve()
    resolved_base_config = base_config or PlannerConfig()
    resolved_floorplan_names = _resolve_floorplan_names(
        floorplan_names,
        repo_root=resolved_repo_root,
    )

    results: list[PresentationFloorplanResult] = []
    for floorplan_name in resolved_floorplan_names:
        floorplan_config = replace(resolved_base_config, floorplan_name=floorplan_name)
        floorplan = load_traced_floorplan(
            get_traced_floorplan_path(
                floorplan_name,
                repo_root=resolved_repo_root,
            )
        )
        phase01_artifacts = resolve_candidate_generation_artifacts(
            floorplan,
            floorplan_config,
            repo_root=resolved_repo_root,
            force=force,
        )
        visualization_artifacts_by_k = resolve_visualization_artifacts_for_k_values(
            floorplan,
            floorplan_config,
            repo_root=resolved_repo_root,
            k_values=k_values,
            force=force,
            solver=solver,
        )
        results.append(
            PresentationFloorplanResult(
                floorplan=floorplan,
                config=floorplan_config,
                phase01_artifacts=phase01_artifacts,
                visualization_artifacts_by_k=visualization_artifacts_by_k,
            )
        )

    return tuple(results)


def build_floorplan_catalog_rows(
    results: Sequence[PresentationFloorplanResult],
) -> list[dict[str, object]]:
    """Build the top-level floor-plan catalog rows shown in the notebook."""

    rows: list[dict[str, object]] = []
    for result in results:
        floorplan = result.floorplan
        rows.append(
            {
                "name": floorplan.name,
                "grid_shape": f"{floorplan.height} x {floorplan.width}",
                "open_cells": floorplan.open_cell_count,
                "solid_cells": floorplan.solid_cell_count,
                "null_cells": floorplan.null_cell_count,
                "grid_cell_size_m": floorplan.grid_cell_size_m,
                "min_k": floorplan.min_k,
            }
        )
    return rows


def build_candidate_summary_rows(
    results: Sequence[PresentationFloorplanResult],
) -> list[dict[str, object]]:
    """Build rows that compare the full eligible and reduced candidate sets."""

    rows: list[dict[str, object]] = []
    for result in results:
        phase01_artifacts = result.phase01_artifacts
        eligible_candidate_count = len(
            phase01_artifacts.eligible_candidate_cell_indices
        )
        candidate_count = len(phase01_artifacts.candidate_cell_indices)
        reduction_pct = 0.0
        if eligible_candidate_count > 0:
            reduction_pct = (
                100.0 * float(eligible_candidate_count - candidate_count)
            ) / eligible_candidate_count
        rows.append(
            {
                "name": result.floorplan.name,
                "eligible_candidates": eligible_candidate_count,
                "optimization_candidates": candidate_count,
                "candidate_reduction_pct": reduction_pct,
            }
        )
    return rows


def build_settings_rows(config: PlannerConfig) -> list[dict[str, object]]:
    """Build the compact implemented-settings table for the notebook."""

    thresholds = config.dori_thresholds
    return [
        {
            "setting": "Camera resolution (px)",
            "value": config.camera_horizontal_resolution_px,
        },
        {"setting": "Horizontal FOV (deg)", "value": config.camera_horizontal_fov_deg},
        {"setting": "Orientation step (deg)", "value": config.orientation_step_deg},
        {"setting": "Orientation angles", "value": list(config.orientation_angles_deg)},
        {
            "setting": "Candidate spacing (cells)",
            "value": config.candidate_spacing_cells,
        },
        {"setting": "Budget sweep (K)", "value": list(config.k_values)},
        {
            "setting": "DORI thresholds (px/m)",
            "value": {
                "detection": thresholds.detection,
                "observation": thresholds.observation,
                "recognition": thresholds.recognition,
                "identification": thresholds.identification,
            },
        },
    ]


def build_budget_sweep_rows(
    results: Sequence[PresentationFloorplanResult],
) -> list[dict[str, object]]:
    """Build per-floorplan rows over the resolved absolute camera budgets."""

    rows: list[dict[str, object]] = []
    for result in results:
        min_k = result.floorplan.min_k
        for artifacts in result.visualization_artifacts_by_k:
            rows.append(_build_metric_row(result, artifacts, min_k=min_k))
    return rows


def compute_common_delta_k_range(
    results: Sequence[PresentationFloorplanResult],
) -> tuple[int, ...]:
    """Return the shared relative-budget offsets available for every floor plan."""

    if not results:
        return ()

    shared_delta_k_values: set[int] | None = None
    for result in results:
        min_k = require_floorplan_min_k(result.floorplan)
        floorplan_delta_k_values = {
            int(artifacts.solved_k) - min_k
            for artifacts in result.visualization_artifacts_by_k
            if int(artifacts.solved_k) >= min_k
        }
        if shared_delta_k_values is None:
            shared_delta_k_values = floorplan_delta_k_values
        else:
            shared_delta_k_values &= floorplan_delta_k_values

    if not shared_delta_k_values:
        return ()
    return tuple(sorted(shared_delta_k_values))


def build_aligned_budget_rows(
    results: Sequence[PresentationFloorplanResult],
) -> list[dict[str, object]]:
    """Build cross-floorplan rows aligned by `delta_k = K - min_k`."""

    shared_delta_k_values = set(compute_common_delta_k_range(results))
    rows: list[dict[str, object]] = []
    for result in results:
        min_k = require_floorplan_min_k(result.floorplan)
        for artifacts in result.visualization_artifacts_by_k:
            solved_k = int(artifacts.solved_k)
            if solved_k < min_k:
                continue
            delta_k = solved_k - min_k
            if delta_k not in shared_delta_k_values:
                continue
            rows.append(_build_metric_row(result, artifacts, min_k=min_k))
    return rows


def build_aligned_baseline_rows(
    results: Sequence[PresentationFloorplanResult],
) -> list[dict[str, object]]:
    """Return one aligned row per floor plan at the `delta_k = 0` baseline."""

    return [
        row
        for row in build_aligned_budget_rows(results)
        if _row_int(row, "delta_k") == 0
    ]


def require_floorplan_min_k(floorplan: FloorPlanInput) -> int:
    """Return the presentation baseline `min_k` or raise a clear notebook error."""

    if floorplan.min_k is None:
        raise ValueError(
            f"Floorplan '{floorplan.name}' is missing metadata.min_k, which is required "
            "for aligned cross-floorplan presentation comparisons."
        )
    return int(floorplan.min_k)


def plot_summary_table(
    rows: Sequence[dict[str, object]],
    *,
    columns: Sequence[str] | None = None,
    column_labels: Sequence[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Render a generic summary table for the notebook."""

    plt, axes_type = _import_matplotlib_primitives()
    plot_axis = _resolve_axes(plt, axes_type, ax)
    plot_axis.axis("off")
    if title is not None:
        plot_axis.set_title(title)

    if not rows:
        plot_axis.text(0.5, 0.5, "No rows to display.", ha="center", va="center")
        return plot_axis

    resolved_columns = list(columns or rows[0].keys())
    resolved_labels = list(column_labels or resolved_columns)
    cell_text = [
        [_format_table_value(row.get(column_name)) for column_name in resolved_columns]
        for row in rows
    ]
    table = plot_axis.table(
        cellText=cell_text,
        colLabels=resolved_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    return plot_axis


def plot_candidate_set_comparison(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    *,
    axes: Sequence[Axes] | None = None,
    eligible_title: str | None = None,
    final_title: str | None = None,
) -> tuple[Axes, Axes]:
    """Render the eligible and reduced candidate sets side by side."""

    plt, axes_type = _import_matplotlib_primitives()
    eligible_axis, final_axis = _resolve_axis_pair(plt, axes_type, axes)

    floorplan.plot(
        ax=eligible_axis,
        title=eligible_title or f"{floorplan.name} eligible candidates",
        show_colorbar=False,
    )
    _plot_candidate_coords(
        eligible_axis,
        phase01_artifacts.eligible_candidate_cell_coords_rc,
        color="#d96c4b",
        size=6.0,
    )

    floorplan.plot(
        ax=final_axis,
        title=final_title or f"{floorplan.name} optimization candidates",
        show_colorbar=False,
    )
    _plot_candidate_coords(
        final_axis,
        phase01_artifacts.candidate_cell_coords_rc,
        color="#2f9e44",
        size=6.0,
    )
    return eligible_axis, final_axis


def plot_metric_by_k(
    results: Sequence[PresentationFloorplanResult],
    metric_key: str,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    ybound: tuple[float | None, float | None] | None = None,
) -> Axes:
    """Plot one metric against the absolute camera budget for each floor plan."""

    plt, axes_type = _import_matplotlib_primitives()
    plot_axis = _resolve_axes(plt, axes_type, ax)

    for result in results:
        rows = sorted(
            build_budget_sweep_rows((result,)),
            key=lambda row: _row_int(row, "k"),
        )
        k_values = [_row_int(row, "k") for row in rows]
        metric_values = [_row_float(row, metric_key) for row in rows]
        plot_axis.plot(
            k_values,
            metric_values,
            marker="o",
            linewidth=1.8,
            label=result.floorplan.name,
        )

    if ybound is not None:
        lower, upper = ybound
        plot_axis.set_ybound(lower, upper)
    plot_axis.set_title(title or metric_key.replace("_", " ").title())
    plot_axis.set_xlabel("Absolute camera budget K")
    plot_axis.set_ylabel(ylabel or metric_key.replace("_", " ").title())
    plot_axis.grid(True, alpha=0.25)
    plot_axis.legend()
    return plot_axis


def select_showcase_k_values(
    result: PresentationFloorplanResult,
    *,
    count: int = 3,
) -> tuple[int, ...]:
    """Pick a small set of representative `K` values from min to max."""

    if count < 2:
        raise ValueError("count must be at least 2.")

    min_k = require_floorplan_min_k(result.floorplan)
    available_k_values = [
        int(artifacts.solved_k)
        for artifacts in result.visualization_artifacts_by_k
        if int(artifacts.solved_k) >= min_k
    ]
    if not available_k_values:
        raise ValueError(
            f"No resolved K values at or above min_k={min_k} for floorplan={result.floorplan.name}."
        )
    if len(available_k_values) <= count:
        return tuple(available_k_values)

    position_values = np.linspace(
        0,
        len(available_k_values) - 1,
        num=count,
        dtype=np.float64,
    )
    showcase_k_values: list[int] = []
    seen_k_values: set[int] = set()
    for position in position_values:
        k_value = available_k_values[int(round(float(position)))]
        if k_value in seen_k_values:
            continue
        showcase_k_values.append(k_value)
        seen_k_values.add(k_value)

    if available_k_values[0] not in seen_k_values:
        showcase_k_values.insert(0, available_k_values[0])
    if available_k_values[-1] not in seen_k_values:
        showcase_k_values.append(available_k_values[-1])
    return tuple(showcase_k_values)


def plot_metric_by_delta_k(
    results: Sequence[PresentationFloorplanResult],
    metric_key: str,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    ybound: tuple[float | None, float | None] | None = None,
) -> Axes:
    """Plot one metric against the aligned relative camera budget `delta_k`."""

    plt, axes_type = _import_matplotlib_primitives()
    plot_axis = _resolve_axes(plt, axes_type, ax)
    aligned_rows = build_aligned_budget_rows(results)

    rows_by_floorplan: dict[str, list[dict[str, object]]] = {}
    for row in aligned_rows:
        rows_by_floorplan.setdefault(str(row["name"]), []).append(row)

    for result in results:
        floorplan_name = result.floorplan.name
        rows = sorted(
            rows_by_floorplan.get(floorplan_name, ()),
            key=lambda row: _row_int(row, "delta_k"),
        )
        if not rows:
            continue
        delta_k_values = [_row_int(row, "delta_k") for row in rows]
        metric_values = [_row_float(row, metric_key) for row in rows]
        plot_axis.plot(
            delta_k_values,
            metric_values,
            marker="o",
            linewidth=1.8,
            label=f"{floorplan_name} (min K={require_floorplan_min_k(result.floorplan)})",
        )

    if ybound is not None:
        lower, upper = ybound
        plot_axis.set_ybound(lower, upper)
    plot_axis.set_title(title or metric_key.replace("_", " ").title())
    plot_axis.set_xlabel("Relative budget delta_k = K - min_k")
    plot_axis.set_ylabel(ylabel or metric_key.replace("_", " ").title())
    plot_axis.grid(True, alpha=0.25)
    plot_axis.legend()
    return plot_axis


def _build_metric_row(
    result: PresentationFloorplanResult,
    artifacts: VisualizationArtifacts,
    *,
    min_k: int | None,
) -> dict[str, object]:
    """Convert one phase-05 artifact into a flat notebook-facing metrics row."""

    metrics = artifacts.metrics
    solved_k = int(artifacts.solved_k)
    return {
        "name": result.floorplan.name,
        "min_k": min_k,
        "k": solved_k,
        "delta_k": None if min_k is None else solved_k - min_k,
        "selected_camera_count": artifacts.selected_camera_count,
        "total_dori_score": metrics.total_dori_score,
        "detection_plus_pct": metrics.detection_plus_pct,
        "observation_plus_pct": metrics.observation_plus_pct,
        "recognition_plus_pct": metrics.recognition_plus_pct,
        "identification_pct": metrics.identification_pct,
        "blind_spot_pct": metrics.blind_spot_pct,
    }


def _resolve_floorplan_names(
    floorplan_names: Sequence[str] | None,
    *,
    repo_root: Path,
) -> tuple[str, ...]:
    """Resolve the requested floorplan names against the traced asset catalog."""

    if floorplan_names is None:
        resolved_floorplan_names = DEFAULT_PRESENTATION_FLOORPLAN_NAMES
    else:
        resolved_floorplan_names = tuple(str(name) for name in floorplan_names)

    available_floorplan_names = set(list_traced_floorplan_names(repo_root=repo_root))
    missing_floorplan_names = [
        floorplan_name
        for floorplan_name in resolved_floorplan_names
        if floorplan_name not in available_floorplan_names
    ]
    if missing_floorplan_names:
        raise ValueError(
            "Presentation floorplans are missing traced PNG+JSON assets: "
            f"{missing_floorplan_names}"
        )
    return resolved_floorplan_names


def _import_matplotlib_primitives():
    """Import matplotlib lazily so data assembly stays usable without plotting."""

    try:
        from matplotlib import pyplot as plt
        from matplotlib.axes import Axes
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render the planner presentation notebook plots."
        ) from exc
    return plt, Axes


def _resolve_axes(plt, axes_type, ax: Axes | None):
    """Return one plotting axis, creating a new figure when needed."""

    if ax is None:
        _, plot_axis = plt.subplots()
        return plot_axis
    if isinstance(ax, axes_type):
        return ax
    raise TypeError("ax must be a matplotlib.axes.Axes instance or None.")


def _resolve_axis_pair(
    plt,
    axes_type,
    axes: Sequence[Axes] | None,
) -> tuple[Axes, Axes]:
    """Return exactly two plotting axes for side-by-side comparisons."""

    if axes is None:
        _, resolved_axes = plt.subplots(1, 2, figsize=(10, 4))
        return resolved_axes[0], resolved_axes[1]
    if len(axes) != 2:
        raise ValueError("axes must contain exactly two matplotlib axes.")
    left_axis, right_axis = axes
    if not isinstance(left_axis, axes_type) or not isinstance(right_axis, axes_type):
        raise TypeError("axes must contain matplotlib.axes.Axes instances.")
    return left_axis, right_axis


def _format_table_value(value: Any) -> str:
    """Format one table value deterministically for notebook display."""

    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, dict):
        return str(value)
    if isinstance(value, list | tuple):
        return str(list(value))
    if value is None:
        return "-"
    return str(value)


def _row_int(row: dict[str, object], key: str) -> int:
    """Read one notebook row field as an integer with a clear type error."""

    value = row[key]
    if not isinstance(value, int):
        raise TypeError(
            f"Expected row[{key!r}] to be an int, got {type(value).__name__}."
        )
    return int(value)


def _row_float(row: dict[str, object], key: str) -> float:
    """Read one notebook row field as a float-compatible metric value."""

    value = row[key]
    if not isinstance(value, int | float):
        raise TypeError(
            f"Expected row[{key!r}] to be numeric, got {type(value).__name__}."
        )
    return float(value)


def _plot_candidate_coords(
    plot_axis: Axes,
    candidate_coords_rc,
    *,
    color: str,
    size: float,
) -> None:
    """Scatter one candidate-coordinate array onto an existing floor-plan axis."""

    if len(candidate_coords_rc) == 0:
        return
    rows = candidate_coords_rc[:, 0].astype(np.float64, copy=False)
    cols = candidate_coords_rc[:, 1].astype(np.float64, copy=False)
    plot_axis.scatter(
        cols,
        rows,
        s=size,
        c=color,
        linewidths=0.35,
        alpha=0.9,
        zorder=4,
    )


__all__ = [
    "DEFAULT_PRESENTATION_FLOORPLAN_NAMES",
    "PresentationFloorplanResult",
    "build_aligned_baseline_rows",
    "build_aligned_budget_rows",
    "build_budget_sweep_rows",
    "build_candidate_summary_rows",
    "build_floorplan_catalog_rows",
    "build_settings_rows",
    "compute_common_delta_k_range",
    "plot_candidate_set_comparison",
    "plot_metric_by_delta_k",
    "plot_metric_by_k",
    "plot_summary_table",
    "require_floorplan_min_k",
    "resolve_presentation_floorplan_results",
    "select_showcase_k_values",
]
