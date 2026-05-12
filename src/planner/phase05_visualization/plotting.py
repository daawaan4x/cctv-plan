"""Matplotlib plotting helpers for phase-05 visualization artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.common.floorplan import FloorPlanInput

from .artifacts import VisualizationArtifacts

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# Public plotting helpers
def plot_dori_map(
    floorplan: FloorPlanInput,
    artifacts: VisualizationArtifacts,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_camera_overlays: bool = True,
) -> Axes:
    """Render the discrete DORI score map over the tri-state floor-plan base."""

    plt, axes_type, BoundaryNorm, ListedColormap = _import_matplotlib_primitives()
    plot_axis = _resolve_axes(plt, axes_type, ax)

    _plot_base_floorplan(
        floorplan,
        plot_axis,
        title=title or f"{floorplan.name} DORI Map (K={artifacts.solved_k})",
    )
    dori_cmap = ListedColormap(
        ["#d96c4b", "#f4d35e", "#3ba99c", "#4f6ddf", "#2f9e44"]
    )
    dori_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], dori_cmap.N)
    overlay = np.ma.array(
        artifacts.final_score_grid,
        mask=~floorplan.open_mask,
    )
    overlay_image = plot_axis.imshow(
        overlay,
        cmap=dori_cmap,
        norm=dori_norm,
        interpolation="nearest",
        origin="upper",
        alpha=0.9,
    )
    colorbar = plot_axis.figure.colorbar(
        overlay_image,
        ax=plot_axis,
        ticks=[0, 1, 2, 3, 4],
        fraction=0.046,
        pad=0.04,
    )
    colorbar.ax.set_yticklabels(
        [
            "0 Blind spot",
            "1 Detection",
            "2 Observation",
            "3 Recognition",
            "4 Identification",
        ]
    )
    if show_camera_overlays:
        _plot_camera_overlays(plot_axis, artifacts)
    return plot_axis


def plot_blind_spot_map(
    floorplan: FloorPlanInput,
    artifacts: VisualizationArtifacts,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_camera_overlays: bool = True,
) -> Axes:
    """Render covered-open and blind-spot-open cells over the tri-state base grid."""

    plt, axes_type, BoundaryNorm, ListedColormap = _import_matplotlib_primitives()
    plot_axis = _resolve_axes(plt, axes_type, ax)

    _plot_base_floorplan(
        floorplan,
        plot_axis,
        title=title or f"{floorplan.name} Blind Spot Map (K={artifacts.solved_k})",
    )

    open_classification = np.full(floorplan.shape, -1, dtype=np.int8)
    open_classification[floorplan.open_mask] = 0
    open_classification[artifacts.blind_spot_mask] = 1
    classification_cmap = ListedColormap(["#dcebd2", "#d96c4b"])
    classification_norm = BoundaryNorm([-0.5, 0.5, 1.5], classification_cmap.N)
    overlay = np.ma.array(open_classification, mask=~floorplan.open_mask)
    overlay_image = plot_axis.imshow(
        overlay,
        cmap=classification_cmap,
        norm=classification_norm,
        interpolation="nearest",
        origin="upper",
        alpha=0.9,
    )
    colorbar = plot_axis.figure.colorbar(
        overlay_image,
        ax=plot_axis,
        ticks=[0, 1],
        fraction=0.046,
        pad=0.04,
    )
    colorbar.ax.set_yticklabels(["Covered open", "Blind spot"])
    if show_camera_overlays:
        _plot_camera_overlays(plot_axis, artifacts)
    return plot_axis


def plot_selected_configurations(
    floorplan: FloorPlanInput,
    artifacts: VisualizationArtifacts,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Render the selected camera positions and orientations over the floor plan."""

    plt, axes_type, _, _ = _import_matplotlib_primitives()
    plot_axis = _resolve_axes(plt, axes_type, ax)
    _plot_base_floorplan(
        floorplan,
        plot_axis,
        title=title or f"{floorplan.name} Selected Cameras (K={artifacts.solved_k})",
    )
    _plot_camera_overlays(plot_axis, artifacts)
    return plot_axis


def plot_metric_summary_table(
    artifacts: VisualizationArtifacts,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Render the phase-05 scalar metrics as a compact Matplotlib summary table."""

    plt, axes_type, _, _ = _import_matplotlib_primitives()
    plot_axis = _resolve_axes(plt, axes_type, ax)
    plot_axis.axis("off")
    plot_axis.set_title(title or f"Coverage Summary (K={artifacts.solved_k})")

    rows = [
        ["Metric", "Value"],
        ["Total DORI score", f"{artifacts.metrics.total_dori_score:.1f}"],
        ["Detection+ coverage", f"{artifacts.metrics.detection_plus_pct:.2f}%"],
        ["Observation+ coverage", f"{artifacts.metrics.observation_plus_pct:.2f}%"],
        ["Recognition+ coverage", f"{artifacts.metrics.recognition_plus_pct:.2f}%"],
        ["Identification coverage", f"{artifacts.metrics.identification_pct:.2f}%"],
        ["Blind spot coverage", f"{artifacts.metrics.blind_spot_pct:.2f}%"],
        ["Selected camera count", str(artifacts.selected_camera_count)],
    ]
    table = plot_axis.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    return plot_axis


# Shared plotting helpers
def _import_matplotlib_primitives():
    """Import matplotlib lazily so non-plotting code paths stay lightweight."""

    try:
        from matplotlib import pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render phase-05 visualization plots."
        ) from exc
    return plt, Axes, BoundaryNorm, ListedColormap


def _resolve_axes(plt, axes_type, ax: Axes | None):
    """Return one plotting axis, creating a new figure when necessary."""

    if ax is None:
        _, plot_axis = plt.subplots()
        return plot_axis
    if isinstance(ax, axes_type):
        return ax
    raise TypeError("ax must be a matplotlib.axes.Axes instance or None.")


def _plot_base_floorplan(
    floorplan: FloorPlanInput,
    plot_axis: Axes,
    *,
    title: str,
) -> None:
    """Render the null/open/solid semantic base layer shared by all plots."""

    _, _, BoundaryNorm, ListedColormap = _import_matplotlib_primitives()

    # The base layer intentionally preserves the locked tri-state occupancy semantics
    # so uncovered open cells never get visually conflated with null or solid cells.
    base_cmap = ListedColormap(["#bdbdbd", "#ffffff", "#111111"])
    base_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], base_cmap.N)
    plot_axis.imshow(
        floorplan.grid,
        cmap=base_cmap,
        norm=base_norm,
        interpolation="nearest",
        origin="upper",
    )
    plot_axis.set_title(title)
    plot_axis.set_xlabel("Column")
    plot_axis.set_ylabel("Row")


def _plot_camera_overlays(
    plot_axis: Axes,
    artifacts: VisualizationArtifacts,
) -> None:
    """Draw selected candidate centers plus deterministic orientation arrows."""

    if artifacts.selected_camera_count == 0:
        return

    rows = artifacts.selected_candidate_coords_rc[:, 0].astype(np.float64, copy=False)
    cols = artifacts.selected_candidate_coords_rc[:, 1].astype(np.float64, copy=False)

    # Row coordinates increase downward on the rendered grid, so the Y component is
    # negated to preserve the locked angle convention where 90 degrees points north.
    theta_rad = np.deg2rad(artifacts.selected_angles_deg.astype(np.float64, copy=False))
    delta_x = np.cos(theta_rad) * 0.75
    delta_y = -np.sin(theta_rad) * 0.75

    plot_axis.scatter(
        cols,
        rows,
        s=34,
        c="#111111",
        edgecolors="#ffffff",
        linewidths=0.8,
        zorder=4,
    )
    plot_axis.quiver(
        cols,
        rows,
        delta_x,
        delta_y,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#111111",
        width=0.008,
        headwidth=4.6,
        headlength=5.5,
        headaxislength=4.6,
        zorder=5,
    )


__all__ = [
    "plot_blind_spot_map",
    "plot_dori_map",
    "plot_metric_summary_table",
    "plot_selected_configurations",
]

