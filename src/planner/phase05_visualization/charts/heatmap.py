"""Render combined DORI heatmap charts for every configured K.

Each saved figure contains the DORI maps for all traced floorplans at one K value.
The chart uses the existing phase-05 artifact loader/resolver path, matching the
same data contract used by the phase-05 notebook.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np


DORI_COLORS = ["#d96c4b", "#f4d35e", "#3ba99c", "#4f6ddf", "#2f9e44"]
DORI_TICKS = [0, 1, 2, 3, 4]
DORI_LABELS = [
    "0 Blind spot",
    "1 Detection",
    "2 Observation",
    "3 Recognition",
    "4 Identification",
]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.planner.phase05_visualization.charts.heatmap",
        description="Render one four-floorplan DORI heatmap chart for each K.",
    )
    parser.add_argument(
        "--floorplan",
        action="append",
        dest="floorplan_names",
        metavar="NAME",
        help=(
            "Floorplan to include. Repeat to choose multiple. "
            "Defaults to all traced floorplans."
        ),
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        metavar="K",
        help="K values to render. Defaults to PlannerConfig.k_values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for rendered charts. Defaults to "
            "<repo>/artifacts/planner/charts/dori_heatmaps."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force phase-05 artifact regeneration before rendering.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="DPI for saved PNG files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    from src.planner._shared.bootstrap import find_repo_root, list_traced_floorplan_names
    from src.planner._shared.config import PlannerConfig

    repo_root = find_repo_root()
    base_config = PlannerConfig()
    floorplan_names = _order_floorplans_for_grid(
        _resolve_floorplan_names(
            parser,
            requested_floorplan_names=args.floorplan_names,
            available_floorplan_names=list_traced_floorplan_names(repo_root=repo_root),
        )
    )
    k_values = _resolve_k_values(parser, requested_k_values=args.k_values, config=base_config)
    output_dir = (
        args.output_dir
        or repo_root / "artifacts" / "planner" / "charts" / "dori_heatmaps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for k in k_values:
        output_path = output_dir / f"dori_heatmaps_k{k}.png"
        render_dori_heatmaps_for_k(
            floorplan_names=floorplan_names,
            base_config=base_config,
            repo_root=repo_root,
            k=k,
            output_path=output_path,
            force=bool(args.force),
            dpi=int(args.dpi),
        )
        print(output_path)

    return 0


def render_dori_heatmaps_for_k(
    *,
    floorplan_names: Sequence[str],
    base_config,
    repo_root: Path,
    k: int,
    output_path: Path,
    force: bool = False,
    dpi: int = 180,
) -> Path:
    """Render one combined DORI heatmap figure for all requested floorplans."""

    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from src.common.floorplan_loader import load_traced_floorplan
    from src.planner._shared.bootstrap import get_traced_floorplan_path
    from src.planner.phase05_visualization import resolve_visualization_artifact

    if not floorplan_names:
        raise ValueError("At least one floorplan is required.")

    rows, cols = _subplot_shape(len(floorplan_names))
    figure_width, figure_height = _figure_size(rows, cols)
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.865,
        top=0.9,
        bottom=0.085,
        wspace=0.12,
        hspace=0.34,
    )
    axes_by_index = axes.ravel()

    dori_cmap = ListedColormap(DORI_COLORS)
    dori_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], dori_cmap.N)

    for index, (axis, floorplan_name) in enumerate(zip(axes_by_index, floorplan_names)):
        config = replace(base_config, floorplan_name=floorplan_name, k_values=(k,))
        floorplan_path = get_traced_floorplan_path(floorplan_name, repo_root=repo_root)
        floorplan = load_traced_floorplan(floorplan_path)
        artifacts = resolve_visualization_artifact(
            floorplan,
            config,
            repo_root=repo_root,
            k=k,
            force=force,
        )
        _plot_dori_heatmap(
            floorplan,
            artifacts,
            ax=axis,
            title=f"{floorplan.name} (K={k})",
            dori_cmap=dori_cmap,
            dori_norm=dori_norm,
        )
        _format_axis_labels(axis, row=index // cols, col=index % cols, rows=rows)

    for axis in axes_by_index[len(floorplan_names) :]:
        axis.axis("off")

    colorbar_axis = figure.add_axes((0.885, 0.16, 0.018, 0.66))
    colorbar = figure.colorbar(
        ScalarMappable(norm=dori_norm, cmap=dori_cmap),
        cax=colorbar_axis,
        ticks=DORI_TICKS,
    )
    colorbar.ax.set_yticklabels(DORI_LABELS)

    figure.suptitle(f"DORI heatmaps for K={k}", fontsize=14, fontweight="bold", y=0.975)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_dori_heatmap(
    floorplan,
    artifacts,
    *,
    ax,
    title: str,
    dori_cmap,
    dori_norm,
) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from src.planner.phase05_visualization.plotting import _plot_camera_overlays

    base_cmap = ListedColormap(["#bdbdbd", "#ffffff", "#111111"])
    base_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], base_cmap.N)
    ax.imshow(
        floorplan.grid,
        cmap=base_cmap,
        norm=base_norm,
        interpolation="nearest",
        origin="upper",
    )

    overlay = np.ma.array(artifacts.final_score_grid, mask=~floorplan.open_mask)
    ax.imshow(
        overlay,
        cmap=dori_cmap,
        norm=dori_norm,
        interpolation="nearest",
        origin="upper",
        alpha=0.9,
    )
    _plot_camera_overlays(ax, artifacts)
    ax.set_title(title, fontsize=11, pad=3)
    ax.tick_params(axis="both", labelsize=8, length=2)


def _resolve_floorplan_names(
    parser: argparse.ArgumentParser,
    *,
    requested_floorplan_names: Sequence[str] | None,
    available_floorplan_names: Sequence[str],
) -> tuple[str, ...]:
    if requested_floorplan_names is None:
        if not available_floorplan_names:
            parser.error("No traced floorplans were found.")
        return tuple(available_floorplan_names)

    requested = tuple(dict.fromkeys(requested_floorplan_names))
    unknown = sorted(set(requested) - set(available_floorplan_names))
    if unknown:
        parser.error(
            "Unknown floorplan name(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(available_floorplan_names)
        )
    return requested


def _resolve_k_values(
    parser: argparse.ArgumentParser,
    *,
    requested_k_values: Sequence[int] | None,
    config,
) -> tuple[int, ...]:
    if requested_k_values is None:
        return tuple(config.k_values)

    resolved = tuple(dict.fromkeys(int(k) for k in requested_k_values))
    if not resolved:
        parser.error("Pass at least one K value.")
    if any(k <= 0 for k in resolved):
        parser.error("Every K value must be positive.")
    return resolved


def _order_floorplans_for_grid(floorplan_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(floorplan_names, key=_floorplan_grid_key))


def _floorplan_grid_key(floorplan_name: str) -> tuple[int, int, str]:
    normalized = floorplan_name.lower()
    row = 0 if "back" in normalized else 1 if "front" in normalized else 2
    col = 0 if "ground" in normalized else 1 if "second" in normalized else 2
    return row, col, normalized


def _figure_size(rows: int, cols: int) -> tuple[float, float]:
    if rows == 1 and cols == 1:
        return 6.2, 3.1
    if rows == 1:
        return 11.2, 3.25
    return 11.4, 5.95


def _format_axis_labels(axis, *, row: int, col: int, rows: int) -> None:
    axis.set_xlabel("Column" if row == rows - 1 else "", fontsize=9, labelpad=2)
    axis.set_ylabel("Row" if col == 0 else "", fontsize=9, labelpad=2)


def _subplot_shape(count: int) -> tuple[int, int]:
    if count <= 1:
        return 1, 1
    if count <= 2:
        return 1, 2
    return 2, 2


if __name__ == "__main__":
    raise SystemExit(main())
