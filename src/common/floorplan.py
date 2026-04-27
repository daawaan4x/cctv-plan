"""Common tri-state floor-plan data structures used by the planner."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Locked tri-state occupancy values used throughout the planner.
NULL_CELL = np.int8(-1)
OPEN_CELL = np.int8(0)
SOLID_CELL = np.int8(1)

PathLikeStr = str | PathLike[str]


class TracedFloorPlanValidationError(ValueError):
    """Raised when a traced floor-plan PNG violates the expected palette."""


@dataclass(frozen=True, slots=True)
class FloorPlanInput:
    """Validated optimization input derived from a traced floor-plan asset."""

    name: str
    source_path: Path
    grid: NDArray[np.int8]
    height: int
    width: int
    null_cell_count: int
    open_cell_count: int
    solid_cell_count: int
    grid_cell_size_m: float | None = None

    def __post_init__(self) -> None:
        """Enforce the locked tri-state grid shape and count invariants."""

        # Downstream planner phases assume a compact 2D `int8` grid with the exact
        # locked cell encoding. Validating that shape here prevents every later phase
        # from repeating the same structural checks.
        if self.grid.dtype != np.int8:
            raise TypeError("FloorPlanInput.grid must use dtype np.int8.")
        if self.grid.ndim != 2:
            raise ValueError("FloorPlanInput.grid must be a 2D array.")
        if self.grid.shape != (self.height, self.width):
            raise ValueError(
                "FloorPlanInput dimensions do not match the provided height and width."
            )

        total_cells = self.height * self.width
        if self.null_cell_count + self.open_cell_count + self.solid_cell_count != total_cells:
            raise ValueError("FloorPlanInput cell counts must sum to the grid size.")
        if self.grid_cell_size_m is not None and self.grid_cell_size_m <= 0:
            raise ValueError("FloorPlanInput.grid_cell_size_m must be positive when set.")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the `(height, width)` shape of the tri-state grid."""

        return self.grid.shape

    @property
    def null_mask(self) -> NDArray[np.bool_]:
        """Return a boolean mask for null or out-of-bounds cells."""

        return self.grid == NULL_CELL

    @property
    def open_mask(self) -> NDArray[np.bool_]:
        """Return a boolean mask for open surveillance target cells."""

        return self.grid == OPEN_CELL

    @property
    def solid_mask(self) -> NDArray[np.bool_]:
        """Return a boolean mask for solid blocking cells."""

        return self.grid == SOLID_CELL

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        show_colorbar: bool = True,
    ) -> Axes:
        """Render the tri-state grid with distinct colors for null, open, and solid."""

        # Import matplotlib lazily so the core data model stays usable in
        # non-plotting contexts such as tests, notebooks, and future CLI stages.
        try:
            from matplotlib import pyplot as plt
            from matplotlib.axes import Axes
            from matplotlib.colors import BoundaryNorm, ListedColormap
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required to display traced floor-plan grids."
            ) from exc

        if ax is None:
            _, plot_axis = plt.subplots()
        elif isinstance(ax, Axes):
            plot_axis = ax
        else:
            raise TypeError("ax must be a matplotlib.axes.Axes instance or None.")

        # The colormap order intentionally matches the fixed cell values
        # `[-1, 0, 1]` so visualization does not silently drift from the planner's
        # semantics when null and blind-spot states are discussed later.
        cmap = ListedColormap(["#bdbdbd", "#ffffff", "#111111"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        image = plot_axis.imshow(
            self.grid,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="upper",
        )
        plot_axis.set_title(title or self.name)
        plot_axis.set_xlabel("Column")
        plot_axis.set_ylabel("Row")

        if show_colorbar:
            colorbar = plot_axis.figure.colorbar(image, ax=plot_axis, ticks=[-1, 0, 1])
            colorbar.ax.set_yticklabels(["Null", "Open", "Solid"])

        return plot_axis


__all__ = [
    "FloorPlanInput",
    "NULL_CELL",
    "OPEN_CELL",
    "PathLikeStr",
    "SOLID_CELL",
    "TracedFloorPlanValidationError",
]
