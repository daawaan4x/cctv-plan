"""Load traced floor-plan assets plus metadata into validated planner inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..floorplan import FloorPlanInput, NULL_CELL, OPEN_CELL, PathLikeStr, SOLID_CELL
from .decoder import decode_traced_rgba, read_rgba_image
from .metadata import load_traced_floorplan_metadata
from .validator import validate_traced_palette


def load_traced_floorplan(path: PathLikeStr) -> FloorPlanInput:
    """Load one traced PNG floor plan and decode it into a `FloorPlanInput`."""

    source_path = Path(path).expanduser().resolve()
    metadata = load_traced_floorplan_metadata(source_path)
    rgba_image = read_rgba_image(source_path)
    validate_traced_palette(rgba_image, source_path=source_path)
    grid = decode_traced_rgba(rgba_image)

    height, width = grid.shape
    null_cell_count = int(np.count_nonzero(grid == NULL_CELL))
    open_cell_count = int(np.count_nonzero(grid == OPEN_CELL))
    solid_cell_count = int(np.count_nonzero(grid == SOLID_CELL))

    return FloorPlanInput(
        name=source_path.stem,
        source_path=source_path,
        grid=grid,
        height=height,
        width=width,
        null_cell_count=null_cell_count,
        open_cell_count=open_cell_count,
        solid_cell_count=solid_cell_count,
        grid_cell_size_m=metadata.grid_cell_size_m,
    )


def load_traced_floorplans(directory: PathLikeStr) -> dict[str, FloorPlanInput]:
    """Load every traced PNG floor plan in a directory keyed by stem name."""

    directory_path = Path(directory).expanduser().resolve()
    if not directory_path.exists():
        raise FileNotFoundError(
            f"Traced floor-plan directory does not exist: {directory_path}"
        )
    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory of traced floor plans, got: {directory_path}"
        )

    floorplans: dict[str, FloorPlanInput] = {}
    for image_path in sorted(directory_path.glob("*.png")):
        floorplan = load_traced_floorplan(image_path)
        floorplans[floorplan.name] = floorplan

    return floorplans


__all__ = [
    "load_traced_floorplan",
    "load_traced_floorplans",
]
