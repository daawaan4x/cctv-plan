"""Read sibling JSON metadata required by traced floor-plan PNG assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from ..floorplan import PathLikeStr


@dataclass(frozen=True, slots=True)
class TracedFloorPlanMetadata:
    """Normalized metadata loaded from a traced floor-plan sibling JSON file."""

    source_path: Path
    metadata_path: Path
    grid_cell_size_m: float | None


def get_traced_floorplan_metadata_path(source_path: PathLikeStr) -> Path:
    """Return the required sibling JSON path for a traced floor-plan PNG."""

    return Path(source_path).expanduser().resolve().with_suffix(".json")


def load_traced_floorplan_metadata(source_path: PathLikeStr) -> TracedFloorPlanMetadata:
    """Load and validate the traced floor-plan metadata JSON payload."""

    # Resolve the PNG path first so both direct file loads and directory scans
    # always derive the sibling JSON location from the same canonical absolute path.
    source_path_resolved = Path(source_path).expanduser().resolve()
    metadata_path = get_traced_floorplan_metadata_path(source_path_resolved)

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Expected traced floor-plan metadata JSON next to the PNG: "
            f"{metadata_path}"
        )

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise ValueError(
            f"Traced floor-plan metadata is not valid JSON: {metadata_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Traced floor-plan metadata must be a JSON object with a "
            f"'grid_cell_size_m' field: {metadata_path}"
        )

    if "grid_cell_size_m" not in payload:
        raise ValueError(
            "Traced floor-plan metadata must define 'grid_cell_size_m': "
            f"{metadata_path}"
        )

    # Reject booleans explicitly even though `bool` is an `int` subtype in Python.
    # The project uses this field as a real geometric scale later, so accepting
    # `true`/`false` would silently turn malformed metadata into `1.0` or `0.0`.
    raw_grid_cell_size_m = payload["grid_cell_size_m"]
    if raw_grid_cell_size_m is None:
        grid_cell_size_m = None
    elif isinstance(raw_grid_cell_size_m, bool) or not isinstance(
        raw_grid_cell_size_m, int | float
    ):
        raise ValueError(
            "Traced floor-plan metadata 'grid_cell_size_m' must be a positive number "
            f"or null: {metadata_path}"
        )
    else:
        grid_cell_size_m = float(raw_grid_cell_size_m)
        if grid_cell_size_m <= 0:
            raise ValueError(
                "Traced floor-plan metadata 'grid_cell_size_m' must be positive when "
                f"provided: {metadata_path}"
            )

    return TracedFloorPlanMetadata(
        source_path=source_path_resolved,
        metadata_path=metadata_path,
        grid_cell_size_m=grid_cell_size_m,
    )


__all__ = [
    "TracedFloorPlanMetadata",
    "get_traced_floorplan_metadata_path",
    "load_traced_floorplan_metadata",
]
