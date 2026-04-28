"""Phase-01 candidate-generation helpers for the CCTV planner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput, OPEN_CELL, SOLID_CELL
from src.planner._shared.cache import get_shared_artifact_dir, write_npz
from src.planner._shared.config import PlannerConfig

PHASE_NAME = "candidate_generation"
PHASE_ARTIFACT_STEM = "01_candidates"

_BOUNDARY_FLAG_NORTH = np.uint8(1)
_BOUNDARY_FLAG_EAST = np.uint8(2)
_BOUNDARY_FLAG_SOUTH = np.uint8(4)
_BOUNDARY_FLAG_WEST = np.uint8(8)

_EXCEPTION_FLAG_ENDPOINT = np.uint8(1)
_EXCEPTION_FLAG_MIDPOINT = np.uint8(2)
_EXCEPTION_FLAG_CORNER = np.uint8(4)
_EXCEPTION_FLAG_JUNCTION = np.uint8(8)


@dataclass(frozen=True, slots=True)
class CandidateGenerationArtifacts:
    """Deterministic phase-01 outputs derived from a tri-state floor-plan grid."""

    grid_shape: tuple[int, int]
    open_cell_indices: NDArray[np.int32]
    open_cell_coords_rc: NDArray[np.int32]
    eligible_candidate_cell_indices: NDArray[np.int32]
    eligible_candidate_cell_coords_rc: NDArray[np.int32]
    eligible_candidate_boundary_flags: NDArray[np.uint8]
    candidate_cell_indices: NDArray[np.int32]
    candidate_cell_coords_rc: NDArray[np.int32]
    candidate_boundary_flags: NDArray[np.uint8]
    candidate_exception_flags: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _DirectionalMasks:
    """Boolean directional masks reused across solid-adjacency wall-run logic."""

    open_mask: NDArray[np.bool_]
    north_solid: NDArray[np.bool_]
    east_solid: NDArray[np.bool_]
    south_solid: NDArray[np.bool_]
    west_solid: NDArray[np.bool_]


# Artifact generation
def generate_candidate_generation_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig | None = None,
) -> CandidateGenerationArtifacts:
    """Build eligible and thinned candidate-camera arrays for phase 01."""

    resolved_config = config or PlannerConfig(floorplan_name=floorplan.name)
    grid = floorplan.grid
    _, width = floorplan.shape

    open_cell_indices = _build_open_cell_indices(grid)
    directional_masks = _build_directional_masks(grid)
    (
        eligible_candidate_cell_indices,
        eligible_candidate_boundary_flags,
        eligible_candidate_solid_flags,
    ) = _build_eligible_candidate_arrays(directional_masks)
    (
        candidate_cell_indices,
        candidate_boundary_flags,
        candidate_exception_flags,
    ) = _thin_candidate_set(
        floorplan,
        resolved_config,
        eligible_candidate_cell_indices,
        eligible_candidate_boundary_flags,
        eligible_candidate_solid_flags,
        directional_masks,
    )

    artifacts = CandidateGenerationArtifacts(
        grid_shape=floorplan.shape,
        open_cell_indices=open_cell_indices,
        open_cell_coords_rc=_flat_indices_to_coords(open_cell_indices, width),
        eligible_candidate_cell_indices=eligible_candidate_cell_indices,
        eligible_candidate_cell_coords_rc=_flat_indices_to_coords(
            eligible_candidate_cell_indices,
            width,
        ),
        eligible_candidate_boundary_flags=eligible_candidate_boundary_flags,
        candidate_cell_indices=candidate_cell_indices,
        candidate_cell_coords_rc=_flat_indices_to_coords(candidate_cell_indices, width),
        candidate_boundary_flags=candidate_boundary_flags,
        candidate_exception_flags=candidate_exception_flags,
    )
    validate_candidate_generation_artifacts(
        floorplan,
        artifacts,
        config=resolved_config,
    )
    return artifacts


def resolve_candidate_generation_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    force: bool = False,
) -> CandidateGenerationArtifacts:
    """Load, validate, or rebuild the canonical cached phase-01 artifact."""

    artifact_path = (
        get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
        / f"{PHASE_ARTIFACT_STEM}.npz"
    )
    if not force and artifact_path.exists():
        try:
            artifacts = load_candidate_generation_artifacts(artifact_path)
            validate_candidate_generation_artifacts(
                floorplan,
                artifacts,
                config=config,
            )
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    artifacts = generate_candidate_generation_artifacts(floorplan, config)
    save_candidate_generation_artifacts(artifact_path, artifacts)
    return artifacts


def validate_candidate_generation_artifacts(
    floorplan: FloorPlanInput,
    artifacts: CandidateGenerationArtifacts,
    *,
    config: PlannerConfig | None = None,
) -> None:
    """Validate structural and semantic consistency for phase-01 artifacts."""

    resolved_config = config or PlannerConfig(floorplan_name=floorplan.name)
    height, width = floorplan.shape
    grid_size = height * width

    if artifacts.grid_shape != floorplan.shape:
        raise ValueError(
            "Candidate-generation grid_shape does not match floorplan.shape."
        )

    _validate_flat_index_array(
        "open_cell_indices",
        artifacts.open_cell_indices,
        size=grid_size,
    )
    _validate_flat_index_array(
        "eligible_candidate_cell_indices",
        artifacts.eligible_candidate_cell_indices,
        size=grid_size,
    )
    _validate_flat_index_array(
        "candidate_cell_indices",
        artifacts.candidate_cell_indices,
        size=grid_size,
    )
    _validate_coordinate_array(
        "open_cell_coords_rc",
        artifacts.open_cell_coords_rc,
        expected_length=len(artifacts.open_cell_indices),
    )
    _validate_coordinate_array(
        "eligible_candidate_cell_coords_rc",
        artifacts.eligible_candidate_cell_coords_rc,
        expected_length=len(artifacts.eligible_candidate_cell_indices),
    )
    _validate_coordinate_array(
        "candidate_cell_coords_rc",
        artifacts.candidate_cell_coords_rc,
        expected_length=len(artifacts.candidate_cell_indices),
    )
    _validate_uint8_array(
        "eligible_candidate_boundary_flags",
        artifacts.eligible_candidate_boundary_flags,
        expected_length=len(artifacts.eligible_candidate_cell_indices),
    )
    _validate_uint8_array(
        "candidate_boundary_flags",
        artifacts.candidate_boundary_flags,
        expected_length=len(artifacts.candidate_cell_indices),
    )
    _validate_uint8_array(
        "candidate_exception_flags",
        artifacts.candidate_exception_flags,
        expected_length=len(artifacts.candidate_cell_indices),
    )

    if len(artifacts.open_cell_indices) != floorplan.open_cell_count:
        raise ValueError(
            "open_cell_indices length does not match floorplan.open_cell_count."
        )

    expected_open_coords = _flat_indices_to_coords(artifacts.open_cell_indices, width)
    expected_eligible_coords = _flat_indices_to_coords(
        artifacts.eligible_candidate_cell_indices,
        width,
    )
    expected_candidate_coords = _flat_indices_to_coords(
        artifacts.candidate_cell_indices,
        width,
    )
    if not np.array_equal(artifacts.open_cell_coords_rc, expected_open_coords):
        raise ValueError("open_cell_coords_rc does not match open_cell_indices.")
    if not np.array_equal(
        artifacts.eligible_candidate_cell_coords_rc,
        expected_eligible_coords,
    ):
        raise ValueError(
            "eligible_candidate_cell_coords_rc does not match "
            "eligible_candidate_cell_indices."
        )
    if not np.array_equal(
        artifacts.candidate_cell_coords_rc, expected_candidate_coords
    ):
        raise ValueError(
            "candidate_cell_coords_rc does not match candidate_cell_indices."
        )

    if not np.isin(
        artifacts.eligible_candidate_cell_indices,
        artifacts.open_cell_indices,
        assume_unique=True,
    ).all():
        raise ValueError(
            "eligible_candidate_cell_indices must be a subset of open_cell_indices."
        )
    if not np.isin(
        artifacts.candidate_cell_indices,
        artifacts.eligible_candidate_cell_indices,
        assume_unique=True,
    ).all():
        raise ValueError(
            "candidate_cell_indices must be a subset of eligible_candidate_cell_indices."
        )
    if artifacts.eligible_candidate_boundary_flags.size and np.any(
        artifacts.eligible_candidate_boundary_flags == 0
    ):
        raise ValueError(
            "eligible_candidate_boundary_flags must be non-zero for every eligible "
            "candidate cell."
        )
    if artifacts.candidate_boundary_flags.size and np.any(
        artifacts.candidate_boundary_flags == 0
    ):
        raise ValueError(
            "candidate_boundary_flags must be non-zero for every candidate cell."
        )

    directional_masks = _build_directional_masks(floorplan.grid)
    (
        expected_eligible_indices,
        expected_eligible_boundary_flags,
        expected_eligible_solid_flags,
    ) = _build_eligible_candidate_arrays(directional_masks)
    if not np.array_equal(
        artifacts.eligible_candidate_cell_indices,
        expected_eligible_indices,
    ):
        raise ValueError(
            "eligible_candidate_cell_indices does not match the locked solid-adjacency rule."
        )
    if not np.array_equal(
        artifacts.eligible_candidate_boundary_flags,
        expected_eligible_boundary_flags,
    ):
        raise ValueError(
            "eligible_candidate_boundary_flags does not match a re-derived solid "
            "boundary bitmask."
        )

    _validate_open_cells_match_grid(floorplan.grid, artifacts.open_cell_coords_rc)
    _validate_candidate_cells_match_grid(
        floorplan.grid,
        artifacts.eligible_candidate_cell_coords_rc,
        artifacts.eligible_candidate_boundary_flags,
    )
    _validate_candidate_cells_match_grid(
        floorplan.grid,
        artifacts.candidate_cell_coords_rc,
        artifacts.candidate_boundary_flags,
    )
    _validate_candidate_spacing_rules(
        floorplan,
        resolved_config,
        artifacts,
        directional_masks,
        expected_eligible_solid_flags,
    )


# Artifact persistence
def save_candidate_generation_artifacts(
    artifact_path: Path,
    artifacts: CandidateGenerationArtifacts,
) -> Path:
    """Persist phase-01 artifacts to the deterministic `01_candidates.npz` schema."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        open_cell_indices=artifacts.open_cell_indices,
        open_cell_coords_rc=artifacts.open_cell_coords_rc,
        eligible_candidate_cell_indices=artifacts.eligible_candidate_cell_indices,
        eligible_candidate_cell_coords_rc=artifacts.eligible_candidate_cell_coords_rc,
        eligible_candidate_boundary_flags=artifacts.eligible_candidate_boundary_flags,
        candidate_cell_indices=artifacts.candidate_cell_indices,
        candidate_cell_coords_rc=artifacts.candidate_cell_coords_rc,
        candidate_boundary_flags=artifacts.candidate_boundary_flags,
        candidate_exception_flags=artifacts.candidate_exception_flags,
    )


def load_candidate_generation_artifacts(
    artifact_path: Path,
) -> CandidateGenerationArtifacts:
    """Load phase-01 artifacts from disk and perform structural validation."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = CandidateGenerationArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            open_cell_indices=payload["open_cell_indices"].astype(np.int32, copy=False),
            open_cell_coords_rc=payload["open_cell_coords_rc"].astype(
                np.int32,
                copy=False,
            ),
            eligible_candidate_cell_indices=payload[
                "eligible_candidate_cell_indices"
            ].astype(np.int32, copy=False),
            eligible_candidate_cell_coords_rc=payload[
                "eligible_candidate_cell_coords_rc"
            ].astype(np.int32, copy=False),
            eligible_candidate_boundary_flags=payload[
                "eligible_candidate_boundary_flags"
            ].astype(np.uint8, copy=False),
            candidate_cell_indices=payload["candidate_cell_indices"].astype(
                np.int32,
                copy=False,
            ),
            candidate_cell_coords_rc=payload["candidate_cell_coords_rc"].astype(
                np.int32,
                copy=False,
            ),
            candidate_boundary_flags=payload["candidate_boundary_flags"].astype(
                np.uint8,
                copy=False,
            ),
            candidate_exception_flags=payload["candidate_exception_flags"].astype(
                np.uint8,
                copy=False,
            ),
        )

    height, width = artifacts.grid_shape
    grid_size = height * width
    _validate_flat_index_array(
        "open_cell_indices",
        artifacts.open_cell_indices,
        size=grid_size,
    )
    _validate_flat_index_array(
        "eligible_candidate_cell_indices",
        artifacts.eligible_candidate_cell_indices,
        size=grid_size,
    )
    _validate_flat_index_array(
        "candidate_cell_indices",
        artifacts.candidate_cell_indices,
        size=grid_size,
    )
    _validate_coordinate_array(
        "open_cell_coords_rc",
        artifacts.open_cell_coords_rc,
        expected_length=len(artifacts.open_cell_indices),
    )
    _validate_coordinate_array(
        "eligible_candidate_cell_coords_rc",
        artifacts.eligible_candidate_cell_coords_rc,
        expected_length=len(artifacts.eligible_candidate_cell_indices),
    )
    _validate_coordinate_array(
        "candidate_cell_coords_rc",
        artifacts.candidate_cell_coords_rc,
        expected_length=len(artifacts.candidate_cell_indices),
    )
    _validate_uint8_array(
        "eligible_candidate_boundary_flags",
        artifacts.eligible_candidate_boundary_flags,
        expected_length=len(artifacts.eligible_candidate_cell_indices),
    )
    _validate_uint8_array(
        "candidate_boundary_flags",
        artifacts.candidate_boundary_flags,
        expected_length=len(artifacts.candidate_cell_indices),
    )
    _validate_uint8_array(
        "candidate_exception_flags",
        artifacts.candidate_exception_flags,
        expected_length=len(artifacts.candidate_cell_indices),
    )
    return artifacts


# Grid traversal helpers
def _build_open_cell_indices(grid: NDArray[np.int8]) -> NDArray[np.int32]:
    """Return all open-cell flat indices in deterministic row-major order."""

    return np.flatnonzero(grid == OPEN_CELL).astype(np.int32, copy=False)


def _build_directional_masks(grid: NDArray[np.int8]) -> _DirectionalMasks:
    """Build reusable directional neighbor masks for solid adjacency."""

    open_mask = grid == OPEN_CELL
    solid_mask = grid == SOLID_CELL
    padded_solid = np.pad(
        solid_mask,
        pad_width=1,
        mode="constant",
        constant_values=False,
    )

    return _DirectionalMasks(
        open_mask=open_mask,
        north_solid=padded_solid[0:-2, 1:-1],
        east_solid=padded_solid[1:-1, 2:],
        south_solid=padded_solid[2:, 1:-1],
        west_solid=padded_solid[1:-1, 0:-2],
    )


def _build_eligible_candidate_arrays(
    directional_masks: _DirectionalMasks,
) -> tuple[NDArray[np.int32], NDArray[np.uint8], NDArray[np.uint8]]:
    """Return eligible solid-adjacent candidate flat indices plus direction bitmasks."""

    eligible_mask = directional_masks.open_mask & (
        directional_masks.north_solid
        | directional_masks.east_solid
        | directional_masks.south_solid
        | directional_masks.west_solid
    )
    solid_flags_full = (
        (directional_masks.north_solid.astype(np.uint8) * _BOUNDARY_FLAG_NORTH)
        | (directional_masks.east_solid.astype(np.uint8) * _BOUNDARY_FLAG_EAST)
        | (directional_masks.south_solid.astype(np.uint8) * _BOUNDARY_FLAG_SOUTH)
        | (directional_masks.west_solid.astype(np.uint8) * _BOUNDARY_FLAG_WEST)
    )

    eligible_candidate_cell_indices = np.flatnonzero(eligible_mask).astype(
        np.int32,
        copy=False,
    )
    eligible_candidate_boundary_flags = solid_flags_full[eligible_mask].astype(
        np.uint8,
        copy=False,
    )
    eligible_candidate_solid_flags = solid_flags_full[eligible_mask].astype(
        np.uint8,
        copy=False,
    )
    return (
        eligible_candidate_cell_indices,
        eligible_candidate_boundary_flags,
        eligible_candidate_solid_flags,
    )


def _thin_candidate_set(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    eligible_candidate_cell_indices: NDArray[np.int32],
    eligible_candidate_boundary_flags: NDArray[np.uint8],
    eligible_candidate_solid_flags: NDArray[np.uint8],
    directional_masks: _DirectionalMasks,
) -> tuple[NDArray[np.int32], NDArray[np.uint8], NDArray[np.uint8]]:
    """Apply deterministic exception-first wall-run spacing to eligible candidates."""

    if len(eligible_candidate_cell_indices) == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint8),
        )

    eligible_boundary_lookup = {
        int(flat_index): int(boundary_flag)
        for flat_index, boundary_flag in zip(
            eligible_candidate_cell_indices,
            eligible_candidate_boundary_flags,
            strict=True,
        )
    }
    eligible_solid_lookup = {
        int(flat_index): int(solid_flag)
        for flat_index, solid_flag in zip(
            eligible_candidate_cell_indices,
            eligible_candidate_solid_flags,
            strict=True,
        )
    }
    exception_flags_by_flat: dict[int, int] = {}
    selected_candidate_flats: set[int] = set()

    wall_runs = _build_wall_runs(floorplan.shape[1], directional_masks)

    for run in wall_runs:
        run_anchor_positions = _build_run_anchor_positions(
            run,
            eligible_solid_lookup,
            exception_flags_by_flat,
        )
        run_selected_positions = set(run_anchor_positions)

        if run_anchor_positions:
            if run_anchor_positions[0] > 0:
                run_selected_positions.update(
                    _build_segment_candidate_positions(
                        0,
                        run_anchor_positions[0] - 1,
                        config.candidate_spacing_cells,
                    )
                )
            for left_anchor, right_anchor in zip(
                run_anchor_positions,
                run_anchor_positions[1:],
            ):
                run_selected_positions.update(
                    _build_anchor_gap_candidate_positions(
                        left_anchor,
                        right_anchor,
                        config.candidate_spacing_cells,
                    )
                )
            if run_anchor_positions[-1] < len(run) - 1:
                run_selected_positions.update(
                    _build_segment_candidate_positions(
                        run_anchor_positions[-1] + 1,
                        len(run) - 1,
                        config.candidate_spacing_cells,
                    )
                )
        else:
            run_selected_positions.update(
                _build_segment_candidate_positions(
                    0,
                    len(run) - 1,
                    config.candidate_spacing_cells,
                )
            )

        for position in run_selected_positions:
            selected_candidate_flats.add(int(run[position]))

    candidate_cell_indices = np.asarray(
        sorted(selected_candidate_flats),
        dtype=np.int32,
    )
    candidate_boundary_flags = np.asarray(
        [
            eligible_boundary_lookup[int(flat_index)]
            for flat_index in candidate_cell_indices
        ],
        dtype=np.uint8,
    )
    candidate_exception_flags = np.asarray(
        [
            exception_flags_by_flat.get(int(flat_index), 0)
            for flat_index in candidate_cell_indices
        ],
        dtype=np.uint8,
    )
    return candidate_cell_indices, candidate_boundary_flags, candidate_exception_flags


def _build_wall_runs(
    width: int,
    directional_masks: _DirectionalMasks,
) -> list[NDArray[np.int32]]:
    """Build deterministic solid-wall runs in row-major or axis-major order."""

    return (
        _collect_horizontal_runs(
            directional_masks.open_mask & directional_masks.north_solid, width
        )
        + _collect_horizontal_runs(
            directional_masks.open_mask & directional_masks.south_solid, width
        )
        + _collect_vertical_runs(
            directional_masks.open_mask & directional_masks.east_solid, width
        )
        + _collect_vertical_runs(
            directional_masks.open_mask & directional_masks.west_solid, width
        )
    )


def _collect_horizontal_runs(
    mask: NDArray[np.bool_],
    width: int,
) -> list[NDArray[np.int32]]:
    """Collect contiguous horizontal run slices from one directional mask."""

    runs: list[NDArray[np.int32]] = []
    for row in range(mask.shape[0]):
        col = 0
        while col < mask.shape[1]:
            if not bool(mask[row, col]):
                col += 1
                continue
            start_col = col
            while col < mask.shape[1] and bool(mask[row, col]):
                col += 1
            run_cols = np.arange(start_col, col, dtype=np.int32)
            run_rows = np.full(run_cols.shape, row, dtype=np.int32)
            runs.append((run_rows * width + run_cols).astype(np.int32, copy=False))
    return runs


def _collect_vertical_runs(
    mask: NDArray[np.bool_],
    width: int,
) -> list[NDArray[np.int32]]:
    """Collect contiguous vertical run slices from one directional mask."""

    runs: list[NDArray[np.int32]] = []
    for col in range(mask.shape[1]):
        row = 0
        while row < mask.shape[0]:
            if not bool(mask[row, col]):
                row += 1
                continue
            start_row = row
            while row < mask.shape[0] and bool(mask[row, col]):
                row += 1
            run_rows = np.arange(start_row, row, dtype=np.int32)
            run_cols = np.full(run_rows.shape, col, dtype=np.int32)
            runs.append((run_rows * width + run_cols).astype(np.int32, copy=False))
    return runs


def _build_run_anchor_positions(
    run: NDArray[np.int32],
    eligible_solid_lookup: dict[int, int],
    exception_flags_by_flat: dict[int, int],
) -> list[int]:
    """Return the sorted run-order anchor positions and record exception bits."""

    anchor_positions: set[int] = set()
    run_length = len(run)
    if run_length == 0:
        return []

    # The performance revision locked these exception classes on permanently, so
    # phase 01 now bakes them into the thinning rule instead of carrying redundant
    # booleans through `PlannerConfig`.
    anchor_positions.add(0)
    anchor_positions.add(run_length - 1)
    exception_flags_by_flat[int(run[0])] = exception_flags_by_flat.get(
        int(run[0]), 0
    ) | int(_EXCEPTION_FLAG_ENDPOINT)
    exception_flags_by_flat[int(run[-1])] = exception_flags_by_flat.get(
        int(run[-1]), 0
    ) | int(_EXCEPTION_FLAG_ENDPOINT)

    midpoint_position = (run_length - 1) // 2
    anchor_positions.add(midpoint_position)
    midpoint_flat = int(run[midpoint_position])
    exception_flags_by_flat[midpoint_flat] = exception_flags_by_flat.get(
        midpoint_flat, 0
    ) | int(_EXCEPTION_FLAG_MIDPOINT)

    for position, flat_index in enumerate(run):
        flat_value = int(flat_index)
        solid_flags = eligible_solid_lookup.get(flat_value, 0)
        solid_dir_count = _count_direction_bits(solid_flags)
        has_vertical_solid = (
            solid_flags & int(_BOUNDARY_FLAG_NORTH | _BOUNDARY_FLAG_SOUTH)
        ) != 0
        has_horizontal_solid = (
            solid_flags & int(_BOUNDARY_FLAG_EAST | _BOUNDARY_FLAG_WEST)
        ) != 0

        if solid_dir_count == 2 and has_vertical_solid and has_horizontal_solid:
            anchor_positions.add(position)
            exception_flags_by_flat[flat_value] = exception_flags_by_flat.get(
                flat_value, 0
            ) | int(_EXCEPTION_FLAG_CORNER)

        if solid_dir_count >= 3:
            anchor_positions.add(position)
            exception_flags_by_flat[flat_value] = exception_flags_by_flat.get(
                flat_value, 0
            ) | int(_EXCEPTION_FLAG_JUNCTION)

    return sorted(anchor_positions)


def _count_direction_bits(boundary_flags: int) -> int:
    """Count the active directional bits in one compact boundary mask."""

    count = 0
    if boundary_flags & int(_BOUNDARY_FLAG_NORTH):
        count += 1
    if boundary_flags & int(_BOUNDARY_FLAG_EAST):
        count += 1
    if boundary_flags & int(_BOUNDARY_FLAG_SOUTH):
        count += 1
    if boundary_flags & int(_BOUNDARY_FLAG_WEST):
        count += 1
    return count


def _build_segment_candidate_positions(
    start: int,
    end: int,
    spacing: int,
) -> set[int]:
    """Distribute the maximum feasible non-exception candidates across one segment."""

    if start > end:
        return set()

    length = (end - start) + 1
    candidate_count = 1 + ((length - 1) // spacing)
    if candidate_count <= 0:
        return set()
    if candidate_count == 1:
        return {start + ((length - 1) // 2)}

    required_span = spacing * (candidate_count - 1)
    slack = (length - 1) - required_span
    positions = {
        start
        + (position_ordinal * spacing)
        + ((position_ordinal * slack) // (candidate_count - 1))
        for position_ordinal in range(candidate_count)
    }
    return positions


def _build_anchor_gap_candidate_positions(
    left_anchor: int,
    right_anchor: int,
    spacing: int,
) -> set[int]:
    """Distribute interior candidates across one anchor-bounded wall-run gap.

    The current thinning rule allows sub-minimum gaps only when one neighbor is
    an exception anchor. To avoid filling a gap by snapping candidates directly
    onto the cells beside both anchors, derive the interior count from the full
    anchor span first, then repair the rounded positions so every
    non-exception-to-non-exception gap still respects `spacing`.
    """

    interior_start = left_anchor + 1
    interior_end = right_anchor - 1
    if interior_start > interior_end:
        return set()

    interior_length = (interior_end - interior_start) + 1
    candidate_count = interior_length // spacing
    if candidate_count <= 0:
        return set()

    full_span = right_anchor - left_anchor
    positions = [
        int(np.floor(left_anchor + ((ordinal + 1) * full_span / (candidate_count + 1))))
        for ordinal in range(candidate_count)
    ]

    min_positions = [
        interior_start + (ordinal * spacing) for ordinal in range(candidate_count)
    ]
    max_positions = [
        interior_end - ((candidate_count - ordinal - 1) * spacing)
        for ordinal in range(candidate_count)
    ]

    for ordinal in range(candidate_count):
        positions[ordinal] = max(positions[ordinal], min_positions[ordinal])
        if ordinal > 0:
            positions[ordinal] = max(
                positions[ordinal], positions[ordinal - 1] + spacing
            )

    for ordinal in range(candidate_count - 1, -1, -1):
        positions[ordinal] = min(positions[ordinal], max_positions[ordinal])
        if ordinal < candidate_count - 1:
            positions[ordinal] = min(
                positions[ordinal], positions[ordinal + 1] - spacing
            )

    return set(positions)


def _flat_indices_to_coords(
    indices: NDArray[np.int32],
    width: int,
) -> NDArray[np.int32]:
    """Decode row-major flat indices into `(row, col)` coordinate pairs."""

    rows = indices // width
    cols = indices % width
    return np.column_stack((rows, cols)).astype(np.int32, copy=False)


# Validation helpers
def _validate_flat_index_array(
    name: str,
    indices: NDArray[np.int32],
    *,
    size: int,
) -> None:
    """Validate dtype, shape, ordering, uniqueness, and bounds for flat indices."""

    if indices.dtype != np.int32:
        raise TypeError(f"{name} must use dtype np.int32.")
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if len(indices) > 1 and not np.all(indices[:-1] < indices[1:]):
        raise ValueError(f"{name} must be strictly increasing and duplicate-free.")
    if indices.size and (indices[0] < 0 or indices[-1] >= size):
        raise ValueError(f"{name} contains an out-of-range flat index.")


def _validate_coordinate_array(
    name: str,
    coords: NDArray[np.int32],
    *,
    expected_length: int,
) -> None:
    """Validate dtype and shape for a `(row, col)` coordinate array."""

    if coords.dtype != np.int32:
        raise TypeError(f"{name} must use dtype np.int32.")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2).")
    if coords.shape[0] != expected_length:
        raise ValueError(f"{name} length must match its paired flat-index array.")


def _validate_uint8_array(
    name: str,
    values: NDArray[np.uint8],
    *,
    expected_length: int,
) -> None:
    """Validate dtype and length for compact uint8 metadata arrays."""

    if values.dtype != np.uint8:
        raise TypeError(f"{name} must use dtype np.uint8.")
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if len(values) != expected_length:
        raise ValueError(f"{name} length must match its paired flat-index array.")


def _validate_open_cells_match_grid(
    grid: NDArray[np.int8],
    open_cell_coords_rc: NDArray[np.int32],
) -> None:
    """Confirm that every stored open-cell coordinate still maps to an open cell."""

    for row, col in open_cell_coords_rc:
        if grid[row, col] != OPEN_CELL:
            raise ValueError("open_cell_coords_rc includes a non-open grid cell.")


def _validate_candidate_cells_match_grid(
    grid: NDArray[np.int8],
    candidate_cell_coords_rc: NDArray[np.int32],
    candidate_boundary_flags: NDArray[np.uint8],
) -> None:
    """Confirm that every candidate is open and satisfies the solid-adjacency rule."""

    for (row, col), boundary_flag in zip(
        candidate_cell_coords_rc,
        candidate_boundary_flags,
        strict=True,
    ):
        if grid[row, col] != OPEN_CELL:
            raise ValueError(
                "candidate coordinate array includes a non-open grid cell."
            )
        if boundary_flag == 0:
            raise ValueError("candidate boundary-flag arrays must be non-zero.")

        has_solid_neighbor = False
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            neighbor_row = row + dr
            neighbor_col = col + dc
            if (
                0 <= neighbor_row < grid.shape[0]
                and 0 <= neighbor_col < grid.shape[1]
                and grid[neighbor_row, neighbor_col] == SOLID_CELL
            ):
                has_solid_neighbor = True
                break

        if not has_solid_neighbor:
            raise ValueError(
                "candidate coordinate array includes an open cell with no solid "
                "4-neighbor."
            )


def _validate_candidate_spacing_rules(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    artifacts: CandidateGenerationArtifacts,
    directional_masks: _DirectionalMasks,
    eligible_candidate_solid_flags: NDArray[np.uint8],
) -> None:
    """Validate anchor retention and the exception-first spacing rule per wall run."""

    eligible_solid_lookup = {
        int(flat_index): int(solid_flag)
        for flat_index, solid_flag in zip(
            artifacts.eligible_candidate_cell_indices,
            eligible_candidate_solid_flags,
            strict=True,
        )
    }
    selected_lookup = {
        int(flat_index): int(exception_flag)
        for flat_index, exception_flag in zip(
            artifacts.candidate_cell_indices,
            artifacts.candidate_exception_flags,
            strict=True,
        )
    }

    for run in _build_wall_runs(floorplan.shape[1], directional_masks):
        anchor_flags_by_flat: dict[int, int] = {}
        anchor_positions = _build_run_anchor_positions(
            run,
            eligible_solid_lookup,
            anchor_flags_by_flat,
        )
        for anchor_position in anchor_positions:
            anchor_flat = int(run[anchor_position])
            if anchor_flat not in selected_lookup:
                raise ValueError(
                    "Exception-anchor candidate was not retained in the final thinned "
                    "candidate set."
                )

        selected_positions = [
            position
            for position, flat_index in enumerate(run)
            if int(flat_index) in selected_lookup
        ]
        for left_position, right_position in zip(
            selected_positions,
            selected_positions[1:],
        ):
            if (right_position - left_position) >= config.candidate_spacing_cells:
                continue

            left_exception_flag = selected_lookup[int(run[left_position])]
            right_exception_flag = selected_lookup[int(run[right_position])]
            if left_exception_flag == 0 and right_exception_flag == 0:
                raise ValueError(
                    "Final thinned candidates contain a sub-minimum wall-run spacing "
                    "gap that does not involve an exception anchor."
                )


__all__ = [
    "CandidateGenerationArtifacts",
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
    "generate_candidate_generation_artifacts",
    "load_candidate_generation_artifacts",
    "resolve_candidate_generation_artifacts",
    "save_candidate_generation_artifacts",
    "validate_candidate_generation_artifacts",
]
