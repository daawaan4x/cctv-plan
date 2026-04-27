"""Tests for phase-02 LOS generation, sparse storage, and corner blocking."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from src.common.floorplan import FloorPlanInput, NULL_CELL, OPEN_CELL, SOLID_CELL
from src.planner.phase01_candidate_generation import generate_candidate_generation_artifacts
from src.planner.phase02_visibility import (
    VisibilityArtifacts,
    generate_visibility_artifacts,
    get_diagonal_blocked_target_ordinals,
    get_visible_target_ordinals,
    load_visibility_artifacts,
    save_visibility_artifacts,
    validate_visibility_artifacts,
)


# Small tri-state fixtures
def _build_floorplan(name: str, rows: list[list[np.int8]]) -> FloorPlanInput:
    """Build a `FloorPlanInput` from a compact nested-list tri-state grid."""

    grid = np.asarray(rows, dtype=np.int8)
    return FloorPlanInput(
        name=name,
        source_path=Path(f"{name}.png"),
        grid=grid,
        height=int(grid.shape[0]),
        width=int(grid.shape[1]),
        null_cell_count=int(np.count_nonzero(grid == NULL_CELL)),
        open_cell_count=int(np.count_nonzero(grid == OPEN_CELL)),
        solid_cell_count=int(np.count_nonzero(grid == SOLID_CELL)),
    )


def _build_all_open_square_floorplan() -> FloorPlanInput:
    """Build a tiny open square where every candidate sees every other target."""

    return _build_floorplan(
        "phase02-open-square",
        [
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
        ],
    )


def _build_diagonal_corner_blocked_floorplan() -> FloorPlanInput:
    """Build a grid where the only cross-cell LOS path is blocked by both sides."""

    return _build_floorplan(
        "phase02-diagonal-blocked",
        [
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, SOLID_CELL, NULL_CELL],
            [NULL_CELL, SOLID_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
        ],
    )


def _build_diagonal_one_side_open_floorplan() -> FloorPlanInput:
    """Build a grid where a diagonal LOS remains valid because one side is open."""

    return _build_floorplan(
        "phase02-diagonal-open",
        [
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, SOLID_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
        ],
    )


class VisibilityTests(unittest.TestCase):
    """Verify phase-02 visibility artifacts against the locked LOS rules."""

    def test_generate_visibility_artifacts_builds_expected_sparse_slices(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)

        artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)

        expected_offsets = np.array([0, 3, 6, 9, 12], dtype=np.int32)
        expected_target_ordinals = np.array(
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            dtype=np.int32,
        )

        self.assertEqual(artifacts.grid_shape, floorplan.shape)
        self.assertEqual(artifacts.candidate_count, 4)
        self.assertEqual(artifacts.open_cell_count, 4)
        np.testing.assert_array_equal(artifacts.los_candidate_offsets, expected_offsets)
        np.testing.assert_array_equal(artifacts.los_target_ordinals, expected_target_ordinals)
        np.testing.assert_array_equal(
            artifacts.diagonal_candidate_offsets,
            np.zeros(5, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            artifacts.diagonal_target_ordinals,
            np.empty(0, dtype=np.int32),
        )

    def test_generate_visibility_artifacts_tracks_diagonal_corner_blocking(self) -> None:
        floorplan = _build_diagonal_corner_blocked_floorplan()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)

        artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)

        np.testing.assert_array_equal(
            artifacts.los_candidate_offsets,
            np.zeros(3, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            artifacts.diagonal_candidate_offsets,
            np.array([0, 1, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            get_diagonal_blocked_target_ordinals(artifacts, 0),
            np.array([1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            get_diagonal_blocked_target_ordinals(artifacts, 1),
            np.array([0], dtype=np.int32),
        )

    def test_generate_visibility_artifacts_keeps_diagonal_visible_when_one_side_is_open(
        self,
    ) -> None:
        floorplan = _build_diagonal_one_side_open_floorplan()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)

        artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)

        np.testing.assert_array_equal(
            get_visible_target_ordinals(artifacts, 0),
            np.array([1, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            get_visible_target_ordinals(artifacts, 2),
            np.array([0, 1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            artifacts.diagonal_target_ordinals,
            np.empty(0, dtype=np.int32),
        )

    def test_generate_visibility_artifacts_blocks_rays_crossing_non_open_cells(self) -> None:
        cases = {
            "null-blocker": [
                [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
                [NULL_CELL, OPEN_CELL, NULL_CELL, OPEN_CELL, NULL_CELL],
                [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            ],
            "solid-blocker": [
                [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
                [NULL_CELL, OPEN_CELL, SOLID_CELL, OPEN_CELL, NULL_CELL],
                [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            ],
        }

        for name, rows in cases.items():
            with self.subTest(name=name):
                floorplan = _build_floorplan(name, rows)
                phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
                artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)

                np.testing.assert_array_equal(
                    artifacts.los_candidate_offsets,
                    np.zeros(3, dtype=np.int32),
                )
                np.testing.assert_array_equal(
                    artifacts.diagonal_candidate_offsets,
                    np.zeros(3, dtype=np.int32),
                )

    def test_save_and_load_visibility_artifacts_round_trip(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
        artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)

        temp_dir = Path.cwd() / f".tmp-phase02-visibility-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
            artifact_path = temp_dir / "02_visibility.npz"
            save_visibility_artifacts(artifact_path, artifacts)
            reloaded = load_visibility_artifacts(artifact_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(reloaded.grid_shape, artifacts.grid_shape)
        self.assertEqual(reloaded.candidate_count, artifacts.candidate_count)
        self.assertEqual(reloaded.open_cell_count, artifacts.open_cell_count)
        np.testing.assert_array_equal(
            reloaded.los_candidate_offsets,
            artifacts.los_candidate_offsets,
        )
        np.testing.assert_array_equal(
            reloaded.los_target_ordinals,
            artifacts.los_target_ordinals,
        )
        np.testing.assert_array_equal(
            reloaded.diagonal_candidate_offsets,
            artifacts.diagonal_candidate_offsets,
        )
        np.testing.assert_array_equal(
            reloaded.diagonal_target_ordinals,
            artifacts.diagonal_target_ordinals,
        )
        validate_visibility_artifacts(floorplan, phase01_artifacts, reloaded)

    def test_validate_visibility_artifacts_rejects_overlapping_visible_and_diagonal_pairs(
        self,
    ) -> None:
        floorplan = _build_all_open_square_floorplan()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
        artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)

        broken = VisibilityArtifacts(
            grid_shape=artifacts.grid_shape,
            candidate_count=artifacts.candidate_count,
            open_cell_count=artifacts.open_cell_count,
            los_candidate_offsets=artifacts.los_candidate_offsets,
            los_target_ordinals=artifacts.los_target_ordinals,
            diagonal_candidate_offsets=np.array([0, 1, 1, 1, 1], dtype=np.int32),
            diagonal_target_ordinals=np.array([1], dtype=np.int32),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"cannot be both visible and diagonal-blocked",
        ):
            validate_visibility_artifacts(floorplan, phase01_artifacts, broken)


if __name__ == "__main__":
    unittest.main()
