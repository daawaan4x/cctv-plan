from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.common.floorplan import FloorPlanInput, NULL_CELL, OPEN_CELL, SOLID_CELL
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    generate_candidate_generation_artifacts,
    load_candidate_generation_artifacts,
    save_candidate_generation_artifacts,
    validate_candidate_generation_artifacts,
)


def _build_floorplan() -> FloorPlanInput:
    grid = np.array(
        [
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, SOLID_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, SOLID_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
        ],
        dtype=np.int8,
    )
    return FloorPlanInput(
        name="phase01-test",
        source_path=Path("phase01-test.png"),
        grid=grid,
        height=5,
        width=5,
        null_cell_count=16,
        open_cell_count=7,
        solid_cell_count=2,
    )


class CandidateGenerationTests(unittest.TestCase):
    def test_generate_candidate_generation_artifacts_builds_expected_sets(self) -> None:
        floorplan = _build_floorplan()

        artifacts = generate_candidate_generation_artifacts(floorplan)

        expected_open_indices = np.array([6, 7, 11, 12, 13, 17, 18], dtype=np.int32)
        expected_open_coords = np.array(
            [[1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 2], [3, 3]],
            dtype=np.int32,
        )
        expected_candidate_indices = np.array([6, 7, 11, 13, 17, 18], dtype=np.int32)
        expected_candidate_coords = np.array(
            [[1, 1], [1, 2], [2, 1], [2, 3], [3, 2], [3, 3]],
            dtype=np.int32,
        )
        expected_boundary_flags = np.array([9, 3, 12, 3, 12, 6], dtype=np.uint8)

        self.assertEqual(artifacts.grid_shape, (5, 5))
        np.testing.assert_array_equal(artifacts.open_cell_indices, expected_open_indices)
        np.testing.assert_array_equal(artifacts.open_cell_coords_rc, expected_open_coords)
        np.testing.assert_array_equal(
            artifacts.candidate_cell_indices,
            expected_candidate_indices,
        )
        np.testing.assert_array_equal(
            artifacts.candidate_cell_coords_rc,
            expected_candidate_coords,
        )
        np.testing.assert_array_equal(
            artifacts.candidate_boundary_flags,
            expected_boundary_flags,
        )

    def test_save_and_load_candidate_generation_artifacts_round_trip(self) -> None:
        floorplan = _build_floorplan()
        artifacts = generate_candidate_generation_artifacts(floorplan)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "01_candidates.npz"
            save_candidate_generation_artifacts(artifact_path, artifacts)
            reloaded = load_candidate_generation_artifacts(artifact_path)

        self.assertEqual(reloaded.grid_shape, artifacts.grid_shape)
        np.testing.assert_array_equal(
            reloaded.open_cell_indices,
            artifacts.open_cell_indices,
        )
        np.testing.assert_array_equal(
            reloaded.open_cell_coords_rc,
            artifacts.open_cell_coords_rc,
        )
        np.testing.assert_array_equal(
            reloaded.candidate_cell_indices,
            artifacts.candidate_cell_indices,
        )
        np.testing.assert_array_equal(
            reloaded.candidate_cell_coords_rc,
            artifacts.candidate_cell_coords_rc,
        )
        np.testing.assert_array_equal(
            reloaded.candidate_boundary_flags,
            artifacts.candidate_boundary_flags,
        )
        validate_candidate_generation_artifacts(floorplan, reloaded)

    def test_validate_candidate_generation_artifacts_rejects_zero_boundary_flag(
        self,
    ) -> None:
        floorplan = _build_floorplan()
        artifacts = generate_candidate_generation_artifacts(floorplan)
        broken = CandidateGenerationArtifacts(
            grid_shape=artifacts.grid_shape,
            open_cell_indices=artifacts.open_cell_indices,
            open_cell_coords_rc=artifacts.open_cell_coords_rc,
            candidate_cell_indices=artifacts.candidate_cell_indices,
            candidate_cell_coords_rc=artifacts.candidate_cell_coords_rc,
            candidate_boundary_flags=np.zeros_like(artifacts.candidate_boundary_flags),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"candidate_boundary_flags must be non-zero",
        ):
            validate_candidate_generation_artifacts(floorplan, broken)


if __name__ == "__main__":
    unittest.main()
