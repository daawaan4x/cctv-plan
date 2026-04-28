"""Tests for deterministic phase-01 open-target and candidate generation."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from src.common.floorplan import FloorPlanInput, NULL_CELL, OPEN_CELL, SOLID_CELL
from src.planner._shared.config import PlannerConfig
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    generate_candidate_generation_artifacts,
    load_candidate_generation_artifacts,
    resolve_candidate_generation_artifacts,
    save_candidate_generation_artifacts,
    validate_candidate_generation_artifacts,
)


# Small tri-state fixture
def _build_floorplan() -> FloorPlanInput:
    """Build a small tri-state grid that exercises the candidate boundary rule."""

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
    """Verify the stored phase-01 artifacts against the locked candidate rules."""

    def test_generate_candidate_generation_artifacts_builds_expected_sets(self) -> None:
        floorplan = _build_floorplan()
        config = PlannerConfig(floorplan_name=floorplan.name)

        # The expected sets are written as flat indices and `(row, col)` pairs so
        # the test locks down both the canonical cross-phase identity and the more
        # human-readable geometry view of the same candidate cells.
        artifacts = generate_candidate_generation_artifacts(floorplan, config)

        expected_open_indices = np.array([6, 7, 11, 12, 13, 17, 18], dtype=np.int32)
        expected_open_coords = np.array(
            [[1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 2], [3, 3]],
            dtype=np.int32,
        )
        expected_eligible_indices = np.array([6, 7, 11, 13, 17, 18], dtype=np.int32)
        expected_eligible_coords = np.array(
            [[1, 1], [1, 2], [2, 1], [2, 3], [3, 2], [3, 3]],
            dtype=np.int32,
        )
        expected_boundary_flags = np.array([9, 3, 12, 3, 12, 6], dtype=np.uint8)
        expected_exception_flags = np.array([0, 3, 3, 3, 3, 0], dtype=np.uint8)

        self.assertEqual(artifacts.grid_shape, (5, 5))
        np.testing.assert_array_equal(artifacts.open_cell_indices, expected_open_indices)
        np.testing.assert_array_equal(artifacts.open_cell_coords_rc, expected_open_coords)
        np.testing.assert_array_equal(
            artifacts.eligible_candidate_cell_indices,
            expected_eligible_indices,
        )
        np.testing.assert_array_equal(
            artifacts.eligible_candidate_cell_coords_rc,
            expected_eligible_coords,
        )
        np.testing.assert_array_equal(
            artifacts.eligible_candidate_boundary_flags,
            expected_boundary_flags,
        )
        np.testing.assert_array_equal(
            artifacts.candidate_cell_indices,
            expected_eligible_indices,
        )
        np.testing.assert_array_equal(
            artifacts.candidate_cell_coords_rc,
            expected_eligible_coords,
        )
        np.testing.assert_array_equal(
            artifacts.candidate_boundary_flags,
            expected_boundary_flags,
        )
        np.testing.assert_array_equal(
            artifacts.candidate_exception_flags,
            expected_exception_flags,
        )

    def test_save_and_load_candidate_generation_artifacts_round_trip(self) -> None:
        floorplan = _build_floorplan()
        config = PlannerConfig(floorplan_name=floorplan.name)
        artifacts = generate_candidate_generation_artifacts(floorplan, config)

        temp_dir = Path.cwd() / f".tmp-phase01-candidates-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
            artifact_path = temp_dir / "01_candidates.npz"
            save_candidate_generation_artifacts(artifact_path, artifacts)
            reloaded = load_candidate_generation_artifacts(artifact_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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
            reloaded.eligible_candidate_cell_indices,
            artifacts.eligible_candidate_cell_indices,
        )
        np.testing.assert_array_equal(
            reloaded.eligible_candidate_cell_coords_rc,
            artifacts.eligible_candidate_cell_coords_rc,
        )
        np.testing.assert_array_equal(
            reloaded.eligible_candidate_boundary_flags,
            artifacts.eligible_candidate_boundary_flags,
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
        np.testing.assert_array_equal(
            reloaded.candidate_exception_flags,
            artifacts.candidate_exception_flags,
        )
        validate_candidate_generation_artifacts(floorplan, reloaded, config=config)

    def test_validate_candidate_generation_artifacts_rejects_zero_boundary_flag(
        self,
    ) -> None:
        floorplan = _build_floorplan()
        config = PlannerConfig(floorplan_name=floorplan.name)
        artifacts = generate_candidate_generation_artifacts(floorplan, config)
        broken = CandidateGenerationArtifacts(
            grid_shape=artifacts.grid_shape,
            open_cell_indices=artifacts.open_cell_indices,
            open_cell_coords_rc=artifacts.open_cell_coords_rc,
            eligible_candidate_cell_indices=artifacts.eligible_candidate_cell_indices,
            eligible_candidate_cell_coords_rc=artifacts.eligible_candidate_cell_coords_rc,
            eligible_candidate_boundary_flags=artifacts.eligible_candidate_boundary_flags,
            candidate_cell_indices=artifacts.candidate_cell_indices,
            candidate_cell_coords_rc=artifacts.candidate_cell_coords_rc,
            candidate_boundary_flags=np.zeros_like(artifacts.candidate_boundary_flags),
            candidate_exception_flags=artifacts.candidate_exception_flags,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"candidate_boundary_flags must be non-zero",
        ):
            validate_candidate_generation_artifacts(floorplan, broken, config=config)

    def test_resolve_candidate_generation_artifacts_ignores_public_k_values_tuple(self) -> None:
        floorplan = _build_floorplan()
        config_left = PlannerConfig(floorplan_name=floorplan.name, k_values=(1, 2))
        config_right = PlannerConfig(floorplan_name=floorplan.name, k_values=(5, 6, 7))

        temp_dir = Path.cwd() / f".tmp-phase01-resolve-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
            left = resolve_candidate_generation_artifacts(
                floorplan,
                config_left,
                repo_root=temp_dir,
            )
            right = resolve_candidate_generation_artifacts(
                floorplan,
                config_right,
                repo_root=temp_dir,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        np.testing.assert_array_equal(left.candidate_cell_indices, right.candidate_cell_indices)


if __name__ == "__main__":
    unittest.main()
