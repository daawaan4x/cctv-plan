"""Tests for phase-03 orientation-aware sparse DORI score generation."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from src.common.floorplan import FloorPlanInput, NULL_CELL, OPEN_CELL, SOLID_CELL
from src.planner._shared.config import PlannerConfig
from src.planner.phase01_candidate_generation import generate_candidate_generation_artifacts
from src.planner.phase02_visibility import generate_visibility_artifacts
from src.planner.phase03_scoring import (
    SparseScoreArtifacts,
    decode_configuration_ordinal,
    generate_sparse_score_artifacts,
    get_configuration_dori_scores,
    get_configuration_target_ordinals,
    load_sparse_score_artifacts,
    save_sparse_score_artifacts,
    validate_sparse_score_artifacts,
)
from src.planner.phase03_scoring.core import (
    _build_scoring_constants,
    _score_distances_to_dori,
)


# Small tri-state fixtures
def _build_floorplan(
    name: str,
    rows: list[list[np.int8]],
    *,
    grid_cell_size_m: float | None = 1.0,
) -> FloorPlanInput:
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
        grid_cell_size_m=grid_cell_size_m,
    )


def _build_all_open_square_floorplan(
    *,
    grid_cell_size_m: float | None = 1.0,
) -> FloorPlanInput:
    """Build a tiny open square where every LOS-positive pair scores at level four."""

    return _build_floorplan(
        "phase03-open-square",
        [
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
        ],
        grid_cell_size_m=grid_cell_size_m,
    )


def _build_test_config() -> PlannerConfig:
    """Build a compact planner config with four orthogonal orientations."""

    return PlannerConfig(
        floorplan_name="phase03-open-square",
        camera_horizontal_resolution_px=1000,
        camera_horizontal_fov_deg=90.0,
        orientation_step_deg=90,
        k_values=(1,),
    )


class SparseScoreTests(unittest.TestCase):
    """Verify phase-03 sparse scores against the locked scoring model."""

    def test_generate_sparse_score_artifacts_builds_expected_configuration_slices(
        self,
    ) -> None:
        floorplan = _build_all_open_square_floorplan()
        config = _build_test_config()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
        phase02_artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)

        artifacts = generate_sparse_score_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase02_artifacts,
        )

        self.assertEqual(artifacts.grid_shape, floorplan.shape)
        self.assertEqual(artifacts.candidate_count, 4)
        self.assertEqual(artifacts.open_cell_count, 4)
        np.testing.assert_array_equal(
            artifacts.orientation_angles_deg,
            np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            artifacts.configuration_candidate_ordinals,
            np.repeat(np.arange(4, dtype=np.int32), 4),
        )
        np.testing.assert_array_equal(
            artifacts.configuration_angle_ordinals,
            np.tile(np.arange(4, dtype=np.int16), 4),
        )

        np.testing.assert_array_equal(
            get_configuration_target_ordinals(artifacts, 0),
            np.array([1, 3], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            get_configuration_dori_scores(artifacts, 0),
            np.array([4, 4], dtype=np.int8),
        )
        np.testing.assert_array_equal(
            get_configuration_target_ordinals(artifacts, 1),
            np.empty(0, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            get_configuration_target_ordinals(artifacts, 3),
            np.array([2, 3], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            get_configuration_dori_scores(artifacts, 3),
            np.array([4, 4], dtype=np.int8),
        )

    def test_decode_configuration_ordinal_uses_candidate_major_order(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config = _build_test_config()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
        phase02_artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)
        artifacts = generate_sparse_score_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase02_artifacts,
        )

        self.assertEqual(
            decode_configuration_ordinal(artifacts, 5),
            (1, 1, 90.0),
        )
        self.assertEqual(
            decode_configuration_ordinal(artifacts, 15),
            (3, 3, 270.0),
        )

    def test_save_and_load_sparse_score_artifacts_round_trip(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config = _build_test_config()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
        phase02_artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)
        artifacts = generate_sparse_score_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase02_artifacts,
        )

        temp_dir = Path.cwd() / f".tmp-phase03-scoring-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
            artifact_path = temp_dir / "03_sparse_scores.npz"
            save_sparse_score_artifacts(artifact_path, artifacts)
            reloaded = load_sparse_score_artifacts(artifact_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(reloaded.grid_shape, artifacts.grid_shape)
        self.assertEqual(reloaded.candidate_count, artifacts.candidate_count)
        self.assertEqual(reloaded.open_cell_count, artifacts.open_cell_count)
        np.testing.assert_array_equal(
            reloaded.orientation_angles_deg,
            artifacts.orientation_angles_deg,
        )
        np.testing.assert_array_equal(
            reloaded.configuration_candidate_ordinals,
            artifacts.configuration_candidate_ordinals,
        )
        np.testing.assert_array_equal(
            reloaded.configuration_angle_ordinals,
            artifacts.configuration_angle_ordinals,
        )
        np.testing.assert_array_equal(
            reloaded.score_configuration_offsets,
            artifacts.score_configuration_offsets,
        )
        np.testing.assert_array_equal(
            reloaded.score_target_ordinals,
            artifacts.score_target_ordinals,
        )
        np.testing.assert_array_equal(reloaded.score_values, artifacts.score_values)
        validate_sparse_score_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase02_artifacts,
            reloaded,
        )

    def test_generate_sparse_score_artifacts_requires_grid_cell_size(self) -> None:
        floorplan = _build_all_open_square_floorplan(grid_cell_size_m=None)
        config = _build_test_config()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
        phase02_artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)

        with self.assertRaisesRegex(
            ValueError,
            r"requires floorplan\.grid_cell_size_m",
        ):
            generate_sparse_score_artifacts(
                floorplan,
                config,
                phase01_artifacts,
                phase02_artifacts,
            )

    def test_validate_sparse_score_artifacts_rejects_zero_score_values(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config = _build_test_config()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
        phase02_artifacts = generate_visibility_artifacts(floorplan, phase01_artifacts)
        artifacts = generate_sparse_score_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase02_artifacts,
        )
        broken = SparseScoreArtifacts(
            grid_shape=artifacts.grid_shape,
            candidate_count=artifacts.candidate_count,
            open_cell_count=artifacts.open_cell_count,
            orientation_angles_deg=artifacts.orientation_angles_deg,
            configuration_candidate_ordinals=artifacts.configuration_candidate_ordinals,
            configuration_angle_ordinals=artifacts.configuration_angle_ordinals,
            score_configuration_offsets=artifacts.score_configuration_offsets,
            score_target_ordinals=artifacts.score_target_ordinals,
            score_values=np.where(
                np.arange(len(artifacts.score_values)) == 0,
                np.int8(0),
                artifacts.score_values,
            ).astype(np.int8, copy=False),
        )

        with self.assertRaisesRegex(ValueError, r"categorical scores 1\.\.4"):
            validate_sparse_score_artifacts(
                floorplan,
                config,
                phase01_artifacts,
                phase02_artifacts,
                broken,
            )

    def test_score_distances_to_dori_honors_exact_threshold_boundaries(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config = _build_test_config()
        scoring_constants = _build_scoring_constants(config, floorplan.grid_cell_size_m or 1.0)
        distances_m = np.asarray(
            [
                config.camera_horizontal_resolution_px
                / (2.0 * config.dori_thresholds.detection),
                config.camera_horizontal_resolution_px
                / (2.0 * config.dori_thresholds.observation),
                config.camera_horizontal_resolution_px
                / (2.0 * config.dori_thresholds.recognition),
                config.camera_horizontal_resolution_px
                / (2.0 * config.dori_thresholds.identification),
            ],
            dtype=np.float64,
        )

        np.testing.assert_array_equal(
            _score_distances_to_dori(distances_m, scoring_constants),
            np.array([1, 2, 3, 4], dtype=np.int8),
        )


if __name__ == "__main__":
    unittest.main()
