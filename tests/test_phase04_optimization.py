"""Tests for phase-04 exact threshold-coverage optimization."""

from __future__ import annotations

from io import StringIO
import json
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
)
from src.planner.phase03_scoring import SparseScoreArtifacts
from src.planner.phase04_optimization import (
    OptimizationArtifacts,
    build_optimization_precompute_artifacts,
    ensure_optimization_precompute_artifacts,
    load_optimization_artifacts,
    load_optimization_precompute_artifacts,
    resolve_optimization_artifact,
    resolve_optimization_artifacts_for_k_values,
    save_optimization_artifacts,
    save_optimization_precompute_artifacts,
    save_optimization_summary,
    solve_for_k_values,
    solve_optimization_artifacts,
    validate_optimization_artifacts,
    validate_optimization_precompute_artifacts,
)


# Small tri-state fixtures
def _build_floorplan(
    name: str,
    rows: list[list[np.int8]],
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
        grid_cell_size_m=1.0,
    )


def _build_all_open_square_floorplan() -> FloorPlanInput:
    """Build a tiny open square used by the synthetic optimization tests."""

    return _build_floorplan(
        "phase04-open-square",
        [
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
        ],
    )


def _build_test_config(*, k_values: tuple[int, ...] = (1, 2)) -> PlannerConfig:
    """Build a compact planner config with two orthogonal orientations."""

    return PlannerConfig(
        floorplan_name="phase04-open-square",
        camera_horizontal_resolution_px=1000,
        camera_horizontal_fov_deg=90.0,
        orientation_step_deg=180,
        k_values=k_values,
    )


def _build_sparse_score_artifacts(
    floorplan: FloorPlanInput,
) -> tuple[PlannerConfig, CandidateGenerationArtifacts, SparseScoreArtifacts]:
    """Build a small deterministic sparse score artifact with one unique `K=2` optimum."""

    config = _build_test_config()
    phase01_artifacts = generate_candidate_generation_artifacts(floorplan)

    # Configuration ordinals follow the phase-03 candidate-major, angle-minor order:
    # candidate 0 -> configs 0,1
    # candidate 1 -> configs 2,3
    # candidate 2 -> configs 4,5
    # candidate 3 -> configs 6,7
    configuration_candidate_ordinals = np.repeat(np.arange(4, dtype=np.int32), 2)
    configuration_angle_ordinals = np.tile(np.arange(2, dtype=np.int16), 4)
    score_configuration_offsets = np.array([0, 2, 4, 6, 6, 7, 8, 8, 8], dtype=np.int32)
    score_target_ordinals = np.array([0, 1, 2, 3, 0, 2, 3, 0], dtype=np.int32)
    score_values = np.array([4, 4, 2, 2, 3, 2, 1, 2], dtype=np.int8)
    phase03_artifacts = SparseScoreArtifacts(
        grid_shape=floorplan.shape,
        candidate_count=4,
        open_cell_count=4,
        orientation_angles_deg=np.array([0.0, 180.0], dtype=np.float32),
        configuration_candidate_ordinals=configuration_candidate_ordinals,
        configuration_angle_ordinals=configuration_angle_ordinals,
        score_configuration_offsets=score_configuration_offsets,
        score_target_ordinals=score_target_ordinals,
        score_values=score_values,
    )
    return config, phase01_artifacts, phase03_artifacts


class OptimizationTests(unittest.TestCase):
    """Verify exact threshold-coverage solving, persistence, and validation."""

    def test_solve_optimization_artifacts_enforces_budget_and_orientation_exclusivity(
        self,
    ) -> None:
        floorplan = _build_all_open_square_floorplan()
        config, phase01_artifacts, phase03_artifacts = _build_sparse_score_artifacts(
            floorplan
        )

        artifacts = solve_optimization_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            k=2,
        )

        self.assertEqual(artifacts.grid_shape, floorplan.shape)
        self.assertEqual(artifacts.solved_k, 2)
        self.assertEqual(artifacts.solver_status, "Optimal")
        self.assertAlmostEqual(artifacts.objective_value, 10.0)
        np.testing.assert_array_equal(
            artifacts.selected_configuration_ordinals,
            np.array([0, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            artifacts.selected_candidate_ordinals,
            np.array([0, 1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            artifacts.selected_angle_ordinals,
            np.array([0, 0], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            artifacts.selected_angles_deg,
            np.array([0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            artifacts.final_open_cell_scores,
            np.array([4, 4, 2, 0], dtype=np.int8),
        )
        np.testing.assert_array_equal(
            artifacts.best_configuration_ordinals,
            np.array([0, 0, 2, -1], dtype=np.int32),
        )

    def test_solve_optimization_artifacts_uses_threshold_objective_without_double_counting(
        self,
    ) -> None:
        floorplan = _build_all_open_square_floorplan()
        config, phase01_artifacts, phase03_artifacts = _build_sparse_score_artifacts(
            floorplan
        )

        artifacts = solve_optimization_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            k=2,
        )

        self.assertAlmostEqual(
            artifacts.objective_value,
            float(np.sum(artifacts.final_open_cell_scores, dtype=np.int64)),
        )
        self.assertEqual(int(artifacts.final_open_cell_scores[0]), 4)
        self.assertEqual(
            int(np.sum(artifacts.final_open_cell_scores, dtype=np.int64)),
            10,
        )

    def test_solve_optimization_artifacts_with_k_one_selects_one_configuration(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config, phase01_artifacts, phase03_artifacts = _build_sparse_score_artifacts(
            floorplan
        )

        artifacts = solve_optimization_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            k=1,
        )

        self.assertEqual(len(artifacts.selected_configuration_ordinals), 1)
        self.assertAlmostEqual(artifacts.objective_value, 8.0)
        self.assertEqual(int(np.sum(artifacts.final_open_cell_scores, dtype=np.int64)), 8)
        self.assertEqual(
            int(np.count_nonzero(artifacts.best_configuration_ordinals == -1)),
            2,
        )

    def test_save_load_and_summary_round_trip(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config, phase01_artifacts, phase03_artifacts = _build_sparse_score_artifacts(
            floorplan
        )
        artifacts = solve_optimization_artifacts(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            k=2,
        )

        temp_dir = Path.cwd() / f".tmp-phase04-optimization-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
            artifact_path = temp_dir / "04_solution_k2.npz"
            summary_path = temp_dir / "04_solution_k2_summary.json"
            save_optimization_artifacts(artifact_path, artifacts)
            save_optimization_summary(summary_path, artifacts)
            reloaded = load_optimization_artifacts(artifact_path)
            validate_optimization_artifacts(
                floorplan,
                config,
                phase01_artifacts,
                phase03_artifacts,
                reloaded,
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        np.testing.assert_array_equal(
            reloaded.selected_configuration_ordinals,
            artifacts.selected_configuration_ordinals,
        )
        np.testing.assert_array_equal(
            reloaded.final_open_cell_scores,
            artifacts.final_open_cell_scores,
        )
        self.assertEqual(summary["phase_name"], "optimization")
        self.assertEqual(summary["solved_k"], 2)
        self.assertEqual(summary["selected_camera_count"], 2)
        self.assertEqual(summary["open_cell_count"], 4)
        self.assertAlmostEqual(summary["objective_value"], 10.0)
        self.assertAlmostEqual(summary["total_dori_score"], 10.0)
        self.assertAlmostEqual(summary["coverage_detection_plus_pct"], 75.0)
        self.assertAlmostEqual(summary["coverage_observation_plus_pct"], 75.0)
        self.assertAlmostEqual(summary["coverage_recognition_plus_pct"], 50.0)
        self.assertAlmostEqual(summary["coverage_identification_pct"], 50.0)
        self.assertAlmostEqual(summary["blind_spot_pct"], 25.0)

    def test_precompute_artifact_round_trip_and_ensure_reuse(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        _, phase01_artifacts, phase03_artifacts = _build_sparse_score_artifacts(
            floorplan
        )
        precompute = build_optimization_precompute_artifacts(
            floorplan,
            phase01_artifacts,
            phase03_artifacts,
        )

        temp_dir = Path.cwd() / f".tmp-phase04-precompute-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
            artifact_path = temp_dir / "04_precompute.npz"
            save_optimization_precompute_artifacts(artifact_path, precompute)
            reloaded = load_optimization_precompute_artifacts(artifact_path)
            validate_optimization_precompute_artifacts(
                floorplan,
                phase01_artifacts,
                phase03_artifacts,
                reloaded,
            )
            ensured = ensure_optimization_precompute_artifacts(
                artifact_path,
                floorplan,
                phase01_artifacts,
                phase03_artifacts,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        np.testing.assert_array_equal(
            reloaded.candidate_configuration_offsets,
            precompute.candidate_configuration_offsets,
        )
        for level_index in range(4):
            np.testing.assert_array_equal(
                reloaded.level_offsets[level_index],
                precompute.level_offsets[level_index],
            )
            np.testing.assert_array_equal(
                ensured.level_configuration_ordinals[level_index],
                precompute.level_configuration_ordinals[level_index],
            )

    def test_solve_for_k_values_returns_one_artifact_per_budget(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config, phase01_artifacts, phase03_artifacts = _build_sparse_score_artifacts(
            floorplan
        )

        artifacts_by_k = solve_for_k_values(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            k_values=(1, 2),
        )

        self.assertEqual([artifacts.solved_k for artifacts in artifacts_by_k], [1, 2])
        self.assertAlmostEqual(artifacts_by_k[0].objective_value, 8.0)
        self.assertAlmostEqual(artifacts_by_k[1].objective_value, 10.0)

    def test_solve_for_k_values_writes_count_based_progress_updates(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config, phase01_artifacts, phase03_artifacts = _build_sparse_score_artifacts(
            floorplan
        )
        progress_buffer = StringIO()

        artifacts_by_k = solve_for_k_values(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            k_values=(1, 2),
            progress_writer=progress_buffer,
            progress_total=3,
            progress_completed_before=1,
        )

        progress_output = progress_buffer.getvalue()
        self.assertEqual([artifacts.solved_k for artifacts in artifacts_by_k], [1, 2])
        self.assertIn("k=1 solve starting", progress_output)
        self.assertIn("2/3 solved k=1", progress_output)
        self.assertIn("3/3 solved k=2", progress_output)

    def test_solve_for_k_values_invokes_callback_after_each_solved_budget(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config, phase01_artifacts, phase03_artifacts = _build_sparse_score_artifacts(
            floorplan
        )
        callback_solved_k_values: list[int] = []

        artifacts_by_k = solve_for_k_values(
            floorplan,
            config,
            phase01_artifacts,
            phase03_artifacts,
            k_values=(1, 2),
            on_solved_artifact=lambda artifacts: callback_solved_k_values.append(
                artifacts.solved_k
            ),
        )

        self.assertEqual(callback_solved_k_values, [1, 2])
        self.assertEqual([artifacts.solved_k for artifacts in artifacts_by_k], [1, 2])

    def test_resolve_optimization_artifact_uses_cached_prior_k_warm_start(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config = _build_test_config(k_values=(1, 3))
        progress_buffer = StringIO()
        temp_dir = Path.cwd() / f".tmp-phase04-resolve-single-{uuid.uuid4().hex}"

        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
            _ = resolve_optimization_artifact(
                floorplan,
                config,
                repo_root=temp_dir,
                k=1,
            )
            artifacts = resolve_optimization_artifact(
                floorplan,
                config,
                repo_root=temp_dir,
                k=3,
                progress_writer=progress_buffer,
            )
            self.assertTrue(any(temp_dir.rglob("04_solution_k2.npz")))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(artifacts.solved_k, 3)
        self.assertIn("k=2 solve starting", progress_buffer.getvalue())
        self.assertIn("k=3 solve starting", progress_buffer.getvalue())
        self.assertIn("warm_start=yes", progress_buffer.getvalue())

    def test_resolve_optimization_artifacts_for_k_values_uses_cached_prior_k_warm_start(
        self,
    ) -> None:
        floorplan = _build_all_open_square_floorplan()
        config = _build_test_config(k_values=(1, 3))
        progress_buffer = StringIO()
        temp_dir = Path.cwd() / f".tmp-phase04-resolve-batch-{uuid.uuid4().hex}"

        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
            _ = resolve_optimization_artifacts_for_k_values(
                floorplan,
                config,
                repo_root=temp_dir,
                k_values=(1,),
            )
            artifacts_by_k = resolve_optimization_artifacts_for_k_values(
                floorplan,
                config,
                repo_root=temp_dir,
                k_values=(3,),
                progress_writer=progress_buffer,
            )
            self.assertTrue(any(temp_dir.rglob("04_solution_k2.npz")))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual([artifacts.solved_k for artifacts in artifacts_by_k], [3])
        self.assertIn("k=2 solve starting", progress_buffer.getvalue())
        self.assertIn("k=3 solve starting", progress_buffer.getvalue())
        self.assertIn("warm_start=yes", progress_buffer.getvalue())

    def test_validate_optimization_artifacts_rejects_duplicate_candidate_selection(
        self,
    ) -> None:
        floorplan = _build_all_open_square_floorplan()
        config, phase01_artifacts, phase03_artifacts = _build_sparse_score_artifacts(
            floorplan
        )
        broken = OptimizationArtifacts(
            grid_shape=floorplan.shape,
            open_cell_count=4,
            candidate_count=4,
            configuration_count=8,
            solved_k=2,
            solver_name="PULP_CBC_CMD",
            solver_status="Optimal",
            objective_value=16.0,
            selected_configuration_ordinals=np.array([0, 1], dtype=np.int32),
            selected_candidate_ordinals=np.array([0, 0], dtype=np.int32),
            selected_angle_ordinals=np.array([0, 1], dtype=np.int16),
            selected_angles_deg=np.array([0.0, 180.0], dtype=np.float32),
            final_open_cell_scores=np.array([4, 4, 4, 4], dtype=np.int8),
            best_configuration_ordinals=np.array([0, 0, 1, 1], dtype=np.int32),
        )

        with self.assertRaisesRegex(ValueError, r"No candidate ordinal may appear"):
            validate_optimization_artifacts(
                floorplan,
                config,
                phase01_artifacts,
                phase03_artifacts,
                broken,
            )


if __name__ == "__main__":
    unittest.main()
