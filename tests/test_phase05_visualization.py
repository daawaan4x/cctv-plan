"""Tests for phase-05 visualization artifacts, summaries, and plotting helpers."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import matplotlib
import numpy as np

from src.common.floorplan import FloorPlanInput, NULL_CELL, OPEN_CELL, SOLID_CELL
from src.planner._shared.cache import get_shared_artifact_dir
from src.planner._shared.config import PlannerConfig
from src.planner.phase01_candidate_generation import generate_candidate_generation_artifacts
from src.planner.phase04_optimization import OptimizationArtifacts
from src.planner.phase05_visualization import (
    build_visualization_artifacts,
    ensure_visualization_artifacts,
    load_visualization_artifacts,
    plot_blind_spot_map,
    plot_dori_map,
    plot_metric_summary_table,
    plot_selected_configurations,
    resolve_visualization_artifact,
    resolve_visualization_artifacts_for_k_values,
    save_visualization_artifacts,
    save_visualization_summary,
    validate_visualization_artifacts,
)

matplotlib.use("Agg")


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
    """Build a tiny open square used by the synthetic phase-05 tests."""

    return _build_floorplan(
        "phase05-open-square",
        [
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
        ],
    )


def _build_test_config() -> PlannerConfig:
    """Build a compact planner config with two orthogonal orientations."""

    return PlannerConfig(
        floorplan_name="phase05-open-square",
        camera_horizontal_resolution_px=1000,
        camera_horizontal_fov_deg=90.0,
        orientation_step_deg=180,
        k_values=(1, 2),
    )


def _build_phase04_artifacts() -> tuple[FloorPlanInput, PlannerConfig, OptimizationArtifacts]:
    """Build a deterministic synthetic phase-04 artifact for phase-05 unit tests."""

    floorplan = _build_all_open_square_floorplan()
    config = _build_test_config()
    artifacts = OptimizationArtifacts(
        grid_shape=floorplan.shape,
        open_cell_count=4,
        candidate_count=4,
        configuration_count=8,
        solved_k=2,
        solver_name="PULP_CBC_CMD",
        solver_status="Optimal",
        objective_value=10.0,
        selected_configuration_ordinals=np.array([0, 2], dtype=np.int32),
        selected_candidate_ordinals=np.array([0, 1], dtype=np.int32),
        selected_angle_ordinals=np.array([0, 0], dtype=np.int16),
        selected_angles_deg=np.array([0.0, 0.0], dtype=np.float32),
        final_open_cell_scores=np.array([4, 4, 2, 0], dtype=np.int8),
        best_configuration_ordinals=np.array([0, 0, 2, -1], dtype=np.int32),
    )
    return floorplan, config, artifacts


class VisualizationTests(unittest.TestCase):
    """Verify phase-05 reconstruction, persistence, plotting, and resolver behavior."""

    def test_build_visualization_artifacts_reconstructs_grid_and_metrics(self) -> None:
        floorplan, _, phase04_artifacts = _build_phase04_artifacts()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)

        artifacts = build_visualization_artifacts(
            floorplan,
            phase01_artifacts,
            phase04_artifacts,
        )

        np.testing.assert_array_equal(
            artifacts.final_score_grid,
            np.array(
                [
                    [-1, -1, -1, -1],
                    [-1, 4, 4, -1],
                    [-1, 2, 0, -1],
                    [-1, -1, -1, -1],
                ],
                dtype=np.int8,
            ),
        )
        np.testing.assert_array_equal(
            artifacts.blind_spot_mask,
            np.array(
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, True, False],
                    [False, False, False, False],
                ],
                dtype=np.bool_,
            ),
        )
        np.testing.assert_array_equal(
            artifacts.selected_candidate_coords_rc,
            np.array([[1, 1], [1, 2]], dtype=np.int32),
        )
        self.assertAlmostEqual(artifacts.metrics.total_dori_score, 10.0)
        self.assertAlmostEqual(artifacts.metrics.detection_plus_pct, 75.0)
        self.assertAlmostEqual(artifacts.metrics.observation_plus_pct, 75.0)
        self.assertAlmostEqual(artifacts.metrics.recognition_plus_pct, 50.0)
        self.assertAlmostEqual(artifacts.metrics.identification_pct, 50.0)
        self.assertAlmostEqual(artifacts.metrics.blind_spot_pct, 25.0)

    def test_save_load_summary_and_ensure_round_trip(self) -> None:
        floorplan, _, phase04_artifacts = _build_phase04_artifacts()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
        artifacts = build_visualization_artifacts(
            floorplan,
            phase01_artifacts,
            phase04_artifacts,
        )

        temp_dir = Path.cwd() / f".tmp-phase05-visualization-{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=False)
            artifact_path = temp_dir / "05_metrics_k2.npz"
            summary_path = temp_dir / "05_metrics_k2_summary.json"
            save_visualization_artifacts(artifact_path, artifacts)
            save_visualization_summary(summary_path, artifacts)
            reloaded = load_visualization_artifacts(artifact_path)
            validate_visualization_artifacts(
                floorplan,
                phase01_artifacts,
                phase04_artifacts,
                reloaded,
            )
            ensured = ensure_visualization_artifacts(
                artifact_path,
                floorplan,
                phase01_artifacts,
                phase04_artifacts,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        np.testing.assert_array_equal(reloaded.final_score_grid, artifacts.final_score_grid)
        np.testing.assert_array_equal(
            ensured.selected_candidate_coords_rc,
            artifacts.selected_candidate_coords_rc,
        )
        self.assertEqual(summary["phase_name"], "visualization")
        self.assertEqual(summary["solved_k"], 2)
        self.assertEqual(summary["selected_camera_count"], 2)
        self.assertEqual(summary["open_cell_count"], 4)
        self.assertEqual(summary["grid_shape"], [4, 4])
        self.assertEqual(
            summary["dori_score_histogram"],
            {"0": 1, "1": 0, "2": 1, "3": 0, "4": 2},
        )

    def test_plotting_helpers_render_without_error(self) -> None:
        from matplotlib import pyplot as plt

        floorplan, _, phase04_artifacts = _build_phase04_artifacts()
        phase01_artifacts = generate_candidate_generation_artifacts(floorplan)
        artifacts = build_visualization_artifacts(
            floorplan,
            phase01_artifacts,
            phase04_artifacts,
        )

        figure, axes = plt.subplots(2, 2, figsize=(10, 8))
        try:
            dori_axis = plot_dori_map(floorplan, artifacts, ax=axes[0, 0])
            blind_axis = plot_blind_spot_map(floorplan, artifacts, ax=axes[0, 1])
            selected_axis = plot_selected_configurations(
                floorplan,
                artifacts,
                ax=axes[1, 0],
            )
            summary_axis = plot_metric_summary_table(artifacts, ax=axes[1, 1])
        finally:
            plt.close(figure)

        self.assertEqual(dori_axis.get_xlabel(), "Column")
        self.assertEqual(blind_axis.get_ylabel(), "Row")
        self.assertIn("Selected Cameras", selected_axis.get_title())
        self.assertIn("Coverage Summary", summary_axis.get_title())

    def test_resolve_visualization_artifacts_persists_phase_outputs(self) -> None:
        floorplan = _build_all_open_square_floorplan()
        config = _build_test_config()
        repo_root = Path.cwd() / f".tmp-phase05-resolve-{uuid.uuid4().hex}"
        k1_artifact_exists = False
        k1_summary_exists = False
        k2_artifact_exists = False
        k2_summary_exists = False
        try:
            repo_root.mkdir(parents=True, exist_ok=False)
            single = resolve_visualization_artifact(
                floorplan,
                config,
                repo_root=repo_root,
                k=1,
            )
            batch = resolve_visualization_artifacts_for_k_values(
                floorplan,
                config,
                repo_root=repo_root,
                k_values=(1, 2),
            )
            artifact_dir = get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
            k1_artifact_exists = (artifact_dir / "05_metrics_k1.npz").exists()
            k1_summary_exists = (artifact_dir / "05_metrics_k1_summary.json").exists()
            k2_artifact_exists = (artifact_dir / "05_metrics_k2.npz").exists()
            k2_summary_exists = (artifact_dir / "05_metrics_k2_summary.json").exists()
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

        self.assertEqual(single.solved_k, 1)
        self.assertEqual([artifacts.solved_k for artifacts in batch], [1, 2])
        self.assertTrue(k1_artifact_exists)
        self.assertTrue(k1_summary_exists)
        self.assertTrue(k2_artifact_exists)
        self.assertTrue(k2_summary_exists)


if __name__ == "__main__":
    unittest.main()
