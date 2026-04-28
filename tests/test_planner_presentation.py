"""Tests for planner presentation helpers used by the main notebook."""

from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib
import numpy as np

from src.common.floorplan import FloorPlanInput, NULL_CELL, OPEN_CELL, SOLID_CELL
from src.planner._shared.config import PlannerConfig
from src.planner.phase01_candidate_generation import generate_candidate_generation_artifacts
from src.planner.phase05_visualization import CoverageMetrics, VisualizationArtifacts
from src.planner.presentation import (
    PresentationFloorplanResult,
    build_aligned_budget_rows,
    build_budget_sweep_rows,
    build_candidate_summary_rows,
    build_floorplan_catalog_rows,
    compute_common_delta_k_range,
    plot_candidate_set_comparison,
    require_floorplan_min_k,
    select_showcase_k_values,
)

matplotlib.use("Agg")


def _build_floorplan(name: str, *, min_k: int | None) -> FloorPlanInput:
    """Build a small synthetic tri-state floor plan for presentation tests."""

    grid = np.array(
        [
            [NULL_CELL, NULL_CELL, NULL_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, OPEN_CELL, OPEN_CELL, NULL_CELL],
            [NULL_CELL, SOLID_CELL, SOLID_CELL, NULL_CELL],
        ],
        dtype=np.int8,
    )
    return FloorPlanInput(
        name=name,
        source_path=Path(f"{name}.png"),
        grid=grid,
        height=int(grid.shape[0]),
        width=int(grid.shape[1]),
        null_cell_count=int(np.count_nonzero(grid == NULL_CELL)),
        open_cell_count=int(np.count_nonzero(grid == OPEN_CELL)),
        solid_cell_count=int(np.count_nonzero(grid == SOLID_CELL)),
        grid_cell_size_m=0.5,
        min_k=min_k,
    )


def _build_visualization_artifacts(
    floorplan: FloorPlanInput,
    *,
    solved_k: int,
    total_dori_score: float,
    detection_plus_pct: float,
    observation_plus_pct: float,
    recognition_plus_pct: float,
    identification_pct: float,
    blind_spot_pct: float,
) -> VisualizationArtifacts:
    """Build a minimal consistent phase-05 artifact for notebook-helper tests."""

    final_score_grid = np.full(floorplan.shape, -1, dtype=np.int8)
    final_score_grid[1, 1] = 4
    final_score_grid[1, 2] = 2
    final_score_grid[2, 1] = 1
    final_score_grid[2, 2] = 0
    blind_spot_mask = floorplan.open_mask & (final_score_grid == 0)
    return VisualizationArtifacts(
        grid_shape=floorplan.shape,
        open_cell_count=floorplan.open_cell_count,
        candidate_count=4,
        configuration_count=8,
        solved_k=solved_k,
        solver_name="HiGHS",
        solver_status="Optimal",
        selected_camera_count=1,
        metrics=CoverageMetrics(
            total_dori_score=total_dori_score,
            detection_plus_pct=detection_plus_pct,
            observation_plus_pct=observation_plus_pct,
            recognition_plus_pct=recognition_plus_pct,
            identification_pct=identification_pct,
            blind_spot_pct=blind_spot_pct,
        ),
        final_open_cell_scores=np.array([4, 2, 1, 0], dtype=np.int8),
        final_score_grid=final_score_grid,
        blind_spot_mask=blind_spot_mask,
        selected_configuration_ordinals=np.array([0], dtype=np.int32),
        selected_candidate_ordinals=np.array([0], dtype=np.int32),
        selected_candidate_coords_rc=np.array([[1, 1]], dtype=np.int32),
        selected_angle_ordinals=np.array([0], dtype=np.int16),
        selected_angles_deg=np.array([0.0], dtype=np.float32),
        best_configuration_ordinals=np.array([0, 0, 0, -1], dtype=np.int32),
    )


def _build_result(
    name: str,
    *,
    min_k: int | None,
    solved_ks: tuple[int, ...],
) -> PresentationFloorplanResult:
    """Build one notebook-ready result bundle backed by synthetic artifacts."""

    floorplan = _build_floorplan(name, min_k=min_k)
    config = PlannerConfig(
        floorplan_name=name,
        orientation_step_deg=180,
        k_values=solved_ks,
    )
    phase01_artifacts = generate_candidate_generation_artifacts(floorplan, config)
    visualization_artifacts_by_k = tuple(
        _build_visualization_artifacts(
            floorplan,
            solved_k=solved_k,
            total_dori_score=float(10 * solved_k),
            detection_plus_pct=60.0 + solved_k,
            observation_plus_pct=50.0 + solved_k,
            recognition_plus_pct=40.0 + solved_k,
            identification_pct=30.0 + solved_k,
            blind_spot_pct=40.0 - solved_k,
        )
        for solved_k in solved_ks
    )
    return PresentationFloorplanResult(
        floorplan=floorplan,
        config=config,
        phase01_artifacts=phase01_artifacts,
        visualization_artifacts_by_k=visualization_artifacts_by_k,
    )


def _row_int(row: dict[str, object], key: str) -> int:
    """Read one presentation row field as an integer for typed test assertions."""

    value = row[key]
    if not isinstance(value, int):
        raise TypeError(f"Expected row[{key!r}] to be an int, got {type(value).__name__}.")
    return int(value)


class PlannerPresentationTests(unittest.TestCase):
    """Verify the notebook-facing presentation data assembly helpers."""

    def test_build_floorplan_catalog_rows_includes_min_k(self) -> None:
        result = _build_result("catalog-demo", min_k=10, solved_ks=(10, 11))

        rows = build_floorplan_catalog_rows((result,))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "catalog-demo")
        self.assertEqual(rows[0]["grid_shape"], "4 x 4")
        self.assertEqual(rows[0]["open_cells"], 4)
        self.assertEqual(rows[0]["solid_cells"], 2)
        self.assertEqual(rows[0]["null_cells"], 10)
        self.assertEqual(rows[0]["min_k"], 10)

    def test_build_candidate_summary_rows_reports_reduced_set(self) -> None:
        result = _build_result("candidate-demo", min_k=10, solved_ks=(10,))

        rows = build_candidate_summary_rows((result,))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "candidate-demo")
        self.assertGreaterEqual(
            _row_int(rows[0], "eligible_candidates"),
            _row_int(rows[0], "optimization_candidates"),
        )

    def test_compute_common_delta_k_range_uses_shared_offsets(self) -> None:
        first = _build_result("first", min_k=10, solved_ks=(10, 11, 12))
        second = _build_result("second", min_k=12, solved_ks=(12, 13, 14))

        delta_k_values = compute_common_delta_k_range((first, second))

        self.assertEqual(delta_k_values, (0, 1, 2))

    def test_build_aligned_budget_rows_uses_floorplan_relative_min_k(self) -> None:
        first = _build_result("first", min_k=10, solved_ks=(10, 11, 12))
        second = _build_result("second", min_k=12, solved_ks=(12, 13, 14))

        rows = build_aligned_budget_rows((first, second))

        self.assertEqual(len(rows), 6)
        first_baseline = next(
            row for row in rows if row["name"] == "first" and _row_int(row, "delta_k") == 0
        )
        second_baseline = next(
            row for row in rows if row["name"] == "second" and _row_int(row, "delta_k") == 0
        )
        self.assertEqual(first_baseline["k"], 10)
        self.assertEqual(second_baseline["k"], 12)

    def test_build_budget_sweep_rows_preserves_absolute_k_values(self) -> None:
        result = _build_result("absolute-demo", min_k=10, solved_ks=(10, 11, 12))

        rows = build_budget_sweep_rows((result,))

        self.assertEqual([_row_int(row, "k") for row in rows], [10, 11, 12])
        self.assertEqual([_row_int(row, "delta_k") for row in rows], [0, 1, 2])

    def test_require_floorplan_min_k_rejects_missing_metadata(self) -> None:
        floorplan = _build_floorplan("missing-min-k", min_k=None)

        with self.assertRaisesRegex(
            ValueError,
            r"missing metadata.min_k",
        ):
            require_floorplan_min_k(floorplan)

    def test_select_showcase_k_values_spans_min_to_max(self) -> None:
        result = _build_result("showcase-demo", min_k=10, solved_ks=(10, 11, 12, 13, 14, 15))

        showcase_k_values = select_showcase_k_values(result, count=3)

        self.assertEqual(showcase_k_values[0], 10)
        self.assertEqual(showcase_k_values[-1], 15)
        self.assertEqual(len(showcase_k_values), 3)

    def test_plot_candidate_set_comparison_renders_side_by_side(self) -> None:
        from matplotlib import pyplot as plt

        result = _build_result("candidate-plot-demo", min_k=10, solved_ks=(10,))
        figure, axes = plt.subplots(1, 2, figsize=(8, 4))
        try:
            eligible_axis, final_axis = plot_candidate_set_comparison(
                result.floorplan,
                result.phase01_artifacts,
                axes=axes,
            )
        finally:
            plt.close(figure)

        self.assertIn("eligible", eligible_axis.get_title().lower())
        self.assertIn("optimization", final_axis.get_title().lower())
        self.assertEqual(eligible_axis.get_xlabel(), "Column")
        self.assertEqual(final_axis.get_ylabel(), "Row")


if __name__ == "__main__":
    unittest.main()
