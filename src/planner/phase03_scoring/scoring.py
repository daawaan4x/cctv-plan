"""DORI scoring math and configuration indexing helpers for phase 03."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput
from src.planner._shared.config import DoriThresholds, PlannerConfig
from src.planner.phase02_visibility import VisibilityArtifacts, get_visible_target_ordinals


@dataclass(frozen=True, slots=True)
class _ScoringConstants:
    """Derived scalar constants reused across the phase-03 scoring loops."""

    grid_cell_size_m: float
    camera_horizontal_resolution_px: float
    camera_horizontal_fov_deg: float
    camera_horizontal_fov_rad: float
    half_fov_deg: float
    detection_distance_max_m: float
    thresholds: DoriThresholds


def _build_scoring_constants(
    config: PlannerConfig,
    grid_cell_size_m: float,
) -> _ScoringConstants:
    """Materialize the scalar scoring constants derived from planner config."""

    camera_horizontal_fov_rad = np.deg2rad(config.camera_horizontal_fov_deg)
    detection_distance_max_m = (
        config.camera_horizontal_resolution_px
        / (
            2.0
            * float(config.dori_thresholds.detection)
            * np.tan(camera_horizontal_fov_rad / 2.0)
        )
    )
    return _ScoringConstants(
        grid_cell_size_m=grid_cell_size_m,
        camera_horizontal_resolution_px=float(config.camera_horizontal_resolution_px),
        camera_horizontal_fov_deg=float(config.camera_horizontal_fov_deg),
        camera_horizontal_fov_rad=float(camera_horizontal_fov_rad),
        half_fov_deg=float(config.camera_horizontal_fov_deg / 2.0),
        detection_distance_max_m=float(detection_distance_max_m),
        thresholds=config.dori_thresholds,
    )


def _build_candidate_base_score_arrays(
    candidate_ordinal: int,
    candidate_coords_rc: NDArray[np.int32],
    open_coords_rc: NDArray[np.int32],
    phase02_artifacts: VisibilityArtifacts,
    scoring_constants: _ScoringConstants,
) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int8]]:
    """Build filtered base target ordinals, angles, and DORI scores for one candidate."""

    los_target_ordinals = get_visible_target_ordinals(phase02_artifacts, candidate_ordinal)
    if len(los_target_ordinals) == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int8),
        )

    candidate_row = int(candidate_coords_rc[candidate_ordinal, 0])
    candidate_col = int(candidate_coords_rc[candidate_ordinal, 1])
    target_coords_rc = open_coords_rc[los_target_ordinals]
    row_deltas = target_coords_rc[:, 0].astype(np.float64) - float(candidate_row)
    col_deltas = target_coords_rc[:, 1].astype(np.float64) - float(candidate_col)
    distances_m = scoring_constants.grid_cell_size_m * np.sqrt(
        np.square(row_deltas) + np.square(col_deltas)
    )

    # Phase 02 already skips self-pairs, but the explicit positive-distance check keeps
    # phase 03 robust against corrupted upstream artifacts and avoids division by zero.
    candidate_mask = (
        (distances_m > 0.0)
        & (distances_m <= scoring_constants.detection_distance_max_m)
    )
    if not np.any(candidate_mask):
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int8),
        )

    filtered_target_ordinals = los_target_ordinals[candidate_mask]
    filtered_row_deltas = row_deltas[candidate_mask]
    filtered_col_deltas = col_deltas[candidate_mask]
    filtered_distances_m = distances_m[candidate_mask]
    filtered_scores = _score_distances_to_dori(
        filtered_distances_m,
        scoring_constants,
    )
    positive_score_mask = filtered_scores > 0
    if not np.any(positive_score_mask):
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int8),
        )

    return (
        filtered_target_ordinals[positive_score_mask].astype(np.int32, copy=False),
        _compute_target_angles_deg(
            filtered_row_deltas[positive_score_mask],
            filtered_col_deltas[positive_score_mask],
        ),
        filtered_scores[positive_score_mask].astype(np.int8, copy=False),
    )


def _compute_target_angles_deg(
    row_deltas: NDArray[np.float64],
    col_deltas: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Convert row and column deltas into the locked planner angle convention."""

    dy_math = -row_deltas
    dx_math = col_deltas
    return (np.degrees(np.arctan2(dy_math, dx_math)) % 360.0).astype(
        np.float32,
        copy=False,
    )


def _build_inside_fov_mask(
    target_angles_deg: NDArray[np.float32],
    orientation_deg: float,
    half_fov_deg: float,
) -> NDArray[np.bool_]:
    """Return the inclusive FOV-membership mask for one discrete orientation."""

    angular_deltas = np.abs(
        ((target_angles_deg.astype(np.float64) - orientation_deg + 180.0) % 360.0)
        - 180.0
    )
    return angular_deltas <= half_fov_deg


def _score_distances_to_dori(
    distances_m: NDArray[np.float64],
    scoring_constants: _ScoringConstants,
) -> NDArray[np.int8]:
    """Convert target distances into categorical DORI scores using the locked model."""

    ppm = scoring_constants.camera_horizontal_resolution_px / (
        2.0
        * distances_m
        * np.tan(scoring_constants.camera_horizontal_fov_rad / 2.0)
    )
    return _score_ppm_to_dori(ppm, scoring_constants.thresholds)


def _score_ppm_to_dori(
    ppm: NDArray[np.float64],
    thresholds: DoriThresholds,
) -> NDArray[np.int8]:
    """Map PPM values into the fixed categorical DORI score bins."""

    scores = np.zeros(ppm.shape, dtype=np.int8)
    scores[ppm >= thresholds.detection] = np.int8(1)
    scores[ppm >= thresholds.observation] = np.int8(2)
    scores[ppm >= thresholds.recognition] = np.int8(3)
    scores[ppm >= thresholds.identification] = np.int8(4)
    return scores


def _require_grid_cell_size_m(floorplan: FloorPlanInput) -> float:
    """Return the positive grid-cell size required for distance-aware scoring."""

    if floorplan.grid_cell_size_m is None or floorplan.grid_cell_size_m <= 0:
        raise ValueError(
            "Phase 03 scoring requires floorplan.grid_cell_size_m to be populated "
            "with a positive meters-per-cell value."
        )
    return float(floorplan.grid_cell_size_m)


def _build_orientation_angles_array(config: PlannerConfig) -> NDArray[np.float32]:
    """Materialize the planner configuration's discrete orientation set."""

    return np.asarray(config.orientation_angles_deg, dtype=np.float32)


def _build_configuration_index_arrays(
    candidate_count: int,
    orientation_count: int,
) -> tuple[NDArray[np.int32], NDArray[np.int16]]:
    """Build deterministic candidate-major configuration index arrays."""

    configuration_candidate_ordinals = np.repeat(
        np.arange(candidate_count, dtype=np.int32),
        orientation_count,
    )
    configuration_angle_ordinals = np.tile(
        np.arange(orientation_count, dtype=np.int16),
        candidate_count,
    )
    return configuration_candidate_ordinals, configuration_angle_ordinals


def _score_one_candidate_target_orientation(
    candidate_row: int,
    candidate_col: int,
    target_row: int,
    target_col: int,
    orientation_deg: float,
    scoring_constants: _ScoringConstants,
) -> int:
    """Recompute one scalar score for sampled semantic validation."""

    row_delta = float(target_row - candidate_row)
    col_delta = float(target_col - candidate_col)
    distance_m = scoring_constants.grid_cell_size_m * np.sqrt(
        (row_delta * row_delta) + (col_delta * col_delta)
    )
    if distance_m <= 0.0 or distance_m > scoring_constants.detection_distance_max_m:
        return 0

    score = int(
        _score_distances_to_dori(
            np.asarray([distance_m], dtype=np.float64),
            scoring_constants,
        )[0]
    )
    if score == 0:
        return 0

    target_angle_deg = float(
        _compute_target_angles_deg(
            np.asarray([row_delta], dtype=np.float64),
            np.asarray([col_delta], dtype=np.float64),
        )[0]
    )
    inside_fov = bool(
        _build_inside_fov_mask(
            np.asarray([target_angle_deg], dtype=np.float32),
            orientation_deg,
            scoring_constants.half_fov_deg,
        )[0]
    )
    return score if inside_fov else 0


__all__ = [
    "_ScoringConstants",
    "_build_candidate_base_score_arrays",
    "_build_configuration_index_arrays",
    "_build_inside_fov_mask",
    "_build_orientation_angles_array",
    "_build_scoring_constants",
    "_compute_target_angles_deg",
    "_require_grid_cell_size_m",
    "_score_distances_to_dori",
    "_score_one_candidate_target_orientation",
]
