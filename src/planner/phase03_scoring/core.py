"""Phase-03 scoring helpers for sparse orientation-aware DORI score generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput
from src.planner._shared.cache import write_npz
from src.planner._shared.config import DoriThresholds, PlannerConfig
from src.planner.phase01_candidate_generation import CandidateGenerationArtifacts
from src.planner.phase02_visibility import (
    VisibilityArtifacts,
    get_visible_target_ordinals,
)

PHASE_NAME = "scoring"
PHASE_ARTIFACT_STEM = "03_sparse_scores"

_MAX_SEMANTIC_VALIDATION_PAIRS: Final[int] = 4096


@dataclass(frozen=True, slots=True)
class SparseScoreArtifacts:
    """Sparse configuration-major DORI score arrays for valid non-zero pairs."""

    grid_shape: tuple[int, int]
    candidate_count: int
    open_cell_count: int
    orientation_angles_deg: NDArray[np.float32]
    configuration_candidate_ordinals: NDArray[np.int32]
    configuration_angle_ordinals: NDArray[np.int16]
    score_configuration_offsets: NDArray[np.int32]
    score_target_ordinals: NDArray[np.int32]
    score_values: NDArray[np.int8]


# Artifact generation
def generate_sparse_score_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase02_artifacts: VisibilityArtifacts,
) -> SparseScoreArtifacts:
    """Build deterministic sparse score artifacts for all candidate orientations."""

    grid_cell_size_m = _require_grid_cell_size_m(floorplan)
    orientation_angles_deg = _build_orientation_angles_array(config)
    _validate_phase_dependencies(
        floorplan,
        config,
        phase01_artifacts,
        phase02_artifacts,
        orientation_angles_deg,
    )

    candidate_count = len(phase01_artifacts.candidate_cell_indices)
    open_coords_rc = phase01_artifacts.open_cell_coords_rc
    candidate_coords_rc = phase01_artifacts.candidate_cell_coords_rc
    orientation_count = len(orientation_angles_deg)
    configuration_candidate_ordinals, configuration_angle_ordinals = (
        _build_configuration_index_arrays(candidate_count, orientation_count)
    )

    scoring_constants = _build_scoring_constants(config, grid_cell_size_m)
    score_counts = np.zeros(candidate_count * orientation_count, dtype=np.int32)

    for candidate_ordinal in range(candidate_count):
        base_target_ordinals, base_target_angles_deg, base_target_scores = (
            _build_candidate_base_score_arrays(
                candidate_ordinal,
                candidate_coords_rc,
                open_coords_rc,
                phase02_artifacts,
                scoring_constants,
            )
        )
        if len(base_target_ordinals) == 0:
            continue

        configuration_base = candidate_ordinal * orientation_count
        for angle_ordinal, orientation_deg in enumerate(orientation_angles_deg):
            inside_fov_mask = _build_inside_fov_mask(
                base_target_angles_deg,
                float(orientation_deg),
                scoring_constants.half_fov_deg,
            )
            score_counts[configuration_base + angle_ordinal] = np.int32(
                np.count_nonzero(inside_fov_mask)
            )

    score_configuration_offsets = _build_offsets_from_counts(score_counts)
    score_target_ordinals = np.empty(
        int(score_configuration_offsets[-1]),
        dtype=np.int32,
    )
    score_values = np.empty(int(score_configuration_offsets[-1]), dtype=np.int8)

    for candidate_ordinal in range(candidate_count):
        base_target_ordinals, base_target_angles_deg, base_target_scores = (
            _build_candidate_base_score_arrays(
                candidate_ordinal,
                candidate_coords_rc,
                open_coords_rc,
                phase02_artifacts,
                scoring_constants,
            )
        )
        configuration_base = candidate_ordinal * orientation_count
        if len(base_target_ordinals) == 0:
            continue

        for angle_ordinal, orientation_deg in enumerate(orientation_angles_deg):
            configuration_ordinal = configuration_base + angle_ordinal
            write_start = int(score_configuration_offsets[configuration_ordinal])
            write_stop = int(score_configuration_offsets[configuration_ordinal + 1])
            if write_start == write_stop:
                continue

            inside_fov_mask = _build_inside_fov_mask(
                base_target_angles_deg,
                float(orientation_deg),
                scoring_constants.half_fov_deg,
            )
            configuration_target_ordinals = base_target_ordinals[inside_fov_mask]
            configuration_scores = base_target_scores[inside_fov_mask]
            score_target_ordinals[write_start:write_stop] = configuration_target_ordinals
            score_values[write_start:write_stop] = configuration_scores

    artifacts = SparseScoreArtifacts(
        grid_shape=floorplan.shape,
        candidate_count=candidate_count,
        open_cell_count=len(phase01_artifacts.open_cell_indices),
        orientation_angles_deg=orientation_angles_deg,
        configuration_candidate_ordinals=configuration_candidate_ordinals,
        configuration_angle_ordinals=configuration_angle_ordinals,
        score_configuration_offsets=score_configuration_offsets,
        score_target_ordinals=score_target_ordinals,
        score_values=score_values,
    )
    validate_sparse_score_artifacts(
        floorplan,
        config,
        phase01_artifacts,
        phase02_artifacts,
        artifacts,
    )
    return artifacts


def validate_sparse_score_artifacts(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase02_artifacts: VisibilityArtifacts,
    artifacts: SparseScoreArtifacts,
) -> None:
    """Validate structural invariants plus sampled score semantics for phase 03."""

    grid_cell_size_m = _require_grid_cell_size_m(floorplan)
    expected_orientation_angles_deg = _build_orientation_angles_array(config)
    _validate_phase_dependencies(
        floorplan,
        config,
        phase01_artifacts,
        phase02_artifacts,
        expected_orientation_angles_deg,
    )

    if artifacts.grid_shape != floorplan.shape:
        raise ValueError("Sparse-score grid_shape does not match floorplan.shape.")
    if artifacts.grid_shape != phase01_artifacts.grid_shape:
        raise ValueError("Sparse-score grid_shape does not match phase-01 grid_shape.")
    if artifacts.grid_shape != phase02_artifacts.grid_shape:
        raise ValueError("Sparse-score grid_shape does not match phase-02 grid_shape.")
    if artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Sparse-score candidate_count does not match the phase-01 candidate count."
        )
    if artifacts.candidate_count != phase02_artifacts.candidate_count:
        raise ValueError(
            "Sparse-score candidate_count does not match the phase-02 candidate count."
        )
    if artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Sparse-score open_cell_count does not match the phase-01 open-cell count."
        )
    if artifacts.open_cell_count != phase02_artifacts.open_cell_count:
        raise ValueError(
            "Sparse-score open_cell_count does not match the phase-02 open-cell count."
        )

    _validate_orientation_angles_array(
        artifacts.orientation_angles_deg,
        expected_orientation_angles_deg,
    )
    _validate_configuration_index_arrays(
        artifacts.configuration_candidate_ordinals,
        artifacts.configuration_angle_ordinals,
        candidate_count=artifacts.candidate_count,
        orientation_count=len(artifacts.orientation_angles_deg),
    )
    _validate_offsets(
        artifacts.score_configuration_offsets,
        expected_configuration_count=len(artifacts.configuration_candidate_ordinals),
        expected_total=len(artifacts.score_target_ordinals),
    )
    _validate_target_ordinals(
        artifacts.score_target_ordinals,
        open_cell_count=artifacts.open_cell_count,
    )
    _validate_score_values(
        artifacts.score_values,
        expected_total=len(artifacts.score_target_ordinals),
    )

    for configuration_ordinal in range(len(artifacts.configuration_candidate_ordinals)):
        target_ordinals = get_configuration_target_ordinals(
            artifacts,
            configuration_ordinal,
        )
        score_values = get_configuration_dori_scores(artifacts, configuration_ordinal)
        _validate_strictly_increasing(
            target_ordinals,
            configuration_ordinal=configuration_ordinal,
        )
        if len(score_values) != len(target_ordinals):
            raise ValueError(
                "Each configuration score slice must match its target-ordinal slice "
                "length."
            )

    scoring_constants = _build_scoring_constants(config, grid_cell_size_m)
    _validate_sampled_pair_semantics(
        phase01_artifacts,
        phase02_artifacts,
        artifacts,
        scoring_constants,
    )


# Artifact persistence
def save_sparse_score_artifacts(
    artifact_path: Path,
    artifacts: SparseScoreArtifacts,
) -> Path:
    """Persist phase-03 artifacts to the deterministic `03_sparse_scores.npz` schema."""

    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        candidate_count=np.asarray(artifacts.candidate_count, dtype=np.int32),
        open_cell_count=np.asarray(artifacts.open_cell_count, dtype=np.int32),
        orientation_angles_deg=artifacts.orientation_angles_deg,
        configuration_candidate_ordinals=artifacts.configuration_candidate_ordinals,
        configuration_angle_ordinals=artifacts.configuration_angle_ordinals,
        score_configuration_offsets=artifacts.score_configuration_offsets,
        score_target_ordinals=artifacts.score_target_ordinals,
        score_values=artifacts.score_values,
    )


def load_sparse_score_artifacts(
    artifact_path: Path,
) -> SparseScoreArtifacts:
    """Load phase-03 artifacts from disk and perform structural validation."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        artifacts = SparseScoreArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            candidate_count=int(payload["candidate_count"].item()),
            open_cell_count=int(payload["open_cell_count"].item()),
            orientation_angles_deg=payload["orientation_angles_deg"].astype(
                np.float32,
                copy=False,
            ),
            configuration_candidate_ordinals=payload[
                "configuration_candidate_ordinals"
            ].astype(np.int32, copy=False),
            configuration_angle_ordinals=payload["configuration_angle_ordinals"].astype(
                np.int16,
                copy=False,
            ),
            score_configuration_offsets=payload["score_configuration_offsets"].astype(
                np.int32,
                copy=False,
            ),
            score_target_ordinals=payload["score_target_ordinals"].astype(
                np.int32,
                copy=False,
            ),
            score_values=payload["score_values"].astype(np.int8, copy=False),
        )

    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")

    _validate_orientation_angles_array(
        artifacts.orientation_angles_deg,
        artifacts.orientation_angles_deg,
    )
    _validate_configuration_index_arrays(
        artifacts.configuration_candidate_ordinals,
        artifacts.configuration_angle_ordinals,
        candidate_count=artifacts.candidate_count,
        orientation_count=len(artifacts.orientation_angles_deg),
    )
    _validate_offsets(
        artifacts.score_configuration_offsets,
        expected_configuration_count=len(artifacts.configuration_candidate_ordinals),
        expected_total=len(artifacts.score_target_ordinals),
    )
    _validate_target_ordinals(
        artifacts.score_target_ordinals,
        open_cell_count=artifacts.open_cell_count,
    )
    _validate_score_values(
        artifacts.score_values,
        expected_total=len(artifacts.score_target_ordinals),
    )
    return artifacts


# Artifact query helpers
def get_configuration_target_ordinals(
    artifacts: SparseScoreArtifacts,
    configuration_ordinal: int,
) -> NDArray[np.int32]:
    """Return the sparse target-ordinal slice for one configuration ordinal."""

    _validate_configuration_ordinal(
        configuration_ordinal,
        configuration_count=len(artifacts.configuration_candidate_ordinals),
    )
    start = int(artifacts.score_configuration_offsets[configuration_ordinal])
    stop = int(artifacts.score_configuration_offsets[configuration_ordinal + 1])
    return artifacts.score_target_ordinals[start:stop]


def get_configuration_dori_scores(
    artifacts: SparseScoreArtifacts,
    configuration_ordinal: int,
) -> NDArray[np.int8]:
    """Return the sparse DORI-score slice for one configuration ordinal."""

    _validate_configuration_ordinal(
        configuration_ordinal,
        configuration_count=len(artifacts.configuration_candidate_ordinals),
    )
    start = int(artifacts.score_configuration_offsets[configuration_ordinal])
    stop = int(artifacts.score_configuration_offsets[configuration_ordinal + 1])
    return artifacts.score_values[start:stop]


def decode_configuration_ordinal(
    artifacts: SparseScoreArtifacts,
    configuration_ordinal: int,
) -> tuple[int, int, float]:
    """Decode one configuration ordinal into candidate ordinal, angle ordinal, and angle."""

    _validate_configuration_ordinal(
        configuration_ordinal,
        configuration_count=len(artifacts.configuration_candidate_ordinals),
    )
    candidate_ordinal = int(
        artifacts.configuration_candidate_ordinals[configuration_ordinal]
    )
    angle_ordinal = int(artifacts.configuration_angle_ordinals[configuration_ordinal])
    angle_deg = float(artifacts.orientation_angles_deg[angle_ordinal])
    return candidate_ordinal, angle_ordinal, angle_deg


# Shared scoring constants
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


# Scoring helpers
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


# Validation helpers
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


def _validate_phase_dependencies(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase02_artifacts: VisibilityArtifacts,
    orientation_angles_deg: NDArray[np.float32],
) -> None:
    """Confirm that phase-01 and phase-02 artifacts still match the current inputs."""

    _ = config
    if phase01_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-01 artifacts do not match the current floorplan shape.")
    if phase02_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-02 artifacts do not match the current floorplan shape.")
    if phase02_artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Phase-02 candidate_count does not match the phase-01 candidate count."
        )
    if phase02_artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Phase-02 open_cell_count does not match the phase-01 open-cell count."
        )
    if orientation_angles_deg.ndim != 1 or len(orientation_angles_deg) == 0:
        raise ValueError("orientation_angles_deg must contain at least one angle.")


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


def _validate_orientation_angles_array(
    orientation_angles_deg: NDArray[np.float32],
    expected_orientation_angles_deg: NDArray[np.float32],
) -> None:
    """Validate dtype, shape, ordering, and config alignment for orientations."""

    if orientation_angles_deg.dtype != np.float32:
        raise TypeError("orientation_angles_deg must use dtype np.float32.")
    if orientation_angles_deg.ndim != 1:
        raise ValueError("orientation_angles_deg must be a 1D array.")
    if len(orientation_angles_deg) == 0:
        raise ValueError("orientation_angles_deg must not be empty.")
    if len(orientation_angles_deg) > 1 and not np.all(
        orientation_angles_deg[:-1] < orientation_angles_deg[1:]
    ):
        raise ValueError("orientation_angles_deg must be strictly increasing.")
    if not np.array_equal(orientation_angles_deg, expected_orientation_angles_deg):
        raise ValueError(
            "orientation_angles_deg does not match the planner configuration's "
            "discrete orientation set."
        )


def _validate_configuration_index_arrays(
    configuration_candidate_ordinals: NDArray[np.int32],
    configuration_angle_ordinals: NDArray[np.int16],
    *,
    candidate_count: int,
    orientation_count: int,
) -> None:
    """Validate deterministic configuration-to-candidate and angle mappings."""

    expected_configuration_count = candidate_count * orientation_count
    if configuration_candidate_ordinals.dtype != np.int32:
        raise TypeError("configuration_candidate_ordinals must use dtype np.int32.")
    if configuration_candidate_ordinals.ndim != 1:
        raise ValueError("configuration_candidate_ordinals must be a 1D array.")
    if len(configuration_candidate_ordinals) != expected_configuration_count:
        raise ValueError(
            "configuration_candidate_ordinals length must equal candidate_count * "
            "orientation_count."
        )
    if configuration_candidate_ordinals.size and (
        configuration_candidate_ordinals[0] < 0
        or configuration_candidate_ordinals[-1] >= candidate_count
    ):
        raise ValueError(
            "configuration_candidate_ordinals contains an out-of-range candidate "
            "ordinal."
        )

    if configuration_angle_ordinals.dtype != np.int16:
        raise TypeError("configuration_angle_ordinals must use dtype np.int16.")
    if configuration_angle_ordinals.ndim != 1:
        raise ValueError("configuration_angle_ordinals must be a 1D array.")
    if len(configuration_angle_ordinals) != expected_configuration_count:
        raise ValueError(
            "configuration_angle_ordinals length must equal candidate_count * "
            "orientation_count."
        )
    if configuration_angle_ordinals.size and (
        configuration_angle_ordinals[0] < 0
        or np.max(configuration_angle_ordinals) >= orientation_count
    ):
        raise ValueError(
            "configuration_angle_ordinals contains an out-of-range angle ordinal."
        )

    expected_candidate_ordinals, expected_angle_ordinals = (
        _build_configuration_index_arrays(candidate_count, orientation_count)
    )
    if not np.array_equal(
        configuration_candidate_ordinals,
        expected_candidate_ordinals,
    ):
        raise ValueError(
            "configuration_candidate_ordinals does not match the required "
            "candidate-major deterministic ordering."
        )
    if not np.array_equal(configuration_angle_ordinals, expected_angle_ordinals):
        raise ValueError(
            "configuration_angle_ordinals does not match the required angle-minor "
            "deterministic ordering."
        )


def _validate_offsets(
    score_configuration_offsets: NDArray[np.int32],
    *,
    expected_configuration_count: int,
    expected_total: int,
) -> None:
    """Validate CSR-style score offsets for configuration-major sparse slices."""

    if score_configuration_offsets.dtype != np.int32:
        raise TypeError("score_configuration_offsets must use dtype np.int32.")
    if score_configuration_offsets.ndim != 1:
        raise ValueError("score_configuration_offsets must be a 1D array.")
    if len(score_configuration_offsets) != expected_configuration_count + 1:
        raise ValueError(
            "score_configuration_offsets length must be configuration_count + 1."
        )
    if score_configuration_offsets[0] != 0:
        raise ValueError("score_configuration_offsets must start at 0.")
    if len(score_configuration_offsets) > 1 and np.any(
        score_configuration_offsets[:-1] > score_configuration_offsets[1:]
    ):
        raise ValueError("score_configuration_offsets must be monotonic nondecreasing.")
    if int(score_configuration_offsets[-1]) != expected_total:
        raise ValueError(
            "score_configuration_offsets final offset must match the paired sparse "
            "array lengths."
        )


def _validate_target_ordinals(
    score_target_ordinals: NDArray[np.int32],
    *,
    open_cell_count: int,
) -> None:
    """Validate the flattened sparse target-ordinal array."""

    if score_target_ordinals.dtype != np.int32:
        raise TypeError("score_target_ordinals must use dtype np.int32.")
    if score_target_ordinals.ndim != 1:
        raise ValueError("score_target_ordinals must be a 1D array.")
    if score_target_ordinals.size and (
        score_target_ordinals[0] < 0 or score_target_ordinals[-1] >= open_cell_count
    ):
        raise ValueError("score_target_ordinals contains an out-of-range target ordinal.")


def _validate_score_values(
    score_values: NDArray[np.int8],
    *,
    expected_total: int,
) -> None:
    """Validate dtype, shape, length, and allowed values for sparse DORI scores."""

    if score_values.dtype != np.int8:
        raise TypeError("score_values must use dtype np.int8.")
    if score_values.ndim != 1:
        raise ValueError("score_values must be a 1D array.")
    if len(score_values) != expected_total:
        raise ValueError(
            "score_values length must match the sparse target-ordinal array length."
        )
    if score_values.size and not np.isin(score_values, np.array([1, 2, 3, 4])).all():
        raise ValueError("score_values must contain only the categorical scores 1..4.")


def _validate_configuration_ordinal(
    configuration_ordinal: int,
    *,
    configuration_count: int,
) -> None:
    """Ensure one public query request references an existing configuration ordinal."""

    if configuration_ordinal < 0 or configuration_ordinal >= configuration_count:
        raise IndexError(
            "configuration_ordinal is out of range for the sparse-score artifact."
        )


def _validate_strictly_increasing(
    target_ordinals: NDArray[np.int32],
    *,
    configuration_ordinal: int,
) -> None:
    """Confirm that one configuration slice stays sorted and duplicate-free."""

    if len(target_ordinals) > 1 and np.any(target_ordinals[:-1] >= target_ordinals[1:]):
        raise ValueError(
            "score_target_ordinals slice for configuration ordinal "
            f"{configuration_ordinal} must be strictly increasing."
        )


def _validate_sampled_pair_semantics(
    phase01_artifacts: CandidateGenerationArtifacts,
    phase02_artifacts: VisibilityArtifacts,
    artifacts: SparseScoreArtifacts,
    scoring_constants: _ScoringConstants,
) -> None:
    """Recompute a bounded sample of stored pairs to confirm scoring semantics."""

    total_pairs = len(artifacts.score_target_ordinals)
    if total_pairs == 0:
        return

    sample_indices = _choose_sample_indices(
        total_pairs,
        max_samples=min(_MAX_SEMANTIC_VALIDATION_PAIRS, total_pairs),
    )
    open_coords_rc = phase01_artifacts.open_cell_coords_rc
    candidate_coords_rc = phase01_artifacts.candidate_cell_coords_rc

    for flat_index in sample_indices:
        configuration_ordinal = int(
            np.searchsorted(
                artifacts.score_configuration_offsets,
                flat_index,
                side="right",
            )
            - 1
        )
        target_ordinal = int(artifacts.score_target_ordinals[flat_index])
        stored_score = int(artifacts.score_values[flat_index])
        candidate_ordinal, angle_ordinal, orientation_deg = decode_configuration_ordinal(
            artifacts,
            configuration_ordinal,
        )
        visible_target_ordinals = get_visible_target_ordinals(
            phase02_artifacts,
            candidate_ordinal,
        )
        if not _sorted_contains(visible_target_ordinals, target_ordinal):
            raise ValueError(
                "Stored sparse-score pair is not present in the phase-02 LOS-positive "
                "visibility slice for its candidate ordinal."
            )

        candidate_row = int(candidate_coords_rc[candidate_ordinal, 0])
        candidate_col = int(candidate_coords_rc[candidate_ordinal, 1])
        target_row = int(open_coords_rc[target_ordinal, 0])
        target_col = int(open_coords_rc[target_ordinal, 1])
        recomputed_score = _score_one_candidate_target_orientation(
            candidate_row,
            candidate_col,
            target_row,
            target_col,
            orientation_deg,
            scoring_constants,
        )
        if recomputed_score != stored_score:
            raise ValueError(
                "Stored sparse-score value does not match a recomputed score for the "
                "sampled configuration-target pair."
            )
        if angle_ordinal < 0 or angle_ordinal >= len(artifacts.orientation_angles_deg):
            raise ValueError("Decoded angle ordinal fell outside the orientation array.")


def _choose_sample_indices(length: int, max_samples: int) -> NDArray[np.int64]:
    """Choose evenly spaced flat-pair indices for semantic revalidation."""

    if length == 0 or max_samples <= 0:
        return np.empty(0, dtype=np.int64)
    if length <= max_samples:
        return np.arange(length, dtype=np.int64)
    sample_indices = np.linspace(0, length - 1, num=max_samples, dtype=np.int64)
    return np.unique(sample_indices)


def _sorted_contains(values: NDArray[np.int32], needle: int) -> bool:
    """Return whether one sorted ordinal array contains the requested value."""

    position = int(np.searchsorted(values, needle))
    return position < len(values) and int(values[position]) == needle


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


def _build_offsets_from_counts(counts: NDArray[np.int32]) -> NDArray[np.int32]:
    """Convert per-configuration non-zero pair counts into CSR-style offsets."""

    if counts.dtype != np.int32:
        raise TypeError("counts must use dtype np.int32.")
    offsets = np.zeros(len(counts) + 1, dtype=np.int32)
    np.cumsum(counts, dtype=np.int32, out=offsets[1:])
    return offsets


__all__ = [
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
    "SparseScoreArtifacts",
    "decode_configuration_ordinal",
    "generate_sparse_score_artifacts",
    "get_configuration_dori_scores",
    "get_configuration_target_ordinals",
    "load_sparse_score_artifacts",
    "save_sparse_score_artifacts",
    "validate_sparse_score_artifacts",
]
