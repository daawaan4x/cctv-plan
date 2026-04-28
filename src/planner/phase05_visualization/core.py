"""Phase-05 visualization artifacts derived from solved optimization outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pulp

from src.common.floorplan import FloorPlanInput
from src.planner._shared.cache import get_shared_artifact_dir, write_json, write_npz
from src.planner._shared.config import PlannerConfig
from src.planner.phase01_candidate_generation import (
    CandidateGenerationArtifacts,
    resolve_candidate_generation_artifacts,
)
from src.planner.phase04_optimization import (
    OptimizationArtifacts,
    resolve_optimization_artifact,
    resolve_optimization_artifacts_for_k_values,
)

PHASE_NAME = "visualization"
PHASE_ARTIFACT_STEM = "05_metrics"
_VALID_DORI_SCORES = np.asarray([0, 1, 2, 3, 4], dtype=np.int8)


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    """Summary metrics over open cells only for one optimization result."""

    total_dori_score: float
    detection_plus_pct: float
    observation_plus_pct: float
    recognition_plus_pct: float
    identification_pct: float
    blind_spot_pct: float


@dataclass(frozen=True, slots=True)
class VisualizationArtifacts:
    """Deterministic phase-05 arrays and metrics for one solved camera budget."""

    grid_shape: tuple[int, int]
    open_cell_count: int
    candidate_count: int
    configuration_count: int
    solved_k: int
    solver_name: str
    solver_status: str
    selected_camera_count: int
    metrics: CoverageMetrics
    final_open_cell_scores: NDArray[np.int8]
    final_score_grid: NDArray[np.int8]
    blind_spot_mask: NDArray[np.bool_]
    selected_configuration_ordinals: NDArray[np.int32]
    selected_candidate_ordinals: NDArray[np.int32]
    selected_candidate_coords_rc: NDArray[np.int32]
    selected_angle_ordinals: NDArray[np.int16]
    selected_angles_deg: NDArray[np.float32]
    best_configuration_ordinals: NDArray[np.int32]


# Public artifact builders
def build_visualization_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase04_artifacts: OptimizationArtifacts,
) -> VisualizationArtifacts:
    """Build the deterministic phase-05 artifact from phase 01 and phase 04 outputs."""

    _validate_phase_dependencies(floorplan, phase01_artifacts, phase04_artifacts)

    # Phase 04 stores all target-facing vectors in the canonical row-major open-cell
    # ordinal order from phase 01. Phase 05's main job is to scatter those vectors
    # back onto the full floor-plan grid without losing the null/open/solid semantics.
    final_score_grid = _reconstruct_final_score_grid(
        floorplan.shape,
        phase01_artifacts.open_cell_coords_rc,
        phase04_artifacts.final_open_cell_scores,
    )
    blind_spot_mask = floorplan.open_mask & (final_score_grid == 0)
    selected_candidate_coords_rc = _decode_selected_candidate_coords(
        phase01_artifacts.candidate_cell_coords_rc,
        phase04_artifacts.selected_candidate_ordinals,
    )
    metrics = compute_coverage_metrics(phase04_artifacts.final_open_cell_scores)

    artifacts = VisualizationArtifacts(
        grid_shape=floorplan.shape,
        open_cell_count=phase04_artifacts.open_cell_count,
        candidate_count=phase04_artifacts.candidate_count,
        configuration_count=phase04_artifacts.configuration_count,
        solved_k=phase04_artifacts.solved_k,
        solver_name=phase04_artifacts.solver_name,
        solver_status=phase04_artifacts.solver_status,
        selected_camera_count=int(
            len(phase04_artifacts.selected_configuration_ordinals)
        ),
        metrics=metrics,
        final_open_cell_scores=phase04_artifacts.final_open_cell_scores.astype(
            np.int8, copy=False
        ),
        final_score_grid=final_score_grid,
        blind_spot_mask=blind_spot_mask.astype(np.bool_, copy=False),
        selected_configuration_ordinals=phase04_artifacts.selected_configuration_ordinals.astype(
            np.int32, copy=False
        ),
        selected_candidate_ordinals=phase04_artifacts.selected_candidate_ordinals.astype(
            np.int32, copy=False
        ),
        selected_candidate_coords_rc=selected_candidate_coords_rc,
        selected_angle_ordinals=phase04_artifacts.selected_angle_ordinals.astype(
            np.int16, copy=False
        ),
        selected_angles_deg=phase04_artifacts.selected_angles_deg.astype(
            np.float32, copy=False
        ),
        best_configuration_ordinals=phase04_artifacts.best_configuration_ordinals.astype(
            np.int32, copy=False
        ),
    )
    validate_visualization_artifacts(
        floorplan,
        phase01_artifacts,
        phase04_artifacts,
        artifacts,
    )
    return artifacts


def compute_coverage_metrics(
    final_open_cell_scores: NDArray[np.int8],
) -> CoverageMetrics:
    """Compute coverage metrics over open cells only from the final score vector."""

    _validate_final_open_cell_scores(final_open_cell_scores, open_cell_count=None)
    open_cell_count = len(final_open_cell_scores)
    if open_cell_count == 0:
        return CoverageMetrics(
            total_dori_score=0.0,
            detection_plus_pct=0.0,
            observation_plus_pct=0.0,
            recognition_plus_pct=0.0,
            identification_pct=0.0,
            blind_spot_pct=0.0,
        )

    total_dori_score = float(np.sum(final_open_cell_scores, dtype=np.int64))
    return CoverageMetrics(
        total_dori_score=total_dori_score,
        detection_plus_pct=(
            100.0 * float(np.count_nonzero(final_open_cell_scores >= 1))
        )
        / open_cell_count,
        observation_plus_pct=(
            100.0 * float(np.count_nonzero(final_open_cell_scores >= 2))
        )
        / open_cell_count,
        recognition_plus_pct=(
            100.0 * float(np.count_nonzero(final_open_cell_scores >= 3))
        )
        / open_cell_count,
        identification_pct=(
            100.0 * float(np.count_nonzero(final_open_cell_scores >= 4))
        )
        / open_cell_count,
        blind_spot_pct=(100.0 * float(np.count_nonzero(final_open_cell_scores == 0)))
        / open_cell_count,
    )


def validate_visualization_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase04_artifacts: OptimizationArtifacts,
    artifacts: VisualizationArtifacts,
) -> None:
    """Validate structural invariants and phase-05 reconstruction semantics."""

    _validate_phase_dependencies(floorplan, phase01_artifacts, phase04_artifacts)
    _validate_visualization_artifact_structure(artifacts)

    if artifacts.grid_shape != floorplan.shape:
        raise ValueError("Visualization grid_shape does not match floorplan.shape.")
    if artifacts.grid_shape != phase01_artifacts.grid_shape:
        raise ValueError("Visualization grid_shape does not match phase-01 grid_shape.")
    if artifacts.grid_shape != phase04_artifacts.grid_shape:
        raise ValueError("Visualization grid_shape does not match phase-04 grid_shape.")
    if artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Visualization open_cell_count does not match phase-01 open-cell count."
        )
    if artifacts.open_cell_count != phase04_artifacts.open_cell_count:
        raise ValueError(
            "Visualization open_cell_count does not match phase-04 open-cell count."
        )
    if artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Visualization candidate_count does not match phase-01 candidate count."
        )
    if artifacts.candidate_count != phase04_artifacts.candidate_count:
        raise ValueError(
            "Visualization candidate_count does not match phase-04 candidate count."
        )
    if artifacts.configuration_count != phase04_artifacts.configuration_count:
        raise ValueError(
            "Visualization configuration_count does not match phase-04."
        )
    if artifacts.solved_k != phase04_artifacts.solved_k:
        raise ValueError("Visualization solved_k does not match phase-04.")
    if artifacts.solver_name != phase04_artifacts.solver_name:
        raise ValueError("Visualization solver_name does not match phase-04.")
    if artifacts.solver_status != phase04_artifacts.solver_status:
        raise ValueError("Visualization solver_status does not match phase-04.")

    expected_selected_camera_count = len(artifacts.selected_configuration_ordinals)
    if artifacts.selected_camera_count != expected_selected_camera_count:
        raise ValueError(
            "selected_camera_count must equal the selected configuration count."
        )
    if artifacts.selected_camera_count != len(artifacts.selected_candidate_ordinals):
        raise ValueError(
            "selected_camera_count must equal the selected candidate count."
        )
    if artifacts.selected_camera_count != len(artifacts.selected_angle_ordinals):
        raise ValueError(
            "selected_camera_count must equal the selected angle-ordinal count."
        )
    if artifacts.selected_camera_count != len(artifacts.selected_angles_deg):
        raise ValueError("selected_camera_count must equal the selected angle count.")
    if artifacts.selected_candidate_coords_rc.shape[0] != artifacts.selected_camera_count:
        raise ValueError(
            "selected_candidate_coords_rc length must equal selected_camera_count."
        )

    if not np.array_equal(
        artifacts.selected_configuration_ordinals,
        phase04_artifacts.selected_configuration_ordinals,
    ):
        raise ValueError(
            "selected_configuration_ordinals does not match phase-04 selection output."
        )
    if not np.array_equal(
        artifacts.selected_candidate_ordinals,
        phase04_artifacts.selected_candidate_ordinals,
    ):
        raise ValueError(
            "selected_candidate_ordinals does not match phase-04 selection output."
        )
    if not np.array_equal(
        artifacts.selected_angle_ordinals,
        phase04_artifacts.selected_angle_ordinals,
    ):
        raise ValueError(
            "selected_angle_ordinals does not match phase-04 selection output."
        )
    if not np.array_equal(
        artifacts.selected_angles_deg,
        phase04_artifacts.selected_angles_deg,
    ):
        raise ValueError(
            "selected_angles_deg does not match phase-04 selection output."
        )
    if not np.array_equal(
        artifacts.final_open_cell_scores,
        phase04_artifacts.final_open_cell_scores,
    ):
        raise ValueError(
            "final_open_cell_scores does not match the phase-04 final score vector."
        )
    if not np.array_equal(
        artifacts.best_configuration_ordinals,
        phase04_artifacts.best_configuration_ordinals,
    ):
        raise ValueError(
            "best_configuration_ordinals does not match the phase-04 best-selection map."
        )

    expected_score_grid = _reconstruct_final_score_grid(
        floorplan.shape,
        phase01_artifacts.open_cell_coords_rc,
        phase04_artifacts.final_open_cell_scores,
    )
    if not np.array_equal(artifacts.final_score_grid, expected_score_grid):
        raise ValueError(
            "final_score_grid does not match the expected open-cell scatter result."
        )
    if np.any(artifacts.final_score_grid[~floorplan.open_mask] != -1):
        raise ValueError(
            "Non-open cells in final_score_grid must keep the sentinel value -1."
        )

    expected_blind_spot_mask = floorplan.open_mask & (artifacts.final_score_grid == 0)
    if not np.array_equal(artifacts.blind_spot_mask, expected_blind_spot_mask):
        raise ValueError(
            "blind_spot_mask must equal the open-cell score-zero mask exactly."
        )
    if np.any(artifacts.blind_spot_mask[floorplan.null_mask | floorplan.solid_mask]):
        raise ValueError("Null or solid cells must never appear in blind_spot_mask.")

    expected_selected_coords = _decode_selected_candidate_coords(
        phase01_artifacts.candidate_cell_coords_rc,
        phase04_artifacts.selected_candidate_ordinals,
    )
    if not np.array_equal(artifacts.selected_candidate_coords_rc, expected_selected_coords):
        raise ValueError(
            "selected_candidate_coords_rc does not decode from the selected "
            "candidate ordinals."
        )

    expected_metrics = compute_coverage_metrics(artifacts.final_open_cell_scores)
    if not _metrics_equal(artifacts.metrics, expected_metrics):
        raise ValueError("Visualization metrics do not match the final score vector.")
    if not (
        artifacts.metrics.identification_pct
        <= artifacts.metrics.recognition_plus_pct
        <= artifacts.metrics.observation_plus_pct
        <= artifacts.metrics.detection_plus_pct
    ):
        raise ValueError("Coverage percentage metrics must be monotone by DORI level.")
    if not np.isclose(
        artifacts.metrics.blind_spot_pct + artifacts.metrics.detection_plus_pct,
        100.0,
    ):
        raise ValueError(
            "blind_spot_pct and detection_plus_pct must sum to 100 over open cells."
        )


# Artifact persistence
def save_visualization_artifacts(
    artifact_path: Path,
    artifacts: VisualizationArtifacts,
) -> Path:
    """Persist phase-05 artifacts to the deterministic `05_metrics_k<K>.npz` schema."""

    _validate_visualization_artifact_structure(artifacts)
    return write_npz(
        artifact_path,
        grid_shape=np.asarray(artifacts.grid_shape, dtype=np.int32),
        open_cell_count=np.asarray(artifacts.open_cell_count, dtype=np.int32),
        candidate_count=np.asarray(artifacts.candidate_count, dtype=np.int32),
        configuration_count=np.asarray(artifacts.configuration_count, dtype=np.int32),
        solved_k=np.asarray(artifacts.solved_k, dtype=np.int32),
        solver_name=np.asarray(artifacts.solver_name),
        solver_status=np.asarray(artifacts.solver_status),
        selected_camera_count=np.asarray(artifacts.selected_camera_count, dtype=np.int32),
        final_open_cell_scores=artifacts.final_open_cell_scores,
        final_score_grid=artifacts.final_score_grid,
        blind_spot_mask=artifacts.blind_spot_mask,
        selected_configuration_ordinals=artifacts.selected_configuration_ordinals,
        selected_candidate_ordinals=artifacts.selected_candidate_ordinals,
        selected_candidate_coords_rc=artifacts.selected_candidate_coords_rc,
        selected_angle_ordinals=artifacts.selected_angle_ordinals,
        selected_angles_deg=artifacts.selected_angles_deg,
        best_configuration_ordinals=artifacts.best_configuration_ordinals,
    )


def load_visualization_artifacts(
    artifact_path: Path,
) -> VisualizationArtifacts:
    """Load phase-05 artifacts from disk and validate their standalone structure."""

    with np.load(artifact_path) as payload:
        raw_grid_shape = payload["grid_shape"].tolist()
        if len(raw_grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two integers.")
        final_open_cell_scores = payload["final_open_cell_scores"].astype(
            np.int8, copy=False
        )
        artifacts = VisualizationArtifacts(
            grid_shape=(int(raw_grid_shape[0]), int(raw_grid_shape[1])),
            open_cell_count=int(payload["open_cell_count"].item()),
            candidate_count=int(payload["candidate_count"].item()),
            configuration_count=int(payload["configuration_count"].item()),
            solved_k=int(payload["solved_k"].item()),
            solver_name=str(payload["solver_name"].item()),
            solver_status=str(payload["solver_status"].item()),
            selected_camera_count=int(payload["selected_camera_count"].item()),
            metrics=compute_coverage_metrics(final_open_cell_scores),
            final_open_cell_scores=final_open_cell_scores,
            final_score_grid=payload["final_score_grid"].astype(np.int8, copy=False),
            blind_spot_mask=payload["blind_spot_mask"].astype(np.bool_, copy=False),
            selected_configuration_ordinals=payload[
                "selected_configuration_ordinals"
            ].astype(np.int32, copy=False),
            selected_candidate_ordinals=payload["selected_candidate_ordinals"].astype(
                np.int32, copy=False
            ),
            selected_candidate_coords_rc=payload["selected_candidate_coords_rc"].astype(
                np.int32, copy=False
            ),
            selected_angle_ordinals=payload["selected_angle_ordinals"].astype(
                np.int16, copy=False
            ),
            selected_angles_deg=payload["selected_angles_deg"].astype(
                np.float32, copy=False
            ),
            best_configuration_ordinals=payload["best_configuration_ordinals"].astype(
                np.int32, copy=False
            ),
        )

    _validate_visualization_artifact_structure(artifacts)
    return artifacts


def ensure_visualization_artifacts(
    artifact_path: Path,
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase04_artifacts: OptimizationArtifacts,
) -> VisualizationArtifacts:
    """Load a persisted phase-05 artifact when present, otherwise build and save it."""

    if artifact_path.exists():
        artifacts = load_visualization_artifacts(artifact_path)
        validate_visualization_artifacts(
            floorplan,
            phase01_artifacts,
            phase04_artifacts,
            artifacts,
        )
        return artifacts

    artifacts = build_visualization_artifacts(
        floorplan,
        phase01_artifacts,
        phase04_artifacts,
    )
    save_visualization_artifacts(artifact_path, artifacts)
    return artifacts


def save_visualization_summary(
    summary_path: Path,
    artifacts: VisualizationArtifacts,
) -> Path:
    """Persist the human-readable per-`K` visualization summary JSON."""

    _validate_visualization_artifact_structure(artifacts)
    summary_payload = {
        "phase_name": PHASE_NAME,
        "solved_k": artifacts.solved_k,
        "solver_name": artifacts.solver_name,
        "solver_status": artifacts.solver_status,
        "selected_camera_count": artifacts.selected_camera_count,
        "open_cell_count": artifacts.open_cell_count,
        "grid_shape": list(artifacts.grid_shape),
        "total_dori_score": artifacts.metrics.total_dori_score,
        "coverage_detection_plus_pct": artifacts.metrics.detection_plus_pct,
        "coverage_observation_plus_pct": artifacts.metrics.observation_plus_pct,
        "coverage_recognition_plus_pct": artifacts.metrics.recognition_plus_pct,
        "coverage_identification_pct": artifacts.metrics.identification_pct,
        "blind_spot_pct": artifacts.metrics.blind_spot_pct,
        "dori_score_histogram": _build_dori_score_histogram(
            artifacts.final_open_cell_scores
        ),
    }
    return write_json(summary_path, summary_payload)


# Resolver entrypoints
def resolve_visualization_artifact(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    k: int,
    force: bool = False,
    solver: pulp.LpSolver | None = None,
) -> VisualizationArtifacts:
    """Load, validate, or build one canonical cached per-`K` phase-05 artifact."""

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    phase04_artifacts = resolve_optimization_artifact(
        floorplan,
        config,
        repo_root=repo_root,
        k=k,
        force=force,
        solver=solver,
    )
    artifact_dir = get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
    artifact_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{k}.npz"
    summary_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{k}_summary.json"

    if not force and artifact_path.exists():
        try:
            artifacts = load_visualization_artifacts(artifact_path)
            validate_visualization_artifacts(
                floorplan,
                phase01_artifacts,
                phase04_artifacts,
                artifacts,
            )
            if not summary_path.exists():
                save_visualization_summary(summary_path, artifacts)
            return artifacts
        except (KeyError, TypeError, ValueError):
            pass

    artifacts = build_visualization_artifacts(
        floorplan,
        phase01_artifacts,
        phase04_artifacts,
    )
    save_visualization_artifacts(artifact_path, artifacts)
    save_visualization_summary(summary_path, artifacts)
    return artifacts


def resolve_visualization_artifacts_for_k_values(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
    k_values: Sequence[int] | None = None,
    force: bool = False,
    solver: pulp.LpSolver | None = None,
) -> tuple[VisualizationArtifacts, ...]:
    """Resolve phase-05 artifacts for one or more `K` values in request order."""

    requested_k_values = tuple(config.k_values if k_values is None else k_values)
    if not requested_k_values:
        return ()

    phase01_artifacts = resolve_candidate_generation_artifacts(
        floorplan,
        config,
        repo_root=repo_root,
        force=force,
    )
    phase04_artifacts_by_k = resolve_optimization_artifacts_for_k_values(
        floorplan,
        config,
        repo_root=repo_root,
        k_values=requested_k_values,
        force=force,
        solver=solver,
    )
    artifact_dir = get_shared_artifact_dir(floorplan, config, repo_root=repo_root)

    resolved_by_k: dict[int, VisualizationArtifacts] = {}
    for phase04_artifacts in phase04_artifacts_by_k:
        solved_k = int(phase04_artifacts.solved_k)
        artifact_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{solved_k}.npz"
        summary_path = artifact_dir / f"{PHASE_ARTIFACT_STEM}_k{solved_k}_summary.json"

        if not force and artifact_path.exists():
            try:
                artifacts = load_visualization_artifacts(artifact_path)
                validate_visualization_artifacts(
                    floorplan,
                    phase01_artifacts,
                    phase04_artifacts,
                    artifacts,
                )
                if not summary_path.exists():
                    save_visualization_summary(summary_path, artifacts)
                resolved_by_k[solved_k] = artifacts
                continue
            except (KeyError, TypeError, ValueError):
                pass

        artifacts = build_visualization_artifacts(
            floorplan,
            phase01_artifacts,
            phase04_artifacts,
        )
        save_visualization_artifacts(artifact_path, artifacts)
        save_visualization_summary(summary_path, artifacts)
        resolved_by_k[solved_k] = artifacts

    return tuple(resolved_by_k[int(requested_k)] for requested_k in requested_k_values)


# Reconstruction helpers
def _reconstruct_final_score_grid(
    grid_shape: tuple[int, int],
    open_cell_coords_rc: NDArray[np.int32],
    final_open_cell_scores: NDArray[np.int8],
) -> NDArray[np.int8]:
    """Scatter the 1D open-cell score vector back onto the full floor-plan grid."""

    final_score_grid = np.full(grid_shape, -1, dtype=np.int8)
    if len(open_cell_coords_rc) == 0:
        return final_score_grid

    rows = open_cell_coords_rc[:, 0].astype(np.int64, copy=False)
    cols = open_cell_coords_rc[:, 1].astype(np.int64, copy=False)
    final_score_grid[rows, cols] = final_open_cell_scores
    return final_score_grid


def _decode_selected_candidate_coords(
    candidate_cell_coords_rc: NDArray[np.int32],
    selected_candidate_ordinals: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Decode selected candidate ordinals into persisted `(row, col)` coordinates."""

    if len(selected_candidate_ordinals) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return candidate_cell_coords_rc[
        selected_candidate_ordinals.astype(np.int64, copy=False)
    ].astype(np.int32, copy=False)


def _build_dori_score_histogram(
    final_open_cell_scores: NDArray[np.int8],
) -> dict[str, int]:
    """Return the deterministic score histogram used by summaries and notebook checks."""

    counts = np.bincount(
        final_open_cell_scores.astype(np.int64, copy=False),
        minlength=5,
    )
    return {str(score): int(counts[score]) for score in range(5)}


# Validation helpers
def _validate_phase_dependencies(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase04_artifacts: OptimizationArtifacts,
) -> None:
    """Confirm that phase 01 and phase 04 still match the current floor plan."""

    if phase01_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-01 artifacts do not match the current floorplan shape.")
    if phase04_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-04 artifacts do not match the current floorplan shape.")
    if phase04_artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Phase-04 open_cell_count does not match the phase-01 open-cell count."
        )
    if phase04_artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Phase-04 candidate_count does not match the phase-01 candidate count."
        )


def _validate_visualization_artifact_structure(
    artifacts: VisualizationArtifacts,
) -> None:
    """Validate the standalone structure, dtypes, and bounds of phase-05 artifacts."""

    if len(artifacts.grid_shape) != 2:
        raise ValueError("grid_shape must contain exactly two integers.")
    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")
    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.configuration_count < 0:
        raise ValueError("configuration_count must be non-negative.")
    if artifacts.solved_k <= 0:
        raise ValueError("solved_k must be positive.")
    if artifacts.selected_camera_count < 0:
        raise ValueError("selected_camera_count must be non-negative.")

    _validate_final_open_cell_scores(
        artifacts.final_open_cell_scores,
        open_cell_count=artifacts.open_cell_count,
    )
    _validate_selected_configuration_ordinals(
        artifacts.selected_configuration_ordinals,
        configuration_count=artifacts.configuration_count,
        solved_k=artifacts.solved_k,
    )
    _validate_selected_candidate_ordinals(
        artifacts.selected_candidate_ordinals,
        candidate_count=artifacts.candidate_count,
        expected_length=artifacts.selected_camera_count,
    )
    _validate_selected_candidate_coords(
        artifacts.selected_candidate_coords_rc,
        expected_length=artifacts.selected_camera_count,
        grid_shape=artifacts.grid_shape,
    )
    _validate_selected_angle_ordinals(
        artifacts.selected_angle_ordinals,
        expected_length=artifacts.selected_camera_count,
    )
    _validate_selected_angles_deg(
        artifacts.selected_angles_deg,
        expected_length=artifacts.selected_camera_count,
    )
    _validate_best_configuration_ordinals(
        artifacts.best_configuration_ordinals,
        open_cell_count=artifacts.open_cell_count,
        selected_configuration_ordinals=artifacts.selected_configuration_ordinals,
    )
    _validate_final_score_grid(
        artifacts.final_score_grid,
        grid_shape=artifacts.grid_shape,
    )
    _validate_blind_spot_mask(
        artifacts.blind_spot_mask,
        grid_shape=artifacts.grid_shape,
    )


def _validate_final_open_cell_scores(
    final_open_cell_scores: NDArray[np.int8],
    *,
    open_cell_count: int | None,
) -> None:
    """Validate dtype, shape, length, and admissible DORI scores for open cells."""

    if final_open_cell_scores.dtype != np.int8:
        raise TypeError("final_open_cell_scores must use dtype np.int8.")
    if final_open_cell_scores.ndim != 1:
        raise ValueError("final_open_cell_scores must be a 1D array.")
    if open_cell_count is not None and len(final_open_cell_scores) != open_cell_count:
        raise ValueError("final_open_cell_scores length must equal open_cell_count.")
    if final_open_cell_scores.size and not np.isin(
        final_open_cell_scores,
        _VALID_DORI_SCORES,
    ).all():
        raise ValueError("final_open_cell_scores must contain only the scores 0..4.")


def _validate_selected_configuration_ordinals(
    selected_configuration_ordinals: NDArray[np.int32],
    *,
    configuration_count: int,
    solved_k: int,
) -> None:
    """Validate dtype, ordering, uniqueness, bounds, and budget for config ordinals."""

    if selected_configuration_ordinals.dtype != np.int32:
        raise TypeError("selected_configuration_ordinals must use dtype np.int32.")
    if selected_configuration_ordinals.ndim != 1:
        raise ValueError("selected_configuration_ordinals must be a 1D array.")
    if len(selected_configuration_ordinals) > solved_k:
        raise ValueError(
            "The selected configuration count must not exceed the solved camera budget."
        )
    if len(selected_configuration_ordinals) > 1 and not np.all(
        selected_configuration_ordinals[:-1] < selected_configuration_ordinals[1:]
    ):
        raise ValueError(
            "selected_configuration_ordinals must be strictly increasing and unique."
        )
    if selected_configuration_ordinals.size and (
        selected_configuration_ordinals[0] < 0
        or selected_configuration_ordinals[-1] >= configuration_count
    ):
        raise ValueError(
            "selected_configuration_ordinals contains an out-of-range configuration "
            "ordinal."
        )


def _validate_selected_candidate_ordinals(
    selected_candidate_ordinals: NDArray[np.int32],
    *,
    candidate_count: int,
    expected_length: int,
) -> None:
    """Validate dtype, shape, length, bounds, and no-duplicate candidate selection."""

    if selected_candidate_ordinals.dtype != np.int32:
        raise TypeError("selected_candidate_ordinals must use dtype np.int32.")
    if selected_candidate_ordinals.ndim != 1:
        raise ValueError("selected_candidate_ordinals must be a 1D array.")
    if len(selected_candidate_ordinals) != expected_length:
        raise ValueError(
            "selected_candidate_ordinals length must equal selected_camera_count."
        )
    if selected_candidate_ordinals.size and (
        np.min(selected_candidate_ordinals) < 0
        or np.max(selected_candidate_ordinals) >= candidate_count
    ):
        raise ValueError(
            "selected_candidate_ordinals contains an out-of-range candidate ordinal."
        )
    if len(np.unique(selected_candidate_ordinals)) != len(selected_candidate_ordinals):
        raise ValueError("No candidate ordinal may appear more than once.")


def _validate_selected_candidate_coords(
    selected_candidate_coords_rc: NDArray[np.int32],
    *,
    expected_length: int,
    grid_shape: tuple[int, int],
) -> None:
    """Validate dtype, shape, length, and in-grid bounds for selected coordinates."""

    if selected_candidate_coords_rc.dtype != np.int32:
        raise TypeError("selected_candidate_coords_rc must use dtype np.int32.")
    if selected_candidate_coords_rc.ndim != 2 or selected_candidate_coords_rc.shape[1] != 2:
        raise ValueError("selected_candidate_coords_rc must have shape (N, 2).")
    if selected_candidate_coords_rc.shape[0] != expected_length:
        raise ValueError(
            "selected_candidate_coords_rc length must equal selected_camera_count."
        )
    if selected_candidate_coords_rc.size == 0:
        return

    rows = selected_candidate_coords_rc[:, 0]
    cols = selected_candidate_coords_rc[:, 1]
    if (
        np.min(rows) < 0
        or np.max(rows) >= grid_shape[0]
        or np.min(cols) < 0
        or np.max(cols) >= grid_shape[1]
    ):
        raise ValueError("selected_candidate_coords_rc contains an out-of-grid cell.")


def _validate_selected_angle_ordinals(
    selected_angle_ordinals: NDArray[np.int16],
    *,
    expected_length: int,
) -> None:
    """Validate dtype, shape, and length for selected angle ordinals."""

    if selected_angle_ordinals.dtype != np.int16:
        raise TypeError("selected_angle_ordinals must use dtype np.int16.")
    if selected_angle_ordinals.ndim != 1:
        raise ValueError("selected_angle_ordinals must be a 1D array.")
    if len(selected_angle_ordinals) != expected_length:
        raise ValueError(
            "selected_angle_ordinals length must equal selected_camera_count."
        )


def _validate_selected_angles_deg(
    selected_angles_deg: NDArray[np.float32],
    *,
    expected_length: int,
) -> None:
    """Validate dtype, shape, and length for selected physical angles."""

    if selected_angles_deg.dtype != np.float32:
        raise TypeError("selected_angles_deg must use dtype np.float32.")
    if selected_angles_deg.ndim != 1:
        raise ValueError("selected_angles_deg must be a 1D array.")
    if len(selected_angles_deg) != expected_length:
        raise ValueError("selected_angles_deg length must equal selected_camera_count.")


def _validate_best_configuration_ordinals(
    best_configuration_ordinals: NDArray[np.int32],
    *,
    open_cell_count: int,
    selected_configuration_ordinals: NDArray[np.int32],
) -> None:
    """Validate target-to-best-configuration bookkeeping against selection bounds."""

    if best_configuration_ordinals.dtype != np.int32:
        raise TypeError("best_configuration_ordinals must use dtype np.int32.")
    if best_configuration_ordinals.ndim != 1:
        raise ValueError("best_configuration_ordinals must be a 1D array.")
    if len(best_configuration_ordinals) != open_cell_count:
        raise ValueError("best_configuration_ordinals length must equal open_cell_count.")

    selected_lookup = set(selected_configuration_ordinals.tolist())
    for configuration_ordinal in best_configuration_ordinals:
        if int(configuration_ordinal) == -1:
            continue
        if int(configuration_ordinal) not in selected_lookup:
            raise ValueError(
                "best_configuration_ordinals must point only to selected "
                "configuration ordinals or -1."
            )


def _validate_final_score_grid(
    final_score_grid: NDArray[np.int8],
    *,
    grid_shape: tuple[int, int],
) -> None:
    """Validate dtype and shape for the reconstructed full-grid score surface."""

    if final_score_grid.dtype != np.int8:
        raise TypeError("final_score_grid must use dtype np.int8.")
    if final_score_grid.shape != grid_shape:
        raise ValueError("final_score_grid shape must equal grid_shape.")


def _validate_blind_spot_mask(
    blind_spot_mask: NDArray[np.bool_],
    *,
    grid_shape: tuple[int, int],
) -> None:
    """Validate dtype and shape for the explicit blind-spot mask."""

    if blind_spot_mask.dtype != np.bool_:
        raise TypeError("blind_spot_mask must use dtype np.bool_.")
    if blind_spot_mask.shape != grid_shape:
        raise ValueError("blind_spot_mask shape must equal grid_shape.")


def _metrics_equal(left: CoverageMetrics, right: CoverageMetrics) -> bool:
    """Return whether two metric bundles match within float tolerance."""

    return bool(
        np.isclose(left.total_dori_score, right.total_dori_score)
        and np.isclose(left.detection_plus_pct, right.detection_plus_pct)
        and np.isclose(left.observation_plus_pct, right.observation_plus_pct)
        and np.isclose(left.recognition_plus_pct, right.recognition_plus_pct)
        and np.isclose(left.identification_pct, right.identification_pct)
        and np.isclose(left.blind_spot_pct, right.blind_spot_pct)
    )


__all__ = [
    "CoverageMetrics",
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
    "VisualizationArtifacts",
    "build_visualization_artifacts",
    "compute_coverage_metrics",
    "ensure_visualization_artifacts",
    "load_visualization_artifacts",
    "resolve_visualization_artifact",
    "resolve_visualization_artifacts_for_k_values",
    "save_visualization_artifacts",
    "save_visualization_summary",
    "validate_visualization_artifacts",
]
