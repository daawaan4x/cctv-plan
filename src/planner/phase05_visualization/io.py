from __future__ import annotations

from pathlib import Path

import numpy as np

from src.planner._shared.cache import write_json, write_npz

from .artifacts import VisualizationArtifacts
from .constants import PHASE_NAME
from .metrics import compute_coverage_metrics
from .transforms import _build_dori_score_histogram
from .validation import _validate_visualization_artifact_structure


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
