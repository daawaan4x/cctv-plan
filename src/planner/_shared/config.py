"""Planner configuration objects aligned with the locked CCTV project model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# Fixed score thresholds
@dataclass(frozen=True, slots=True)
class DoriThresholds:
    """Fixed DORI thresholds expressed in pixels per meter."""

    detection: int = 25
    observation: int = 63
    recognition: int = 125
    identification: int = 250

    def as_dict(self) -> dict[str, int]:
        """Return the thresholds in a JSON-serializable dictionary form."""

        return asdict(self)


# Shared planner configuration
@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Shared planner configuration for floorplan selection, scoring, and caching."""

    floorplan_name: str = "ground-back"
    camera_horizontal_resolution_px: int = 1920
    camera_horizontal_fov_deg: float = 90.0
    orientation_step_deg: int = 15
    candidate_min_spacing_cells: int = 3
    k_values: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    artifact_cache_root: str = "artifacts/planner"
    dori_thresholds: DoriThresholds = field(default_factory=DoriThresholds)

    def __post_init__(self) -> None:
        """Validate the configurable values that shape the planning pipeline."""

        # These checks defend the finite candidate/configuration model described in
        # the handoff document. In particular, `orientation_step_deg` must tile 360
        # exactly so later phases can build the discrete orientation set without
        # carrying angle-wrap edge cases throughout the codebase.
        if self.camera_horizontal_resolution_px <= 0:
            raise ValueError("camera_horizontal_resolution_px must be positive.")
        if self.camera_horizontal_fov_deg <= 0 or self.camera_horizontal_fov_deg >= 360:
            raise ValueError("camera_horizontal_fov_deg must be between 0 and 360.")
        if self.orientation_step_deg <= 0 or 360 % self.orientation_step_deg != 0:
            raise ValueError("orientation_step_deg must be a positive divisor of 360.")
        if self.candidate_min_spacing_cells < 1:
            raise ValueError("candidate_min_spacing_cells must be at least 1.")
        if not self.k_values:
            raise ValueError("k_values must not be empty.")
        if any(k <= 0 for k in self.k_values):
            raise ValueError("Every K value must be positive.")

    @property
    def orientation_angles_deg(self) -> tuple[int, ...]:
        """Expand the locked orientation step into the full angle set."""

        return tuple(range(0, 360, self.orientation_step_deg))

    def as_shared_cache_dict(self) -> dict[str, object]:
        """Return the cache payload shared by phases 01-04 precompute artifacts."""

        return {
            "camera_horizontal_resolution_px": self.camera_horizontal_resolution_px,
            "camera_horizontal_fov_deg": self.camera_horizontal_fov_deg,
            "orientation_step_deg": self.orientation_step_deg,
            "orientation_angles_deg": list(self.orientation_angles_deg),
            "candidate_min_spacing_cells": self.candidate_min_spacing_cells,
            "dori_thresholds": self.dori_thresholds.as_dict(),
        }

    def as_solution_cache_dict(self, k: int) -> dict[str, object]:
        """Return the per-solution cache payload for one scalar camera budget."""

        if k <= 0:
            raise ValueError("Solution cache payload requires a positive k.")

        payload = self.as_shared_cache_dict()
        payload["k"] = int(k)
        return payload

    def as_batch_request_dict(self) -> dict[str, object]:
        """Return the public batch-facing planner request payload."""

        # Materialize tuple fields as lists so manifests and JSON payloads remain
        # stable plain-JSON objects, independent of Python-specific tuple encoding.
        return {
            "floorplan_name": self.floorplan_name,
            "artifact_cache_root": self.artifact_cache_root,
            "shared_config": self.as_shared_cache_dict(),
            "k_values": list(self.k_values),
        }

    def as_dict(self) -> dict[str, object]:
        """Return the full JSON-serializable planner request view."""

        return self.as_batch_request_dict()


__all__ = [
    "DoriThresholds",
    "PlannerConfig",
]
