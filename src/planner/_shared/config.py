from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class DoriThresholds:
    detection: int = 25
    observation: int = 63
    recognition: int = 125
    identification: int = 250

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    floorplan_name: str = "ground-back"
    camera_horizontal_resolution_px: int = 1920
    camera_horizontal_fov_deg: float = 90.0
    orientation_step_deg: int = 15
    k_values: tuple[int, ...] = (1, 2, 3, 4, 5)
    artifact_cache_root: str = "artifacts/planner"
    dori_thresholds: DoriThresholds = field(default_factory=DoriThresholds)

    def __post_init__(self) -> None:
        if self.camera_horizontal_resolution_px <= 0:
            raise ValueError("camera_horizontal_resolution_px must be positive.")
        if self.camera_horizontal_fov_deg <= 0 or self.camera_horizontal_fov_deg >= 360:
            raise ValueError("camera_horizontal_fov_deg must be between 0 and 360.")
        if self.orientation_step_deg <= 0 or 360 % self.orientation_step_deg != 0:
            raise ValueError("orientation_step_deg must be a positive divisor of 360.")
        if not self.k_values:
            raise ValueError("k_values must not be empty.")
        if any(k <= 0 for k in self.k_values):
            raise ValueError("Every K value must be positive.")

    @property
    def orientation_angles_deg(self) -> tuple[int, ...]:
        return tuple(range(0, 360, self.orientation_step_deg))

    def as_dict(self) -> dict[str, object]:
        return {
            "floorplan_name": self.floorplan_name,
            "camera_horizontal_resolution_px": self.camera_horizontal_resolution_px,
            "camera_horizontal_fov_deg": self.camera_horizontal_fov_deg,
            "orientation_step_deg": self.orientation_step_deg,
            "orientation_angles_deg": list(self.orientation_angles_deg),
            "k_values": list(self.k_values),
            "artifact_cache_root": self.artifact_cache_root,
            "dori_thresholds": self.dori_thresholds.as_dict(),
        }


__all__ = [
    "DoriThresholds",
    "PlannerConfig",
]
