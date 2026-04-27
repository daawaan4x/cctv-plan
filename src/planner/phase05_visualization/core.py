from __future__ import annotations

from dataclasses import dataclass

PHASE_NAME = "visualization"
PHASE_ARTIFACT_STEM = "05_metrics"


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    total_dori_score: float
    detection_plus_pct: float
    observation_plus_pct: float
    recognition_plus_pct: float
    identification_pct: float
    blind_spot_pct: float


__all__ = [
    "CoverageMetrics",
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
]
