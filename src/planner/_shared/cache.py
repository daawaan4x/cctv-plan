"""Deterministic artifact-cache helpers for the planner pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common.floorplan import FloorPlanInput

from .config import PlannerConfig

ARTIFACT_SCHEMA_VERSION = 1


# Cache key derivation
def build_config_fingerprint(floorplan: FloorPlanInput, config: PlannerConfig) -> str:
    """Hash the floor-plan identity and planner config into a short cache key."""

    # The fingerprint deliberately includes both the source floor-plan identity and
    # every planner-facing config field so cached artifacts remain invalidated when
    # either the geometry inputs or the locked scoring/orientation settings change.
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "floorplan_name": floorplan.name,
        "grid_shape": list(floorplan.shape),
        "grid_cell_size_m": floorplan.grid_cell_size_m,
        "config": config.as_dict(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


# Cache directory helpers
def get_artifact_dir(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
    """Return the deterministic artifact directory for one floorplan/config pair."""

    fingerprint = build_config_fingerprint(floorplan, config)
    return repo_root / config.artifact_cache_root / floorplan.name / fingerprint


def ensure_artifact_dir(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
    """Create the deterministic artifact directory when it does not yet exist."""

    artifact_dir = get_artifact_dir(floorplan, config, repo_root=repo_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


# Manifest construction
def build_artifact_manifest(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Build the manifest payload describing one cached planner workspace."""

    artifact_dir = get_artifact_dir(floorplan, config, repo_root=repo_root)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_dir": str(artifact_dir.relative_to(repo_root)),
        "floorplan": {
            "name": floorplan.name,
            "source_path": str(floorplan.source_path),
            "shape": list(floorplan.shape),
            "null_cell_count": floorplan.null_cell_count,
            "open_cell_count": floorplan.open_cell_count,
            "solid_cell_count": floorplan.solid_cell_count,
            "grid_cell_size_m": floorplan.grid_cell_size_m,
        },
        "config": config.as_dict(),
        "fingerprint": build_config_fingerprint(floorplan, config),
    }


# File writers
def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON file with stable formatting and a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_npz(path: Path, **arrays: np.ndarray) -> Path:
    """Write one compressed NumPy `.npz` bundle after ensuring the parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # NumPy accepts arbitrary `**kwargs` here, but pinning the annotation to ndarray
    # keeps the planner artifact layer intentionally narrow and predictable.
    named_arrays: dict[str, Any] = arrays
    np.savez_compressed(path, **named_arrays)
    return path


def write_manifest(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
    """Persist the standard artifact manifest into the workspace cache directory."""

    artifact_dir = ensure_artifact_dir(floorplan, config, repo_root=repo_root)
    manifest = build_artifact_manifest(floorplan, config, repo_root=repo_root)
    return write_json(artifact_dir / "manifest.json", manifest)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "build_artifact_manifest",
    "build_config_fingerprint",
    "ensure_artifact_dir",
    "get_artifact_dir",
    "write_json",
    "write_manifest",
    "write_npz",
]
