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
def build_shared_config_fingerprint(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
) -> str:
    """Hash the shared floor-plan and non-`K` planner inputs into one cache key."""

    # The shared fingerprint intentionally excludes the public `k_values` batch tuple
    # so phase 01 through phase 04 precompute artifacts remain reusable when only the
    # requested solve batch changes.
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "floorplan_name": floorplan.name,
        "grid_shape": list(floorplan.shape),
        "grid_cell_size_m": floorplan.grid_cell_size_m,
        "config": config.as_shared_cache_dict(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def build_solution_fingerprint(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    k: int,
) -> str:
    """Hash one scalar `k` on top of the shared artifact payload."""

    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "floorplan_name": floorplan.name,
        "grid_shape": list(floorplan.shape),
        "grid_cell_size_m": floorplan.grid_cell_size_m,
        "config": config.as_solution_cache_dict(k),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def build_config_fingerprint(floorplan: FloorPlanInput, config: PlannerConfig) -> str:
    """Backward-compatible alias for the shared artifact fingerprint."""

    return build_shared_config_fingerprint(floorplan, config)


# Cache directory helpers
def get_shared_artifact_dir(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
    """Return the deterministic shared artifact directory for one workspace."""

    fingerprint = build_shared_config_fingerprint(floorplan, config)
    return repo_root / config.artifact_cache_root / floorplan.name / fingerprint


def get_artifact_dir(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
    """Backward-compatible alias for the shared artifact directory."""

    return get_shared_artifact_dir(floorplan, config, repo_root=repo_root)


def ensure_shared_artifact_dir(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
    """Create the deterministic shared artifact directory when it is missing."""

    artifact_dir = get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def ensure_artifact_dir(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
    """Backward-compatible alias for the shared artifact directory creator."""

    return ensure_shared_artifact_dir(floorplan, config, repo_root=repo_root)


# Manifest construction
def build_artifact_manifest(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Build the manifest payload describing one cached planner workspace."""

    artifact_dir = get_shared_artifact_dir(floorplan, config, repo_root=repo_root)
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
        "batch_request": config.as_batch_request_dict(),
        "shared_config": config.as_shared_cache_dict(),
        "shared_fingerprint": build_shared_config_fingerprint(floorplan, config),
        "solution_fingerprints": {
            str(k): build_solution_fingerprint(floorplan, config, k=k)
            for k in config.k_values
        },
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

    artifact_dir = ensure_shared_artifact_dir(floorplan, config, repo_root=repo_root)
    manifest = build_artifact_manifest(floorplan, config, repo_root=repo_root)
    return write_json(artifact_dir / "manifest.json", manifest)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "build_artifact_manifest",
    "build_config_fingerprint",
    "build_shared_config_fingerprint",
    "build_solution_fingerprint",
    "ensure_artifact_dir",
    "ensure_shared_artifact_dir",
    "get_artifact_dir",
    "get_shared_artifact_dir",
    "write_json",
    "write_manifest",
    "write_npz",
]
