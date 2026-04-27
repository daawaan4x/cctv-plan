from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common.floorplan import FloorPlanInput

from .config import PlannerConfig

ARTIFACT_SCHEMA_VERSION = 1


def build_config_fingerprint(floorplan: FloorPlanInput, config: PlannerConfig) -> str:
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "floorplan_name": floorplan.name,
        "grid_shape": list(floorplan.shape),
        "grid_cell_size_m": floorplan.grid_cell_size_m,
        "config": config.as_dict(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def get_artifact_dir(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
    fingerprint = build_config_fingerprint(floorplan, config)
    return repo_root / config.artifact_cache_root / floorplan.name / fingerprint


def ensure_artifact_dir(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
    artifact_dir = get_artifact_dir(floorplan, config, repo_root=repo_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def build_artifact_manifest(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> dict[str, Any]:
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


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_npz(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    named_arrays: dict[str, Any] = arrays
    np.savez_compressed(path, **named_arrays)
    return path


def write_manifest(
    floorplan: FloorPlanInput,
    config: PlannerConfig,
    *,
    repo_root: Path,
) -> Path:
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
