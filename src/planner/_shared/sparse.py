"""Shared sparse-array helpers for planner phase artifacts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_offsets_from_counts(counts: NDArray[np.int32]) -> NDArray[np.int32]:
    """Convert per-entry counts into CSR-style offsets."""

    if counts.dtype != np.int32:
        raise TypeError("counts must use dtype np.int32.")

    offsets = np.zeros(len(counts) + 1, dtype=np.int32)
    np.cumsum(counts, dtype=np.int32, out=offsets[1:])
    return offsets


def choose_sample_indices(length: int, max_samples: int) -> NDArray[np.int64]:
    """Choose evenly spaced flat-pair indices for bounded semantic checks."""

    if length == 0 or max_samples <= 0:
        return np.empty(0, dtype=np.int64)
    if length <= max_samples:
        return np.arange(length, dtype=np.int64)
    sample_indices = np.linspace(0, length - 1, num=max_samples, dtype=np.int64)
    return np.unique(sample_indices)


__all__ = [
    "build_offsets_from_counts",
    "choose_sample_indices",
]
