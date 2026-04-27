"""Decode validated traced PNG assets into the locked tri-state grid format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..floorplan import NULL_CELL, OPEN_CELL, SOLID_CELL, TracedFloorPlanValidationError

# The traced loader accepts only opaque black and opaque white in addition to
# transparency, so these constants define the full allowed visible palette.
_BLACK_RGB = np.array([0, 0, 0], dtype=np.uint8)
_WHITE_RGB = np.array([255, 255, 255], dtype=np.uint8)


def read_rgba_image(path: Path) -> NDArray[np.uint8]:
    """Load a traced PNG file and normalize it to an RGBA `uint8` array."""

    if not path.exists():
        raise FileNotFoundError(f"Traced floor-plan image does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a traced floor-plan file, got: {path}")

    try:
        from matplotlib import image as mpl_image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to load traced floor-plan PNG assets."
        ) from exc

    image = np.asarray(mpl_image.imread(path))
    return ensure_rgba_uint8(image=image, source_path=path)


def ensure_rgba_uint8(
    *,
    image: NDArray[np.generic],
    source_path: Path,
) -> NDArray[np.uint8]:
    """Convert RGB or RGBA image data into a clipped `uint8` RGBA array."""

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise TracedFloorPlanValidationError(
            f"Expected RGB or RGBA PNG data for {source_path}, got shape {image.shape!r}."
        )

    # Matplotlib may return either float images in `[0, 1]` or integer images,
    # depending on the input file and backend. The loader normalizes both cases
    # into a single `uint8` RGBA representation so the validator and decoder can
    # reason about exact palette matches without dtype-specific branches.
    if np.issubdtype(image.dtype, np.floating):
        float_image = np.asarray(image, dtype=np.float64)
        clipped = np.clip(float_image, 0.0, 1.0)
        image_uint8 = np.rint(clipped * 255.0).astype(np.uint8)
    elif np.issubdtype(image.dtype, np.integer):
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        raise TracedFloorPlanValidationError(
            f"Unsupported image dtype {image.dtype!s} for {source_path}."
        )

    # RGB assets are treated as fully opaque so the downstream palette validator
    # can enforce the same black/white semantics regardless of whether the source
    # file happened to include an explicit alpha channel.
    if image_uint8.shape[2] == 3:
        alpha_channel = np.full(image_uint8.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate((image_uint8, alpha_channel), axis=2)

    return image_uint8


def decode_traced_rgba(rgba_image: NDArray[np.uint8]) -> NDArray[np.int8]:
    """Map transparent, white, and black pixels onto null, open, and solid cells."""

    # The grid defaults to null so that fully transparent pixels and any future
    # non-opaque values rejected earlier both naturally map to out-of-scope cells.
    grid = np.full(rgba_image.shape[:2], NULL_CELL, dtype=np.int8)
    alpha_channel = rgba_image[..., 3]
    opaque_mask = alpha_channel == 255
    opaque_rgb = rgba_image[..., :3]

    black_mask = opaque_mask & np.all(opaque_rgb == _BLACK_RGB, axis=2)
    white_mask = opaque_mask & np.all(opaque_rgb == _WHITE_RGB, axis=2)

    grid[black_mask] = SOLID_CELL
    grid[white_mask] = OPEN_CELL
    return grid


__all__ = [
    "decode_traced_rgba",
    "ensure_rgba_uint8",
    "read_rgba_image",
]
