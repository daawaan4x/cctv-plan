"""Validate that traced floor-plan PNGs use only the locked palette semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..floorplan import TracedFloorPlanValidationError

# Only fully opaque black/white pixels are meaningful floor-plan content.
_BLACK_RGB = np.array([0, 0, 0], dtype=np.uint8)
_WHITE_RGB = np.array([255, 255, 255], dtype=np.uint8)


def validate_traced_palette(
    rgba_image: NDArray[np.uint8],
    *,
    source_path: Path | None = None,
) -> None:
    """Reject partial alpha and unexpected opaque colors in traced PNG assets."""

    if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise TracedFloorPlanValidationError("Expected an RGBA image array with shape (H, W, 4).")

    # Partial transparency is rejected first because it is visually ambiguous:
    # unlike fully transparent pixels, semi-transparent pixels do not have a stable
    # interpretation as null, open, or solid cells in the locked tri-state model.
    alpha_channel = rgba_image[..., 3]
    partial_alpha_mask = (alpha_channel != 0) & (alpha_channel != 255)
    if np.any(partial_alpha_mask):
        row, col = _first_true_index(partial_alpha_mask)
        pixel = _pixel_to_rgba_tuple(rgba_image[row, col])
        raise TracedFloorPlanValidationError(
            _format_validation_error(
                "partial transparency is not allowed",
                source_path=source_path,
                row=row,
                col=col,
                pixel=pixel,
            )
        )

    opaque_mask = alpha_channel == 255
    opaque_rgb = rgba_image[..., :3]
    black_mask = np.all(opaque_rgb == _BLACK_RGB, axis=2)
    white_mask = np.all(opaque_rgb == _WHITE_RGB, axis=2)
    # At this point every surviving opaque pixel must be either black or white.
    # Anything else would create an implicit fourth semantic state, which the
    # planner intentionally forbids.
    invalid_opaque_mask = opaque_mask & ~(black_mask | white_mask)
    if np.any(invalid_opaque_mask):
        row, col = _first_true_index(invalid_opaque_mask)
        pixel = _pixel_to_rgba_tuple(rgba_image[row, col])
        raise TracedFloorPlanValidationError(
            _format_validation_error(
                "unexpected opaque color",
                source_path=source_path,
                row=row,
                col=col,
                pixel=pixel,
            )
        )


def _first_true_index(mask: NDArray[np.bool_]) -> tuple[int, int]:
    """Return the first `(row, col)` coordinate where a boolean mask is true."""

    row, col = np.argwhere(mask)[0]
    return int(row), int(col)


def _pixel_to_rgba_tuple(pixel: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    """Convert one RGBA pixel slice into a plain Python integer tuple."""

    if pixel.shape != (4,):
        raise TracedFloorPlanValidationError(
            f"Expected an RGBA pixel with shape (4,), got {pixel.shape!r}."
        )

    rgba_pixel = cast(tuple[Any, ...], tuple(pixel.tolist()))
    return (
        int(rgba_pixel[0]),
        int(rgba_pixel[1]),
        int(rgba_pixel[2]),
        int(rgba_pixel[3]),
    )


def _format_validation_error(
    message: str,
    *,
    source_path: Path | None,
    row: int,
    col: int,
    pixel: tuple[int, int, int, int],
) -> str:
    """Build a location-aware validation error message for one offending pixel."""

    source_label = str(source_path) if source_path is not None else "<in-memory image>"
    return (
        f"{message} in traced floor-plan image {source_label} at row {row}, col {col}: "
        f"pixel={pixel}"
    )


__all__ = ["validate_traced_palette"]
