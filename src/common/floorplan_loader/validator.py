from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..floorplan import TracedFloorPlanValidationError

_BLACK_RGB = np.array([0, 0, 0], dtype=np.uint8)
_WHITE_RGB = np.array([255, 255, 255], dtype=np.uint8)


def validate_traced_palette(
    rgba_image: NDArray[np.uint8],
    *,
    source_path: Path | None = None,
) -> None:
    if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise TracedFloorPlanValidationError("Expected an RGBA image array with shape (H, W, 4).")

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
    row, col = np.argwhere(mask)[0]
    return int(row), int(col)


def _pixel_to_rgba_tuple(pixel: NDArray[np.uint8]) -> tuple[int, int, int, int]:
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
    source_label = str(source_path) if source_path is not None else "<in-memory image>"
    return (
        f"{message} in traced floor-plan image {source_label} at row {row}, col {col}: "
        f"pixel={pixel}"
    )


__all__ = ["validate_traced_palette"]
