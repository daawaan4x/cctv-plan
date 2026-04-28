"""Tests for traced floor-plan loading, palette validation, and plotting."""

from __future__ import annotations

import shutil
import struct
import unittest
import zlib
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from src.common.floorplan import (
    FloorPlanInput,
    NULL_CELL,
    OPEN_CELL,
    SOLID_CELL,
    TracedFloorPlanValidationError,
)
from src.common.floorplan_loader import load_traced_floorplan, load_traced_floorplans

# Repo-local test scratch space
_TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-test-workdir"


def _write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    """Write a minimal RGBA PNG file for loader tests without extra dependencies."""

    if rgba.dtype != np.uint8:
        raise TypeError("PNG test helper expects uint8 RGBA input.")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("PNG test helper expects shape (H, W, 4).")

    height, width, _ = rgba.shape
    raw_scanlines = bytearray()
    for row in rgba:
        raw_scanlines.append(0)
        raw_scanlines.extend(row.tobytes())

    compressed = zlib.compress(bytes(raw_scanlines))

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = struct.pack(">I", len(data)) + tag + data
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return payload + struct.pack(">I", crc)

    # Building the PNG by hand keeps the tests lightweight and deterministic. The
    # loader only needs a valid RGBA PNG container, so depending on Pillow here
    # would add packaging noise without improving test intent.
    png_bytes = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)),
            chunk(b"IDAT", compressed),
            chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png_bytes)


def _write_floorplan_metadata(
    path: Path,
    *,
    grid_cell_size_m: float | None,
    min_k: int | None = None,
) -> None:
    """Write the sibling JSON metadata file expected by the traced loader."""

    metadata_path = path.with_suffix(".json")
    metadata_payload: dict[str, float | int | None] = {
        "grid_cell_size_m": grid_cell_size_m,
    }
    if min_k is not None:
        metadata_payload["min_k"] = min_k
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _workspace_temp_dir() -> Iterator[Path]:
    """Provide an isolated temporary workspace under the repo-local test root."""

    _TEST_TEMP_ROOT.mkdir(exist_ok=True)
    # These tests avoid the system temp directory because the sandbox user does not
    # always have stable write/reopen permissions there on Windows.
    temp_dir = _TEST_TEMP_ROOT / f"case-{uuid4().hex}"
    temp_dir.mkdir()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class FloorPlanLoaderTests(unittest.TestCase):
    """Exercise the traced PNG loading path and its validation guarantees."""

    def test_load_traced_floorplan_decodes_tri_state_grid(self) -> None:
        # Use one tiny fixture that hits all three locked cell semantics at once:
        # transparent -> null, white -> open, black -> solid.
        with _workspace_temp_dir() as temp_dir:
            image_path = temp_dir / "sample.png"
            rgba = np.array(
                [
                    [[0, 0, 0, 0], [255, 255, 255, 255]],
                    [[0, 0, 0, 255], [255, 255, 255, 255]],
                ],
                dtype=np.uint8,
            )
            _write_rgba_png(image_path, rgba)
            _write_floorplan_metadata(image_path, grid_cell_size_m=0.5, min_k=3)
            floorplan = load_traced_floorplan(image_path)

        expected_grid = np.array(
            [
                [NULL_CELL, OPEN_CELL],
                [SOLID_CELL, OPEN_CELL],
            ],
            dtype=np.int8,
        )
        np.testing.assert_array_equal(floorplan.grid, expected_grid)
        self.assertEqual(floorplan.shape, (2, 2))
        self.assertEqual(floorplan.null_cell_count, 1)
        self.assertEqual(floorplan.open_cell_count, 2)
        self.assertEqual(floorplan.solid_cell_count, 1)
        self.assertEqual(floorplan.grid_cell_size_m, 0.5)
        self.assertEqual(floorplan.min_k, 3)
        np.testing.assert_array_equal(
            floorplan.open_mask,
            np.array([[False, True], [False, True]]),
        )

    def test_load_traced_floorplans_returns_sorted_names(self) -> None:
        with _workspace_temp_dir() as traced_dir:
            transparent = np.array([0, 0, 0, 0], dtype=np.uint8)
            white = np.array([255, 255, 255, 255], dtype=np.uint8)

            _write_rgba_png(
                traced_dir / "b-room.png",
                np.array([[transparent]], dtype=np.uint8),
            )
            _write_floorplan_metadata(traced_dir / "b-room.png", grid_cell_size_m=1.0)
            _write_rgba_png(
                traced_dir / "a-room.png",
                np.array([[white]], dtype=np.uint8),
            )
            _write_floorplan_metadata(traced_dir / "a-room.png", grid_cell_size_m=0.5)

            floorplans = load_traced_floorplans(traced_dir)

        self.assertEqual(list(floorplans.keys()), ["a-room", "b-room"])
        self.assertEqual(floorplans["a-room"].open_cell_count, 1)
        self.assertEqual(floorplans["b-room"].null_cell_count, 1)

    def test_load_traced_floorplan_rejects_unexpected_opaque_color(self) -> None:
        with _workspace_temp_dir() as temp_dir:
            image_path = temp_dir / "invalid-color.png"
            rgba = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
            _write_rgba_png(image_path, rgba)
            _write_floorplan_metadata(image_path, grid_cell_size_m=0.5)

            with self.assertRaisesRegex(
                TracedFloorPlanValidationError,
                r"unexpected opaque color.*row 0, col 0.*pixel=\(255, 0, 0, 255\)",
            ):
                load_traced_floorplan(image_path)

    def test_load_traced_floorplan_rejects_partial_transparency(self) -> None:
        with _workspace_temp_dir() as temp_dir:
            image_path = temp_dir / "partial-alpha.png"
            rgba = np.array([[[255, 255, 255, 128]]], dtype=np.uint8)
            _write_rgba_png(image_path, rgba)
            _write_floorplan_metadata(image_path, grid_cell_size_m=0.5)

            with self.assertRaisesRegex(
                TracedFloorPlanValidationError,
                r"partial transparency is not allowed.*row 0, col 0.*pixel=\(255, 255, 255, 128\)",
            ):
                load_traced_floorplan(image_path)

    def test_existing_ground_back_asset_decodes_to_only_locked_cell_values(
        self,
    ) -> None:
        asset_path = (
            Path(__file__).resolve().parents[1]
            / "static"
            / "floor-plan"
            / "traced"
            / "ground-back.png"
        )
        floorplan = load_traced_floorplan(asset_path)

        unique_values = set(np.unique(floorplan.grid).tolist())
        self.assertEqual(
            unique_values, {int(NULL_CELL), int(OPEN_CELL), int(SOLID_CELL)}
        )
        self.assertEqual(
            floorplan.null_cell_count
            + floorplan.open_cell_count
            + floorplan.solid_cell_count,
            floorplan.height * floorplan.width,
        )
        self.assertAlmostEqual(float(floorplan.grid_cell_size_m or 0.0), 0.34653822)
        self.assertEqual(floorplan.min_k, 14)

    def test_load_traced_floorplan_requires_sibling_metadata_json(self) -> None:
        with _workspace_temp_dir() as temp_dir:
            image_path = temp_dir / "missing-metadata.png"
            rgba = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
            _write_rgba_png(image_path, rgba)

            with self.assertRaisesRegex(
                FileNotFoundError,
                r"Expected traced floor-plan metadata JSON next to the PNG",
            ):
                load_traced_floorplan(image_path)

    def test_load_traced_floorplan_rejects_invalid_grid_cell_size_metadata(self) -> None:
        with _workspace_temp_dir() as temp_dir:
            image_path = temp_dir / "invalid-metadata.png"
            rgba = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
            _write_rgba_png(image_path, rgba)
            image_path.with_suffix(".json").write_text(
                json.dumps({"grid_cell_size_m": "large"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"grid_cell_size_m' must be a positive number or null",
            ):
                load_traced_floorplan(image_path)

    def test_load_traced_floorplan_accepts_missing_min_k(self) -> None:
        with _workspace_temp_dir() as temp_dir:
            image_path = temp_dir / "missing-min-k.png"
            rgba = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
            _write_rgba_png(image_path, rgba)
            _write_floorplan_metadata(image_path, grid_cell_size_m=0.5)

            floorplan = load_traced_floorplan(image_path)

        self.assertIsNone(floorplan.min_k)

    def test_load_traced_floorplan_rejects_non_integer_min_k(self) -> None:
        with _workspace_temp_dir() as temp_dir:
            image_path = temp_dir / "invalid-min-k-type.png"
            rgba = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
            _write_rgba_png(image_path, rgba)
            image_path.with_suffix(".json").write_text(
                json.dumps({"grid_cell_size_m": 0.5, "min_k": "ten"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"'min_k' must be a positive integer or null",
            ):
                load_traced_floorplan(image_path)

    def test_load_traced_floorplan_rejects_non_positive_min_k(self) -> None:
        with _workspace_temp_dir() as temp_dir:
            image_path = temp_dir / "invalid-min-k-value.png"
            rgba = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
            _write_rgba_png(image_path, rgba)
            image_path.with_suffix(".json").write_text(
                json.dumps({"grid_cell_size_m": 0.5, "min_k": 0}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"'min_k' must be positive when provided",
            ):
                load_traced_floorplan(image_path)

    def test_floorplan_plot_displays_loaded_grid(self) -> None:
        floorplan = FloorPlanInput(
            name="plot-check",
            source_path=Path("plot-check.png"),
            grid=np.array(
                [
                    [NULL_CELL, OPEN_CELL],
                    [SOLID_CELL, OPEN_CELL],
                ],
                dtype=np.int8,
            ),
            height=2,
            width=2,
            null_cell_count=1,
            open_cell_count=2,
            solid_cell_count=1,
        )

        figure, axis = plt.subplots()
        returned_axis = floorplan.plot(
            ax=axis, title="Loader Check", show_colorbar=False
        )

        self.assertIs(returned_axis, axis)
        self.assertEqual(axis.get_title(), "Loader Check")
        self.assertEqual(len(axis.images), 1)
        np.testing.assert_array_equal(
            np.asarray(axis.images[0].get_array()), floorplan.grid
        )
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
