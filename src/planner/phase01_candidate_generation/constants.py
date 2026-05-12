from __future__ import annotations

import numpy as np

PHASE_NAME = "candidate_generation"
PHASE_ARTIFACT_STEM = "01_candidates"

_BOUNDARY_FLAG_NORTH = np.uint8(1)
_BOUNDARY_FLAG_EAST = np.uint8(2)
_BOUNDARY_FLAG_SOUTH = np.uint8(4)
_BOUNDARY_FLAG_WEST = np.uint8(8)

_EXCEPTION_FLAG_ENDPOINT = np.uint8(1)
_EXCEPTION_FLAG_MIDPOINT = np.uint8(2)
_EXCEPTION_FLAG_CORNER = np.uint8(4)
_EXCEPTION_FLAG_JUNCTION = np.uint8(8)
