"""WO-0 config.py - Determinism guards, version asserts, constants, dtypes.

Enforces:
- Single-thread env (OMP/BLAS/MKL/NUMEXPR=1, PYTHONHASHSEED=0)
- Pinned library versions (Python 3.11.x, numpy 2.1.x, scipy 1.13.x, ortools 9.10.x)
- Fixed dtypes and constants per anchors 03_annex.md & 05_contracts.md
"""
from __future__ import annotations
import os
import sys
import numpy as np


# ============================================================================
# Determinism env (must be set before heavy libs import)
# ============================================================================
def _set_determinism_env() -> None:
    """Set threading and hash seed env vars for determinism."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")


_set_determinism_env()


# ============================================================================
# Version requirements (contracts 05_contracts.md / Global)
# ============================================================================
REQUIRED_VERSIONS = {
    "python_major_minor": (3, 11),
    "numpy": (2, 1),  # accept 2.1.x
    "scipy": (1, 13),  # accept 1.13.x
    "ortools": (9, 10),  # accept 9.10.x
}


def _assert_versions() -> None:
    """Assert pinned library versions match requirements."""
    import scipy

    # Check Python version
    py_ver = sys.version_info
    req_py = REQUIRED_VERSIONS["python_major_minor"]
    if (py_ver.major, py_ver.minor) != req_py:
        raise RuntimeError(
            f"Python must be {req_py[0]}.{req_py[1]}.x, "
            f"got {py_ver.major}.{py_ver.minor}.{py_ver.micro}"
        )

    # Check numpy version
    np_parts = tuple(map(int, np.__version__.split(".")[:2]))
    req_np = REQUIRED_VERSIONS["numpy"]
    if np_parts != req_np:
        raise RuntimeError(
            f"numpy must be {req_np[0]}.{req_np[1]}.x, got {np.__version__}"
        )

    # Check scipy version
    sp_parts = tuple(map(int, scipy.__version__.split(".")[:2]))
    req_sp = REQUIRED_VERSIONS["scipy"]
    if sp_parts != req_sp:
        raise RuntimeError(
            f"scipy must be {req_sp[0]}.{req_sp[1]}.x, got {scipy.__version__}"
        )

    # Check OR-Tools is importable (version check is weaker for OR-Tools)
    try:
        from ortools.graph.python import min_cost_flow  # noqa: F401
    except ImportError as e:
        raise RuntimeError("OR-Tools not properly installed") from e


_assert_versions()


# ============================================================================
# Global constants (anchors 03_annex.md A.1-A.3, 04_engg_spec.md §0)
# ============================================================================

# Palette
PALETTE_C = 10

# Semantic values (contracts 05 / Global)
BACKGROUND = np.int32(0)  # Background color in grids
PAD_SENTINEL = np.int32(-1)  # Padding during embedding (treat as background for structure)

# Dtypes (contracts 05 / Global)
GRID_DTYPE = np.int32  # Grid values in {-1, 0..9}
INT_DTYPE = np.int64  # Counts, costs, indices
SCORE_DTYPE = np.float64  # Scores ŝ (internal)

# Cost scaling (contracts 05 / Scores & Costs)
SCALE = 1_000_000  # cost = round(-ŝ * SCALE)

# Hard bounds to keep int64 safe (contracts 05 / Scores & Costs)
MAX_ABS_SCORE = 1_000_000  # max |ŝ| ≤ 10^6 so Σ|cost| ≪ 2^63

# Lex orders (anchors 03_annex.md A.3, 04_engg_spec.md §0, contracts 05 / Global)
PALETTE_ORDER = tuple(range(PALETTE_C))  # 0 < 1 < ... < 9
PIXEL_LEX = "row_col_asc"  # raster (row, col) ascending
PERIOD_LEX = "py_px_asc"  # (p_y, p_x) ascending
CANVAS_LEX = "H_W_asc"  # (H, W) ascending
SIG_LEX_FIELDS = (
    "neg_count",
    "row_hist",
    "col_hist",
    "bin_hist",
    "color_id",
)  # signature lex order


# ============================================================================
# Dtype enforcement helpers
# ============================================================================
def enforce_dtype(arr: np.ndarray, kind: str) -> np.ndarray:
    """Enforce dtype for given array kind.

    Args:
        arr: Input array
        kind: One of 'grid', 'int', 'count', 'cost', 'idx', 'score'

    Returns:
        Array with correct dtype

    Raises:
        ValueError: If kind is unknown
    """
    if kind == "grid":
        return np.asarray(arr, dtype=GRID_DTYPE)
    elif kind in ("int", "count", "cost", "idx"):
        return np.asarray(arr, dtype=INT_DTYPE)
    elif kind == "score":
        return np.asarray(arr, dtype=SCORE_DTYPE)
    else:
        raise ValueError(f"Unknown dtype kind: {kind}")


def assert_score_bounds(shat: np.ndarray) -> None:
    """Assert score array is within safe bounds for int64 cost conversion.

    Args:
        shat: Score array (float64)

    Raises:
        RuntimeError: If any |ŝ| > MAX_ABS_SCORE
    """
    if not np.all(np.abs(shat) <= MAX_ABS_SCORE):
        max_val = np.max(np.abs(shat))
        raise RuntimeError(
            f"ŝ out of bounds: max |ŝ| = {max_val:.2e} > {MAX_ABS_SCORE}; "
            "violates int64 cost budget (contracts 05 / Scores & Costs)"
        )
