"""WO-0 hash_utils.py - Byte-stable hashing for receipts.

Provides deterministic, cross-platform hashing for:
- Integer numpy arrays (byte-exact)
- JSON-serializable objects (canonical key order)

All hashes use SHA256 and are stable across runs and platforms.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any
import numpy as np


def sha256_bytes(b: bytes) -> str:
    """Compute SHA256 hex digest of bytes.

    Args:
        b: Input bytes

    Returns:
        Hex string (64 chars)
    """
    return hashlib.sha256(b).hexdigest()


def hash_ndarray_int(a: np.ndarray) -> str:
    """Hash integer numpy array (byte-exact).

    Args:
        a: Integer numpy array

    Returns:
        SHA256 hex digest

    Raises:
        RuntimeError: If array is not integer dtype
    """
    if not np.issubdtype(a.dtype, np.integer):
        raise RuntimeError(
            f"hash_ndarray_int requires integer dtype, got {a.dtype}"
        )
    # Use C-order to ensure consistent byte layout
    return sha256_bytes(a.tobytes(order="C"))


def hash_json_canonical(obj: Any) -> str:
    """Hash JSON-serializable object with canonical serialization.

    Args:
        obj: JSON-serializable object (dict, list, primitives)

    Returns:
        SHA256 hex digest of canonical JSON representation

    Notes:
        - Keys are sorted
        - No whitespace
        - UTF-8 encoding
        - Deterministic across platforms
    """
    # Canonical JSON: sorted keys, no spaces, UTF-8
    json_str = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256_bytes(json_str.encode("utf-8"))


def hash_grid(grid_data: np.ndarray) -> str:
    """Hash grid data (convenience wrapper for hash_ndarray_int).

    Args:
        grid_data: Grid array (int32)

    Returns:
        SHA256 hex digest
    """
    return hash_ndarray_int(grid_data)
