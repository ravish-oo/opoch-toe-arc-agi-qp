"""WO-2 embed.py - Embedding & Period tests.

Implements deterministic, byte-exact embedding and period detection:
- embed_to_canvas: topleft/center placement with -1 padding
- periods_2d_exact: byte-exact torus period detection
- reembed_round_trip_ok: Π-idempotence check

All operations are Π-safe (byte-exact, deterministic).
"""
from __future__ import annotations
import numpy as np
from .config import GRID_DTYPE


def embed_to_canvas(
    Y: np.ndarray, H_out: int, W_out: int, mode: str
) -> np.ndarray:
    """Embed grid Y onto canvas with deterministic placement.

    Args:
        Y: Input grid (H_in, W_in) with int32 dtype, values in {-1, 0..9}
        H_out: Output canvas height
        W_out: Output canvas width
        mode: 'center' or 'topleft'

    Returns:
        Embedded grid (H_out, W_out) with int32 dtype, -1 padding

    Raises:
        ValueError: If Y doesn't fit on canvas (no clipping allowed)
        ValueError: If mode is not 'center' or 'topleft'

    Notes:
        - Topleft: Y placed at (0, 0), padded bottom/right
        - Center: Y centered with integer offsets, padded around
        - Padding value: -1 (PAD_SENTINEL)
        - Byte-exact, deterministic
        - Anchors: 03_annex.md A.1, 04_engg_spec.md §2, 05_contracts.md
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y.shape}")
    if Y.dtype != GRID_DTYPE:
        raise ValueError(f"Y must have dtype {GRID_DTYPE}, got {Y.dtype}")

    H_in, W_in = Y.shape

    # Check if Y fits on canvas
    if H_in > H_out or W_in > W_out:
        raise ValueError(
            f"Grid size ({H_in}, {W_in}) does not fit on canvas ({H_out}, {W_out}). "
            "No clipping allowed (UNSAT: EmbeddingSize)."
        )

    # Create output canvas filled with -1 (PAD_SENTINEL)
    out = np.full((H_out, W_out), -1, dtype=GRID_DTYPE)

    if mode == "topleft":
        # Place Y at (0, 0)
        out[:H_in, :W_in] = Y
    elif mode == "center":
        # Center Y with integer offsets
        r0 = (H_out - H_in) // 2
        c0 = (W_out - W_in) // 2
        out[r0 : r0 + H_in, c0 : c0 + W_in] = Y
    else:
        raise ValueError(f"mode must be 'center' or 'topleft', got '{mode}'")

    return out


def periods_2d_exact(A: np.ndarray) -> tuple[int, int]:
    """Find minimal torus periods in y and x directions using byte-exact equality.

    Args:
        A: Grid (H, W) with int32 dtype

    Returns:
        (p_y, p_x): Minimal periods in y and x directions
            - p_y: minimal shift in axis=0 where A == roll(A, p_y, axis=0)
            - p_x: minimal shift in axis=1 where A == roll(A, p_x, axis=1)
            - Defaults to H or W if no period found

    Notes:
        - Uses np.roll for shifting
        - Uses np.array_equal for byte-exact comparison (no tolerance)
        - Treats -1 as just another value (no special case)
        - Complexity: O(H*W*(H+W)) worst case (fine for ARC sizes)
        - Anchors: 03_annex.md A.1, 04_engg_spec.md §7
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    H, W = A.shape

    # Find minimal period in y direction (axis=0)
    p_y = H  # Default to full height
    for shift in range(1, H + 1):
        rolled = np.roll(A, shift=shift, axis=0)
        if np.array_equal(A, rolled):
            p_y = shift
            break

    # Find minimal period in x direction (axis=1)
    p_x = W  # Default to full width
    for shift in range(1, W + 1):
        rolled = np.roll(A, shift=shift, axis=1)
        if np.array_equal(A, rolled):
            p_x = shift
            break

    return (p_y, p_x)


def reembed_round_trip_ok(
    Y: np.ndarray, H_out: int, W_out: int, mode: str
) -> bool:
    """Check Π-idempotence: embed(embed(Y)) == embed(Y)?

    Args:
        Y: Input grid
        H_out: Canvas height
        W_out: Canvas width
        mode: 'center' or 'topleft'

    Returns:
        True if re-embedding is idempotent (byte-exact equality)

    Notes:
        - Tests that embedding is a projector (Π∘Π = Π)
        - Uses byte-exact np.array_equal
        - Anchors: 05_contracts.md (Idempotence)
    """
    try:
        # First embed
        Y1 = embed_to_canvas(Y, H_out, W_out, mode)
        # Second embed (should be identical)
        Y2 = embed_to_canvas(Y1, H_out, W_out, mode)
        # Check byte-exact equality
        return np.array_equal(Y1, Y2)
    except Exception:
        # If embedding fails, return False
        return False
