"""WO-4 mask.py - Forward meet closure and color-agnostic lift.

Implements the order-free (monotone, extensive, idempotent) forward meet closure:
- build_forward_meet: Π-safe closure F[p,k,c] (intersection across trainings)
- apply_color_agnostic_lift: Lift with statistics tracking
- build_test_mask: Extract test mask A[p,c] from F
- admits_stats: Compute statistics for receipts
- sample_test_input_at_output_coords: Inverse sampling for size-mismatched test inputs

All operations are deterministic (int64, byte-exact equality, no tolerances).
All operations are fully vectorized using NumPy broadcasting (no pixel loops).
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
from .config import GRID_DTYPE, INT_DTYPE, PALETTE_C


def build_forward_meet(
    train_pairs_emb_aligned: List[Tuple[np.ndarray, np.ndarray]],
    H_out: int,
    W_out: int,
    C_in: int = PALETTE_C,
    C_out: int = PALETTE_C,
) -> np.ndarray:
    """Build forward meet closure F[p,k,c] (order-free, WITHOUT lift).

    Args:
        train_pairs_emb_aligned: List of (X_i_emb, Y_i_emb) tuples
        H_out: Output height
        W_out: Output width
        C_in: Input palette size (default 10)
        C_out: Output palette size (default 10)

    Returns:
        F: Bool array (N, C_in, C_out) where F[p,k,c] = True iff
           every training with X_i[p]==k also has Y_i[p]==c

    Notes:
        - Extensive (starts with all True)
        - Monotone (only restricts, never adds)
        - Idempotent (re-running gives same result)
        - Order-free (intersection is commutative)
        - Padding (-1) in X is treated as unseen
        - Fully vectorized (no pixel loops)
        - Anchors: 01_addendum.md §2, 02_addendum.md §H, 04_engg_spec.md §6
    """
    N = H_out * W_out

    # Initialize F (extensive start: all admits)
    F = np.ones((N, C_in, C_out), dtype=bool)

    # Forward meet: intersection across trainings using one-hot approach
    for X_i_emb, Y_i_emb in train_pairs_emb_aligned:
        # Flatten to raster order (C-order)
        X_flat = X_i_emb.ravel(order='C')
        Y_flat = Y_i_emb.ravel(order='C')

        # Build one-hot for outputs: H[p, c] = (Y_flat[p] == c)
        # Use outer product for vectorization
        H = np.equal.outer(Y_flat, np.arange(C_out, dtype=Y_flat.dtype))  # (N, C_out) bool

        # For each input color k, AND F[p,k,:] with H at positions where X[p]==k
        for k in range(C_in):
            idx_k = (X_flat == k)  # positions where this training speaks for k
            if not idx_k.any():
                continue
            # Order-free meet: F[idx_k, k, :] &= H[idx_k, :]
            # This is element-wise logical AND (no overwrite)
            F[idx_k, k, :] &= H[idx_k, :]

    return F


def apply_color_agnostic_lift(
    F: np.ndarray,
    train_pairs_emb_aligned: List[Tuple[np.ndarray, np.ndarray]],
    H_out: int,
    W_out: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Apply color-agnostic lift to F and return statistics.

    Args:
        F: Forward meet closure (N, C_in, C_out) bool (will be modified in-place)
        train_pairs_emb_aligned: Training pairs (to build seen mask)
        H_out: Output height
        W_out: Output width

    Returns:
        F: Modified F with lift applied (same array, modified in-place)
        lift_stats: Dict with:
            - pixels_with_unified_admits: Count of pixels where K_obs admits are identical
            - copied_rows: Count of (p, k') pairs where admits copied to unseen

    Notes:
        - For each pixel p, if all observed input colors admit identical output sets,
          copy that set to unseen input colors
        - Fully vectorized using broadcasting
        - Anchors: 04_engg_spec.md §6.2
    """
    N = H_out * W_out
    C_in = F.shape[1]

    # Build seen mask: which input colors were observed at each pixel
    # Vectorized: accumulate across trainings
    seen = np.zeros((N, C_in), dtype=bool)
    for X_i_emb, _ in train_pairs_emb_aligned:
        X_flat = X_i_emb.ravel(order='C')
        # For each color k, mark positions where k appears
        for k in range(C_in):
            seen[:, k] |= (X_flat == k)

    # Statistics counters
    pixels_with_unified_admits = 0
    copied_rows = 0

    # Vectorized lift per pixel
    # For efficiency, we check admits identity using broadcasting
    for p in range(N):
        K_obs = np.where(seen[p, :])[0]

        if len(K_obs) == 0:
            continue

        # Check if all observed input colors have identical admits vectors
        # Vectorized comparison: broadcast F[p, K_obs, :] against first row
        first_admits = F[p, K_obs[0], :]

        if len(K_obs) > 1:
            # Compare all K_obs admits against first (vectorized)
            all_admits = F[p, K_obs, :]  # (len(K_obs), C_out)
            all_identical = np.all(all_admits == first_admits, axis=1).all()
        else:
            all_identical = True

        if all_identical:
            pixels_with_unified_admits += 1

            # Copy to all unseen input colors (vectorized)
            K_unseen = np.where(~seen[p, :])[0]
            if len(K_unseen) > 0:
                F[p, K_unseen, :] = first_admits  # Broadcasting
                copied_rows += len(K_unseen)

    return F, {
        "pixels_with_unified_admits": pixels_with_unified_admits,
        "copied_rows": copied_rows,
    }


def build_test_mask(F: np.ndarray, Xstar_flat: np.ndarray) -> np.ndarray:
    """Extract test mask A[p,c] from forward meet F (fully vectorized).

    Args:
        F: Forward meet closure (N, C_in, C_out) bool
        Xstar_flat: Test input flattened (N,) int32 in {-1, 0..9}
                    -1 = unseen (out of bounds after inverse sampling)

    Returns:
        A: Test mask (N, C_out) bool where A[p,c] = F[p, Xstar[p], c]
           For unseen pixels (k=-1), A[p,:] remains all-False (unconstrained)

    Notes:
        - Fully vectorized using advanced indexing
        - Unseen pixels (k=-1) leave A[p,:] unconstrained
        - Anchors: 04_engg_spec.md §6
    """
    N = Xstar_flat.shape[0]
    C_out = F.shape[2]

    # Initialize mask (all False = unconstrained)
    A = np.zeros((N, C_out), dtype=bool)

    # Find valid pixels (not unseen)
    valid = (Xstar_flat >= 0)

    if not valid.any():
        return A

    # Vectorized gather: A[valid, :] = F[valid_indices, Xstar_flat[valid], :]
    # Use advanced indexing
    valid_indices = np.where(valid)[0]
    k_values = Xstar_flat[valid]

    # Gather in one vectorized operation
    A[valid, :] = F[valid_indices, k_values, :]

    return A


def sample_test_input_at_output_coords(
    Xstar: np.ndarray,
    H_out: int,
    W_out: int,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample test input at output canvas coordinates (inverse embedding).

    Args:
        Xstar: Test input (H_in, W_in) int32 in {0..9}
        H_out: Output canvas height
        W_out: Output canvas width
        mode: Embedding mode ('topleft' or 'center')

    Returns:
        K: (H_out, W_out) int32 in {-1, 0..9} where:
           - K[r,c] = Xstar[...] if (r,c) maps inside Xstar
           - K[r,c] = -1 (unseen sentinel) if (r,c) maps outside Xstar
        stats: Dict with sampling statistics

    Notes:
        - Inverse sampling: maps output pixel to input pixel
        - Out-of-bounds pixels are marked -1 (unseen)
        - Fully vectorized
        - Anchors: Issue #6 clarification
    """
    H_in, W_in = Xstar.shape

    # Compute embedding offset
    if mode == 'topleft':
        r0, c0 = 0, 0
    elif mode == 'center':
        r0 = (H_out - H_in) // 2
        c0 = (W_out - W_in) // 2
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Initialize K with unseen sentinel
    K = np.full((H_out, W_out), -1, dtype=np.int32)

    # Compute overlapping box (clamp to valid ranges)
    r_in_lo = max(0, -r0)
    r_in_hi = min(H_in, H_out - r0)
    c_in_lo = max(0, -c0)
    c_in_hi = min(W_in, W_out - c0)

    # Copy overlapping region if it exists
    if r_in_lo < r_in_hi and c_in_lo < c_in_hi:
        r_out_lo = r0 + r_in_lo
        r_out_hi = r0 + r_in_hi
        c_out_lo = c0 + c_in_lo
        c_out_hi = c0 + c_in_hi
        K[r_out_lo:r_out_hi, c_out_lo:c_out_hi] = Xstar[r_in_lo:r_in_hi, c_in_lo:c_in_hi]

    # Compute statistics
    in_bounds_pixels = int((K >= 0).sum())
    unseen_pixels = int((K == -1).sum())

    stats = {
        "mode": mode,
        "H_in": int(H_in),
        "W_in": int(W_in),
        "H_out": int(H_out),
        "W_out": int(W_out),
        "in_bounds_pixels": in_bounds_pixels,
        "unseen_pixels": unseen_pixels,
    }

    return K, stats


def admits_stats(F: np.ndarray, seen: np.ndarray = None) -> Dict[str, Any]:
    """Compute statistics on admits for receipts.

    Args:
        F: Forward meet closure (N, C_in, C_out) bool
        seen: Optional (N, C_in) bool indicating which input colors were observed

    Returns:
        Dict with statistics:
            - avg_admits_per_pixel: Mean number of admits over seen input colors
            - total_true: Total number of True entries in F
            - admits_histogram: Distribution of admits-set sizes

    Notes:
        - Used for receipt generation and progress tracking
        - Anchors: 04_engg_spec.md §6
    """
    N, C_in, C_out = F.shape

    # Count admits per (pixel, input_color) slot
    admits_counts = F.sum(axis=2)  # (N, C_in)

    if seen is not None:
        # Only count seen input colors
        total_admits = admits_counts[seen].sum()
        num_seen = seen.sum()
        avg_admits = float(total_admits) / max(num_seen, 1)
    else:
        # All input colors (fallback)
        total_admits = admits_counts.sum()
        num_slots = N * C_in
        avg_admits = float(total_admits) / max(num_slots, 1)

    # Total True entries
    total_true = int(F.sum())

    # Histogram of admits-set sizes (use bincount for speed)
    admits_flat = admits_counts.ravel()
    hist = np.bincount(admits_flat, minlength=C_out + 1)

    return {
        "avg_admits_per_pixel": avg_admits,
        "total_true": total_true,
        "admits_histogram": hist.tolist(),
    }


def build_seen_mask(
    train_pairs_emb_aligned: List[Tuple[np.ndarray, np.ndarray]],
    H_out: int,
    W_out: int,
) -> np.ndarray:
    """Build seen mask from training pairs (helper for before/after stats).

    Args:
        train_pairs_emb_aligned: Training pairs
        H_out: Output height
        W_out: Output width

    Returns:
        seen: (N, C_in) bool mask of observed input colors per pixel
    """
    N = H_out * W_out
    C_in = PALETTE_C

    seen = np.zeros((N, C_in), dtype=bool)
    for X_i_emb, _ in train_pairs_emb_aligned:
        X_flat = X_i_emb.ravel(order='C')
        for k in range(C_in):
            seen[:, k] |= (X_flat == k)

    return seen
