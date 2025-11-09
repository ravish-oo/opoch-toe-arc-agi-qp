#!/usr/bin/env python3
"""
WO-6: Π-safe Scores and FREE Predicate

This module implements:
- Π-safe scoring: scores depend ONLY on bins/mask, NEVER on raw color IDs
- FREE predicate: byte-exact invariance checks for symmetries
- Integer cost conversion: float scores → int64 costs
"""
import numpy as np
from arcsolver.config import SCALE, assert_score_bounds


def build_scores_pi_safe(
    H: int,
    W: int,
    A_mask: np.ndarray,
    bin_ids: np.ndarray,
    stage_features: dict | None = None
) -> np.ndarray:
    """
    Build Π-safe scores: ŝ ∈ ℝ^(N×C) that depends ONLY on bins/mask.

    Π-safe guarantee: If two tasks differ only by a color permutation,
    they produce identical scores (up to channel reordering).

    Args:
        H, W: output grid dimensions
        A_mask: constraint mask of shape (rows, N*C) from WO-4
        bin_ids: bin assignment array of shape (N,) from WO-1
        stage_features: optional dict with additional features from previous WOs

    Returns:
        ŝ: scores of shape (N, C) as float64

    CRITICAL: Never use raw color IDs - only use bin_ids, mask patterns, etc.
    """
    N = H * W
    C = 10  # ARC color palette size

    # Initialize scores (Π-safe: uniform across all cells initially)
    scores = np.zeros((N, C), dtype=np.float64)

    # Feature 1: Bin-based scoring
    # Cells in the same bin get correlated scores
    # This is Π-safe because bin_ids don't depend on color values
    for bin_id in np.unique(bin_ids):
        mask_bin = (bin_ids == bin_id)
        bin_size = np.sum(mask_bin)
        # Give slight preference to cells in smaller bins (more constrained)
        if bin_size > 0:
            scores[mask_bin, :] += 1.0 / np.sqrt(bin_size + 1.0)

    # Feature 2: Per-position mask indicator (Π-safe template from 02_addendum.md)
    # Use indicator α·1{A_{p,c}=1} per position, NOT aggregated counts
    # A_mask is shape (N, C) where A_mask[i, c] indicates if cell i can be color c
    if A_mask is not None and A_mask.shape == (N, C):
        # Direct per-position indicator: add weight where mask allows this color
        # This is the strict Π-safe formula: α_1·1{A_{p,c}=1}
        scores += 0.5 * A_mask.astype(np.float64)  # α_1 = 0.5

    # Feature 3: Spatial position bias (Π-safe: geometry only)
    # Prefer central cells slightly (common in ARC puzzles)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    y_coords = y_coords.flatten()
    x_coords = x_coords.flatten()
    center_y, center_x = H / 2.0, W / 2.0
    distance_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    if max_dist > 0:
        centrality = 1.0 - (distance_from_center / max_dist)
        scores += centrality[:, np.newaxis] * 0.5  # broadcast to all colors

    # Verify scores are within safe bounds for int64 conversion
    assert_score_bounds(scores)

    return scores


def to_int_costs(scores: np.ndarray) -> np.ndarray:
    """
    Convert float scores to integer costs: cost = round(-ŝ * SCALE).

    Args:
        scores: float64 array of shape (N, C)

    Returns:
        costs: int64 array of shape (N, C)
    """
    costs = np.round(-scores * SCALE).astype(np.int64)
    return costs


def apply_U_to_scores(scores: np.ndarray, U: dict, H: int, W: int) -> np.ndarray:
    """
    Apply symmetry U to scores tensor.

    Args:
        scores: shape (N, C) where N = H*W
        U: symmetry dict with type="roll" or type="perm"
        H, W: grid dimensions

    Returns:
        transformed scores of shape (N, C)
    """
    if U["type"] == "roll":
        # Spatial torus shift
        dy, dx = U["dy"], U["dx"]
        scores_grid = scores.reshape(H, W, -1)  # (H, W, C)
        rolled = np.roll(scores_grid, shift=(dy, dx), axis=(0, 1))
        return rolled.reshape(-1, scores.shape[1])

    elif U["type"] == "perm":
        # Channel permutation
        perm = np.array(U["perm"], dtype=np.int64)
        # Apply permutation to color axis
        return np.take(scores, perm, axis=1)

    else:
        raise ValueError(f"Unknown symmetry type: {U['type']}")


def apply_U_to_mask(A_mask: np.ndarray, U: dict, H: int, W: int) -> np.ndarray:
    """
    Apply symmetry U to answer mask A.

    Args:
        A_mask: answer mask of shape (N, C) where A_mask[i, c] indicates feasibility
        U: symmetry dict with type="roll" or type="perm"
        H, W: grid dimensions

    Returns:
        transformed mask of shape (N, C)
    """
    N = H * W
    C = 10

    # A_mask should be shape (N, C)
    if A_mask.shape != (N, C):
        raise ValueError(f"Expected A_mask shape ({N}, {C}), got {A_mask.shape}")

    if U["type"] == "roll":
        # Spatial torus shift on cell dimension
        dy, dx = U["dy"], U["dx"]
        # Reshape to (H, W, C)
        A_spatial = A_mask.reshape(H, W, C)
        # Roll spatial dimensions
        rolled = np.roll(A_spatial, shift=(dy, dx), axis=(0, 1))
        return rolled.reshape(N, C)

    elif U["type"] == "perm":
        # Channel permutation on color dimension
        perm = np.array(U["perm"], dtype=np.int64)
        # Apply permutation to color axis
        return np.take(A_mask, perm, axis=1)

    else:
        raise ValueError(f"Unknown symmetry type: {U['type']}")


def apply_U_to_equalizers(equalizer_rows: list, U: dict, H: int, W: int) -> list:
    """
    Apply symmetry U to equalizer rows.

    Equalizer rows are (p_i, p_j, c) tuples representing cell ties.
    - Roll: map pixel indices via φ(r,c) = ((r+dy)%H, (c+dx)%W)
    - Perm: map color index c → σ(c)
    """
    if U["type"] == "roll":
        dy, dx = U["dy"], U["dx"]
        transformed = []
        for p_i, p_j, c in equalizer_rows:
            # Map pixel indices
            r_i, c_i = divmod(p_i, W)
            r_j, c_j = divmod(p_j, W)

            r_i_new = (r_i + dy) % H
            c_i_new = (c_i + dx) % W
            r_j_new = (r_j + dy) % H
            c_j_new = (c_j + dx) % W

            p_i_new = r_i_new * W + c_i_new
            p_j_new = r_j_new * W + c_j_new

            # Re-canonicalize as (min, max, c)
            transformed.append((min(p_i_new, p_j_new), max(p_i_new, p_j_new), c))
        return transformed

    elif U["type"] == "perm":
        perm = np.array(U["perm"], dtype=np.int64)
        # Map color index
        return [(p_i, p_j, int(perm[c])) for p_i, p_j, c in equalizer_rows]

    else:
        raise ValueError(f"Unknown symmetry type: {U['type']}")


def apply_U_to_faces(faces: dict, U: dict, H: int, W: int) -> dict:
    """
    Apply symmetry U to faces (row/col totals per color).

    Faces: {"R": R_rc (H×C), "S": S_jc (W×C)}
    - Roll: shift row/col indices circularly
    - Perm: permute color axis
    """
    if "R" not in faces or "S" not in faces:
        return faces

    R = faces["R"]  # (H, C)
    S = faces["S"]  # (W, C)

    if U["type"] == "roll":
        dy, dx = U["dy"], U["dx"]
        # Shift row indices by dy, col indices by dx
        R_new = np.roll(R, shift=dy, axis=0)
        S_new = np.roll(S, shift=dx, axis=0)
        return {"R": R_new, "S": S_new}

    elif U["type"] == "perm":
        perm = np.array(U["perm"], dtype=np.int64)
        # Permute color axis
        R_new = np.take(R, perm, axis=1)
        S_new = np.take(S, perm, axis=1)
        return {"R": R_new, "S": S_new}

    else:
        raise ValueError(f"Unknown symmetry type: {U['type']}")


def _equalizer_sets_equal(rows1: list, rows2: list) -> bool:
    """Byte-exact equality of equalizer row sets (sorted)."""
    if len(rows1) != len(rows2):
        return False
    # Sort both and compare
    sorted1 = sorted(rows1)
    sorted2 = sorted(rows2)
    return sorted1 == sorted2


def _faces_equal(faces1: dict, faces2: dict) -> bool:
    """Byte-exact equality of face constraints."""
    if set(faces1.keys()) != set(faces2.keys()):
        return False

    for key in faces1:
        if not np.array_equal(faces1[key], faces2[key]):
            return False

    return True


def check_free_predicate(
    scores: np.ndarray,
    A_mask: np.ndarray,
    U: dict,
    H: int,
    W: int,
    equalizer_rows: list | None = None,
    faces: dict | None = None
) -> tuple[bool, bool]:
    """
    Check FREE predicate for symmetry U.

    A symmetry U is FREE iff:
    1. Cost invariance: ŝ∘U == ŝ (byte-exact)
    2. Constraint invariance: A·U == A (byte-exact)

    Args:
        scores: shape (N, C) float64
        A_mask: constraint matrix from WO-4
        U: symmetry dict (type="roll" or "perm")
        H, W: grid dimensions
        equalizer_rows: optional list of equalizer constraint rows
        faces: optional dict of face constraints

    Returns:
        (cost_invariance_ok, constraint_invariance_ok)
    """
    # Check 1: Cost invariance ŝ∘U == ŝ
    scores_transformed = apply_U_to_scores(scores, U, H, W)
    cost_invariance_ok = np.array_equal(scores, scores_transformed)

    # Check 2: Constraint invariance A·U == A
    # Only check if we have constraints
    if A_mask is not None and A_mask.shape[0] > 0:
        A_transformed = apply_U_to_mask(A_mask, U, H, W)

        # Convert both to dense for byte-exact comparison
        if hasattr(A_mask, 'toarray'):
            A_dense = A_mask.toarray()
        else:
            A_dense = A_mask

        if hasattr(A_transformed, 'toarray'):
            A_trans_dense = A_transformed.toarray()
        else:
            A_trans_dense = A_transformed

        constraint_invariance_ok = np.array_equal(A_dense, A_trans_dense)
    else:
        # No constraints → trivially invariant
        constraint_invariance_ok = True

    # Check 3: Equalizer rows invariance (if present)
    if equalizer_rows is not None and len(equalizer_rows) > 0:
        eq_transformed = apply_U_to_equalizers(equalizer_rows, U, H, W)
        # Compare as sorted sets (byte-exact)
        eq_ok = _equalizer_sets_equal(equalizer_rows, eq_transformed)
        constraint_invariance_ok = constraint_invariance_ok and eq_ok

    # Check 4: Faces invariance (if present)
    if faces is not None and len(faces) > 0:
        faces_transformed = apply_U_to_faces(faces, U, H, W)
        faces_ok = _faces_equal(faces, faces_transformed)
        constraint_invariance_ok = constraint_invariance_ok and faces_ok

    return cost_invariance_ok, constraint_invariance_ok


def assert_pi_safety_equivariance(
    scores: np.ndarray,
    A_mask: np.ndarray,
    bin_ids: np.ndarray,
    H: int,
    W: int,
    stage_features: dict | None = None
) -> None:
    """
    Runtime self-test: Verify Π-safe equivariance property.

    Tests that build_scores_pi_safe(π·A) = π·build_scores_pi_safe(A)
    for a fixed test permutation π = [0,2,1,3,4,5,6,7,8,9] (swap colors 1↔2).

    Raises:
        RuntimeError: If equivariance check fails
    """
    # Test permutation: swap colors 1 and 2
    perm = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)

    # Apply permutation to mask: A_perm[p, π(c)] = A[p, c]
    A_perm = np.take(A_mask, perm, axis=1)

    # Rebuild scores with permuted mask
    scores_rebuilt = build_scores_pi_safe(H, W, A_perm, bin_ids, stage_features)

    # Apply permutation to original scores: scores_perm[p, π(c)] = scores[p, c]
    scores_perm = np.take(scores, perm, axis=1)

    # Check byte-exact equality
    if not np.array_equal(scores_rebuilt, scores_perm):
        raise RuntimeError(
            "Π-safe equivariance check FAILED: "
            f"build_scores_pi_safe(π·A) ≠ π·build_scores_pi_safe(A) "
            f"for permutation {perm.tolist()}"
        )


def check_all_free_maps(
    scores: np.ndarray,
    A_mask: np.ndarray,
    free_maps_candidates: list[dict],
    H: int,
    W: int
) -> list[dict]:
    """
    Check all candidate FREE maps and return those that pass the predicate.

    Args:
        scores: shape (N, C) float64
        A_mask: constraint matrix from WO-4
        free_maps_candidates: list of symmetry dicts from WO-2/WO-3
        H, W: grid dimensions

    Returns:
        List of verified FREE maps with check results
    """
    verified_maps = []

    for U in free_maps_candidates:
        cost_ok, constraint_ok = check_free_predicate(scores, A_mask, U, H, W)

        result = {
            "symmetry": U,
            "cost_invariance": cost_ok,
            "constraint_invariance": constraint_ok,
            "is_free": cost_ok and constraint_ok
        }
        verified_maps.append(result)

    return verified_maps
