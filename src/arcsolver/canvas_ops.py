#!/usr/bin/env python3
"""
Canvas Operations: Per-canvas WO-4/5/6 callables for WO-A & WO-B

Refactored from harness.py to support per-test per-canvas pack enumeration
for linear size laws (anchor §1).
"""
from __future__ import annotations
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_tree


GRID_DTYPE = np.int32


def align_to_canvas(
    Y_i: np.ndarray,
    H_target: int,
    W_target: int,
    embed_mode: str,
) -> np.ndarray:
    """
    Embed training output Y_i onto target canvas per anchor §2.

    Args:
        Y_i: Training output grid (H_i, W_i)
        H_target: Target canvas height
        W_target: Target canvas width
        embed_mode: "topleft" or "center"

    Returns:
        Embedded grid (H_target, W_target) with -1 padding
    """
    H_i, W_i = Y_i.shape

    # If already correct size, return as-is
    if H_i == H_target and W_i == W_target:
        return Y_i.copy()

    # Initialize canvas with -1 (undefined/background per anchor §2)
    canvas = np.full((H_target, W_target), -1, dtype=GRID_DTYPE)

    if embed_mode == "topleft":
        # Place at (0,0)
        h_copy = min(H_i, H_target)
        w_copy = min(W_i, W_target)
        canvas[:h_copy, :w_copy] = Y_i[:h_copy, :w_copy]

    elif embed_mode == "center":
        # Center Y_i in target canvas
        offset_r = (H_target - H_i) // 2
        offset_c = (W_target - W_i) // 2

        # Compute valid ranges (handle case where Y_i is larger than target)
        src_r_start = max(0, -offset_r)
        src_r_end = min(H_i, H_target - offset_r)
        src_c_start = max(0, -offset_c)
        src_c_end = min(W_i, W_target - offset_c)

        dst_r_start = max(0, offset_r)
        dst_r_end = dst_r_start + (src_r_end - src_r_start)
        dst_c_start = max(0, offset_c)
        dst_c_end = dst_c_start + (src_c_end - src_c_start)

        canvas[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
            Y_i[src_r_start:src_r_end, src_c_start:src_c_end]

    else:
        raise ValueError(f"Unknown embed_mode: {embed_mode}")

    return canvas


def forward_meet_closure_lift(
    train_inputs_embedded: List[np.ndarray],
    train_outputs_embedded: List[np.ndarray],
    test_input_embedded: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Compute forward meet with closure and color-agnostic lift per anchor §5.

    Implements the three-step algorithm:
    1. Forward meet: F[p,k,c]=1 iff every training with X_i(p)=k has Y_i(p)=c
    2. Sufficiency closure: Iterate to ensure order-independence and idempotence
    3. Color-agnostic lift: Extend consensus admits to unseen input colors

    Args:
        train_inputs_embedded: List of training inputs aligned to canvas (H, W)
        train_outputs_embedded: List of training outputs aligned to canvas (H, W)
        test_input_embedded: Test input aligned to canvas (H, W)
        H: Canvas height
        W: Canvas width

    Returns:
        A_mask: (N, C) bool array where A_mask[p,c] = F[p, test_input(p), c]

    Complexity: O(N·m + N·C) where m = num trainings
    """
    N = H * W
    C = 10  # ARC palette size

    # === STEP 1: FORWARD MEET (GATHER) ===
    # For each (pixel p, input color k), collect output colors across trainings
    seen = np.zeros((N, C), dtype=bool)        # Have we seen (p,k)?
    unique = -np.ones((N, C), dtype=np.int16)  # First output color for (p,k)
    amb = np.zeros((N, C), dtype=bool)         # Ambiguous mapping?

    for X_i, Y_i in zip(train_inputs_embedded, train_outputs_embedded):
        x = X_i.flatten()
        y = Y_i.flatten()

        # Only process valid pixels (not padded -1)
        valid = (x >= 0) & (y >= 0)

        for p in np.nonzero(valid)[0]:
            k = int(x[p])
            c = int(y[p])

            if not seen[p, k]:
                # First time seeing X(p)=k → record output c
                seen[p, k] = True
                unique[p, k] = c
            elif unique[p, k] != c:
                # Saw X(p)=k → different output → ambiguous
                amb[p, k] = True

    # Build forward meet F[p,k,c]
    F = np.ones((N, C, C), dtype=bool)  # Initially all True

    for p in range(N):
        for k in range(C):
            if not seen[p, k]:
                # Never saw this (p,k) → leave broad (all True)
                continue

            if amb[p, k]:
                # Ambiguous: multiple outputs for same input → forbid all
                F[p, k, :] = False
            else:
                # Unique mapping k→c*
                c_star = unique[p, k]
                F[p, k, :] = False
                F[p, k, c_star] = True

    # === STEP 2: SUFFICIENCY CLOSURE ===
    # Enforce order-independence: for each training, set F[p,k,c*]=1, F[p,k,≠c*]=0
    # CRITICAL: Skip ambiguous (p,k) pairs (they remain all False from Step 1)
    for X_i, Y_i in zip(train_inputs_embedded, train_outputs_embedded):
        x = X_i.flatten()
        y = Y_i.flatten()

        valid = (x >= 0) & (y >= 0)
        idx = np.nonzero(valid)[0]

        for i, p in enumerate(idx):
            k = int(x[idx][i])
            c = int(y[idx][i])

            if not amb[p, k]:  # Only if not ambiguous
                F[p, k, :] = False
                F[p, k, c] = True

    # === STEP 3: COLOR-AGNOSTIC LIFT ===
    # If all observed k at pixel p share same admits row → lift to unseen k'
    for p in range(N):
        obs_k = np.nonzero(seen[p, :])[0]

        if len(obs_k) == 0:
            continue  # No evidence at this pixel

        # Check if all observed k have same admits row
        first_row = F[p, obs_k[0], :]
        consensus = all(np.array_equal(F[p, k, :], first_row) for k in obs_k[1:])

        if consensus:
            # Lift to unseen k'
            unseen_k = np.setdiff1d(np.arange(C), obs_k, assume_unique=True)
            F[p, unseen_k, :] = first_row

    # === STEP 4: EXTRACT TEST MASK ===
    # A_mask[p,c] = F[p, test_input(p), c]
    A_mask = np.zeros((N, C), dtype=bool)
    test_flat = test_input_embedded.flatten()

    for p in range(N):
        k_test = test_flat[p]
        if k_test >= 0:
            A_mask[p, :] = F[p, int(k_test), :]
        else:
            # Padded position → broad (all True)
            A_mask[p, :] = True

    return A_mask


def wo4_per_canvas(
    train_inputs_raw: List[np.ndarray],
    train_outputs_embedded: List[np.ndarray],
    test_input_raw: np.ndarray,
    H: int,
    W: int,
    embed_mode: str,
) -> Dict:
    """
    WO-4 per-canvas: Forward meet + closure + lift for given canvas dimensions.

    Phase 3 implementation: Input-conditioned A_mask per anchor §5.

    Args:
        train_inputs_raw: List of raw training input grids (NOT embedded yet)
        train_outputs_embedded: List of training outputs already embedded to (H, W)
        test_input_raw: Raw test input grid (NOT embedded yet)
        H: Canvas height
        W: Canvas width
        embed_mode: "topleft" or "center"

    Returns:
        Dict with:
        - A_mask: (N, C) bool array
        - bin_ids: (N,) int array
        - meta: {H, W, N, num_trains, num_bins, forward_meet_applied, hash}
    """
    N = H * W
    C = 10

    # Embed inputs to canvas (sample at output coords)
    train_inputs_embedded = [align_to_canvas(X_i, H, W, embed_mode) for X_i in train_inputs_raw]
    test_input_embedded = align_to_canvas(test_input_raw, H, W, embed_mode)

    # Compute forward meet + closure + lift
    A_mask = forward_meet_closure_lift(
        train_inputs_embedded=train_inputs_embedded,
        train_outputs_embedded=train_outputs_embedded,
        test_input_embedded=test_input_embedded,
        H=H,
        W=W,
    )

    # Build bin_ids (periphery-parity bins per anchor §4)
    # Bins: intersection of {top/bottom/left/right/interior} × {row_parity, col_parity}
    bin_ids = np.zeros(N, dtype=np.int32)

    for r in range(H):
        for c in range(W):
            p = r * W + c

            # Edge flags
            is_top = (r == 0)
            is_bottom = (r == H - 1)
            is_left = (c == 0)
            is_right = (c == W - 1)

            # Parity
            row_parity = r % 2
            col_parity = c % 2

            # Bin index: encode edge flags + parity
            # Use a simple encoding: (edge_type, row_parity, col_parity)
            if is_top and is_left:
                edge_type = 0  # corner TL
            elif is_top and is_right:
                edge_type = 1  # corner TR
            elif is_bottom and is_left:
                edge_type = 2  # corner BL
            elif is_bottom and is_right:
                edge_type = 3  # corner BR
            elif is_top:
                edge_type = 4  # edge top
            elif is_bottom:
                edge_type = 5  # edge bottom
            elif is_left:
                edge_type = 6  # edge left
            elif is_right:
                edge_type = 7  # edge right
            else:
                edge_type = 8  # interior

            # Bin id: edge_type * 4 + row_parity * 2 + col_parity
            bin_ids[p] = edge_type * 4 + row_parity * 2 + col_parity

    # WO-B Fix: Renumber bin_ids to be consecutive [0, 1, ..., num_bins-1]
    # Original IDs are non-consecutive (e.g., [0, 4, 8, 12, 16, 17, ...])
    # Downstream WO-05 quota building expects consecutive range
    unique_bin_ids = np.unique(bin_ids)
    bin_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_bin_ids)}
    bin_ids = np.array([bin_id_map[bid] for bid in bin_ids], dtype=np.int32)

    num_bins = len(np.unique(bin_ids))

    # Compute hash for determinism
    wo4_hash = hashlib.sha256(
        A_mask.tobytes() + bin_ids.tobytes()
    ).hexdigest()

    meta = {
        "H": H,
        "W": W,
        "N": N,
        "num_trains": len(train_outputs_embedded),
        "num_bins": num_bins,
        "forward_meet_applied": True,  # Phase 3 flag
        "hash": wo4_hash,
    }

    return {
        "A_mask": A_mask,
        "bin_ids": bin_ids,
        "meta": meta,
    }


def wo5_per_canvas(
    train_outputs_embedded: List[np.ndarray],
    A_mask: np.ndarray,
    bin_ids: np.ndarray,
    H: int,
    W: int,
    train_output_sizes: List[Tuple[int, int]],
) -> Dict:
    """
    WO-5 per-canvas: Quotas, faces, equalizers for given canvas dimensions.

    Phase 2 implementation: Only compute faces when all trainings naturally match
    canvas size (no embedding).

    Args:
        train_outputs_embedded: List of training outputs embedded to (H, W)
        A_mask: (N, C) bool array from WO-4
        bin_ids: (N,) int array from WO-4
        H: Canvas height
        W: Canvas width
        train_output_sizes: List of (H_i, W_i) native sizes BEFORE embedding

    Returns:
        Dict with:
        - quotas: (num_bins, C) int64 array
        - faces_R: (H, C) int64 array or None
        - faces_S: (W, C) int64 array or None
        - equalizer_edges: dict {(s,c): [(p_i,p_j), ...]}
        - meta: {H, W, num_bins, has_faces, native_size_match, faces_R_sum, faces_S_sum}
    """
    N = H * W
    C = 10
    num_bins = len(np.unique(bin_ids))

    # === QUOTAS (anchor §8) ===
    # Per-bin quotas: q[s,c] = min_i #{p∈B_s: Y_i(p)=c}
    quotas = np.full((num_bins, C), np.iinfo(np.int64).max, dtype=np.int64)

    for Y_i in train_outputs_embedded:
        Y_i_flat = Y_i.flatten()
        for s in range(num_bins):
            bin_mask = (bin_ids == s)
            for c in range(C):
                count = np.sum((Y_i_flat[bin_mask] == c))
                quotas[s, c] = min(quotas[s, c], count)

    # Replace inf with 0 (bins never seen)
    quotas[quotas == np.iinfo(np.int64).max] = 0

    # === FACES (anchor §8) ===
    # Phase 2 fix: Only compute faces when ALL trainings naturally match canvas
    # Check if all trainings have native size == canvas size
    all_native_match = all(
        H_i == H and W_i == W
        for H_i, W_i in train_output_sizes
    )

    if all_native_match:
        # All trainings naturally match canvas → compute faces (no -1 padding)
        # Faces: meet of per-row/column color counts across trainings
        faces_R = np.full((H, C), np.iinfo(np.int64).max, dtype=np.int64)
        faces_S = np.full((W, C), np.iinfo(np.int64).max, dtype=np.int64)

        for Y_i in train_outputs_embedded:
            # Row faces
            for r in range(H):
                row = Y_i[r, :]
                for c in range(C):
                    count = np.sum(row == c)
                    faces_R[r, c] = min(faces_R[r, c], count)

            # Column faces
            for j in range(W):
                col = Y_i[:, j]
                for c in range(C):
                    count = np.sum(col == c)
                    faces_S[j, c] = min(faces_S[j, c], count)

        # Replace inf with 0
        faces_R[faces_R == np.iinfo(np.int64).max] = 0
        faces_S[faces_S == np.iinfo(np.int64).max] = 0

        faces_R_sum = int(np.sum(faces_R))
        faces_S_sum = int(np.sum(faces_S))
        has_faces = (faces_R_sum > 0 or faces_S_sum > 0)
    else:
        # Trainings have different native sizes → no faces
        # Cannot compute meaningful "meet of counts" when comparing embedded outputs
        faces_R = None
        faces_S = None
        faces_R_sum = 0
        faces_S_sum = 0
        has_faces = False

    # === EQUALIZER EDGES (anchor §4) ===
    # For each (bin, color), if trainings prove "constant over the bin" on allowed set,
    # create spanning tree edges
    equalizer_edges = {}

    for s in range(num_bins):
        bin_mask = (bin_ids == s)
        bin_pixels = np.where(bin_mask)[0]

        if len(bin_pixels) < 2:
            continue  # Skip singleton bins

        for c in range(C):
            # Check if all trainings have constant color c on this bin (allowed pixels only)
            allowed_in_bin = [p for p in bin_pixels if A_mask[p, c]]

            if len(allowed_in_bin) < 2:
                continue  # Skip if <2 allowed pixels

            # Check constancy across trainings
            is_constant = True
            for Y_i in train_outputs_embedded:
                Y_i_flat = Y_i.flatten()
                values = [Y_i_flat[p] for p in allowed_in_bin if Y_i_flat[p] >= 0]

                # Must have values and all must equal c
                # Empty bins or non-constant values → fail constancy check
                if len(values) == 0 or not all(v == c for v in values):
                    is_constant = False
                    break

            if is_constant and len(allowed_in_bin) >= 2:
                # Create spanning tree edges (breadth-first from first pixel)
                # Build adjacency graph (4-connected on canvas)
                adj_matrix = np.zeros((N, N), dtype=bool)

                for p in allowed_in_bin:
                    r, col = divmod(p, W)
                    # 4-connected neighbors
                    neighbors = []
                    if r > 0:
                        neighbors.append((r - 1) * W + col)
                    if r < H - 1:
                        neighbors.append((r + 1) * W + col)
                    if col > 0:
                        neighbors.append(r * W + col - 1)
                    if col < W - 1:
                        neighbors.append(r * W + col + 1)

                    for q in neighbors:
                        if q in allowed_in_bin:
                            adj_matrix[p, q] = True

                # Build spanning tree via BFS
                adj_sparse = csr_matrix(adj_matrix)
                root = allowed_in_bin[0]

                # Use scipy BFS to get spanning tree
                tree = breadth_first_tree(adj_sparse, root, directed=False)

                # Extract edges from tree
                edges = []
                tree_coo = tree.tocoo()
                for i, j in zip(tree_coo.row, tree_coo.col):
                    if i < j:  # Canonicalize edge direction
                        edges.append((int(i), int(j)))

                if len(edges) > 0:
                    equalizer_edges[(s, c)] = edges

    meta = {
        "H": H,
        "W": W,
        "num_bins": num_bins,
        "has_faces": has_faces,
        "native_size_match": all_native_match,  # Phase 2 metadata
        "faces_R_sum": faces_R_sum,
        "faces_S_sum": faces_S_sum,
    }

    return {
        "quotas": quotas,
        "faces_R": faces_R,
        "faces_S": faces_S,
        "equalizer_edges": equalizer_edges,
        "meta": meta,
    }


def wo6_per_canvas(
    train_outputs_embedded: List[np.ndarray],
    A_mask: np.ndarray,
    bin_ids: np.ndarray,
    H: int,
    W: int,
    free_maps_verified: List[Dict],
) -> Dict:
    """
    WO-6 per-canvas: Π-safe scores + FREE projection + int64 costs.

    Implements WO-B Phase 1: Per-canvas cost computation with verified FREE maps.

    Args:
        train_outputs_embedded: List of training outputs embedded to (H, W)
        A_mask: (N, C) bool array from WO-4
        bin_ids: (N,) int array from WO-4
        H: Canvas height
        W: Canvas width
        free_maps_verified: List of verified FREE map dicts from task-level WO-6

    Returns:
        Dict with:
        - costs: (N, C) int64 array
        - meta: {H, W, N, C, pi_safe_ok, free_invariance_ok, costs_hash}
    """
    from . import scores as scores_module
    from .config import SCALE

    N = H * W
    C = 10

    # === BUILD Π-SAFE SCORES ===
    # Scores depend ONLY on bins/mask (anchor §6, 01_addendum.md §B)
    stage_features = None  # Can be extended later with harmonic/gravity features

    scores = scores_module.build_scores_pi_safe(
        H=H,
        W=W,
        A_mask=A_mask,
        bin_ids=bin_ids,
        stage_features=stage_features,
    )

    # Mark Π-safe (all inputs are geometry-only)
    pi_safe_ok = True

    # === APPLY FREE MAP PROJECTION ===
    # Project scores using Haar projector (Reynolds operator) over verified FREE symmetries
    # This ensures ŝ∘U = ŝ for all verified U (anchor 01_addendum.md §B)
    free_invariance_ok = True

    if free_maps_verified and len(free_maps_verified) > 0:
        # Apply each FREE map and average (Haar projector)
        projected_scores = np.zeros_like(scores, dtype=np.float64)

        for U in free_maps_verified:
            scores_transformed = scores_module.apply_U_to_scores(scores, U, H, W)
            projected_scores += scores_transformed

        # Average over symmetry group
        projected_scores /= len(free_maps_verified)
        scores = projected_scores

        # Verify FREE invariance (cost invariance after projection)
        for U in free_maps_verified:
            costs_test = scores_module.to_int_costs(scores)
            costs_transformed = scores_module.apply_U_to_costs(costs_test, U, H, W)
            if not np.array_equal(costs_test, costs_transformed):
                free_invariance_ok = False
                break

    # === CONVERT TO INTEGER COSTS ===
    # Per Annex A.2: costs = round(-ŝ * SCALE) as int64
    costs = scores_module.to_int_costs(scores)

    # Compute hash for determinism
    costs_hash = hashlib.sha256(costs.tobytes()).hexdigest()

    meta = {
        "H": H,
        "W": W,
        "N": N,
        "C": C,
        "pi_safe_ok": pi_safe_ok,
        "free_invariance_ok": free_invariance_ok,
        "costs_hash": costs_hash,
    }

    return {
        "costs": costs,
        "meta": meta,
    }
