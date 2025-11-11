#!/usr/bin/env python3
"""
Canvas Operations: Per-canvas WO-4/5 callables for WO-A

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
    """
    N = H * W
    C = 10  # ARC-AGI palette size

    # === STEP 1: GATHER (Forward Meet) ===
    # For each (p,k), track: seen, unique output color, ambiguous flag
    seen = np.zeros((N, C), dtype=bool)
    unique = -np.ones((N, C), dtype=np.int16)
    amb = np.zeros((N, C), dtype=bool)

    for X_i, Y_i in zip(train_inputs_embedded, train_outputs_embedded):
        x = X_i.flatten()
        y = Y_i.flatten()
        valid = (x >= 0) & (y >= 0)  # Skip -1 padding in both input and output

        for p in np.nonzero(valid)[0]:
            k = int(x[p])
            c = int(y[p])

            if not seen[p, k]:
                seen[p, k] = True
                unique[p, k] = c
            elif unique[p, k] != c:
                # Different output for same input → ambiguous
                amb[p, k] = True

    # === BUILD F[p,k,c] ===
    F = np.ones((N, C, C), dtype=bool)  # Start with all allowed

    for p in range(N):
        for k in range(C):
            if not seen[p, k]:
                # No evidence for this (p,k) → leave broad (will be lifted or left open)
                continue

            if amb[p, k]:
                # Ambiguous: multiple outputs for same input → forbid all
                F[p, k, :] = False
            else:
                # Unique mapping: only one output color allowed
                c_star = unique[p, k]
                F[p, k, :] = False
                F[p, k, c_star] = True

    # === STEP 2: SUFFICIENCY CLOSURE ===
    # One pass to enforce order-independence and idempotence
    # This ensures F is monotone (only 1→0 besides the unique 1)
    # CRITICAL: Skip ambiguous (p,k) pairs - they should remain all False
    for X_i, Y_i in zip(train_inputs_embedded, train_outputs_embedded):
        x = X_i.flatten()
        y = Y_i.flatten()
        valid = (x >= 0) & (y >= 0)

        idx = np.nonzero(valid)[0]
        if len(idx) == 0:
            continue

        kvec = x[idx].astype(int)
        cvec = y[idx].astype(int)

        # Enforce: F[p,k,c*]=1, F[p,k,≠c*]=0
        # BUT: skip if ambiguous (Step 1 already set all to False)
        for i, p in enumerate(idx):
            k = kvec[i]
            c = cvec[i]
            if not amb[p, k]:  # Only enforce if not ambiguous
                F[p, k, :] = False
                F[p, k, c] = True

    # === STEP 3: COLOR-AGNOSTIC LIFT ===
    # If all observed k at pixel p share same admits row, extend to unseen k
    for p in range(N):
        obs_k = np.nonzero(seen[p, :])[0]

        if len(obs_k) == 0:
            # No evidence at this pixel → leave broad
            continue

        # Check if all observed k share the same admits row
        first_row = F[p, obs_k[0], :]
        consensus = all(np.array_equal(F[p, k, :], first_row) for k in obs_k[1:])

        if consensus:
            # Lift consensus to unseen input colors
            unseen_k = np.setdiff1d(np.arange(C), obs_k, assume_unique=True)
            F[p, unseen_k, :] = first_row

    # === STEP 4: EXTRACT TEST MASK ===
    # A_mask[p,c] = F[p, test_input(p), c]
    A_mask = np.zeros((N, C), dtype=bool)
    test_flat = test_input_embedded.flatten()

    for p in range(N):
        k_test = test_flat[p]
        if k_test >= 0:  # Skip -1 padding in test input
            A_mask[p, :] = F[p, int(k_test), :]
        else:
            # Padded position in test → leave broad (all allowed)
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
    WO-4 per-canvas: Compute A_mask and bin_ids on given canvas.

    CRITICAL (Phase 3 fix): A_mask now computed via forward meet + closure + lift
    per anchor §5, conditioned on input colors. This is TEST-SPECIFIC - different
    tests produce different masks.

    Args:
        train_inputs_raw: List of training input grids (native sizes, before embedding)
        train_outputs_embedded: List of embedded training outputs on this canvas
        test_input_raw: Test input grid (native size, before embedding)
        H: Canvas height
        W: Canvas width
        embed_mode: Embedding mode used

    Returns:
        Dict with:
            - A_mask: (N, C) bool - test-specific mask from forward meet
            - bin_ids: (N,) int64
            - meta: {H, W, embed_mode, num_trains, hash, forward_meet_applied}
    """
    N = H * W
    C = 10  # ARC-AGI palette size

    # === EMBED INPUTS TO CANVAS (Phase 3) ===
    train_inputs_embedded = [
        align_to_canvas(X_i, H, W, embed_mode)
        for X_i in train_inputs_raw
    ]

    test_input_embedded = align_to_canvas(test_input_raw, H, W, embed_mode)

    # === FORWARD MEET + CLOSURE + LIFT (anchor §5) ===
    A_mask = forward_meet_closure_lift(
        train_inputs_embedded=train_inputs_embedded,
        train_outputs_embedded=train_outputs_embedded,
        test_input_embedded=test_input_embedded,
        H=H,
        W=W,
    )

    # === BIN ASSIGNMENT (periphery-parity bins per anchor §4) ===
    # Bins are intersections of edge flags (top/bottom/left/right/interior) with parity
    bin_ids = np.zeros(N, dtype=np.int64)

    for r in range(H):
        for c in range(W):
            p = r * W + c

            # Edge flags
            is_top = (r == 0)
            is_bottom = (r == H - 1)
            is_left = (c == 0)
            is_right = (c == W - 1)

            # Parity
            r_parity = r % 2
            c_parity = c % 2

            # Bin encoding (stable across runs)
            # Use: (edge_flags × 16 + r_parity × 2 + c_parity)
            edge_flags = (int(is_top) << 3) | (int(is_bottom) << 2) | \
                        (int(is_left) << 1) | int(is_right)
            bin_id = edge_flags * 4 + r_parity * 2 + c_parity

            bin_ids[p] = bin_id

    # Compute hash for determinism (Phase 3: include inputs and test input)
    hash_input = f"wo4_v3:{H}x{W}:{embed_mode}:{len(train_inputs_raw)}"
    for X_i in train_inputs_raw:
        hash_input += f":X{hashlib.sha256(X_i.tobytes()).hexdigest()[:8]}"
    for Y_i in train_outputs_embedded:
        hash_input += f":Y{hashlib.sha256(Y_i.tobytes()).hexdigest()[:8]}"
    hash_input += f":test{hashlib.sha256(test_input_raw.tobytes()).hexdigest()[:8]}"
    wo4_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    meta = {
        "H": H,
        "W": W,
        "N": N,
        "embed_mode": embed_mode,
        "num_trains": len(train_inputs_raw),
        "num_bins": int(bin_ids.max()) + 1,
        "forward_meet_applied": True,  # Phase 3: flag for audit
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
    WO-5 per-canvas: Compute quotas, faces, equalizer edges on given canvas.

    Extracted from harness.py WO-5 stage. Computes:
    - quotas: (num_bins, C) int64 quota targets per bin/color
    - faces_R: (H, C) int64 per-row color counts (meet across trainings)
    - faces_S: (W, C) int64 per-column color counts (meet across trainings)
    - equalizer_edges: Dict[(s,c)] -> edges array (E×2)

    CRITICAL (Phase 2 fix): Faces are ONLY computed when all training outputs
    naturally match the target canvas size (no embedding needed). When trainings
    have different native sizes, faces are set to None per anchor §8.

    Args:
        train_outputs_embedded: List of embedded training outputs on this canvas
        A_mask: (N, C) bool allowed colors
        bin_ids: (N,) int64 bin assignments
        H: Canvas height
        W: Canvas width
        train_output_sizes: List of (H_i, W_i) native sizes BEFORE embedding

    Returns:
        Dict with:
            - quotas: (num_bins, C) int64
            - faces_R: (H, C) int64 or None
            - faces_S: (W, C) int64 or None
            - equalizer_edges: Dict[(s,c)] -> (E×2) array
            - meta: {num_bins, has_faces, native_size_match, hash}
    """
    N = H * W
    C = 10
    num_bins = int(bin_ids.max()) + 1

    # === QUOTAS (anchor §8 line 124-127) ===
    # q[s,c] = min_i #{p ∈ B_s : Y_i(p)=c}
    quotas = np.zeros((num_bins, C), dtype=np.int64)

    # Build bins list
    bins_list = []
    for s in range(num_bins):
        bin_pixels = np.where(bin_ids == s)[0]
        bins_list.append(bin_pixels)

    for c in range(C):
        for s, bin_pixels in enumerate(bins_list):
            counts_per_example = []
            for Y_i in train_outputs_embedded:
                Y_i_flat = Y_i.flatten()
                # Count pixels in this bin with color c (skip -1 padding)
                count_in_output = sum(
                    1 for p in bin_pixels
                    if p < len(Y_i_flat) and Y_i_flat[p] == c
                )
                counts_per_example.append(count_in_output)

            quota = min(counts_per_example) if counts_per_example else 0
            quotas[s, c] = quota

    # === FACES (anchor §8 line 128) ===
    # CRITICAL (Phase 2 fix): Only compute faces when ALL training outputs
    # naturally match the target canvas size (no embedding needed).
    # When trainings have different native sizes, comparing counts across
    # embedded outputs with -1 padding violates "meet of counts" semantics.

    # Check if all trainings naturally match canvas
    all_native_match = all(
        H_i == H and W_i == W
        for H_i, W_i in train_output_sizes
    )

    if all_native_match:
        # All trainings naturally match canvas → compute faces
        faces_R = np.zeros((H, C), dtype=np.int64)
        faces_S = np.zeros((W, C), dtype=np.int64)

        for c in range(C):
            # Per-row faces
            for r in range(H):
                counts_per_example = []
                for Y_i in train_outputs_embedded:
                    # Count color c in row r (no -1 padding since all native)
                    count_in_row = np.sum(Y_i[r, :] == c)
                    counts_per_example.append(count_in_row)
                faces_R[r, c] = min(counts_per_example) if counts_per_example else 0

            # Per-column faces
            for j in range(W):
                counts_per_example = []
                for Y_i in train_outputs_embedded:
                    # Count color c in column j (no -1 padding since all native)
                    count_in_col = np.sum(Y_i[:, j] == c)
                    counts_per_example.append(count_in_col)
                faces_S[j, c] = min(counts_per_example) if counts_per_example else 0

        # Check if faces are non-trivial
        faces_R_sum = int(faces_R.sum())
        faces_S_sum = int(faces_S.sum())
        has_faces = (faces_R_sum > 0) or (faces_S_sum > 0)

        # Return None if no faces (matches WO-A spec)
        if not has_faces:
            faces_R = None
            faces_S = None
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
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, col + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            q = nr * W + nc
                            if q in allowed_in_bin:
                                adj_matrix[p, q] = True

                # Use BFS to get spanning tree
                csr_adj = csr_matrix(adj_matrix)
                bfs_tree = breadth_first_tree(csr_adj, allowed_in_bin[0], directed=False)

                # Extract edges
                edges_list = []
                for i in range(bfs_tree.shape[0]):
                    for j in bfs_tree.getrow(i).indices:
                        if i < j:  # Avoid duplicates
                            edges_list.append([i, j])

                if len(edges_list) > 0:
                    equalizer_edges[(s, c)] = np.array(edges_list, dtype=np.int64)

    # Compute hash for determinism
    hash_input = f"wo5:{H}x{W}:{num_bins}:{len(train_outputs_embedded)}"
    hash_input += f":quotas={hashlib.sha256(quotas.tobytes()).hexdigest()[:8]}"
    if faces_R is not None:
        hash_input += f":faces_R={hashlib.sha256(faces_R.tobytes()).hexdigest()[:8]}"
    if faces_S is not None:
        hash_input += f":faces_S={hashlib.sha256(faces_S.tobytes()).hexdigest()[:8]}"
    wo5_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    meta = {
        "H": H,
        "W": W,
        "num_bins": num_bins,
        "has_faces": has_faces,
        "native_size_match": all_native_match,  # Phase 2: document why faces present/absent
        "faces_R_sum": faces_R_sum if has_faces else 0,
        "faces_S_sum": faces_S_sum if has_faces else 0,
        "hash": wo5_hash,
    }

    return {
        "quotas": quotas,
        "faces_R": faces_R,
        "faces_S": faces_S,
        "equalizer_edges": equalizer_edges,
        "meta": meta,
    }
