"""WO-3 color_align.py - Color signatures + Hungarian alignment.

Implements Π-safe per-color signatures and deterministic palette alignment:
- build_color_signatures: Π-safe signatures (count, histograms)
- canonical_palette_and_cost: Lex-sorted canonical order + cost matrices
- align_one_training: Hungarian assignment wrapper
- align_colors: Public API for color alignment

All operations are deterministic (int64, stable sorting, lex tie-break).
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment
import hashlib
from .config import INT_DTYPE, PALETTE_C


# Lex tie-break scale for cost matrix (safe margin so j never changes order)
LEX_SCALE = 1024


def build_color_signatures(
    Y_emb: np.ndarray, bin_ids: np.ndarray, num_bins: int
) -> Dict[int, Tuple]:
    """Build Π-safe per-color signatures.

    Args:
        Y_emb: Embedded grid (H_out, W_out) with int32, values in {-1, 0..9}
        bin_ids: Flattened bin IDs (H_out*W_out,) from WO-1, int64, raster order
        num_bins: Number of bins

    Returns:
        Dict mapping color → signature tuple:
            Σ(c) = (-count, row_hist, col_hist, bin_hist, color_id)
            where histograms are lists of ints

    Notes:
        - -count makes "more pixels" lex-smaller
        - Ignores -1 (padding sentinel)
        - All counts computed as int64
        - Anchors: 03_annex.md A.3, 04_engg_spec.md §3, 05_contracts.md
    """
    if Y_emb.ndim != 2:
        raise ValueError(f"Y_emb must be 2D, got shape {Y_emb.shape}")

    H_out, W_out = Y_emb.shape

    if bin_ids.shape != (H_out * W_out,):
        raise ValueError(
            f"bin_ids must be flattened (H*W,), got {bin_ids.shape} for canvas {Y_emb.shape}"
        )

    signatures = {}

    for c in range(PALETTE_C):  # 0..9
        # Mask for color c (ignore -1 padding)
        M = (Y_emb == c)

        # Count
        count = int(M.sum(dtype=INT_DTYPE))

        # Row histogram
        rows = np.nonzero(M)[0]
        row_hist = np.bincount(rows, minlength=H_out).astype(INT_DTYPE)

        # Col histogram
        cols = np.nonzero(M)[1]
        col_hist = np.bincount(cols, minlength=W_out).astype(INT_DTYPE)

        # Bin histogram
        # Flatten M and select bin_ids where M is True
        Mf = M.ravel(order="C")
        bin_indices = bin_ids[Mf]
        bin_hist = np.bincount(bin_indices, minlength=num_bins).astype(INT_DTYPE)

        # Build signature tuple (JSON-serializable)
        signature = (
            -count,  # Negative so more pixels → lex-smaller
            row_hist.tolist(),
            col_hist.tolist(),
            bin_hist.tolist(),
            c,  # Color ID for lex tie-break
        )

        signatures[c] = signature

    return signatures


def canonical_palette_and_cost(
    Y_emb_list: List[np.ndarray], bin_ids: np.ndarray, num_bins: int
) -> Tuple[List[Tuple], List[np.ndarray], List[Dict[int, Tuple]]]:
    """Compute canonical palette order and cost matrices for Hungarian.

    Args:
        Y_emb_list: List of embedded training outputs
        bin_ids: Flattened bin IDs from WO-1
        num_bins: Number of bins

    Returns:
        canonical_sigs: List of 10 canonical signature tuples (lex-sorted)
        costs_per_training: List of int64 cost matrices (10×10) per training
        sigs_per_training: List of signature dicts per training

    Notes:
        - Canonical signatures: lex-sort all signatures across trainings, take first 10
        - Cost matrix: C[i,j] = base_cost * LEX_SCALE + j (deterministic tie-break)
        - Base cost: L1 distance between flattened signature vectors
        - Anchors: 04_engg_spec.md §3, 05_contracts.md
    """
    # Compute signatures for each training
    sigs_per_training = []
    for Y_emb in Y_emb_list:
        sigs = build_color_signatures(Y_emb, bin_ids, num_bins)
        sigs_per_training.append(sigs)

    # Build canonical palette by lex-sorting all signatures
    all_sigs = []
    for sigs in sigs_per_training:
        for c in range(PALETTE_C):
            all_sigs.append(sigs[c])

    # Stable lex sort to get canonical order
    sorted_sigs = sorted(all_sigs, key=lambda x: x)

    # Take first 10 signatures as canonical representatives (one per slot)
    canonical_sigs = sorted_sigs[:PALETTE_C]

    # Build cost matrices
    costs_per_training = []
    for sigs in sigs_per_training:
        # Build 10×10 cost matrix
        C = np.zeros((PALETTE_C, PALETTE_C), dtype=INT_DTYPE)

        for i in range(PALETTE_C):
            sig_i = sigs[i]
            for j in range(PALETTE_C):
                sig_j = canonical_sigs[j]

                # Base cost: L1 distance between flattened signatures
                # Flatten: [-count, row_hist..., col_hist..., bin_hist...]
                vec_i = np.concatenate([
                    np.array([sig_i[0]], dtype=INT_DTYPE),
                    np.array(sig_i[1], dtype=INT_DTYPE),
                    np.array(sig_i[2], dtype=INT_DTYPE),
                    np.array(sig_i[3], dtype=INT_DTYPE),
                ])
                vec_j = np.concatenate([
                    np.array([sig_j[0]], dtype=INT_DTYPE),
                    np.array(sig_j[1], dtype=INT_DTYPE),
                    np.array(sig_j[2], dtype=INT_DTYPE),
                    np.array(sig_j[3], dtype=INT_DTYPE),
                ])

                base_cost = int(np.abs(vec_i - vec_j).sum())

                # Encode lex tie-break: C[i,j] = base_cost * LEX_SCALE + j
                C[i, j] = base_cost * LEX_SCALE + j

        costs_per_training.append(C)

    return canonical_sigs, costs_per_training, sigs_per_training


def align_one_training(cost_matrix: np.ndarray) -> np.ndarray:
    """Solve Hungarian assignment for one training (deterministic).

    Args:
        cost_matrix: int64 (10, 10) cost matrix with lex tie-break encoded

    Returns:
        perm: Permutation array (10,) where perm[orig_color] = assigned_slot

    Notes:
        - Uses scipy.optimize.linear_sum_assignment
        - Asserts bijection (all 0-9 appear exactly once)
        - Deterministic via encoded +j tie-break in costs
        - Anchors: 04_engg_spec.md §3
    """
    if cost_matrix.shape != (PALETTE_C, PALETTE_C):
        raise ValueError(
            f"Cost matrix must be ({PALETTE_C}, {PALETTE_C}), got {cost_matrix.shape}"
        )

    # Solve Hungarian
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build permutation: perm[orig_color] = assigned_slot
    perm = np.zeros(PALETTE_C, dtype=INT_DTYPE)
    for i in range(PALETTE_C):
        perm[row_ind[i]] = col_ind[i]

    # Assert bijection
    if not (set(perm) == set(range(PALETTE_C))):
        raise RuntimeError(
            f"Hungarian result is not a bijection: perm={perm}, "
            f"row_ind={row_ind}, col_ind={col_ind}"
        )

    return perm


def align_colors(
    train_outputs_emb: List[np.ndarray], bin_ids: np.ndarray, num_bins: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[int, Tuple]], List[Tuple], List[str], List[int]]:
    """Align colors across training outputs using Hungarian algorithm.

    Args:
        train_outputs_emb: List of embedded training outputs
        bin_ids: Flattened bin IDs from WO-1
        num_bins: Number of bins

    Returns:
        aligned_outputs: List of relabeled grids
        perms: List of permutation arrays (one per training)
        sigs: List of signature dicts (one per training)
        canonical_sigs: List of 10 canonical signature tuples (lex-sorted)
        cost_hashes: List of SHA-256 hashes of cost matrices
        total_costs: List of total costs per training

    Notes:
        - Computes canonical signatures by lex-sorting all signatures
        - Solves Hungarian per training
        - Relabels channels according to permutation
        - Hashes cost matrices for receipt
        - Anchors: 04_engg_spec.md §3, 05_contracts.md
    """
    if len(train_outputs_emb) == 0:
        raise ValueError("Need at least one training output")

    # Compute canonical signatures and cost matrices
    canonical_sigs, costs_per_training, sigs = canonical_palette_and_cost(
        train_outputs_emb, bin_ids, num_bins
    )

    # Solve Hungarian per training
    perms = []
    cost_hashes = []
    total_costs = []
    for cost_matrix in costs_per_training:
        perm = align_one_training(cost_matrix)
        perms.append(perm)

        # Hash cost matrix
        cost_hash = hashlib.sha256(cost_matrix.tobytes(order="C")).hexdigest()
        cost_hashes.append(cost_hash)

        # Compute total cost (sum of selected entries)
        # Hungarian returns row_ind, col_ind; we need to reconstruct them from perm
        row_ind = np.arange(PALETTE_C, dtype=INT_DTYPE)
        col_ind = perm
        total_cost = int(cost_matrix[row_ind, col_ind].sum())
        total_costs.append(total_cost)

    # Relabel channels (create aligned outputs)
    aligned_outputs = []
    for Y_emb, perm in zip(train_outputs_emb, perms):
        # Create inverse permutation for relabeling
        inv_perm = np.zeros(PALETTE_C, dtype=INT_DTYPE)
        for orig_c in range(PALETTE_C):
            inv_perm[perm[orig_c]] = orig_c

        # Relabel: for each pixel, map old color to new color
        aligned = np.full_like(Y_emb, -1)
        for new_c in range(PALETTE_C):
            orig_c = inv_perm[new_c]
            aligned[Y_emb == orig_c] = new_c

        # Preserve padding
        aligned[Y_emb == -1] = -1

        aligned_outputs.append(aligned)

    return aligned_outputs, perms, sigs, canonical_sigs, cost_hashes, total_costs
