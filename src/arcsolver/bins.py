"""WO-1 bins.py - Periphery-parity bins, bbox, and center predicate.

Implements canonical bin structure and geometric predicates for ARC grids:
- Periphery-parity bins (disjoint, deterministic)
- Content bounding box (content = values != 0)
- Center predicate for embedding mode selection

All operations are Π-safe (byte-exact, deterministic).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.ndimage import center_of_mass
from .config import GRID_DTYPE, INT_DTYPE


def build_bins(H: int, W: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build periphery-parity bins for an H×W canvas.

    Args:
        H: Canvas height (rows)
        W: Canvas width (columns)

    Returns:
        bin_ids: np.ndarray[int64] shape (H*W,) giving bin index per pixel
        bins: list of np.ndarray[int64], each containing pixel indices in that bin

    Notes:
        - Bins are intersections of (periphery flag) × (parity pair)
        - Periphery: top row, bottom row, left col, right col, interior
        - Parity: (r%2, c%2)
        - Empty bins are dropped
        - All pixel lists sorted in raster order
        - Anchors: 03_annex.md A.3, 04_engg_spec.md §5
    """
    if H <= 0 or W <= 0:
        raise ValueError(f"Canvas dimensions must be positive, got H={H}, W={W}")

    N = H * W

    # Create coordinate grids (raster order)
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    rows = rows.ravel()  # Shape (H*W,)
    cols = cols.ravel()  # Shape (H*W,)

    # Periphery flags (mutually exclusive, priority order)
    # Priority: top/bottom rows first, then left/right cols, then interior
    is_top = (rows == 0)
    is_bottom = (rows == H - 1) & ~is_top
    is_left = (cols == 0) & ~is_top & ~is_bottom
    is_right = (cols == W - 1) & ~is_top & ~is_bottom & ~is_left
    is_interior = ~(is_top | is_bottom | is_left | is_right)

    periphery_flags = [
        ('top', is_top),
        ('bottom', is_bottom),
        ('left', is_left),
        ('right', is_right),
        ('interior', is_interior),
    ]

    # Parity flags
    parity_r = rows % 2
    parity_c = cols % 2
    parity_pairs = [
        ((0, 0), (parity_r == 0) & (parity_c == 0)),
        ((0, 1), (parity_r == 0) & (parity_c == 1)),
        ((1, 0), (parity_r == 1) & (parity_c == 0)),
        ((1, 1), (parity_r == 1) & (parity_c == 1)),
    ]

    # Build bins as intersections
    bin_ids = np.full(N, -1, dtype=INT_DTYPE)
    bins = []
    bin_idx = 0

    # Iterate in deterministic order (periphery, then parity)
    for periph_name, periph_mask in periphery_flags:
        for (pr, pc), parity_mask in parity_pairs:
            # Intersection
            combined_mask = periph_mask & parity_mask
            pixel_indices = np.where(combined_mask)[0]

            # Drop empty bins
            if len(pixel_indices) == 0:
                continue

            # Assign bin ID (already in raster order from np.where)
            bin_ids[pixel_indices] = bin_idx
            bins.append(pixel_indices)
            bin_idx += 1

    # Verify all pixels assigned
    if np.any(bin_ids == -1):
        raise RuntimeError("Some pixels were not assigned to any bin")

    return bin_ids, bins


def bbox_content(grid: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Find bounding box of content (values != 0) in grid.

    Args:
        grid: Grid array (H, W) with int32 values

    Returns:
        (r0, r1, c0, c1) in Python slice convention (end-exclusive), or None if no content

    Notes:
        - Content: values != 0 (background is 0)
        - Padding sentinel -1 is treated as 0 (not content)
        - Byte-exact integer operations only
        - Anchors: 04_engg_spec.md §1 (content definition), 05_contracts.md
    """
    if grid.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape {grid.shape}")

    # Content mask: values != 0 (treat -1 as 0)
    content_mask = (grid != 0) & (grid != -1)

    # Find content pixels
    rows, cols = np.where(content_mask)

    # No content
    if len(rows) == 0:
        return None

    # Compute bbox (Python slice convention: end-exclusive)
    r0 = int(rows.min())
    r1 = int(rows.max()) + 1
    c0 = int(cols.min())
    c1 = int(cols.max()) + 1

    return (r0, r1, c0, c1)


def center_predicate_all(
    train_outputs_emb: list[np.ndarray], H_out: int, W_out: int
) -> bool:
    """Check if all training outputs satisfy centering predicate.

    Args:
        train_outputs_emb: List of embedded training output grids
        H_out: Output canvas height
        W_out: Output canvas width

    Returns:
        True if all training outputs have content centroid within 0.5 of canvas center
        (implies 'center' embedding mode); False otherwise (implies 'topleft')

    Notes:
        - Canvas center: ((H_out-1)/2, (W_out-1)/2)
        - Content: values != 0 (padding -1 treated as 0)
        - Uses scipy.ndimage.center_of_mass for sub-pixel centroids
        - Threshold: |delta_row| ≤ 0.5 AND |delta_col| ≤ 0.5 for ALL trainings
        - Anchors: 05_contracts.md (Canonical predicates)
    """
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Canvas dimensions must be positive, got H={H_out}, W={W_out}")

    if len(train_outputs_emb) == 0:
        raise ValueError("Need at least one training output to check predicate")

    # Canvas center
    center_r = (H_out - 1) / 2.0
    center_c = (W_out - 1) / 2.0

    # Check each training
    for grid in train_outputs_emb:
        if grid.shape != (H_out, W_out):
            raise ValueError(
                f"Training grid shape {grid.shape} does not match canvas ({H_out}, {W_out})"
            )

        # Content mask: values != 0 (treat -1 as 0)
        content_mask = (grid != 0) & (grid != -1)

        # Handle empty content (all zeros)
        if not np.any(content_mask):
            # No content → centroid undefined, fails centering predicate
            return False

        # Compute centroid using scipy
        com = center_of_mass(content_mask)
        com_r, com_c = com

        # Check threshold
        delta_r = abs(com_r - center_r)
        delta_c = abs(com_c - center_c)

        if delta_r > 0.5 or delta_c > 0.5:
            return False

    # All trainings satisfy predicate
    return True
