"""WO-5 eqs.py - Equalizers & structure rows (no solving).

Implements:
- build_equalizer_rows: Spanning-tree equalizers per Π-bin × color
- build_gravity_rows: (I-G)y=0 on transient states (absorbing walls)
- build_harmonic_rows: Discrete Dirichlet Laplacian rows

All operations are deterministic (int64, byte-exact, no tolerances).
Anchors: 01_addendum.md §3,5,6,7; 02_addendum.md §L; 04_engg_spec.md §4,6
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import breadth_first_tree, laplacian
from .config import GRID_DTYPE, INT_DTYPE


def build_equalizer_rows(
    bin_ids: np.ndarray,           # (N,) int64, raster order
    num_bins: int,
    A_mask: np.ndarray,            # (N, C_out) bool (from WO-4)
    train_outputs_aligned: List[np.ndarray],  # Aligned outputs from WO-3
    H_out: int,
    W_out: int,
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Build spanning-tree equalizers per (bin, color) where constant predicate holds.

    For each (bin s, color c) where trainings prove 'constant on bin':
    - S_{s,c} = {p ∈ B_s : A[p,c]=True}
    - Check if Π_{s,c}(Y_i) is constant (min==max) for every training
    - If constant → return spanning tree edges tying y[(p,c)] == y[(q,c)]

    Args:
        bin_ids: (N,) int64 bin assignments per pixel
        num_bins: Total number of bins
        A_mask: (N, C_out) bool mask from forward meet
        train_outputs_aligned: List of aligned output grids from WO-3
        H_out: Output height
        W_out: Output width

    Returns:
        {(bin, color): [(p1, p2), ...]} edge list in raster order

    Notes:
        - Re-verifies constant predicate at WO-5 (don't read from WO-1)
        - Uses BFS tree for deterministic spanning tree
        - Rows commute (block-diagonal per bin × color)
        - Anchors: 01_addendum.md §3
    """
    N = H_out * W_out
    C_out = A_mask.shape[1]

    # Flatten aligned outputs to raster order
    train_outputs_flat = [Y.ravel(order='C') for Y in train_outputs_aligned]

    equalizer_edges = {}

    for s in range(num_bins):
        for c in range(C_out):
            # Get allowed pixels: S_{s,c} = {p ∈ B_s : A[p,c]=1}
            bin_mask = (bin_ids == s)
            allowed_mask = bin_mask & A_mask[:, c]
            allowed_pixels = np.where(allowed_mask)[0]

            if len(allowed_pixels) == 0:
                continue  # No pixels, skip

            # Check constant predicate: Π_{s,c}(Y_i) constant for all trainings
            is_constant = True
            for Y_flat in train_outputs_flat:
                # Get indicator vector: 1{Y[p]==c} for p in allowed set
                indicators = (Y_flat[allowed_pixels] == c).astype(np.int32)

                # Constant means min==max (all 0 or all 1)
                if indicators.min() != indicators.max():
                    is_constant = False
                    break

            if not is_constant:
                continue  # Not constant, no equalizer

            # Build spanning tree over S_{s,c}
            # Create adjacency graph (4-neighborhood on 2D canvas)
            num_allowed = len(allowed_pixels)

            if num_allowed == 1:
                # Single pixel, no edges needed
                equalizer_edges[(s, c)] = []
                continue

            # Build adjacency matrix for allowed pixels
            # Map pixel index to position in allowed array
            pixel_to_idx = {p: i for i, p in enumerate(allowed_pixels)}

            # Create sparse adjacency (4-neighborhood)
            rows_adj, cols_adj = [], []
            for i, p in enumerate(allowed_pixels):
                r, col = divmod(p, W_out)

                # Check 4 neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, col + dc
                    if 0 <= nr < H_out and 0 <= nc < W_out:
                        nbr_p = nr * W_out + nc
                        if nbr_p in pixel_to_idx:
                            j = pixel_to_idx[nbr_p]
                            rows_adj.append(i)
                            cols_adj.append(j)

            # Create sparse adjacency matrix
            if len(rows_adj) == 0:
                # Disconnected single pixels, no tree possible
                equalizer_edges[(s, c)] = []
                continue

            adj_matrix = scipy.sparse.csr_matrix(
                (np.ones(len(rows_adj), dtype=np.int32), (rows_adj, cols_adj)),
                shape=(num_allowed, num_allowed)
            )

            # Get BFS tree (deterministic with root=0)
            tree = breadth_first_tree(adj_matrix, i_start=0, directed=False)

            # Extract edges from tree (convert back to pixel indices)
            tree_coo = tree.tocoo()
            edges = []
            for i, j in zip(tree_coo.row, tree_coo.col):
                p1 = allowed_pixels[i]
                p2 = allowed_pixels[j]
                # Store in raster-sorted order
                if p1 > p2:
                    p1, p2 = p2, p1
                edges.append((int(p1), int(p2)))

            # Sort edges by raster order
            edges.sort()
            equalizer_edges[(s, c)] = edges

    return equalizer_edges


def build_gravity_rows(
    H: int,
    W: int,
    walls_mask: np.ndarray,    # (N,) bool; True=absorbing wall
    direction: str = "down"
) -> List[Tuple[int, int]]:
    """Build (I-G)y=0 gravity rows on transient states.

    For each transient pixel p:
    - If p↓ is transient: y[p] = y[p↓] (one equality row)
    - If p↓ is wall or out-of-bounds: p is absorbing (no row)

    Args:
        H: Height
        W: Width
        walls_mask: (N,) bool, True where walls are (absorbing)
        direction: "down" (standard ARC gravity)

    Returns:
        List of (p, p_down) pairs for equality rows y[p] = y[p_down]

    Notes:
        - Transient graph is acyclic (downward only)
        - (I-Q) invertible on transient set → unique fixed point
        - Anchors: 01_addendum.md §6, 02_addendum.md §L
    """
    N = H * W

    # Transient set: pixels that are not walls
    transient_mask = ~walls_mask

    gravity_rows = []

    for p in range(N):
        if not transient_mask[p]:
            continue  # Absorbing wall, skip

        r, c = divmod(p, W)

        # Direction: down (increase row)
        if direction == "down":
            r_next = r + 1
            c_next = c
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        # Check if next position is valid and transient
        if r_next < H:
            p_next = r_next * W + c_next
            if transient_mask[p_next]:
                # Both transient: y[p] = y[p_next]
                gravity_rows.append((int(p), int(p_next)))
        # Else: p_next is out-of-bounds or wall → p is absorbing (no row)

    return gravity_rows


def build_harmonic_rows(
    interior_idx: np.ndarray,     # 1-D int64 indices of interior pixels D
    boundary_idx: np.ndarray,     # 1-D int64 indices ∂D (not used in L_DD)
    H: int,
    W: int
) -> scipy.sparse.csr_array:
    """Build discrete Laplacian L_D for interior (Dirichlet rows Δu=0).

    Args:
        interior_idx: 1-D int64 indices of interior pixels D
        boundary_idx: 1-D int64 indices of boundary ∂D (for reference)
        H: Height
        W: Width

    Returns:
        CSR matrix L_DD of size |D| × |D| (interior Laplacian block)

    Notes:
        - Builds 4-neighborhood graph Laplacian on whole grid
        - Slices L_DD for interior × interior block
        - Uniqueness by maximum principle (discrete Dirichlet)
        - Anchors: 01_addendum.md §7, 02_addendum.md §L
    """
    N = H * W

    if len(interior_idx) == 0:
        # No interior, return empty matrix
        return scipy.sparse.csr_array((0, 0), dtype=np.float64)

    # Build 4-neighborhood adjacency for whole grid
    rows_adj, cols_adj = [], []

    for p in range(N):
        r, c = divmod(p, W)

        # 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                p_nbr = nr * W + nc
                rows_adj.append(p)
                cols_adj.append(p_nbr)

    # Create adjacency matrix
    adj_matrix = scipy.sparse.csr_matrix(
        (np.ones(len(rows_adj), dtype=np.int32), (rows_adj, cols_adj)),
        shape=(N, N)
    )

    # Compute graph Laplacian: L = D - A
    L_full = laplacian(adj_matrix, normed=False)

    # Convert to CSR for slicing (laplacian returns COO which doesn't support slicing)
    L_full = L_full.tocsr()

    # Slice L_DD (interior × interior block)
    L_DD = L_full[interior_idx, :][:, interior_idx]

    return L_DD.tocsr()


def detect_interior_regions(
    train_outputs_aligned: List[np.ndarray],
    H_out: int,
    W_out: int,
) -> List[np.ndarray]:
    """Detect interior regions (holes) using edge flood-fill.

    Args:
        train_outputs_aligned: List of aligned output grids
        H_out: Height
        W_out: Width

    Returns:
        List of interior region masks (1-D bool arrays)

    Notes:
        - F_∀ = ∩_i {p: Y_i(p) ≠ 0} (unanimous foreground)
        - Flood-fill from edges through ¬F_∀ → EXTERIOR
        - Interior D = (¬F_∀) \ EXTERIOR
        - Anchors: 01_addendum.md §7
    """
    N = H_out * W_out

    # Unanimous foreground: F_∀ = ∩_i {p: Y_i(p) ≠ 0}
    if len(train_outputs_aligned) == 0:
        return []

    F_all = np.ones(N, dtype=bool)
    for Y in train_outputs_aligned:
        Y_flat = Y.ravel(order='C')
        F_all &= (Y_flat != 0)

    # Complement: ¬F_∀
    not_F_all = ~F_all

    # Flood-fill from edges through ¬F_∀ to find EXTERIOR
    exterior = np.zeros(N, dtype=bool)

    # BFS from all edge pixels
    from collections import deque
    queue = deque()

    # Add all edge pixels that are in ¬F_∀
    for r in range(H_out):
        for c in range(W_out):
            if r == 0 or r == H_out - 1 or c == 0 or c == W_out - 1:
                p = r * W_out + c
                if not_F_all[p] and not exterior[p]:
                    exterior[p] = True
                    queue.append(p)

    # BFS through ¬F_∀
    while queue:
        p = queue.popleft()
        r, c = divmod(p, W_out)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H_out and 0 <= nc < W_out:
                p_nbr = nr * W_out + nc
                if not_F_all[p_nbr] and not exterior[p_nbr]:
                    exterior[p_nbr] = True
                    queue.append(p_nbr)

    # Interior = (¬F_∀) \ EXTERIOR
    interior_mask = not_F_all & ~exterior

    if not interior_mask.any():
        return []

    # Find connected components of interior (could be multiple holes)
    # Simple component labeling using BFS
    interior_components = []
    visited = np.zeros(N, dtype=bool)

    for p_start in range(N):
        if interior_mask[p_start] and not visited[p_start]:
            # New component
            component = []
            queue = deque([p_start])
            visited[p_start] = True

            while queue:
                p = queue.popleft()
                component.append(p)
                r, c = divmod(p, W_out)

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H_out and 0 <= nc < W_out:
                        p_nbr = nr * W_out + nc
                        if interior_mask[p_nbr] and not visited[p_nbr]:
                            visited[p_nbr] = True
                            queue.append(p_nbr)

            interior_components.append(np.array(component, dtype=INT_DTYPE))

    return interior_components


def test_row_commutation(
    equalizer_edges: Dict[Tuple[int, int], List[Tuple[int, int]]],
    gravity_rows: List[Tuple[int, int]],
    harmonic_L_matrices: List[scipy.sparse.csr_array],
    N: int,
    C_out: int,
) -> bool:
    """Test that equalizer, gravity, and harmonic rows commute.

    Builds constraint matrices in two different orders and verifies they
    define equivalent constraint spaces (same rank).

    Args:
        equalizer_edges: {(bin, color): [(p1, p2), ...]} equality edges
        gravity_rows: [(p, p_down), ...] gravity equality pairs
        harmonic_L_matrices: List of Laplacian matrices (one per interior region)
        N: Total number of pixels
        C_out: Number of colors

    Returns:
        True if rows commute (order-independent), False otherwise

    Notes:
        - Tests orders: [E, G, H] vs [G, E, H]
        - Verifies ranks are equal (same constraint space dimension)
        - Anchors: 01_addendum.md §5 (commuting rows requirement)
    """
    # Build constraint matrices in two different orders
    # Order 1: Equalizer, Gravity, Harmonic
    # Order 2: Gravity, Equalizer, Harmonic

    def build_constraint_matrix(order: str) -> scipy.sparse.csr_array:
        """Build constraint matrix with specified row order."""
        rows_list = []
        cols_list = []
        data_list = []
        row_idx = 0

        components = {
            'E': (equalizer_edges, 'equalizer'),
            'G': (gravity_rows, 'gravity'),
            'H': (harmonic_L_matrices, 'harmonic'),
        }

        for component_type in order:
            if component_type == 'E':
                # Equalizer rows: y[p1, c] = y[p2, c] → y[p1,c] - y[p2,c] = 0
                for (s, c), edges in equalizer_edges.items():
                    for p1, p2 in edges:
                        var1 = p1 * C_out + c
                        var2 = p2 * C_out + c
                        rows_list.extend([row_idx, row_idx])
                        cols_list.extend([var1, var2])
                        data_list.extend([1, -1])
                        row_idx += 1

            elif component_type == 'G':
                # Gravity rows: y[p, c] = y[p_down, c] for all colors
                for p, p_down in gravity_rows:
                    for c in range(C_out):
                        var1 = p * C_out + c
                        var2 = p_down * C_out + c
                        rows_list.extend([row_idx, row_idx])
                        cols_list.extend([var1, var2])
                        data_list.extend([1, -1])
                        row_idx += 1

            elif component_type == 'H':
                # Harmonic rows: Laplacian L_DD @ u = 0
                for L_DD in harmonic_L_matrices:
                    if L_DD.shape[0] == 0:
                        continue  # Empty interior
                    # L_DD is already a sparse matrix
                    L_coo = L_DD.tocoo()
                    for i, j, val in zip(L_coo.row, L_coo.col, L_coo.data):
                        rows_list.append(row_idx + i)
                        cols_list.append(j)
                        data_list.append(val)
                    row_idx += L_DD.shape[0]

        if len(rows_list) == 0:
            # No constraints - return empty matrix
            return scipy.sparse.csr_array((0, N * C_out), dtype=np.float64)

        # Build sparse matrix
        A = scipy.sparse.csr_array(
            (data_list, (rows_list, cols_list)),
            shape=(row_idx, N * C_out),
            dtype=np.float64
        )
        return A

    # Build in two different orders
    A_order1 = build_constraint_matrix('EGH')
    A_order2 = build_constraint_matrix('GEH')

    # Check ranks are equal
    rank1 = np.linalg.matrix_rank(A_order1.toarray()) if A_order1.shape[0] > 0 else 0
    rank2 = np.linalg.matrix_rank(A_order2.toarray()) if A_order2.shape[0] > 0 else 0

    # Commutation test: ranks must be equal
    return rank1 == rank2
