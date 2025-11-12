"""
WO-7: Unified pixel-level min-cost flow with shared cell capacities.

Uses OR-Tools SimpleMinCostFlow (single-commodity, integer-only).
Graph structure (no-faces baseline):
    U[s,c] (supply=q[s,c]) → P[p] (cap=1, cost=cost[p,c]) → C[r,j] (shared) → T (cap=n_{r,j})

Enforces:
- Mask constraints (A_mask)
- Bin quotas (q[s,c])
- One-of-10 per pixel (P→C cap=1)
- Shared cell capacities (C→T cap=n_{r,j})

Checks:
- Primal feasibility (balance, capacity, mask, cell caps)
- Cost equality (OptimalCost == Σ flow×cost)
- One-of-10 per pixel
- Idempotence (rebuild + resolve → same flows)
"""

import numpy as np
from ortools.graph.python import min_cost_flow
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import hashlib

from .config import SCALE
from .scores import to_int_costs


def load_wo7_inputs(
    task_id: str,
    data_root: Path,
    test_idx: Optional[int] = None,
    canvas_id: Optional[str] = None,
) -> Dict:
    """
    Load all required inputs from WO-04/05/06 caches.

    WO-B (per-canvas): Requires test_idx and canvas_id to load per-canvas caches.
    Legacy (task-level): Omit test_idx and canvas_id for backwards compatibility.

    Args:
        task_id: Task ID
        data_root: Path to data directory
        test_idx: Test index (required for per-canvas)
        canvas_id: Canvas ID (required for per-canvas)

    Returns dict with:
        H, W, C, N: grid dimensions
        bins: {bin_id -> [pixel_indices]}
        bin_ids: np.array of shape (N,) mapping pixel to bin
        A_mask: np.array of shape (N, C) bool
        costs: np.array of shape (N, C) int64
        cell_caps: {(r,j) -> n_{r,j}} shared capacity per cell
        quotas: {(s,c) -> q_{s,c}} bin quota per color
        equalizer_edges: {(s,c) -> edges_array} equalizer edges from WO-05
    """
    from . import cache as cache_module

    cache_root = data_root.parent / ".cache"
    per_canvas_mode = (test_idx is not None and canvas_id is not None)

    # === LOAD WO-04 CACHE (A_mask, bin_ids) ===
    if per_canvas_mode:
        # Per-canvas: .cache/wo04/{task_id}.{test_idx}.{canvas_id}.npz
        wo4_cache_path = cache_root / "wo04" / f"{task_id}.{test_idx}.{canvas_id}.npz"
        if not wo4_cache_path.exists():
            raise FileNotFoundError(f"WO-04 per-canvas cache not found: {wo4_cache_path}")
        cache_wo4 = np.load(wo4_cache_path)
        # Extract H, W from metadata
        H = int(cache_wo4.get("meta_H", 0))
        W = int(cache_wo4.get("meta_W", 0))
        C = 10
        N = H * W
    else:
        # Task-level (legacy): .cache/wo04/{task_id}.npz
        cache_wo4 = cache_module.load_cache(4, task_id, data_root)
        if cache_wo4 is None:
            raise FileNotFoundError(f"WO-04 cache not found for task {task_id}. Run WO-04 first.")
        manifest = cache_wo4.get("_manifest", {})
        H = manifest.get("H")
        W = manifest.get("W")
        C = manifest.get("C", 10)
        N = manifest.get("N")

    if H is None or W is None or N is None or H == 0 or W == 0 or N == 0:
        raise ValueError(f"WO-04 cache missing dimensions for task {task_id}")

    A_mask = cache_wo4["A_mask"]
    bin_ids = cache_wo4["bin_ids"]

    # === LOAD WO-05 CACHE (quotas, cell_caps, equalizers, faces) ===
    if per_canvas_mode:
        # Per-canvas: .cache/wo05/{task_id}.{test_idx}.{canvas_id}.npz
        wo5_cache_path = cache_root / "wo05" / f"{task_id}.{test_idx}.{canvas_id}.npz"
        if not wo5_cache_path.exists():
            raise FileNotFoundError(f"WO-05 per-canvas cache not found: {wo5_cache_path}")
        cache_wo5 = np.load(wo5_cache_path)
    else:
        # Task-level (legacy)
        cache_wo5 = cache_module.load_cache(5, task_id, data_root)
        if cache_wo5 is None:
            raise FileNotFoundError(f"WO-05 cache not found for task {task_id}. Run WO-05 first.")

    # Extract WO-5 artifacts
    quotas_array = cache_wo5["quotas"]

    # Reconstruct bins from bin_ids
    bins_dict = {}
    num_bins = int(bin_ids.max()) + 1
    for s in range(num_bins):
        bins_dict[s] = []

    for p in range(N):
        s = int(bin_ids[p])
        bins_dict[s].append(p)

    # Reconstruct quotas from array
    quotas = {}
    for row in quotas_array:
        s, c, count = int(row[0]), int(row[1]), int(row[2])
        quotas[(s, c)] = count

    # Reconstruct cell_caps from array (if present)
    cell_caps = {}
    if "cell_caps" in cache_wo5:
        cell_caps_array = cache_wo5["cell_caps"]
        for r in range(H):
            for j in range(W):
                cell_caps[(r, j)] = int(cell_caps_array[r, j])
    else:
        # Default: all cells have capacity 1 (for tasks without cell_caps)
        for r in range(H):
            for j in range(W):
                cell_caps[(r, j)] = 1

    # === LOAD WO-06 CACHE (costs) ===
    if per_canvas_mode:
        # Per-canvas: .cache/wo06/{task_id}.{test_idx}.{canvas_id}.npz
        wo6_cache_path = cache_root / "wo06" / f"{task_id}.{test_idx}.{canvas_id}.npz"
        if not wo6_cache_path.exists():
            raise FileNotFoundError(f"WO-06 per-canvas cache not found: {wo6_cache_path}")
        cache_wo6 = np.load(wo6_cache_path)
    else:
        # Task-level (legacy)
        cache_wo6 = cache_module.load_cache(6, task_id, data_root)
        if cache_wo6 is None:
            raise FileNotFoundError(
                f"WO-06 cache not found for task {task_id}. "
                f"WO-07 requires WO-06 costs to be cached. Run WO-06 first."
            )

    costs = cache_wo6["costs"]

    # Extract constant-on-bin pairs and equalizer edges from WO-05 cache keys
    # NEW FORMAT: Only trust "eq_{s}_{c}" prefixed keys (cache format v1)
    # OLD FORMAT: Keys like "0,0", "1,0" are IGNORED (stale cache)
    constant_on_bin = set()
    equalizer_edges = {}  # Dict[(s,c)] → edges array for precheck
    for key in cache_wo5.keys():
        if key.startswith("eq_"):
            try:
                parts = key.split('_')  # "eq_{s}_{c}" → ["eq", "{s}", "{c}"]
                if len(parts) == 3:
                    s, c = int(parts[1]), int(parts[2])
                    # Validate array is non-empty before trusting
                    edges = cache_wo5[key]
                    if edges.ndim == 2 and edges.shape[0] >= 1:
                        constant_on_bin.add((s, c))
                        equalizer_edges[(s, c)] = edges  # Store for precheck
            except (ValueError, IndexError, KeyError):
                pass  # Skip malformed keys

    # Load faces from WO-05 cache (Issue #3)
    # faces_R: int64[H×C] - per-row color totals (optional)
    # faces_S: int64[W×C] - per-column color totals (optional)
    faces_R = cache_wo5.get("faces_R", None)
    faces_S = cache_wo5.get("faces_S", None)

    # Determine faces mode
    if faces_R is not None and faces_S is not None:
        faces_mode = "both"
    elif faces_R is not None:
        faces_mode = "rows_as_supply"
    elif faces_S is not None:
        faces_mode = "cols_as_supply"
    else:
        faces_mode = "none"

    inputs_dict = {
        "H": H,
        "W": W,
        "C": C,
        "N": N,
        "bins": bins_dict,
        "bin_ids": bin_ids,
        "A_mask": A_mask,
        "costs": costs,
        "cell_caps": cell_caps,
        "quotas": quotas,
        "constant_on_bin": constant_on_bin,
        "equalizer_edges": equalizer_edges,  # For precheck
        "faces_R": faces_R,
        "faces_S": faces_S,
        "faces_mode": faces_mode,
    }

    # WO-B: Skip pre-check for per-canvas mode
    # Pre-check happens in try_pack() after faces are filtered by pack.faces_mode
    if not per_canvas_mode:
        # Legacy task-level mode: run pre-checks before returning
        check_input_preconditions(inputs_dict)

    return inputs_dict


def check_input_preconditions(inputs: Dict) -> None:
    """
    Check WO-07 pre-conditions on quotas, constant-on-bin, and faces.

    Pre-checks (Issue #2 from reviewer):
    1. All quotas must be >= 0
    2. For constant-on-bin (s,c): quota must be 0 or |allowed(B_s,c)|
       where |allowed(B_s,c)| = count of pixels p in bin s where A[p,c]=1

    Pre-checks (Issue #3 from reviewer):
    3. If faces_R present: verify ∑_s q[s,c] = ∑_r R[r,c] for all c
    4. If faces_S present: verify ∑_s q[s,c] = ∑_j S[j,c] for all c

    Raises ValueError on violation (IIS@faces/quotas tier per spec).
    """
    bins = inputs["bins"]
    A_mask = inputs["A_mask"]
    quotas = inputs["quotas"]
    constant_on_bin = inputs["constant_on_bin"]
    faces_R = inputs.get("faces_R", None)
    faces_S = inputs.get("faces_S", None)
    H = inputs["H"]
    W = inputs["W"]
    C = inputs["C"]

    # Check 1: All quotas >= 0
    for (s, c), q in quotas.items():
        if q < 0:
            raise ValueError(
                f"WO-07 pre-check failed: quota[{s},{c}] = {q} < 0. "
                f"IIS@faces/quotas tier."
            )

    # Check 2: Constant-on-bin constraint
    for (s, c) in constant_on_bin:
        if (s, c) not in quotas:
            # No quota specified, skip
            continue

        q = quotas[(s, c)]

        # Compute |allowed(B_s,c)| = count of pixels in bin s where A[p,c]=1
        allowed_count = sum(1 for p in bins[s] if A_mask[p, c])

        # Quota must be 0 or |allowed(B_s,c)|
        if q != 0 and q != allowed_count:
            raise ValueError(
                f"WO-07 pre-check failed: constant-on-bin ({s},{c}) has quota={q} "
                f"but |allowed(B_{s},{c})|={allowed_count}. "
                f"Quota must be 0 or {allowed_count}. IIS@faces/quotas tier."
            )

    # Check 3: Faces consistency - row totals
    if faces_R is not None:
        for c in range(C):
            # Sum all quotas for color c
            quota_sum = sum(quotas.get((s, c), 0) for s in bins.keys())

            # Sum all row targets for color c
            row_sum = sum(int(faces_R[r, c]) for r in range(H))

            if quota_sum != row_sum:
                raise ValueError(
                    f"WO-07 pre-check failed: faces_R inconsistency for color {c}. "
                    f"Quota sum = {quota_sum}, Row faces sum = {row_sum}. "
                    f"IIS@faces tier."
                )

    # Check 4: Faces consistency - column totals
    if faces_S is not None:
        for c in range(C):
            # Sum all quotas for color c
            quota_sum = sum(quotas.get((s, c), 0) for s in bins.keys())

            # Sum all column targets for color c
            col_sum = sum(int(faces_S[j, c]) for j in range(W))

            if quota_sum != col_sum:
                raise ValueError(
                    f"WO-07 pre-check failed: faces_S inconsistency for color {c}. "
                    f"Quota sum = {quota_sum}, Column faces sum = {col_sum}. "
                    f"IIS@faces tier."
                )


def compute_row_quotas(inputs: Dict) -> Dict:
    """
    Compute per-row quotas q[s,r,c] from faces_R targets.

    For each (r,c), distribute R[r,c] across bins s proportionally to
    the number of allowed pixels in row r for each bin.

    Returns dict {(s,r,c): quota}
    """
    H = inputs["H"]
    W = inputs["W"]
    C = inputs["C"]
    bins = inputs["bins"]
    A_mask = inputs["A_mask"]
    faces_R = inputs["faces_R"]

    row_quotas = {}

    for c in range(C):
        for r in range(H):
            target = int(faces_R[r, c])
            if target == 0:
                continue

            # Count allowed pixels in row r for each bin
            bin_counts = {}
            total_allowed = 0

            for s in bins.keys():
                count = 0
                for p in bins[s]:
                    if p // W == r and A_mask[p, c]:
                        count += 1
                if count > 0:
                    bin_counts[s] = count
                    total_allowed += count

            if total_allowed == 0:
                raise ValueError(
                    f"WO-07 faces enforcement failed: row {r}, color {c} "
                    f"requires {target} pixels but has 0 allowed. IIS@faces tier."
                )

            # Issue #3 fix: Greedy cap-and-fill respecting bin capacities
            # Per reviewer: proportional rounding can violate bin capacity
            # Use greedy allocation sorted by descending capacity
            remaining = target
            for s in sorted(bin_counts.keys(), key=lambda x: (-bin_counts[x], x)):
                # Allocate min(remaining, capacity) to this bin
                take = min(remaining, bin_counts[s])
                if take > 0:
                    row_quotas[(s, r, c)] = take
                    remaining -= take
                if remaining == 0:
                    break

            # If we couldn't allocate the full target, it's infeasible
            if remaining > 0:
                raise ValueError(
                    f"WO-07 faces enforcement failed: row {r}, color {c} "
                    f"target={target} exceeds total_allowed={total_allowed}. IIS@faces tier."
                )

    return row_quotas


def compute_col_quotas(inputs: Dict) -> Dict:
    """
    Compute per-column quotas q[s,j,c] from faces_S targets.

    Similar to compute_row_quotas but for columns.

    Returns dict {(s,j,c): quota}
    """
    H = inputs["H"]
    W = inputs["W"]
    C = inputs["C"]
    bins = inputs["bins"]
    A_mask = inputs["A_mask"]
    faces_S = inputs["faces_S"]

    col_quotas = {}

    for c in range(C):
        for j in range(W):
            target = int(faces_S[j, c])
            if target == 0:
                continue

            # Count allowed pixels in column j for each bin
            bin_counts = {}
            total_allowed = 0

            for s in bins.keys():
                count = 0
                for p in bins[s]:
                    if p % W == j and A_mask[p, c]:
                        count += 1
                if count > 0:
                    bin_counts[s] = count
                    total_allowed += count

            if total_allowed == 0:
                raise ValueError(
                    f"WO-07 faces enforcement failed: column {j}, color {c} "
                    f"requires {target} pixels but has 0 allowed. IIS@faces tier."
                )

            # Issue #3 fix: Greedy cap-and-fill respecting bin capacities
            # Per reviewer: proportional rounding can violate bin capacity
            # Use greedy allocation sorted by descending capacity
            remaining = target
            for s in sorted(bin_counts.keys(), key=lambda x: (-bin_counts[x], x)):
                # Allocate min(remaining, capacity) to this bin
                take = min(remaining, bin_counts[s])
                if take > 0:
                    col_quotas[(s, j, c)] = take
                    remaining -= take
                if remaining == 0:
                    break

            # If we couldn't allocate the full target, it's infeasible
            if remaining > 0:
                raise ValueError(
                    f"WO-07 faces enforcement failed: column {j}, color {c} "
                    f"target={target} exceeds total_allowed={total_allowed}. IIS@faces tier."
                )

    return col_quotas


def build_flow_graph(inputs: Dict) -> Tuple[min_cost_flow.SimpleMinCostFlow, Dict]:
    """
    Build unified min-cost flow graph with optional faces support.

    Graph structures:
    - No faces: U[s,c] (supply=q[s,c]) → P[p] → C[r,j] → T
    - Rows as supply: U[s,r,c] (supply=q[s,r,c]) → P[p in row r] → C[r,j] → T
    - Cols as supply: U[s,j,c] (supply=q[s,j,c]) → P[p in col j] → C[r,j] → T
    - Both: Rows hard (supply split), columns checked post-solve

    Returns:
        (mcf, metadata) where metadata contains node/arc mappings for checks
    """
    H = inputs["H"]
    W = inputs["W"]
    C = inputs["C"]
    N = inputs["N"]
    bins = inputs["bins"]
    bin_ids = inputs["bin_ids"]
    A_mask = inputs["A_mask"]
    costs = inputs["costs"]
    cell_caps = inputs["cell_caps"]
    quotas = inputs["quotas"]
    faces_R = inputs.get("faces_R", None)
    faces_S = inputs.get("faces_S", None)
    faces_mode = inputs["faces_mode"]

    mcf = min_cost_flow.SimpleMinCostFlow()

    # Node indexing (deterministic, stable)
    node_id = {}
    supply_by_node = {}  # Track supplies for primal balance check (Issue #1)
    base = 0

    # Decide on bin node structure based on faces mode
    if faces_mode == "rows_as_supply" or faces_mode == "both":
        # Split bin sources by row: U[s,r,c]
        # Compute per-row quotas from bin quotas
        row_quotas = compute_row_quotas(inputs)

        for c in range(C):
            for s in sorted(bins.keys()):
                for r in range(H):
                    node_id[("U", s, r, c)] = base
                    base += 1

    elif faces_mode == "cols_as_supply":
        # Split bin sources by column: U[s,j,c]
        col_quotas = compute_col_quotas(inputs)

        for c in range(C):
            for s in sorted(bins.keys()):
                for j in range(W):
                    node_id[("U", s, j, c)] = base
                    base += 1

    else:  # faces_mode == "none"
        # Baseline: bin nodes per color U[s,c]
        for c in range(C):
            for s in sorted(bins.keys()):
                node_id[("U", s, c)] = base
                base += 1

    # Pixel nodes P[p] (shared across colors)
    for p in range(N):
        node_id[("P", p)] = base
        base += 1

    # Cell nodes C[r,j] (shared across colors)
    for r in range(H):
        for j in range(W):
            node_id[("C", r, j)] = base
            base += 1

    # Sink T
    node_id["T"] = base
    base += 1

    total_nodes = base

    # Set supplies/demands (faces-aware)
    total_supply = 0
    supply_by_color = [0] * C

    if faces_mode == "rows_as_supply" or faces_mode == "both":
        # Supplies from row-split quotas
        for c in range(C):
            for s in sorted(bins.keys()):
                for r in range(H):
                    q = row_quotas.get((s, r, c), 0)
                    if q > 0:
                        node_key = ("U", s, r, c)
                        mcf.set_node_supply(node_id[node_key], int(q))
                        supply_by_node[node_id[node_key]] = int(q)
                        total_supply += q
                        supply_by_color[c] += q

    elif faces_mode == "cols_as_supply":
        # Supplies from column-split quotas
        for c in range(C):
            for s in sorted(bins.keys()):
                for j in range(W):
                    q = col_quotas.get((s, j, c), 0)
                    if q > 0:
                        node_key = ("U", s, j, c)
                        mcf.set_node_supply(node_id[node_key], int(q))
                        supply_by_node[node_id[node_key]] = int(q)
                        total_supply += q
                        supply_by_color[c] += q

    else:  # faces_mode == "none"
        # Baseline: supplies from bin quotas
        for c in range(C):
            for s in sorted(bins.keys()):
                q = quotas.get((s, c), 0)
                if q > 0:
                    node_key = ("U", s, c)
                    mcf.set_node_supply(node_id[node_key], int(q))
                    supply_by_node[node_id[node_key]] = int(q)
                    total_supply += q
                    supply_by_color[c] += q

    # Sink demand = -total_supply
    mcf.set_node_supply(node_id["T"], -int(total_supply))
    supply_by_node[node_id["T"]] = -int(total_supply)

    # Add arcs (faces-aware)
    arc_count = 0
    arc_metadata = []

    # 1. Bin → Pixel arcs (cost arcs, topology depends on faces mode)
    if faces_mode == "rows_as_supply" or faces_mode == "both":
        # Wire U[s,r,c] only to pixels in row r
        for c in range(C):
            for s in sorted(bins.keys()):
                for r in range(H):
                    for p in bins[s]:
                        p_row = p // W
                        if p_row == r and A_mask[p, c]:  # Only pixels in this row
                            tail = node_id[("U", s, r, c)]
                            head = node_id[("P", p)]
                            cap = 1
                            unit_cost = int(costs[p, c])

                            mcf.add_arc_with_capacity_and_unit_cost(tail, head, cap, unit_cost)
                            arc_metadata.append({
                                "type": "bin_to_pixel",
                                "tail": ("U", s, r, c),
                                "head": ("P", p),
                                "cap": cap,
                                "cost": unit_cost,
                                "p": p,
                                "c": c,
                                "s": s,
                                "r": r,  # Include row for faces mode
                                "arc_id": arc_count,
                            })
                            arc_count += 1

    elif faces_mode == "cols_as_supply":
        # Wire U[s,j,c] only to pixels in column j
        for c in range(C):
            for s in sorted(bins.keys()):
                for j in range(W):
                    for p in bins[s]:
                        p_col = p % W
                        if p_col == j and A_mask[p, c]:  # Only pixels in this column
                            tail = node_id[("U", s, j, c)]
                            head = node_id[("P", p)]
                            cap = 1
                            unit_cost = int(costs[p, c])

                            mcf.add_arc_with_capacity_and_unit_cost(tail, head, cap, unit_cost)
                            arc_metadata.append({
                                "type": "bin_to_pixel",
                                "tail": ("U", s, j, c),
                                "head": ("P", p),
                                "cap": cap,
                                "cost": unit_cost,
                                "p": p,
                                "c": c,
                                "s": s,
                                "j": j,  # Include column for faces mode
                                "arc_id": arc_count,
                            })
                            arc_count += 1

    else:  # faces_mode == "none"
        # Baseline: U[s,c] wires to all pixels in bin s (no row/col restriction)
        for c in range(C):
            for s in sorted(bins.keys()):
                for p in bins[s]:
                    if A_mask[p, c]:  # Only add arc if mask allows
                        tail = node_id[("U", s, c)]
                        head = node_id[("P", p)]
                        cap = 1
                        unit_cost = int(costs[p, c])

                        mcf.add_arc_with_capacity_and_unit_cost(tail, head, cap, unit_cost)
                        arc_metadata.append({
                            "type": "bin_to_pixel",
                            "tail": ("U", s, c),
                            "head": ("P", p),
                            "cap": cap,
                            "cost": unit_cost,
                            "p": p,
                            "c": c,
                            "s": s,
                            "arc_id": arc_count,
                        })
                        arc_count += 1

    # 2. Pixel → Cell arcs (one-of-10 enforcement, cap=1, cost=0)
    for p in range(N):
        r = p // W
        j = p % W
        tail = node_id[("P", p)]
        head = node_id[("C", r, j)]
        cap = 1
        unit_cost = 0

        mcf.add_arc_with_capacity_and_unit_cost(tail, head, cap, unit_cost)
        arc_metadata.append({
            "type": "pixel_to_cell",
            "tail": ("P", p),
            "head": ("C", r, j),
            "cap": cap,
            "cost": unit_cost,
            "p": p,
            "r": r,
            "j": j,
            "arc_id": arc_count,
        })
        arc_count += 1

    # 3. Cell → Sink arcs (shared cell capacity enforcement)
    for r in range(H):
        for j in range(W):
            n_rj = cell_caps.get((r, j), 1)  # Default to 1 if not specified
            tail = node_id[("C", r, j)]
            head = node_id["T"]
            cap = int(n_rj)
            unit_cost = 0

            mcf.add_arc_with_capacity_and_unit_cost(tail, head, cap, unit_cost)
            arc_metadata.append({
                "type": "cell_to_sink",
                "tail": ("C", r, j),
                "head": "T",
                "cap": cap,
                "cost": unit_cost,
                "r": r,
                "j": j,
                "arc_id": arc_count,
            })
            arc_count += 1

    metadata = {
        "node_id": node_id,
        "supply_by_node": supply_by_node,  # For primal balance check (Issue #1)
        "total_nodes": total_nodes,
        "total_arcs": arc_count,
        "arc_metadata": arc_metadata,
        "total_supply": total_supply,
        "supply_by_color": supply_by_color,
        "faces_mode": faces_mode,
        "faces_R": faces_R,
        "faces_S": faces_S,
        "H": H,
        "W": W,
        "C": C,
    }

    return mcf, metadata


def solve_and_extract(mcf: min_cost_flow.SimpleMinCostFlow, metadata: Dict) -> Dict:
    """
    Solve the min-cost flow and extract solution.

    Returns dict with:
        status: "OPTIMAL" | "INFEASIBLE" | ...
        optimal_cost: int
        flows: list of {arc_id, flow, tail, head, cap, cost, ...}
    """
    status = mcf.solve()

    status_map = {
        mcf.OPTIMAL: "OPTIMAL",
        mcf.NOT_SOLVED: "NOT_SOLVED",
        mcf.FEASIBLE: "FEASIBLE",
        mcf.INFEASIBLE: "INFEASIBLE",
        mcf.UNBALANCED: "UNBALANCED",
        mcf.BAD_RESULT: "BAD_RESULT",
        mcf.BAD_COST_RANGE: "BAD_COST_RANGE",
    }

    status_str = status_map.get(status, f"UNKNOWN({status})")

    if status != mcf.OPTIMAL:
        return {
            "status": status_str,
            "optimal_cost": None,
            "flows": [],
        }

    # Extract flows
    flows = []
    for arc_data in metadata["arc_metadata"]:
        arc_id = arc_data["arc_id"]
        flow_val = mcf.flow(arc_id)
        flows.append({
            **arc_data,
            "flow": int(flow_val),
        })

    return {
        "status": status_str,
        "optimal_cost": int(mcf.optimal_cost()),
        "flows": flows,
    }


def check_primal_feasibility(solution: Dict, inputs: Dict, metadata: Dict) -> Dict:
    """
    Check all primal feasibility conditions:
    - Node balance
    - Capacity constraints
    - Mask constraints
    - One-of-10 per pixel
    - Shared cell caps

    Returns dict with boolean checks.
    """
    if solution["status"] != "OPTIMAL":
        return {
            "primal_balance_ok": False,
            "capacity_ok": False,
            "mask_ok": False,
            "one_of_10_ok": False,
            "cell_caps_ok": False,
        }

    flows = solution["flows"]
    node_id = metadata["node_id"]
    N = inputs["N"]
    H = inputs["H"]
    W = inputs["W"]
    C = inputs["C"]
    A_mask = inputs["A_mask"]
    cell_caps = inputs["cell_caps"]

    # Build flow maps
    outflow = {}
    inflow = {}

    for flow_data in flows:
        tail = flow_data["tail"]
        head = flow_data["head"]
        flow_val = flow_data["flow"]

        outflow[tail] = outflow.get(tail, 0) + flow_val
        inflow[head] = inflow.get(head, 0) + flow_val

    # Check 1: Node balance (Issue #1 fix: actually verify conservation)
    # Per Anchor 03 Annex A.2: "Residuals...are integers; require **exact 0**"
    # For each node: outflow - inflow must equal supply (exact integer equality)
    supply_by_node = metadata.get("supply_by_node", {})
    primal_balance_ok = True

    for node_tuple, node_int_id in node_id.items():
        expected_supply = supply_by_node.get(node_int_id, 0)
        actual_net_flow = outflow.get(node_tuple, 0) - inflow.get(node_tuple, 0)
        if actual_net_flow != expected_supply:
            primal_balance_ok = False
            break

    # Check 2: Capacity constraints
    capacity_ok = True
    for flow_data in flows:
        if flow_data["flow"] > flow_data["cap"]:
            capacity_ok = False
            break

    # Check 3: Mask constraints
    mask_ok = True
    for flow_data in flows:
        if flow_data["type"] == "bin_to_pixel":
            p = flow_data["p"]
            c = flow_data["c"]
            if flow_data["flow"] > 0 and not A_mask[p, c]:
                mask_ok = False
                break

    # Check 4: One-of-10 per pixel (sum of flows into each cell ≤ 1)
    one_of_10_ok = True
    for p in range(N):
        r = p // W
        j = p % W
        pixel_node = ("P", p)
        # Flow out of pixel node = flow into cell node
        pixel_outflow = outflow.get(pixel_node, 0)
        if pixel_outflow > 1:
            one_of_10_ok = False
            break

    # Check 5: Cell caps (sum of flows through each cell ≤ n_{r,j})
    cell_caps_ok = True
    for r in range(H):
        for j in range(W):
            cell_node = ("C", r, j)
            cell_flow = outflow.get(cell_node, 0)  # Flow out to sink
            n_rj = cell_caps.get((r, j), 1)
            if cell_flow > n_rj:
                cell_caps_ok = False
                break
        if not cell_caps_ok:
            break

    return {
        "primal_balance_ok": primal_balance_ok,
        "capacity_ok": capacity_ok,
        "mask_ok": mask_ok,
        "one_of_10_ok": one_of_10_ok,
        "cell_caps_ok": cell_caps_ok,
    }


def check_cost_equality(solution: Dict) -> bool:
    """
    Check that recomputed cost equals OptimalCost.
    """
    if solution["status"] != "OPTIMAL":
        return False

    recomputed_cost = sum(f["flow"] * f["cost"] for f in solution["flows"])
    return recomputed_cost == solution["optimal_cost"]


def check_kkt_reduced_costs(mcf: min_cost_flow.SimpleMinCostFlow, solution: Dict, metadata: Dict) -> bool:
    """
    Attempt to check KKT reduced cost optimality conditions (Issue #5).

    If OR-Tools exposes Potential() API, verify:
    - For all arcs: reduced_cost = cost - potential[tail] + potential[head]
    - If flow > 0: reduced_cost == 0 (complementary slackness)
    - If flow == 0: reduced_cost >= 0 (feasibility)

    Returns:
        True if KKT check passes
        None if Potential() API not available (no AttributeError)
        False if KKT check fails

    This is a best-effort check - returns None with documentation if API unavailable.
    """
    if solution["status"] != "OPTIMAL":
        return False

    try:
        # Attempt to access node potentials from OR-Tools
        # This may not be exposed in SimpleMinCostFlow API
        total_nodes = metadata["total_nodes"]
        potentials = {}

        for node_id in range(total_nodes):
            potentials[node_id] = mcf.Potential(node_id)

        # Verify KKT conditions for all arcs
        for arc_data in metadata["arc_metadata"]:
            arc_id = arc_data["arc_id"]
            tail = arc_data["tail"]
            head = arc_data["head"]
            cost = arc_data["cost"]

            # Find flow for this arc
            flow = next(f["flow"] for f in solution["flows"] if f["arc_id"] == arc_id)

            # Compute reduced cost
            reduced_cost = cost - potentials[tail] + potentials[head]

            # KKT conditions:
            # If flow > 0: reduced_cost must be 0 (arc at lower bound, shadow price = 0)
            # If flow == 0: reduced_cost must be >= 0 (dual feasibility)
            if flow > 0:
                if reduced_cost != 0:
                    return False
            else:
                if reduced_cost < 0:
                    return False

        return True

    except (AttributeError, TypeError):
        # Potential() method not available in this OR-Tools version
        # Return None to indicate "not checkable" rather than "failed"
        return None


def check_idempotence(mcf: min_cost_flow.SimpleMinCostFlow, metadata: Dict, solution: Dict, inputs: Dict) -> bool:
    """
    Rebuild identical graph from scratch, resolve, check flows and cost are identical.

    Tests that graph construction is deterministic by rebuilding the entire
    graph from the same inputs and verifying byte-identical flows.
    """
    if solution["status"] != "OPTIMAL":
        return False

    # Rebuild graph from scratch using same inputs
    # This tests graph construction determinism, not just solver determinism
    mcf2, metadata2 = build_flow_graph(inputs)

    # Solve the rebuilt graph
    status2 = mcf2.solve()

    if status2 != mcf2.OPTIMAL:
        return False

    # Check cost equality
    if int(mcf2.optimal_cost()) != solution["optimal_cost"]:
        return False

    # Check flows are byte-identical (same arc_id -> same flow value)
    for arc_data in metadata["arc_metadata"]:
        arc_id = arc_data["arc_id"]
        flow1 = next(f["flow"] for f in solution["flows"] if f["arc_id"] == arc_id)
        flow2 = int(mcf2.flow(arc_id))
        if flow1 != flow2:
            return False

    return True


def check_faces_postsolve(solution: Dict, metadata: Dict) -> bool:
    """
    Post-solve check for column totals when faces_mode is "both".

    When faces_mode is "both", rows are enforced via supply split (hard constraint
    via graph topology), but columns need post-solve verification.

    For each column j and color c, verify that the sum of pixels assigned to
    column j with color c equals faces_S[j, c].

    Returns True if all column totals match, False otherwise.
    """
    faces_mode = metadata.get("faces_mode", "none")
    faces_S = metadata.get("faces_S", None)

    # Only check columns when faces_mode is "both"
    if faces_mode != "both":
        return True  # Trivially true for other modes

    if faces_S is None:
        return False  # Should not happen if faces_mode is "both"

    if solution["status"] != "OPTIMAL":
        return False

    W = metadata["W"]
    C = metadata["C"]

    # Accumulate column totals from solution
    # col_totals[j][c] = sum of flow to pixels in column j with color c
    col_totals = {}
    for j in range(W):
        col_totals[j] = {}
        for c in range(C):
            col_totals[j][c] = 0

    # Sum flows from bin→pixel arcs by column and color
    for flow_data in solution["flows"]:
        if flow_data["type"] != "bin_to_pixel":
            continue

        p = flow_data["p"]
        c = flow_data["c"]
        flow = flow_data["flow"]

        j = p % W  # Column index
        col_totals[j][c] += flow

    # Verify each column total matches faces_S
    for j in range(W):
        for c in range(C):
            actual = col_totals[j][c]
            expected = int(faces_S[j, c])

            if actual != expected:
                return False

    return True


def run_wo7_for_task(task_dir: Path) -> Dict:
    """
    Run WO-7 for a single task: load inputs, build flow, solve, check.

    Delegates to stages_wo07.run_wo07 for unified implementation.

    Returns receipt dict.
    """
    from .stages_wo07 import run_wo07

    task_id = task_dir.name
    data_root = Path("data")
    return run_wo07(task_id, data_root)
