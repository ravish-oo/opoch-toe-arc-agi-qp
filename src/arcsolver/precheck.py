"""
WO-07 pre-check: Deterministic feasibility screening before solver.

Implements capacity and constant-on-bin checks per §8/§10/§11.
Detects quota/mask conflicts and equalizer infeasibility without running the solver.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


def wo7_precheck(
    H: int,
    W: int,
    bin_ids: np.ndarray,  # (N,) int64, raster
    A_mask: np.ndarray,  # (N,C) bool
    quotas: np.ndarray,  # (S,C) int64, as per §8 (min over trainings)
    eq_rows: Dict[Tuple[int, int], np.ndarray] | None = None,
    costs: np.ndarray | None = None,  # (N,C) int64, optional for diagnostic
) -> Dict:
    """
    Pre-check WO-7 feasibility without running the solver.

    Checks:
    1. Capacity: quota[s,c] <= |{p ∈ B_s : A[p,c]==1}| (§8)
    2. Constant-on-bin: For equalizers with |S|>=2, quota must be 0 or |S| (§3/§4/§10)

    WO-B Fix C: Also computes diagnostic values:
    3. supplies_total: sum of all quotas (§11/§16 zero-supply gate)
    4. costs_nonzero_count: count of non-zero costs (degenerate objective check)

    Args:
        H, W: Grid dimensions
        bin_ids: Bin assignment for each pixel (N,)
        A_mask: Allowed colors per pixel (N, C)
        quotas: Minimum color counts per bin (S, C)
        eq_rows: Equalizer edges dict {(s,c): edges_array} from WO-05 cache
        costs: Optional cost matrix for degenerate objective check

    Returns:
        {
            "infeasibility_expected": bool,
            "capacity_ok": bool,
            "constant_bin_ok": bool,
            "capacity_conflicts": [{"bin":s,"color":c,"q":int,"allowed":int}...],
            "eq_conflicts": [{"bin":s,"color":c,"q":int,"allowed":int,"eq_size":int}...],
            "supplies_total": int,  # WO-B diagnostic
            "zero_supply_pack": bool,  # WO-B diagnostic
            "costs_nonzero_count": int | None,  # WO-B diagnostic
            "degenerate_objective": bool,  # WO-B diagnostic
        }
    """
    S = quotas.shape[0]
    C = quotas.shape[1]
    N = bin_ids.shape[0]

    # WO-B Fix C: Compute diagnostic values
    supplies_total = int(np.sum(quotas))
    zero_supply_pack = (supplies_total == 0)

    costs_nonzero_count = None
    degenerate_objective = False
    if costs is not None:
        costs_nonzero_count = int(np.count_nonzero(costs))
        # Degenerate if supplies > 0 but all costs are zero
        degenerate_objective = (supplies_total > 0 and costs_nonzero_count == 0)

    # Count allowed[s,c] = |{p ∈ B_s : A[p,c]==1}|
    allowed = np.zeros_like(quotas, dtype=np.int64)
    for s in range(S):
        sel = (bin_ids == s)
        if sel.any():
            # Sum over pixels in bin s: how many allow each color
            allowed[s, :] = A_mask[sel].sum(axis=0).astype(np.int64)

    # Check capacity: q[s,c] <= allowed[s,c]
    cap_conf = []
    for s in range(S):
        for c in range(C):
            q = int(quotas[s, c])
            cap = int(allowed[s, c])
            if q > cap:
                cap_conf.append({
                    "bin": int(s),
                    "color": int(c),
                    "q": q,
                    "allowed": cap
                })
    capacity_ok = (len(cap_conf) == 0)

    # Check constant-on-bin: equalizers with |S|>=2 require q ∈ {0, |S|}
    eq_conf = []
    if eq_rows:
        for (s, c), edges in eq_rows.items():
            if edges.ndim != 2 or edges.shape[1] != 2:
                continue
            # Tree with |edges| edges has |edges|+1 nodes
            size_S = edges.shape[0] + 1
            if size_S < 2:
                continue
            q = int(quotas[s, c])
            cap = int(allowed[s, c])
            # If mask already forbids, already counted in capacity check
            if q > cap:
                continue
            # With equalizer active, only q ∈ {0, |S|} is feasible (§3/§4 + §10)
            if q not in (0, size_S):
                eq_conf.append({
                    "bin": int(s),
                    "color": int(c),
                    "q": q,
                    "allowed": cap,
                    "eq_size": int(size_S)
                })
    constant_bin_ok = (len(eq_conf) == 0)

    infeasible = (not capacity_ok) or (not constant_bin_ok)

    return {
        "infeasibility_expected": bool(infeasible),
        "capacity_ok": capacity_ok,
        "constant_bin_ok": constant_bin_ok,
        "capacity_conflicts": cap_conf,
        "eq_conflicts": eq_conf,
        # WO-B Fix C diagnostics
        "supplies_total": supplies_total,
        "zero_supply_pack": zero_supply_pack,
        "costs_nonzero_count": costs_nonzero_count,
        "degenerate_objective": degenerate_objective,
    }
