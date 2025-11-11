#!/usr/bin/env python3
"""
WO-9B: Laminar greedy relaxation and minimal IIS construction.

Pure functions for pack trials, faces dropping, quota reduction, and IIS
extraction using ddmin algorithm with WO-7 as feasibility oracle.
"""
from __future__ import annotations
import copy
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Callable
from dataclasses import dataclass

from . import flows
from . import packs as packs_module


def infer_failure_tier_from_precheck(
    pack: packs_module.Pack,
    precheck: Dict,
    primal_checks: Optional[Dict] = None,
) -> str:
    """
    Infer failure tier from structured precheck and primal results.

    Returns "faces" | "quota" | "hard" based on which tier failed.
    Avoids string matching on exception messages.
    """
    # Check precheck results first
    if not precheck.get("capacity_ok", True):
        # Has capacity conflicts - could be quota or hard tier
        capacity_conflicts = precheck.get("capacity_conflicts", [])

        # Check for quota conflicts (q > allowed where allowed > 0)
        # These are relaxable via quota reduction
        has_quota_conflict = any(
            c.get("q", 0) > c.get("allowed", 0) and c.get("allowed", 0) > 0
            for c in capacity_conflicts
        )

        # Check mask conflicts FIRST (allowed==0 means never-relax per §10)
        # Any mask violation → hard tier (go straight to IIS)
        has_mask_violation = any(c.get("allowed", 0) == 0 for c in capacity_conflicts)
        if has_mask_violation:
            return "hard"

        # Only if NO mask conflicts, check quota-tier conflicts (allowed > 0)
        # Quota excess with allowed>0 is relaxable by reducing quotas
        if has_quota_conflict:
            return "quota"

        # Default to quota if neither (shouldn't happen if capacity_ok=false)
        return "quota"

    if not precheck.get("constant_bin_ok", True):
        # Equalizer conflicts are hard tier
        return "hard"

    # Check solve-phase results if available
    if primal_checks:
        if not primal_checks.get("mask_ok", True):
            return "hard"
        if not primal_checks.get("cell_caps_ok", True):
            return "hard"
        if not primal_checks.get("capacity_ok", True):
            return "quota"

    # If faces mode is active, check if faces could be the issue
    if pack.faces_mode != "none":
        # Check if quick checks indicated faces conflict
        if hasattr(pack, 'quick') and pack.quick.faces_conflict:
            return "faces"

    # Default to hard tier
    return "hard"


@dataclass
class PackResult:
    """Result from trying a pack with WO-7"""
    status: str  # "OPTIMAL" | "INFEASIBLE" | ...
    primal_balance_ok: bool
    capacity_ok: bool
    mask_ok: bool
    one_of_10_ok: bool
    cell_caps_ok: bool
    cost_equal_ok: bool
    optimal_cost: Optional[int]
    failure_tier: Optional[str]  # "faces" | "quota" | "hard" | None
    precheck: Dict
    solution: Optional[Dict] = None  # Add solution for cell cap checking


def try_pack(
    pack: packs_module.Pack,
    inputs: Dict,
    A_mask: np.ndarray,
    bin_ids: np.ndarray,
) -> PackResult:
    """
    Try a pack configuration with WO-7 flow solver.

    Args:
        pack: Pack configuration from WO-9A
        inputs: Base inputs dict from flows.load_wo7_inputs()
        A_mask: (N, C) mask array
        bin_ids: (N,) bin assignments

    Returns:
        PackResult with status, checks, and failure tier
    """
    # Modify inputs for this pack's configuration
    pack_inputs = inputs.copy()
    pack_inputs["faces_mode"] = pack.faces_mode

    # Set faces based on mode
    if pack.faces_mode == "rows_as_supply":
        pack_inputs["faces_R"] = inputs.get("faces_R", None)
        pack_inputs["faces_S"] = None
    elif pack.faces_mode == "cols_as_supply":
        pack_inputs["faces_R"] = None
        pack_inputs["faces_S"] = inputs.get("faces_S", None)
    else:  # "none"
        pack_inputs["faces_R"] = None
        pack_inputs["faces_S"] = None

    # Explicit faces precondition check BEFORE building graph (deterministic)
    if pack.faces_mode != "none":
        try:
            flows.check_input_preconditions(pack_inputs)
        except ValueError:
            # Faces precondition failed - return deterministically
            from .precheck import wo7_precheck

            # Convert quotas dict to array for precheck
            quotas_dict = pack_inputs["quotas"]
            S = len(pack_inputs["bins"])
            C = pack_inputs["C"]
            quotas_array = np.zeros((S, C), dtype=np.int64)
            for (s, c), q in quotas_dict.items():
                quotas_array[s, c] = q

            precheck_result = wo7_precheck(
                H=pack_inputs["H"],
                W=pack_inputs["W"],
                bin_ids=bin_ids,
                A_mask=A_mask,
                quotas=quotas_array,
                eq_rows=pack_inputs.get("equalizer_edges", {}),
            )

            return PackResult(
                status="INFEASIBLE",
                primal_balance_ok=False,
                capacity_ok=False,
                mask_ok=False,
                one_of_10_ok=False,
                cell_caps_ok=False,
                cost_equal_ok=False,
                optimal_cost=None,
                failure_tier="faces",  # Deterministic - we know it's faces
                precheck=precheck_result,
                solution=None,
            )

    # Build flow graph and solve
    mcf, metadata = flows.build_flow_graph(pack_inputs)
    solution = flows.solve_and_extract(mcf, metadata)

    # Run checks
    if solution["status"] == "OPTIMAL":
        primal_checks = flows.check_primal_feasibility(solution, pack_inputs, metadata)
        cost_equal_ok = flows.check_cost_equality(solution)

        # Infer failure tier (if any check fails) using structured approach
        failure_tier = None
        if not (primal_checks["primal_balance_ok"] and
               primal_checks["capacity_ok"] and
               primal_checks["mask_ok"] and
               primal_checks["one_of_10_ok"] and
               primal_checks["cell_caps_ok"] and
               cost_equal_ok):
            # Use structured inference instead of ad-hoc logic
            failure_tier = infer_failure_tier_from_precheck(pack, {}, primal_checks)

        return PackResult(
            status=solution["status"],
            primal_balance_ok=primal_checks["primal_balance_ok"],
            capacity_ok=primal_checks["capacity_ok"],
            mask_ok=primal_checks["mask_ok"],
            one_of_10_ok=primal_checks["one_of_10_ok"],
            cell_caps_ok=primal_checks["cell_caps_ok"],
            cost_equal_ok=cost_equal_ok,
            optimal_cost=solution["optimal_cost"],
            failure_tier=failure_tier,
            precheck={},
            solution=solution,
        )
    else:
        # INFEASIBLE - need to determine tier from precheck
        from .precheck import wo7_precheck

        # Convert quotas dict to array for precheck
        quotas_dict = pack_inputs["quotas"]
        S = len(pack_inputs["bins"])
        C = pack_inputs["C"]
        quotas_array = np.zeros((S, C), dtype=np.int64)
        for (s, c), q in quotas_dict.items():
            quotas_array[s, c] = q

        precheck_result = wo7_precheck(
            H=pack_inputs["H"],
            W=pack_inputs["W"],
            bin_ids=bin_ids,
            A_mask=A_mask,
            quotas=quotas_array,
            eq_rows=pack_inputs.get("equalizer_edges", {}),
        )

        # Use structured failure tier inference
        failure_tier = infer_failure_tier_from_precheck(pack, precheck_result)

        return PackResult(
            status=solution["status"],
            primal_balance_ok=False,
            capacity_ok=False,
            mask_ok=False,
            one_of_10_ok=False,
            cell_caps_ok=False,
            cost_equal_ok=False,
            optimal_cost=None,
            failure_tier=failure_tier,
            precheck=precheck_result,
            solution=solution,
        )


def drop_faces(pack: packs_module.Pack) -> packs_module.Pack:
    """
    Create new pack with faces_mode set to "none".

    Args:
        pack: Original pack

    Returns:
        New pack with faces dropped
    """
    # Create new pack with modified faces_mode
    new_pack_id = packs_module.build_pack_id(
        pack.size_law,
        "none",
        pack.free_maps,
    )

    return packs_module.Pack(
        pack_id=new_pack_id,
        size_law=pack.size_law,
        faces_mode="none",
        free_maps=pack.free_maps,
        quick=pack.quick,
    )


def reduce_quotas_minimally(
    quotas: Dict[Tuple[int, int], int],
    A_mask: np.ndarray,
    bin_ids: np.ndarray,
    bins: Dict[int, List[int]],
) -> Tuple[Dict[Tuple[int, int], int], List[Dict]]:
    """
    Reduce quotas minimally to fix capacity conflicts.

    For each (s,c) where q[s,c] > allowed[s,c], reduce to allowed[s,c].

    Args:
        quotas: Original quotas dict {(s,c): q}
        A_mask: (N, C) mask array
        bin_ids: (N,) bin assignments
        bins: Dict of bin -> pixel list

    Returns:
        (new_quotas, drops) where drops is list of {tier, s, c, drop}
    """
    new_quotas = quotas.copy()
    drops = []

    for (s, c), q in quotas.items():
        # Count allowed pixels in bin s for color c
        allowed_count = 0
        for p in bins[s]:
            if A_mask[p, c]:
                allowed_count += 1

        # Skip mask conflicts (allowed==0) - never relax these per §10
        if allowed_count == 0:
            continue

        if q > allowed_count:
            drop = q - allowed_count
            new_quotas[(s, c)] = allowed_count
            drops.append({
                "tier": "quota",
                "s": int(s),
                "c": int(c),
                "drop": int(drop),
            })

    return new_quotas, drops


# === IIS Construction (ddmin algorithm) ===

@dataclass
class ConstraintRow:
    """A hard-tier constraint row for IIS"""
    type: str  # "mask" | "equalizer" | "cell_cap"
    # For mask: p, c
    p: Optional[int] = None
    c: Optional[int] = None
    # For equalizer: bin, color, p, q (edge)
    bin: Optional[int] = None
    # For cell: r, j
    r: Optional[int] = None
    j: Optional[int] = None
    # For equalizer edge
    q: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization"""
        result = {"type": self.type}
        if self.type == "mask":
            result["p"] = self.p
            result["c"] = self.c
        elif self.type == "equalizer":
            result["bin"] = self.bin
            result["color"] = self.c
            result["p"] = self.p
            result["q"] = self.q
        elif self.type == "cell_cap":
            result["r"] = self.r
            result["j"] = self.j
        return result

    def sort_key(self) -> Tuple:
        """Deterministic sort key for IIS ordering"""
        if self.type == "mask":
            return (0, self.p or 0, self.c or 0)
        elif self.type == "equalizer":
            return (1, self.bin or 0, self.c or 0, self.p or 0, self.q or 0)
        elif self.type == "cell_cap":
            return (2, self.r or 0, self.j or 0)
        return (3,)


def collect_hard_tier_constraints(
    precheck: Dict,
    inputs: Dict,
    solution: Optional[Dict] = None,
) -> List[ConstraintRow]:
    """
    Collect hard-tier constraint violations from precheck and solve results.

    Args:
        precheck: Precheck result from wo7_precheck
        inputs: Inputs dict with equalizer_edges, bins, cell_caps
        solution: Optional solve result for cell capacity violations

    Returns:
        List of ConstraintRow objects
    """
    constraints = []
    bins = inputs.get("bins", {})
    A_mask = inputs.get("A_mask")

    # Mask conflicts (capacity conflicts where allowed=0)
    for conflict in precheck.get("capacity_conflicts", []):
        if conflict["allowed"] == 0:
            # This is a mask violation (q > 0 but allowed = 0)
            # All pixels in this bin are masked out for this color
            s = conflict["bin"]
            c = conflict["color"]

            # Find a representative pixel from this bin
            # Pick first pixel in bin (deterministic)
            if s in bins and len(bins[s]) > 0:
                p = bins[s][0]  # First pixel in bin
                constraints.append(ConstraintRow(
                    type="mask",
                    p=p,
                    c=c,
                    bin=s,
                ))

    # Equalizer conflicts
    for conflict in precheck.get("eq_conflicts", []):
        s = conflict["bin"]
        c = conflict["color"]
        # Get equalizer edges for this (s,c)
        eq_edges = inputs.get("equalizer_edges", {}).get((s, c), None)
        if eq_edges is not None and eq_edges.shape[0] > 0:
            # Add each edge as a constraint
            for edge_idx in range(eq_edges.shape[0]):
                p, q = eq_edges[edge_idx, 0], eq_edges[edge_idx, 1]
                constraints.append(ConstraintRow(
                    type="equalizer",
                    bin=s,
                    c=c,
                    p=int(p),
                    q=int(q),
                ))

    # Cell capacity violations (from solve results)
    if solution is not None:
        H = inputs.get("H", 0)
        W = inputs.get("W", 0)
        cell_caps = inputs.get("cell_caps", {})

        # Build cell outflow map from solution flows
        cell_outflow = {}
        for flow_data in solution.get("flows", []):
            if flow_data.get("type") == "cell_to_sink":
                r = flow_data.get("r")
                j = flow_data.get("j")
                if r is not None and j is not None:
                    cell_outflow[(r, j)] = flow_data.get("flow", 0)

        # Check for violations
        for r in range(H):
            for j in range(W):
                n_rj = cell_caps.get((r, j), 1)
                flow = cell_outflow.get((r, j), 0)
                if flow > n_rj:
                    constraints.append(ConstraintRow(
                        type="cell_cap",
                        r=r,
                        j=j,
                    ))

    # Sort deterministically
    constraints.sort(key=lambda row: row.sort_key())

    return constraints


def build_iis_ddmin(
    pack: packs_module.Pack,
    inputs: Dict,
    A_mask: np.ndarray,
    bin_ids: np.ndarray,
    initial_result: PackResult,
) -> Dict:
    """
    Build minimal IIS using ddmin algorithm.

    Args:
        pack: Pack configuration that failed
        inputs: Base inputs dict
        A_mask: (N, C) mask array
        bin_ids: (N,) bin assignments
        initial_result: Initial failure result from try_pack

    Returns:
        IIS dict with rows and minimality_checks
    """
    # Collect hard-tier constraint violations (including cell caps from solution)
    constraints = collect_hard_tier_constraints(
        initial_result.precheck,
        inputs,
        initial_result.solution,
    )

    if len(constraints) == 0:
        # Compute audit counts for empty IIS (Fix D)
        mask_count = sum(1 for c in initial_result.precheck.get("capacity_conflicts", []) if c.get("allowed", 0) == 0)
        eq_count = len(initial_result.precheck.get("eq_conflicts", []))
        cell_checked = initial_result.solution is not None

        return {
            "present": False,
            "note": "No hard-tier constraints identified",
            "audit": {
                "mask_conflicts": mask_count,
                "eq_conflicts": eq_count,
                "cell_cap_checked": cell_checked,
            }
        }

    # Define FEASIBLE oracle that actually tests feasibility
    def feasible_without(removed_indices: Set[int]) -> bool:
        """
        Check if system is FEASIBLE when certain constraints are NOT enforced.

        Args:
            removed_indices: Indices of constraints to NOT enforce (remove/relax)

        Returns:
            True if WO-7 returns OPTIMAL with all checks passing
        """
        # Clone inputs and masks - use deep copy to avoid shared state
        modified_inputs = inputs.copy()
        modified_A_mask = A_mask.copy()  # numpy array copy is deep

        # Deep copy equalizer_edges dict AND its numpy array values
        modified_eq_edges = {}
        for key, edges in inputs.get("equalizer_edges", {}).items():
            modified_eq_edges[key] = edges.copy()  # Copy each numpy array

        # Remove constraints from the system
        for idx in removed_indices:
            if idx >= len(constraints):
                continue
            constraint = constraints[idx]

            if constraint.type == "mask":
                # Allow this (p, c) pair by setting mask to True
                p = constraint.p
                c = constraint.c
                if p is not None and c is not None and p < len(modified_A_mask):
                    modified_A_mask[p, c] = True

            elif constraint.type == "equalizer":
                # Remove this edge from equalizer_edges
                s = constraint.bin
                c = constraint.c
                key = (s, c)

                if key in modified_eq_edges:
                    edges = modified_eq_edges[key]
                    # Filter out this specific edge (p, q)
                    p_edge = constraint.p
                    q_edge = constraint.q

                    if edges.ndim == 2 and edges.shape[0] > 0:
                        mask = ~((edges[:, 0] == p_edge) & (edges[:, 1] == q_edge))
                        edges_filtered = edges[mask]

                        if len(edges_filtered) == 0:
                            # No edges left, remove the key
                            del modified_eq_edges[key]
                        else:
                            modified_eq_edges[key] = edges_filtered

        # Update modified inputs with deep-copied structures
        modified_inputs["A_mask"] = modified_A_mask
        modified_inputs["equalizer_edges"] = modified_eq_edges

        # Deep copy bins dict to avoid shared state
        modified_inputs["bins"] = copy.deepcopy(inputs.get("bins", {}))

        # Rebuild flow graph and solve
        try:
            mcf, metadata = flows.build_flow_graph(modified_inputs)
            solution = flows.solve_and_extract(mcf, metadata)

            if solution["status"] == "OPTIMAL":
                primal_checks = flows.check_primal_feasibility(solution, modified_inputs, metadata)
                cost_equal_ok = flows.check_cost_equality(solution)

                # System is feasible if all checks pass
                return all([
                    primal_checks["primal_balance_ok"],
                    primal_checks["capacity_ok"],
                    primal_checks["mask_ok"],
                    primal_checks["one_of_10_ok"],
                    primal_checks["cell_caps_ok"],
                    cost_equal_ok,
                ])
            else:
                return False
        except Exception:
            # Any exception means infeasible
            return False

    # Run ddmin to find minimal conflict set
    minimal_constraints = ddmin(constraints, feasible_without)

    # Generate minimality checks - verify each row is necessary
    minimality_checks = []
    for idx in range(len(minimal_constraints)):
        # Map back to original constraint indices
        original_idx = constraints.index(minimal_constraints[idx])

        # Check if removing JUST this row makes system feasible
        removed = {original_idx}
        is_feasible = feasible_without(removed)
        minimality_checks.append({
            "row_idx": idx,
            "feasible_when_removed": is_feasible,
        })

    # Verify minimality: all checks should be True for valid IIS
    all_minimal = all(check["feasible_when_removed"] for check in minimality_checks)

    # Build IIS dict
    iis = {
        "present": True,
        "rows": [row.to_dict() for row in minimal_constraints],
        "minimality_checks": minimality_checks,
        "is_minimal": all_minimal,
    }

    return iis


def ddmin(
    candidates: List[ConstraintRow],
    oracle: Callable[[Set[int]], bool],
) -> List[ConstraintRow]:
    """
    Delta debugging algorithm for minimal conflict set.

    Finds minimal subset K ⊆ candidates such that:
    - System is infeasible with K enforced
    - System is feasible when any single element of K is removed (1-minimal)

    Algorithm from "Simplifying and Isolating Failure-Inducing Input" (Zeller & Hildebrandt)

    Args:
        candidates: List of constraint rows (deterministically ordered)
        oracle: Function oracle(R) that returns True if system is FEASIBLE
                when constraints R are NOT enforced (removed)

    Returns:
        Minimal subset that is still infeasible (1-minimal IIS)
    """
    # Work with indices for efficiency
    C = list(range(len(candidates)))
    k = 2  # Start with 2 chunks

    while k <= len(C):
        # Split C into k chunks
        chunk_size = max(1, len(C) // k)
        chunks = []
        for i in range(0, len(C), chunk_size):
            chunks.append(C[i:i + chunk_size])

        progress = False

        for chunk in chunks:
            # Test if C \ chunk is still infeasible
            # remaining = indices we keep enforced
            remaining = [idx for idx in C if idx not in chunk]

            # removed = indices we don't enforce (relax)
            # We want to know: is system infeasible with only 'remaining' enforced?
            # Oracle returns True if FEASIBLE when given indices are NOT enforced
            # So we ask: is system FEASIBLE when we remove everything except 'remaining'?
            # That means: is system FEASIBLE when we remove complement(remaining)?
            removed = set([idx for idx in range(len(candidates)) if idx not in remaining])

            # If oracle(removed) returns False, it means:
            # System is INFEASIBLE even when we remove complement(remaining)
            # i.e., system is INFEASIBLE with only 'remaining' enforced
            # So we can safely drop the 'chunk' and continue with smaller 'remaining'
            if not oracle(removed):
                # Still infeasible - we can drop this chunk
                C = remaining
                k = 2  # Restart with smaller set
                progress = True
                break

        if not progress:
            # No chunk could be removed - increase granularity
            if k == len(C):
                # Can't subdivide further - we're done
                break
            k = min(len(C), 2 * k)

    # Return minimal constraint subset
    return [candidates[idx] for idx in sorted(C)]
