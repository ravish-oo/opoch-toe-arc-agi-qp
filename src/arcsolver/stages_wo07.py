"""
WO-07: Unified min-cost flow stage using OR-Tools SimpleMinCostFlow.

Loads cached artifacts from WO-04, WO-05, WO-06 and solves the pixel-level
flow problem with shared cell capacities.

Implements all mandatory checks from WO-07 spec with pre-check gating.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict

from . import flows
from .precheck import wo7_precheck


def run_wo07(task_id: str, data_root: Path) -> Dict:
    """
    Run WO-07 for a single task: load from cache, precheck, build flow, solve, check.

    Args:
        task_id: Task identifier
        data_root: Path to data directory

    Returns:
        Receipt dict with precheck, solve status and all checks
    """
    # Load inputs from WO-04/05/06 caches
    inputs = flows.load_wo7_inputs(task_id, data_root)

    # Convert quotas dict to array for precheck
    quotas_dict = inputs["quotas"]
    S = len(inputs["bins"])
    C = inputs["C"]
    quotas_array = np.zeros((S, C), dtype=np.int64)
    for (s, c), q in quotas_dict.items():
        quotas_array[s, c] = q

    # Run pre-check before building solver (§8/§10/§11)
    precheck_result = wo7_precheck(
        H=inputs["H"],
        W=inputs["W"],
        bin_ids=inputs["bin_ids"],
        A_mask=inputs["A_mask"],
        quotas=quotas_array,
        eq_rows=inputs.get("equalizer_edges", {}),
    )

    # Build min-cost flow graph
    mcf, metadata = flows.build_flow_graph(inputs)

    # Calculate supplies_total for budget preservation check
    supplies_total = sum(quotas_dict.values())

    # Solve and extract solution
    solution = flows.solve_and_extract(mcf, metadata)

    # Calculate flow_to_sink for budget preservation
    if solution["status"] == "OPTIMAL":
        # Sum all flows going to sink (sink node stored as "T" in metadata)
        flow_to_sink = 0
        for flow_entry in solution["flows"]:
            if flow_entry["head"] == "T":
                flow_to_sink += flow_entry["flow"]

        # Budget preservation: all supplies reach sink (§8)
        budget_preservation_ok = (supplies_total == flow_to_sink)
    else:
        flow_to_sink = 0
        budget_preservation_ok = False

    # Run all checks
    if solution["status"] == "OPTIMAL":
        primal_checks = flows.check_primal_feasibility(solution, inputs, metadata)
        cost_equal_ok = flows.check_cost_equality(solution)
        kkt_ok = flows.check_kkt_reduced_costs(mcf, solution, metadata)
        idempotent_ok = flows.check_idempotence(mcf, metadata, solution, inputs)
        faces_ok = flows.check_faces_postsolve(solution, metadata)
    else:
        primal_checks = {
            "primal_balance_ok": False,
            "capacity_ok": False,
            "mask_ok": False,
            "one_of_10_ok": False,
            "cell_caps_ok": False,
        }
        cost_equal_ok = False
        kkt_ok = None
        idempotent_ok = False
        faces_ok = False

    # Recompute cost for verification
    if solution["status"] == "OPTIMAL":
        recomputed_cost = sum(f["flow"] * f["cost"] for f in solution["flows"])
        recomputed_cost_ok = (recomputed_cost == solution["optimal_cost"])
    else:
        recomputed_cost = None
        recomputed_cost_ok = False

    # Build receipt per WO-07 spec with precheck and budget preservation
    receipt = {
        "stage": "wo07",
        "precheck": {
            "infeasibility_expected": precheck_result["infeasibility_expected"],
            "capacity_ok": precheck_result["capacity_ok"],
            "constant_bin_ok": precheck_result["constant_bin_ok"],
            "capacity_conflicts": precheck_result["capacity_conflicts"][:16],  # Truncate for receipt
            "eq_conflicts": precheck_result["eq_conflicts"][:16],
        },
        "graph": {
            "H": inputs["H"],
            "W": inputs["W"],
            "C": inputs["C"],
            "N": inputs["N"],
            "nodes": {
                "total": metadata["total_nodes"],
                "bins": S,
                "pixels": inputs["N"],
                "cells": inputs["N"],
                "sinks": 1,
            },
            "arcs": metadata["total_arcs"],
            "supplies": {
                "total": supplies_total,
                "by_color": [
                    sum(quotas_dict.get((s, c), 0) for s in range(S))
                    for c in range(C)
                ],
            },
            "faces_mode": metadata["faces_mode"],
        },
        "solve": {
            "status": solution["status"],
            "optimal_cost": solution["optimal_cost"],
            "recomputed_cost": recomputed_cost,
            "recomputed_cost_ok": recomputed_cost_ok,
            "supplies_total": supplies_total,
            "flow_to_sink": flow_to_sink,
            "budget_preservation_ok": budget_preservation_ok,
            "primal_balance_ok": primal_checks["primal_balance_ok"],
            "capacity_ok": primal_checks["capacity_ok"],
            "mask_ok": primal_checks["mask_ok"],
            "one_of_10_ok": primal_checks["one_of_10_ok"],
            "cell_caps_ok": primal_checks["cell_caps_ok"],
            "cost_equal_ok": cost_equal_ok,
            "kkt_ok": kkt_ok,  # Renamed from kkt_reduced_cost_ok
            "faces_ok": faces_ok,
            "idempotent_ok": idempotent_ok,
        },
        "iis": {
            "present": False,
            "note": "IIS extraction deferred to WO-10 per §10/§11",
        },
    }

    # Write receipt to disk
    receipt_dir = data_root.parent / "receipts" / task_id
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / "wo07.json"

    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)

    return receipt
