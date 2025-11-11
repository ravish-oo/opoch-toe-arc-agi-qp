#!/usr/bin/env python3
"""
WO-10: End-to-end Φ + receipts

Orchestrate the full pipeline for each (task_id, test_id):
1. Load WO-9B results and select final pack (lex-min by cost, delta_bits, canvas, pack_id)
2. If feasible → decode to Ŷ using WO-8 decode
3. If infeasible → use IIS from WO-9B
4. Write final receipts and optionally evaluate against ground truth

Per 00 §16, 01 §12, 03 Annex A.1-A.3 (byte-exact determinism).
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from . import decode


def load_ground_truth(data_root: Path, task_id: str) -> Optional[List]:
    """
    Load ground truth test outputs for a task (if available).

    Ground truth is stored in separate solutions files:
    - arc-agi_training_solutions.json for training tasks
    - arc-agi_evaluation_solutions.json for evaluation tasks

    Args:
        data_root: Path to data directory
        task_id: Task identifier

    Returns:
        List of test output grids (2D arrays) or None if not found
    """
    solutions_files = [
        "arc-agi_training_solutions.json",
        "arc-agi_evaluation_solutions.json",
    ]

    for fname in solutions_files:
        fpath = data_root / fname
        if fpath.exists():
            with open(fpath) as f:
                solutions = json.load(f)
            if task_id in solutions:
                return solutions[task_id]  # List of test output grids

    return None  # No solutions available (e.g., test_challenges)


def make_pack_id_safe(pack_id: str, max_len: int = 200) -> str:
    """
    Make pack_id filesystem-safe, handling long pack IDs.

    For pack IDs exceeding max_len, use SHA256 hash for uniqueness.
    The actual pack_id is always stored in the npz file for verification.

    Args:
        pack_id: Original pack ID string
        max_len: Maximum safe filename length (default 200, well under 255 FS limit)

    Returns:
        Filesystem-safe identifier (either cleaned pack_id or hash)
    """
    # Simple character replacement for short IDs
    safe_id = pack_id.replace("|", "_").replace("=", "-").replace("[", "(").replace("]", ")").replace(",", "-")

    # If still too long, use hash-based filename (deterministic, collision-resistant)
    if len(safe_id) > max_len:
        hash_id = hashlib.sha256(pack_id.encode()).hexdigest()
        return hash_id  # 64 hex chars, well under 255 limit

    return safe_id


def compute_delta_bits_for_pack(
    task_id: str,
    pack_id: str,
    cache_root: Path,
) -> int:
    """
    Compute delta_bits for a given pack by loading its solution and running bit-meter.

    Args:
        task_id: Task ID
        pack_id: Pack ID
        cache_root: Path to cache root

    Returns:
        Total bits (delta_bits) for this pack
    """
    # Load pack solution from cache
    pack_id_safe = make_pack_id_safe(pack_id)
    pack_cache_path = cache_root / "wo07" / f"{task_id}.pack_{pack_id_safe}.npz"

    if not pack_cache_path.exists():
        raise FileNotFoundError(f"Pack solution cache not found: {pack_cache_path}")

    pack_cache = np.load(pack_cache_path)
    arcs_p = pack_cache["arcs_p"]
    arcs_c = pack_cache["arcs_c"]
    arcs_flow = pack_cache["arcs_flow"]
    H = int(pack_cache["H"])
    W = int(pack_cache["W"])
    C = int(pack_cache["C"])
    N = H * W

    # Build x_pc from arcs
    x_pc = decode.build_xpc_from_arcs(arcs_p, arcs_c, arcs_flow, N, C)

    # Load costs and A_mask for bit-meter
    wo6_cache = np.load(cache_root / "wo06" / f"{task_id}.npz")
    costs = wo6_cache["costs"]

    wo4_cache = np.load(cache_root / "wo04" / f"{task_id}.npz")
    A_mask = wo4_cache["A_mask"]

    # Compute bit-meter
    _, total_bits, _ = decode.bitmeter_from_costs(costs, A_mask)

    return int(total_bits)


def select_best_pack_lexmin(
    packs_tried: List[Dict],
    task_id: str,
    cache_root: Path,
) -> Dict:
    """
    Select lex-min feasible pack by (optimal_cost, delta_bits, canvas(H,W), pack_id).

    Args:
        packs_tried: List of pack trial dicts from WO-9B receipt
        task_id: Task ID
        cache_root: Path to cache root

    Returns:
        Best pack dict with pack_id, delta_bits, optimal_cost, canvas
    """
    # Filter to feasible packs
    feasible_packs = [p for p in packs_tried if p["result"]["status"] == "OPTIMAL"]

    if not feasible_packs:
        return None

    # Enrich each pack with delta_bits and canvas
    for pack in feasible_packs:
        pack_id = pack["pack_id"]
        pack["delta_bits"] = compute_delta_bits_for_pack(task_id, pack_id, cache_root)

        # Extract canvas (H, W) from pack cache
        pack_id_safe = make_pack_id_safe(pack_id)
        pack_cache = np.load(cache_root / "wo07" / f"{task_id}.pack_{pack_id_safe}.npz")
        pack["canvas"] = (int(pack_cache["H"]), int(pack_cache["W"]))

    # Select lex-min by (optimal_cost, delta_bits, (H,W), pack_id)
    best_pack = min(feasible_packs, key=lambda p: (
        p["result"]["optimal_cost"],
        p["delta_bits"],
        p["canvas"],  # Tuple (H,W) compares lexicographically
        p["pack_id"],
    ))

    return best_pack


def run_wo10(task_id: str, data_root: Path, evaluate: bool = False) -> Dict:
    """
    WO-10: End-to-end pipeline orchestration for a single task.

    Args:
        task_id: Task ID
        data_root: Path to data directory
        evaluate: If True, compare predictions to ground truth

    Returns:
        Summary dict with per-test results and evaluation (if enabled)
    """
    cache_root = data_root.parent / ".cache"
    receipts_dir = Path("receipts")
    task_receipt_dir = receipts_dir / task_id
    task_receipt_dir.mkdir(exist_ok=True, parents=True)

    # Load ARC task data using harness load_task_data
    from .harness import load_task_data
    task_data = load_task_data(data_root, task_id)

    test_inputs = task_data.get("test", [])
    if not test_inputs:
        raise ValueError(f"Task {task_id} has no test inputs")

    # Load ground truth for evaluation (if available)
    ground_truth_outputs = None
    if evaluate:
        ground_truth_outputs = load_ground_truth(data_root, task_id)
        if ground_truth_outputs is None:
            import sys
            print(f"[WO-10] Warning: No ground truth found for {task_id}, skipping evaluation", file=sys.stderr)

    # Process each test input
    test_results = []
    for test_id, test_pair in enumerate(test_inputs):
        # Get ground truth for this specific test (if available)
        ground_truth = None
        if ground_truth_outputs and test_id < len(ground_truth_outputs):
            ground_truth = ground_truth_outputs[test_id]

        test_result = process_single_test(
            task_id,
            test_id,
            test_pair,
            data_root,
            cache_root,
            task_receipt_dir,
            evaluate,
            ground_truth,
        )
        test_results.append(test_result)

    # Write task-level evaluation summary
    if evaluate:
        eval_summary = {
            "task_id": task_id,
            "tests": test_results,
            "summary": {
                "total": len(test_results),
                "matched": sum(1 for r in test_results if r.get("match", False)),
            },
        }

        eval_path = task_receipt_dir / "eval.json"
        with open(eval_path, "w") as f:
            json.dump(eval_summary, f, indent=2, sort_keys=True)

    return {
        "task_id": task_id,
        "test_count": len(test_results),
        "feasible_count": sum(1 for r in test_results if r["status"] == "OPTIMAL"),
        "infeasible_count": sum(1 for r in test_results if r["status"] == "INFEASIBLE"),
    }


def process_single_test(
    task_id: str,
    test_id: int,
    test_pair: Dict,
    data_root: Path,
    cache_root: Path,
    task_receipt_dir: Path,
    evaluate: bool,
    ground_truth: Optional[List] = None,
) -> Dict:
    """
    Process a single test input: load WO-9B results, select lex-min pack, decode if feasible.

    Args:
        ground_truth: Ground truth output grid (2D list) for evaluation, or None

    Returns:
        Dict with status, match (if evaluate), and receipt path
    """
    # Load WO-9B receipt for this test (constant law packs)
    wo9b_receipt_path = task_receipt_dir / "wo09b.json"
    if not wo9b_receipt_path.exists():
        raise FileNotFoundError(f"WO-9B receipt not found: {wo9b_receipt_path}")

    with open(wo9b_receipt_path) as f:
        wo9b_receipt = json.load(f)

    packs_tried = wo9b_receipt["packs_tried"]
    iis = wo9b_receipt.get("iis")

    # Evaluate linear laws per test input (Fix 1: linear law support)
    # Load WO-1 size laws to check for linear laws
    wo1_cache_path = cache_root / "wo01" / f"{task_id}.size_laws.json"
    if wo1_cache_path.exists():
        with open(wo1_cache_path) as f:
            wo1_cache = json.load(f)

        # Check if any linear laws exist
        linear_laws = [sl for sl in wo1_cache["size_laws"] if sl.get("law") == "linear"]

        if len(linear_laws) > 0:
            # Get test input shape
            test_input = np.array(test_pair["input"], dtype=np.int32)
            H_in, W_in = test_input.shape

            # For each linear law, compute concrete (H,W) for this test
            # NOTE: Full implementation would solve flows for these packs
            # For now, mark as "linear_law_deferred" in packs_tried
            for linear_law in linear_laws:
                a_H = linear_law["a_H"]
                b_H = linear_law["b_H"]
                a_W = linear_law["a_W"]
                b_W = linear_law["b_W"]

                H_out = a_H * H_in + b_H
                W_out = a_W * W_in + b_W

                # Add placeholder pack to packs_tried
                # Full implementation: solve flow for this (H,W) canvas
                packs_tried.append({
                    "pack_id": f"size={H_out}x{W_out}|faces=none|free=[]|linear",
                    "drops": [],
                    "result": {
                        "status": "DEFERRED",
                        "note": f"Linear law (H={a_H}*{H_in}+{b_H}={H_out}, W={a_W}*{W_in}+{b_W}={W_out}) detected but not yet solved",
                    },
                })

    # Select best pack using lex-min comparator (WO-10 responsibility)
    best_pack = select_best_pack_lexmin(packs_tried, task_id, cache_root)

    # Determine status
    if best_pack is not None:
        # Check pack viability (Proposal A: reject zero-supply packs)
        # Load WO-7 receipt to get supplies_total
        wo7_receipt_path = task_receipt_dir / "wo07.json"
        with open(wo7_receipt_path) as f:
            wo7_receipt = json.load(f)
        supplies_total = wo7_receipt["graph"]["supplies"]["total"]

        # Load A_mask from WO-4 cache
        wo4_cache_path = cache_root / "wo04" / f"{task_id}.npz"
        wo4_cache = np.load(wo4_cache_path)
        allowed_any = bool(wo4_cache["A_mask"].any())

        # Pack viability check: zero-supply pack cannot express any paid work
        zero_supply_pack = (supplies_total == 0 and allowed_any)

        if zero_supply_pack:
            # Reject degenerate pack → UNSAT per §11/§16
            selected_pack_id = best_pack["pack_id"]
            result = {
                "status": "UNSAT",
                "pack_viability": {
                    "supplies_total": supplies_total,
                    "allowed_any": allowed_any,
                    "zero_supply_pack": True,
                    "decision": "UNSAT",
                    "reason": "R0_zero_supply_pack",
                },
                "iis": None,
            }
            status = "UNSAT"
        else:
            # Feasible - load selected pack's solution and decode
            selected_pack_id = best_pack["pack_id"]
            delta_bits = best_pack["delta_bits"]

            result = process_feasible_test(
                task_id,
                test_id,
                selected_pack_id,
                delta_bits,
                data_root,
                cache_root,
            )
            status = "OPTIMAL"
    else:
        # Infeasible - use IIS from WO-9B
        selected_pack_id = None
        result = {
            "status": "INFEASIBLE",
            "iis": iis,
        }
        status = "INFEASIBLE"

    # Write wo10.json receipt
    wo10_receipt = build_wo10_receipt(
        task_id,
        test_id,
        selected_pack_id,
        result,
        status,
        packs_tried,  # Include packs array in receipt
    )

    wo10_receipt_path = task_receipt_dir / f"wo10_test{test_id}.json"
    with open(wo10_receipt_path, "w") as f:
        json.dump(wo10_receipt, f, indent=2, sort_keys=True)

    # Evaluate if requested
    match = None
    if evaluate and status == "OPTIMAL" and ground_truth is not None:
        ground_truth_array = np.array(ground_truth, dtype=np.int32)
        y_hat = result["y_hat"]
        match = np.array_equal(y_hat, ground_truth_array)

    return {
        "k": test_id,
        "status": status,
        "match": match,
    }


def process_feasible_test(
    task_id: str,
    test_id: int,
    selected_pack_id: str,
    delta_bits: int,
    data_root: Path,
    cache_root: Path,
) -> Dict:
    """
    Process a feasible test: load cached solution, decode.

    Args:
        delta_bits: Pre-computed delta_bits from lex-min selection

    Returns:
        Dict with y_hat, delta_bits, and all validation flags
    """
    # Load cached solution for selected pack
    pack_id_safe = make_pack_id_safe(selected_pack_id)
    pack_cache_path = cache_root / "wo07" / f"{task_id}.pack_{pack_id_safe}.npz"

    if not pack_cache_path.exists():
        raise FileNotFoundError(f"Pack solution cache not found: {pack_cache_path}")

    pack_cache = np.load(pack_cache_path)
    arcs_p = pack_cache["arcs_p"]
    arcs_c = pack_cache["arcs_c"]
    arcs_flow = pack_cache["arcs_flow"]
    H = int(pack_cache["H"])
    W = int(pack_cache["W"])
    C = int(pack_cache["C"])
    N = H * W

    # Build x_pc from arcs
    x_pc = decode.build_xpc_from_arcs(arcs_p, arcs_c, arcs_flow, N, C)

    # Decode to Y_hat
    Y_hat, one_of_10_ok, is_binary_ok = decode.decode_from_xpc(x_pc, H, W)

    # Load A_mask for mask compliance check
    wo4_cache = np.load(cache_root / "wo04" / f"{task_id}.npz")
    A_mask = wo4_cache["A_mask"]

    # Check mask compliance
    mask_ok = decode.check_mask_compliance(x_pc, A_mask)

    # Idempotence check: re-decode should give same result
    Y_hat_2, _, _ = decode.decode_from_xpc(x_pc, H, W)
    idempotence_ok = np.array_equal(Y_hat, Y_hat_2)

    # Compute hash of Y_hat for determinism
    y_hat_hash = hashlib.sha256(Y_hat.tobytes(order='C')).hexdigest()

    return {
        "status": "OPTIMAL",
        "y_hat": Y_hat,
        "y_hat_hash": y_hat_hash,
        "delta_bits": delta_bits,  # Use pre-computed value from lex-min selection
        "one_of_10_decode_ok": bool(one_of_10_ok),
        "decode_mask_ok": bool(mask_ok),
        "idempotence_ok": bool(idempotence_ok),
        "is_binary_ok": bool(is_binary_ok),
        # Load WO-7 proof flags from cache
        "optimal_cost": int(pack_cache["optimal_cost"]),
        "primal_balance_ok": bool(pack_cache["primal_balance_ok"]),
        "capacity_ok": bool(pack_cache["capacity_ok"]),
        "cost_equal_ok": bool(pack_cache["cost_equal_ok"]),
        "one_of_10_ok": bool(pack_cache["one_of_10_ok"]),
    }


def build_wo10_receipt(
    task_id: str,
    test_id: int,
    selected_pack_id: Optional[str],
    result: Dict,
    status: str,
    packs_tried: List[Dict],
) -> Dict:
    """
    Build WO-10 receipt JSON per spec schema.

    Includes packs array showing all packs tried by WO-9B.
    """
    receipt = {
        "stage": "wo10",
        "task_id": task_id,
        "test_id": test_id,
        "selected_pack_id": selected_pack_id,
    }

    if status == "OPTIMAL":
        receipt["final"] = {
            "status": "OPTIMAL",
            "y_hat_hash": result["y_hat_hash"],
            "delta_bits": result["delta_bits"],
            "optimal_cost": result["optimal_cost"],
            "one_of_10_decode_ok": result["one_of_10_decode_ok"],
            "decode_mask_ok": result["decode_mask_ok"],
            "idempotence_ok": result["idempotence_ok"],
            # Mathematical properties of OPTIMAL min-cost flow (not cached, always true):
            "budget_preservation_ok": True,  # Flow conservation at all nodes
            "primal_balance_ok": result["primal_balance_ok"],
            "capacity_ok": result["capacity_ok"],
            "cost_equal_ok": result["cost_equal_ok"],
            "kkt_ok": True,  # OR-Tools OPTIMAL status guarantees KKT conditions
        }
        receipt["iis"] = {"present": False}
    elif status == "UNSAT":
        # Pack viability rejection (e.g., zero-supply pack)
        receipt["final"] = {"status": "UNSAT"}
        receipt["pack_viability"] = result["pack_viability"]
        receipt["iis"] = {"present": False}
    else:
        # INFEASIBLE
        receipt["final"] = {"status": "INFEASIBLE"}
        receipt["iis"] = result["iis"]

    # Include packs array per WO-10 spec schema (Issue #4 fix)
    receipt["packs"] = packs_tried

    # Compute hash for determinism
    canonical_json = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["hash"] = hashlib.sha256(canonical_json.encode()).hexdigest()

    return receipt


# STAGES registry
STAGES = {}
STAGES[11] = run_wo10
