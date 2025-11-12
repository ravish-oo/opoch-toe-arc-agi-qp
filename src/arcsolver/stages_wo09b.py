#!/usr/bin/env python3
"""
WO-9B Stage: Laminar Greedy Relax + IIS

Loads packs from WO-9A cache, tries each in lex order with laminar precedence:
1. Try as-is
2. Drop faces (if needed)
3. Reduce quotas minimally (if needed)
4. Build IIS if still infeasible

Selects first feasible pack or emits minimal IIS.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from . import flows
from . import packs as packs_module
from . import relax
from .stages_wo10 import make_pack_id_safe


def cache_pack_solution(
    task_id: str,
    data_root: Path,
    result: relax.PackResult,
    pack_id: str,
    H: int,
    W: int,
    C: int,
    A_mask: np.ndarray,
    test_idx: int,
    canvas_id: str,
) -> None:
    """
    Cache a feasible pack's solution for WO-10 lex-min selection.

    WO-B: Saves per-test cache at .cache/wo07/<task_id>.<test_idx>.pack_<pack_id_safe>.npz

    Includes:
    - bin_to_pixel arc flows (for building x_pc)
    - Canvas dimensions (H, W, C)
    - Pack ID
    - Hashes for determinism (costs, A_mask)
    - WO-7 proof flags and optimal_cost
    """
    cache_root = data_root.parent / ".cache"

    # Extract bin_to_pixel arcs from solution
    arcs_p = []
    arcs_c = []
    arcs_flow = []

    for flow_data in result.solution["flows"]:
        if flow_data.get("type") == "bin_to_pixel":
            arcs_p.append(flow_data["p"])
            arcs_c.append(flow_data["c"])
            arcs_flow.append(flow_data["flow"])

    arcs_p = np.array(arcs_p, dtype=np.int32)
    arcs_c = np.array(arcs_c, dtype=np.uint8)
    arcs_flow = np.array(arcs_flow, dtype=np.uint8)

    # Compute hashes for determinism checks
    A_hash = hashlib.sha256(A_mask.tobytes(order='C')).hexdigest()

    # Load WO-6 per-canvas costs for hash
    wo6_cache_path = cache_root / "wo06" / f"{task_id}.{test_idx}.{canvas_id}.npz"
    wo6_cache = np.load(wo6_cache_path)
    costs = wo6_cache["costs"]
    costs_hash = hashlib.sha256(costs.tobytes(order='C')).hexdigest()

    # Save to per-test cache with pack_id in filename (safe encoding)
    cache_dir = cache_root / "wo07"
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Make pack_id filesystem-safe (handles long pack IDs via hashing)
    pack_id_safe = make_pack_id_safe(pack_id)
    cache_path = cache_dir / f"{task_id}.{test_idx}.pack_{pack_id_safe}.npz"

    np.savez(
        cache_path,
        arcs_p=arcs_p,
        arcs_c=arcs_c,
        arcs_flow=arcs_flow,
        H=H,
        W=W,
        C=C,
        pack_id=pack_id,
        costs_hash=costs_hash,
        A_hash=A_hash,
        # WO-7 proof flags
        optimal_cost=result.optimal_cost,
        primal_balance_ok=result.primal_balance_ok,
        capacity_ok=result.capacity_ok,
        cost_equal_ok=result.cost_equal_ok,
        one_of_10_ok=result.one_of_10_ok,
    )


def _process_test_packs(
    task_id: str,
    data_root: Path,
    test_idx: int,
) -> Dict:
    """
    Process packs for a single test: try each with laminar relaxation.

    Args:
        task_id: Task ID
        data_root: Path to data directory
        test_idx: Test index

    Returns:
        Dict with packs_tried, any_feasible, iis
    """
    cache_root = data_root.parent / ".cache"

    # === LOAD PER-TEST PACKS FROM WO-9A′ CACHE ===
    wo9a_cache_path = cache_root / "wo09" / f"{task_id}.{test_idx}.packs.json"
    if not wo9a_cache_path.exists():
        raise FileNotFoundError(f"WO-9A per-test cache not found: {wo9a_cache_path}")

    with open(wo9a_cache_path) as f:
        packs_cache = json.load(f)

    packs_data = packs_cache.get("packs", [])

    # Handle SIZE_LAW_EMPTY from WO-9A' (per 04_engg_spec.md:43)
    # If WO-1 proved no size law, WO-9A' emits empty packs → propagate UNSAT
    if packs_cache.get("packs_count", 0) == 0:
        size_inference = packs_cache.get("size_inference", {})
        return {
            "test_idx": test_idx,
            "any_feasible": False,
            "packs_tried": [],
            "iis": {"present": False},
            "status": "UNSAT",
            "reason": size_inference.get("code", "SIZE_LAW_EMPTY"),
            "note": size_inference.get("note", "No packs to try; propagating UNSAT from WO-9A'"),
        }

    # Convert to Pack objects (with canvas_id metadata)
    packs_with_canvas = []
    for pack_dict in packs_data:
        # Extract canvas_id from pack dict
        canvas_id = pack_dict.get("canvas_id")
        if canvas_id is None:
            raise ValueError(f"Pack missing canvas_id: {pack_dict.get('pack_id')}")

        # Reconstruct SizeLaw
        size_law_dict = pack_dict["size_law"]
        if size_law_dict["law"] == "constant":
            size_law = packs_module.SizeLaw(
                law="constant",
                H=size_law_dict["H"],
                W=size_law_dict["W"],
            )
        elif size_law_dict["law"] == "linear":
            size_law = packs_module.SizeLaw(
                law="linear",
                a_H=size_law_dict["a_H"],
                b_H=size_law_dict["b_H"],
                a_W=size_law_dict["a_W"],
                b_W=size_law_dict["b_W"],
            )
        else:
            raise ValueError(f"Unknown size law: {size_law_dict['law']}")

        # Reconstruct FreeMap objects
        free_maps = []
        for fm_dict in pack_dict["free_maps"]:
            if fm_dict["type"] == "roll":
                free_maps.append(packs_module.FreeMap(
                    type="roll",
                    dy=fm_dict.get("dy"),
                    dx=fm_dict.get("dx"),
                    verified_all_trainings=fm_dict.get("verified_all_trainings", True),
                ))
            elif fm_dict["type"] == "perm":
                free_maps.append(packs_module.FreeMap(
                    type="perm",
                    perm=fm_dict.get("perm", []),
                    verified_all_trainings=fm_dict.get("verified_all_trainings", True),
                ))

        # Reconstruct QuickChecks
        quick_dict = pack_dict["quick"]
        quick = packs_module.QuickChecks(
            capacity_ok=quick_dict["capacity_ok"],
            faces_conflict=quick_dict["faces_conflict"],
            capacity_conflicts=quick_dict["capacity_conflicts"],
            trivial_period=quick_dict.get("trivial_period", False),
        )

        pack = packs_module.Pack(
            pack_id=pack_dict["pack_id"],
            size_law=size_law,
            faces_mode=pack_dict["faces_mode"],
            free_maps=free_maps,
            quick=quick,
        )
        packs_with_canvas.append((pack, canvas_id))

    # === TRY PACKS IN LEX ORDER ===
    packs_tried = []
    iis = None
    any_feasible = False

    for pack, canvas_id in packs_with_canvas:
        # Load per-canvas base inputs
        base_inputs = flows.load_wo7_inputs(
            task_id,
            data_root,
            test_idx=test_idx,
            canvas_id=canvas_id,
        )

        # Extract A_mask and bin_ids for quota reduction
        A_mask = base_inputs["A_mask"]
        bin_ids = base_inputs["bin_ids"]
        pack_trial = {
            "pack_id": pack.pack_id,
            "drops": [],
            "result": None,
        }

        # Track current state through relaxation (laminar precedence)
        current_pack = pack
        current_inputs = base_inputs.copy()

        # Step 1: Try as-is and capture initial state (for IIS if needed)
        initial_result = relax.try_pack(current_pack, current_inputs, A_mask, bin_ids)
        result = initial_result

        if result.status == "OPTIMAL" and all([
            result.primal_balance_ok,
            result.capacity_ok,
            result.mask_ok,
            result.one_of_10_ok,
            result.cell_caps_ok,
            result.cost_equal_ok,
        ]):
            # Feasible! Cache for WO-10 lex-min selection
            # Store the MODIFIED pack_id (after drops) for WO-10 to load correct cache
            pack_trial["pack_id"] = current_pack.pack_id
            pack_trial["canvas_id"] = canvas_id  # For WO-10 per-canvas loading
            pack_trial["result"] = {
                "status": result.status,
                "primal_balance_ok": result.primal_balance_ok,
                "capacity_ok": result.capacity_ok,
                "cost_equal_ok": result.cost_equal_ok,
                "one_of_10_ok": result.one_of_10_ok,
                "optimal_cost": result.optimal_cost,
            }
            packs_tried.append(pack_trial)
            any_feasible = True

            # Cache this pack's solution for WO-10
            cache_pack_solution(
                task_id,
                data_root,
                result,
                current_pack.pack_id,
                base_inputs["H"],
                base_inputs["W"],
                base_inputs["C"],
                A_mask,
                test_idx,
                canvas_id,
            )
            # Continue to try remaining packs (WO-10 needs all feasible packs for lex-min)
            continue

        # Step 2a: Drop faces if failure is faces tier
        if result.failure_tier == "faces" and current_pack.faces_mode != "none":
            pack_trial["drops"].append({
                "tier": "faces",
                "mode": current_pack.faces_mode,
            })

            # Update current_pack: drop faces
            current_pack = relax.drop_faces(current_pack)
            result = relax.try_pack(current_pack, current_inputs, A_mask, bin_ids)

            if result.status == "OPTIMAL" and all([
                result.primal_balance_ok,
                result.capacity_ok,
                result.mask_ok,
                result.one_of_10_ok,
                result.cell_caps_ok,
                result.cost_equal_ok,
            ]):
                # Feasible after dropping faces
                # Store the MODIFIED pack_id (after drops)
                pack_trial["pack_id"] = current_pack.pack_id
                pack_trial["canvas_id"] = canvas_id  # For WO-10 per-canvas loading
                pack_trial["result"] = {
                    "status": result.status,
                    "primal_balance_ok": result.primal_balance_ok,
                    "capacity_ok": result.capacity_ok,
                    "cost_equal_ok": result.cost_equal_ok,
                    "one_of_10_ok": result.one_of_10_ok,
                    "optimal_cost": result.optimal_cost,
                }
                packs_tried.append(pack_trial)
                any_feasible = True

                # Cache this pack's solution for WO-10
                cache_pack_solution(
                    task_id,
                    data_root,
                    result,
                    current_pack.pack_id,
                    base_inputs["H"],
                    base_inputs["W"],
                    base_inputs["C"],
                    A_mask,
                    test_idx,
                    canvas_id,
                )
                # Continue to try remaining packs
                continue

        # Step 2b: Reduce quotas if failure is quota tier
        # (Uses current_pack, which may have faces already dropped)
        if result.failure_tier == "quota":
            # Reduce quotas minimally
            new_quotas, quota_drops = relax.reduce_quotas_minimally(
                current_inputs["quotas"],
                A_mask,
                bin_ids,
                current_inputs["bins"],
            )

            if len(quota_drops) > 0:
                pack_trial["drops"].extend(quota_drops)

                # Update current_inputs: reduce quotas
                current_inputs = current_inputs.copy()
                current_inputs["quotas"] = new_quotas
                result = relax.try_pack(current_pack, current_inputs, A_mask, bin_ids)

                if result.status == "OPTIMAL" and all([
                    result.primal_balance_ok,
                    result.capacity_ok,
                    result.mask_ok,
                    result.one_of_10_ok,
                    result.cell_caps_ok,
                    result.cost_equal_ok,
                ]):
                    # Feasible after quota reduction
                    # Store the MODIFIED pack_id (after drops)
                    pack_trial["pack_id"] = current_pack.pack_id
                    pack_trial["canvas_id"] = canvas_id  # For WO-10 per-canvas loading
                    pack_trial["result"] = {
                        "status": result.status,
                        "primal_balance_ok": result.primal_balance_ok,
                        "capacity_ok": result.capacity_ok,
                        "cost_equal_ok": result.cost_equal_ok,
                        "one_of_10_ok": result.one_of_10_ok,
                        "optimal_cost": result.optimal_cost,
                    }
                    packs_tried.append(pack_trial)
                    any_feasible = True

                    # Cache this pack's solution for WO-10
                    cache_pack_solution(
                        task_id,
                        data_root,
                        result,
                        current_pack.pack_id,
                        base_inputs["H"],
                        base_inputs["W"],
                        base_inputs["C"],
                        A_mask,
                        test_idx,
                        canvas_id,
                    )
                    # Continue to try remaining packs
                    continue

        # Step 3: If still infeasible and hard tier, build IIS
        if result.failure_tier == "hard":
            # Build IIS using ORIGINAL pack/inputs to explain why initial problem failed
            # (not the post-relaxation state - per Fix C)
            iis = relax.build_iis_ddmin(
                pack,  # ORIGINAL pack, not current_pack
                base_inputs,  # ORIGINAL inputs, not current_inputs
                A_mask,
                bin_ids,
                initial_result,  # Has ORIGINAL precheck before relaxations
            )

            pack_trial["result"] = {
                "status": result.status,
                "primal_balance_ok": result.primal_balance_ok,
                "capacity_ok": result.capacity_ok,
                "cost_equal_ok": result.cost_equal_ok,
                "one_of_10_ok": result.one_of_10_ok,
                "selected": False,
            }
            packs_tried.append(pack_trial)

            # Stop after first IIS
            break

        # Record failure
        pack_trial["result"] = {
            "status": result.status,
            "primal_balance_ok": result.primal_balance_ok,
            "capacity_ok": result.capacity_ok,
            "cost_equal_ok": result.cost_equal_ok,
            "one_of_10_ok": result.one_of_10_ok,
            "selected": False,
        }
        packs_tried.append(pack_trial)

    # Return test results (no receipt here - done by run_wo09b)
    return {
        "test_idx": test_idx,
        "packs_tried": packs_tried,
        "any_feasible": any_feasible,
        "iis": iis,
    }


def run_wo09b(task_id: str, data_root: Path) -> Dict:
    """
    WO-9B: Laminar greedy relaxation for all tests in task.

    WO-B: Iterates over all test cases, processes packs per-test with per-canvas WO-4/5/6.

    Args:
        task_id: Task ID
        data_root: Path to data directory

    Returns:
        Receipt dict with tests_processed, total_any_feasible, hash
    """
    cache_root = data_root.parent / ".cache"

    # === DISCOVER ALL TESTS ===
    # Find all per-test pack caches to determine number of tests
    wo9_cache_dir = cache_root / "wo09"
    test_indices = []
    for pack_file in wo9_cache_dir.glob(f"{task_id}.*.packs.json"):
        # Extract test_idx from filename: {task_id}.{test_idx}.packs.json
        parts = pack_file.stem.split(".")
        if len(parts) >= 2:
            try:
                test_idx = int(parts[1])
                test_indices.append(test_idx)
            except ValueError:
                pass  # Skip non-numeric test indices

    if len(test_indices) == 0:
        raise FileNotFoundError(f"No per-test packs found for task {task_id} in {wo9_cache_dir}")

    test_indices = sorted(test_indices)

    # === PROCESS EACH TEST ===
    tests_processed = []
    for test_idx in test_indices:
        test_result = _process_test_packs(task_id, data_root, test_idx)
        tests_processed.append(test_result)

        # Save per-test receipt
        receipts_dir = Path("receipts")
        task_receipt_dir = receipts_dir / task_id
        task_receipt_dir.mkdir(exist_ok=True, parents=True)

        test_receipt = {
            "stage": "wo09b",
            "test_idx": test_idx,
            "packs_tried": test_result["packs_tried"],
            "any_feasible": test_result["any_feasible"],
            "iis": test_result["iis"],
        }

        # Include UNSAT reason/note if present (SIZE_LAW_EMPTY propagation)
        if "status" in test_result:
            test_receipt["status"] = test_result["status"]
        if "reason" in test_result:
            test_receipt["reason"] = test_result["reason"]
        if "note" in test_result:
            test_receipt["note"] = test_result["note"]

        # Compute hash for per-test determinism
        canonical_json = json.dumps(test_receipt, sort_keys=True, separators=(",", ":"))
        test_receipt["hash"] = hashlib.sha256(canonical_json.encode()).hexdigest()

        wo09b_test_receipt_path = task_receipt_dir / f"wo09b.{test_idx}.json"
        with open(wo09b_test_receipt_path, "w") as f:
            json.dump(test_receipt, f, indent=2, sort_keys=True)

    # === BUILD TASK-LEVEL RECEIPT ===
    total_any_feasible = any(t["any_feasible"] for t in tests_processed)

    receipt = {
        "stage": "wo09b",
        "task_id": task_id,
        "tests_processed": [{"test_idx": t["test_idx"], "any_feasible": t["any_feasible"]} for t in tests_processed],
        "total_any_feasible": total_any_feasible,
        "total_tests": len(test_indices),
    }

    # Compute hash for task-level determinism
    canonical_json = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["hash"] = hashlib.sha256(canonical_json.encode()).hexdigest()

    # === WRITE TASK-LEVEL RECEIPT ===
    receipts_dir = Path("receipts")
    task_receipt_dir = receipts_dir / task_id
    task_receipt_dir.mkdir(exist_ok=True, parents=True)

    wo09b_receipt_path = task_receipt_dir / "wo09b.json"
    with open(wo09b_receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)

    return receipt


# STAGES registry
STAGES = {}
STAGES[10] = run_wo09b
