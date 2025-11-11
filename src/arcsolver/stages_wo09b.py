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


def run_wo09b(task_id: str, data_root: Path) -> Dict:
    """
    WO-9B: Laminar greedy relaxation for a single task.

    Args:
        task_id: Task ID
        data_root: Path to data directory

    Returns:
        Receipt dict with selected_pack_id, packs_tried, iis, hash
    """
    cache_root = data_root.parent / ".cache"

    # === LOAD PACKS FROM WO-9A CACHE ===
    wo9a_cache_path = cache_root / "wo09" / f"{task_id}.packs.json"
    if not wo9a_cache_path.exists():
        raise FileNotFoundError(f"WO-9A cache not found: {wo9a_cache_path}")

    with open(wo9a_cache_path) as f:
        packs_data = json.load(f)

    # Convert to Pack objects
    packs = []
    for pack_dict in packs_data:
        # Reconstruct SizeLaw
        size_law_dict = pack_dict["size_law"]
        size_law = packs_module.SizeLaw(
            law=size_law_dict["law"],
            H=size_law_dict["H"],
            W=size_law_dict["W"],
        )

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
        packs.append(pack)

    # === LOAD BASE INPUTS FROM CACHES ===
    # Load using flows module
    base_inputs = flows.load_wo7_inputs(task_id, data_root)

    # Also load A_mask and bin_ids for quota reduction
    cache_wo4 = np.load(cache_root / "wo04" / f"{task_id}.npz")
    A_mask = cache_wo4["A_mask"]
    bin_ids = cache_wo4["bin_ids"]

    # === TRY PACKS IN LEX ORDER ===
    packs_tried = []
    selected_pack_id = None
    iis = None

    for pack in packs:
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
            # Success! Select this pack
            pack_trial["result"] = {
                "status": result.status,
                "primal_balance_ok": result.primal_balance_ok,
                "capacity_ok": result.capacity_ok,
                "cost_equal_ok": result.cost_equal_ok,
                "one_of_10_ok": result.one_of_10_ok,
                "selected": True,
            }
            packs_tried.append(pack_trial)
            selected_pack_id = current_pack.pack_id
            break

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
                # Success after dropping faces
                pack_trial["result"] = {
                    "status": result.status,
                    "primal_balance_ok": result.primal_balance_ok,
                    "capacity_ok": result.capacity_ok,
                    "cost_equal_ok": result.cost_equal_ok,
                    "one_of_10_ok": result.one_of_10_ok,
                    "selected": True,
                }
                packs_tried.append(pack_trial)
                selected_pack_id = current_pack.pack_id
                break

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
                    # Success after quota reduction
                    pack_trial["result"] = {
                        "status": result.status,
                        "primal_balance_ok": result.primal_balance_ok,
                        "capacity_ok": result.capacity_ok,
                        "cost_equal_ok": result.cost_equal_ok,
                        "one_of_10_ok": result.one_of_10_ok,
                        "selected": True,
                    }
                    packs_tried.append(pack_trial)
                    selected_pack_id = current_pack.pack_id
                    break

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

    # === BUILD RECEIPT ===
    receipt = {
        "stage": "wo09b",
        "packs_tried": packs_tried,
        "selected_pack_id": selected_pack_id,
        "iis": iis,
    }

    # Compute hash for determinism
    canonical_json = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["hash"] = hashlib.sha256(canonical_json.encode()).hexdigest()

    # === WRITE RECEIPT ===
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
