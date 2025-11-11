#!/usr/bin/env python3
"""
WO-9A Stage: Packs & Size Law

Loads caches from WO-1/2/4/5/6 and enumerates deterministic packs for WO-9B.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import numpy as np

from . import packs as packs_module


def run_wo09a(task_id: str, data_root: Path) -> Dict:
    """
    WO-9A: Enumerate packs for a single task.

    Args:
        task_id: Task ID
        data_root: Path to data directory

    Returns:
        Receipt dict with packs, packs_count, hash
    """
    cache_root = data_root.parent / ".cache"

    # === LOAD WO-1 CACHE (size laws) ===
    # Per spec: Load all proven size laws from WO-1 enumeration
    wo1_cache_path = cache_root / "wo01" / f"{task_id}.size_laws.json"
    if not wo1_cache_path.exists():
        raise FileNotFoundError(f"WO-1 size_laws cache not found: {wo1_cache_path}")

    with open(wo1_cache_path) as f:
        wo1_cache = json.load(f)

    # Convert size law dicts to SizeLaw objects
    size_laws = []
    for law_dict in wo1_cache["size_laws"]:
        if law_dict["law"] == "constant":
            size_law = packs_module.SizeLaw(
                law="constant",
                H=law_dict["H"],
                W=law_dict["W"],
                proof_hash=law_dict["proof_hash"],
            )
            size_laws.append(size_law)
        elif law_dict["law"] == "linear":
            # Linear laws are test-specific (need H_in, W_in from test input)
            # They will be evaluated in WO-10, not here at WO-9A
            size_law = packs_module.SizeLaw(
                law="linear",
                a_H=law_dict["a_H"],
                b_H=law_dict["b_H"],
                a_W=law_dict["a_W"],
                b_W=law_dict["b_W"],
                proof_hash=law_dict["proof_hash"],
            )
            size_laws.append(size_law)
        # content-based laws also deferred to WO-10

    # If no size laws proven (shouldn't happen for well-formed tasks), fallback to WO-2
    if len(size_laws) == 0:
        # Fallback: Load from WO-2 for backward compatibility
        wo2_cache_path = cache_root / "wo02" / f"{task_id}.json"
        if wo2_cache_path.exists():
            with open(wo2_cache_path) as f:
                wo2_metadata = json.load(f)
            H_out = wo2_metadata["H_out"]
            W_out = wo2_metadata["W_out"]
            size_law = packs_module.SizeLaw(
                law="constant",
                H=H_out,
                W=W_out,
                proof_hash="fallback",
            )
            size_laws = [size_law]
        else:
            raise ValueError(f"No size laws found for task {task_id} in WO-1 or WO-2")

    # === LOAD WO-4 CACHE (A_mask, bin_ids) ===
    wo4_cache_path = cache_root / "wo04" / f"{task_id}.npz"
    if not wo4_cache_path.exists():
        raise FileNotFoundError(f"WO-4 cache not found: {wo4_cache_path}")

    wo4_data = np.load(wo4_cache_path)
    A_mask = wo4_data["A_mask"]
    bin_ids = wo4_data["bin_ids"]

    # === LOAD WO-5 CACHE (quotas, faces) ===
    wo5_cache_path = cache_root / "wo05" / f"{task_id}.npz"
    if not wo5_cache_path.exists():
        raise FileNotFoundError(f"WO-5 cache not found: {wo5_cache_path}")

    wo5_data = np.load(wo5_cache_path)
    quotas = wo5_data["quotas"]

    # Load faces if present (optional per 00 §8)
    faces_R = wo5_data.get("faces_R", None)
    faces_S = wo5_data.get("faces_S", None)

    # === LOAD WO-6 CACHE (free maps) ===
    # Per spec: "Inputs (from cache; never from receipts)"
    wo6_free_maps_path = cache_root / "wo06" / f"{task_id}.free_maps.json"
    if not wo6_free_maps_path.exists():
        raise FileNotFoundError(f"WO-6 free_maps cache not found: {wo6_free_maps_path}")

    with open(wo6_free_maps_path) as f:
        free_maps_data = json.load(f)

    # Extract verified FREE maps (filter by verified_all_trainings=True)
    free_maps_verified = [
        fm for fm in free_maps_data.get("free_maps_verified", [])
        if fm.get("verified_all_trainings", False)
    ]

    # === ENUMERATE PACKS ===
    # Only enumerate packs for constant laws (with concrete H, W)
    # Linear/content laws are test-specific and will be handled in WO-10
    constant_size_laws = [sl for sl in size_laws if sl.law == "constant"]

    packs = packs_module.enumerate_packs(
        size_laws=constant_size_laws,
        faces_R=faces_R,
        faces_S=faces_S,
        free_maps_verified=free_maps_verified,
        quotas=quotas,
        A_mask=A_mask,
        bin_ids=bin_ids,
    )

    # === BUILD RECEIPT ===
    packs_dicts = [packs_module.pack_to_dict(p) for p in packs]

    receipt = {
        "stage": "wo09a",
        "task_id": task_id,
        "packs": packs_dicts,
        "packs_count": len(packs),
        "hash": packs_module.compute_packs_hash(packs),
        "size_laws_complete": True,  # All proven size laws from WO-1 enumeration
        "size_laws_count": len(size_laws),
        "size_laws_constant_count": len(constant_size_laws),
        "note": "Linear/content laws are test-specific; evaluated in WO-10",
    }

    # === WRITE RECEIPT ===
    receipts_dir = Path("receipts")
    task_receipt_dir = receipts_dir / task_id
    task_receipt_dir.mkdir(exist_ok=True, parents=True)

    wo09a_receipt_path = task_receipt_dir / "wo09a.json"
    with open(wo09a_receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)

    # === WRITE CACHE (identical to receipt packs) ===
    cache_dir = cache_root / "wo09"
    cache_dir.mkdir(exist_ok=True, parents=True)

    packs_cache_path = cache_dir / f"{task_id}.packs.json"
    with open(packs_cache_path, "w") as f:
        json.dump(packs_dicts, f, indent=2, sort_keys=True)

    return receipt


# STAGES registry
STAGES = {}
STAGES[9] = run_wo09a
