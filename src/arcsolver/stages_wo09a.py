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

    # === LOAD WO-2 CACHE (size) ===
    # Per spec: "Inputs (from cache; never from receipts)"
    wo2_cache_path = cache_root / "wo02" / f"{task_id}.json"
    if not wo2_cache_path.exists():
        raise FileNotFoundError(f"WO-2 cache not found: {wo2_cache_path}")

    with open(wo2_cache_path) as f:
        wo2_metadata = json.load(f)

    H_out = wo2_metadata["H_out"]
    W_out = wo2_metadata["W_out"]
    mode = wo2_metadata.get("mode", "topleft")

    # INTERIM: single size law (spec prefers extending WO-1 to enumerate all proven laws)
    # Per WO-9A spec, only three valid law types: "constant" | "linear" | "content"
    # Periods affect internal structure but not output size variation → "constant"
    if mode in ["topleft", "center"]:
        law_type = "constant"  # bbox-based sizing
    else:
        law_type = "content"  # content-dependent

    size_law = packs_module.SizeLaw(
        law=law_type,
        H=H_out,
        W=W_out,
        proof_hash="",  # Would come from WO-1 enumeration
    )
    size_laws = [size_law]

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
    packs = packs_module.enumerate_packs(
        size_laws=size_laws,
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
        "size_laws_complete": False,  # INTERIM: single size law from WO-2; full enumeration requires WO-1 extension
        "note": "interim single size law; WO-1 extension pending",
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
