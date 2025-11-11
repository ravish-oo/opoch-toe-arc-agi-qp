#!/usr/bin/env python3
"""
WO-9A′ Stage: Per-test concrete pack builder + per-canvas WO-4/5

Refactored WO-9A to support linear size laws. Enumerates packs per (task,test,canvas).
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Dict, List
import numpy as np

from . import packs as packs_module
from . import canvas_ops


def run_wo09a_prime(task_id: str, data_root: Path) -> Dict:
    """
    WO-9A′: Enumerate packs per test with per-canvas WO-4/5.

    Args:
        task_id: Task ID
        data_root: Path to data directory

    Returns:
        Receipt dict with per-test pack enumeration summary
    """
    cache_root = data_root.parent / ".cache"

    # === LOAD TASK ===
    # Load task data from ARC JSON files
    from .harness import load_task_data
    task = load_task_data(data_root, task_id)

    train_pairs = task["train"]
    test_pairs = task.get("test", [])

    if len(test_pairs) == 0:
        raise ValueError(f"Task {task_id} has no test pairs")

    # === LOAD WO-1 CACHE (size laws) ===
    wo1_cache_path = cache_root / "wo01" / f"{task_id}.size_laws.json"
    if not wo1_cache_path.exists():
        raise FileNotFoundError(f"WO-1 size_laws cache not found: {wo1_cache_path}")

    with open(wo1_cache_path) as f:
        wo1_cache = json.load(f)

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
            size_law = packs_module.SizeLaw(
                law="linear",
                a_H=law_dict["a_H"],
                b_H=law_dict["b_H"],
                a_W=law_dict["a_W"],
                b_W=law_dict["b_W"],
                proof_hash=law_dict["proof_hash"],
            )
            size_laws.append(size_law)

    if len(size_laws) == 0:
        raise ValueError(f"No size laws found for task {task_id} in WO-1")

    # === LOAD WO-6 CACHE (FREE maps) ===
    wo6_free_maps_path = cache_root / "wo06" / f"{task_id}.free_maps.json"
    if not wo6_free_maps_path.exists():
        raise FileNotFoundError(f"WO-6 free_maps cache not found: {wo6_free_maps_path}")

    with open(wo6_free_maps_path) as f:
        free_maps_data = json.load(f)

    # Extract verified FREE maps
    free_maps_verified = [
        fm for fm in free_maps_data.get("free_maps_verified", [])
        if fm.get("verified_all_trainings", False)
    ]

    # === DETERMINE EMBED MODE (from task or WO-3) ===
    # Try to load from WO-3 receipt if available
    wo3_receipt_path = Path("receipts") / task_id / "wo03.json"
    embed_mode = "topleft"  # default

    if wo3_receipt_path.exists():
        with open(wo3_receipt_path) as f:
            wo3_receipt = json.load(f)
        embed_mode = wo3_receipt.get("embed_mode", "topleft")

    # === LOAD TRAINING INPUTS AND OUTPUTS (Phase 3) ===
    # Raw grids, not aligned yet
    train_inputs_raw = []
    train_outputs_raw = []
    for pair in train_pairs:
        X_i = np.array(pair["input"], dtype=np.int32)
        Y_i = np.array(pair["output"], dtype=np.int32)
        train_inputs_raw.append(X_i)
        train_outputs_raw.append(Y_i)

    # === ENUMERATE PACKS PER TEST ===
    total_packs_count = 0
    tests_processed = []

    for test_idx, test_pair in enumerate(test_pairs):
        test_input = np.array(test_pair["input"], dtype=np.int32)
        H_in, W_in = test_input.shape

        packs_for_this_test = []
        canvases_processed = []

        # For each size law, concretize canvas
        for size_law in size_laws:
            # Concretize canvas dimensions
            if size_law.law == "constant":
                H_out = size_law.H
                W_out = size_law.W
            elif size_law.law == "linear":
                H_out = size_law.a_H * H_in + size_law.b_H
                W_out = size_law.a_W * W_in + size_law.b_W
            else:
                # Content-based laws deferred
                continue

            # Validate canvas size
            if H_out <= 0 or W_out <= 0 or H_out > 100 or W_out > 100:
                continue  # Skip invalid canvas sizes

            # Canvas ID must include law type and proof hash to avoid collisions
            # E.g., constant(9×9) vs linear(1H+0,1W+0) with H_in=9 both produce 9×9
            canvas_id = f"{size_law.law}_{H_out}x{W_out}_{size_law.proof_hash[:8]}"

            # Check if this canvas already processed for this test
            # (should not happen with proof hash in canvas_id, but keep for safety)
            if canvas_id in canvases_processed:
                continue
            canvases_processed.append(canvas_id)

            # === EXTRACT ORIGINAL SIZES (Phase 2) ===
            # Needed for faces computation: only compute faces when all trainings
            # naturally match canvas size (no embedding)
            train_output_sizes = [(Y.shape[0], Y.shape[1]) for Y in train_outputs_raw]

            # === ALIGN TRAINING OUTPUTS TO THIS CANVAS ===
            train_outputs_embedded = []
            for Y_i in train_outputs_raw:
                Y_embedded = canvas_ops.align_to_canvas(Y_i, H_out, W_out, embed_mode)
                train_outputs_embedded.append(Y_embedded)

            # === COMPUTE WO-4 PER CANVAS (Phase 3: with forward meet) ===
            wo4_result = canvas_ops.wo4_per_canvas(
                train_inputs_raw=train_inputs_raw,           # Phase 3: training inputs
                train_outputs_embedded=train_outputs_embedded,
                test_input_raw=test_input,                   # Phase 3: test input for forward meet
                H=H_out,
                W=W_out,
                embed_mode=embed_mode,
            )

            A_mask = wo4_result["A_mask"]
            bin_ids = wo4_result["bin_ids"]

            # Save WO-4 cache
            wo4_cache_dir = cache_root / "wo04"
            wo4_cache_dir.mkdir(parents=True, exist_ok=True)
            wo4_cache_path = wo4_cache_dir / f"{task_id}.{test_idx}.{canvas_id}.npz"

            np.savez_compressed(
                wo4_cache_path,
                A_mask=A_mask,
                bin_ids=bin_ids,
                **{f"meta_{k}": v for k, v in wo4_result["meta"].items() if isinstance(v, (int, float, bool))},
            )

            # === COMPUTE WO-5 PER CANVAS ===
            wo5_result = canvas_ops.wo5_per_canvas(
                train_outputs_embedded=train_outputs_embedded,
                A_mask=A_mask,
                bin_ids=bin_ids,
                H=H_out,
                W=W_out,
                train_output_sizes=train_output_sizes,  # Phase 2: pass original sizes
            )

            quotas = wo5_result["quotas"]
            faces_R = wo5_result["faces_R"]
            faces_S = wo5_result["faces_S"]
            equalizer_edges = wo5_result["equalizer_edges"]

            # Save WO-5 cache
            wo5_cache_dir = cache_root / "wo05"
            wo5_cache_dir.mkdir(parents=True, exist_ok=True)
            wo5_cache_path = wo5_cache_dir / f"{task_id}.{test_idx}.{canvas_id}.npz"

            # Serialize equalizer edges
            equalizer_edges_serialized = {}
            for (s, c), edges in equalizer_edges.items():
                key = f"eq_{s}_{c}"
                equalizer_edges_serialized[key] = edges

            save_dict = {
                "quotas": quotas,
                **equalizer_edges_serialized,
                **{f"meta_{k}": v for k, v in wo5_result["meta"].items() if isinstance(v, (int, float, bool))},
            }

            # Add faces if present
            if faces_R is not None:
                save_dict["faces_R"] = faces_R
            if faces_S is not None:
                save_dict["faces_S"] = faces_S

            np.savez_compressed(wo5_cache_path, **save_dict)

            # === ENUMERATE FACES MODES ===
            has_faces = (faces_R is not None and faces_R.sum() > 0) or \
                       (faces_S is not None and faces_S.sum() > 0)

            faces_modes = ["rows_as_supply", "cols_as_supply", "none"] if has_faces else ["none"]

            # === ENUMERATE PACKS FOR THIS CANVAS ===
            # Convert free maps to FreeMap objects
            free_maps_objects = []
            for fm_dict in free_maps_verified:
                if fm_dict["type"] == "roll":
                    free_maps_objects.append(packs_module.FreeMap(
                        type="roll",
                        dy=fm_dict.get("dy"),
                        dx=fm_dict.get("dx"),
                        verified_all_trainings=True,
                    ))
                elif fm_dict["type"] == "perm":
                    free_maps_objects.append(packs_module.FreeMap(
                        type="perm",
                        perm=list(fm_dict.get("perm", [])),
                        verified_all_trainings=True,
                    ))

            for faces_mode in faces_modes:
                # Build pack ID
                pack_id = packs_module.build_pack_id(size_law, faces_mode, free_maps_objects)
                pack_id_with_canvas = f"{pack_id}|canvas={canvas_id}"

                # Compute quick checks
                quick = packs_module.compute_quick_checks(
                    quotas=quotas,
                    A_mask=A_mask,
                    bin_ids=bin_ids,
                    faces_R=faces_R,
                    faces_S=faces_S,
                    faces_mode=faces_mode,
                    free_maps=free_maps_objects,
                    H=H_out,
                    W=W_out,
                )

                # Create pack with canvas_id
                pack = packs_module.Pack(
                    pack_id=pack_id_with_canvas,
                    size_law=size_law,
                    faces_mode=faces_mode,
                    free_maps=free_maps_objects,
                    quick=quick,
                )

                packs_for_this_test.append(pack)

        # === SAVE PACKS FOR THIS TEST ===
        wo9_cache_dir = cache_root / "wo09"
        wo9_cache_dir.mkdir(parents=True, exist_ok=True)
        wo9_cache_path = wo9_cache_dir / f"{task_id}.{test_idx}.packs.json"

        packs_dicts = [packs_module.pack_to_dict(p) for p in packs_for_this_test]

        # Add canvas_id to each pack dict
        for pack_dict, pack in zip(packs_dicts, packs_for_this_test):
            # Extract canvas_id from pack_id (format: "...|canvas=HxW")
            if "|canvas=" in pack.pack_id:
                canvas_id = pack.pack_id.split("|canvas=")[-1]
                pack_dict["canvas_id"] = canvas_id

        packs_cache = {
            "packs": packs_dicts,
            "packs_count": len(packs_for_this_test),
            "hash": packs_module.compute_packs_hash(packs_for_this_test),
            "test_idx": test_idx,
            "size_laws_complete": True,
            "canvases_processed": canvases_processed,
        }

        with open(wo9_cache_path, "w") as f:
            json.dump(packs_cache, f, indent=2, sort_keys=True)

        total_packs_count += len(packs_for_this_test)
        tests_processed.append({
            "test_idx": test_idx,
            "packs_count": len(packs_for_this_test),
            "canvases": canvases_processed,
        })

    # === BUILD RECEIPT ===
    receipt = {
        "stage": "wo09a_prime",
        "task_id": task_id,
        "tests_processed": tests_processed,
        "total_packs_count": total_packs_count,
        "total_tests": len(test_pairs),
        "size_laws_count": len(size_laws),
        "embed_mode": embed_mode,
        "note": "Per-test concrete pack builder with per-canvas WO-4/5",
    }

    # Compute receipt hash
    canonical_json = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["hash"] = hashlib.sha256(canonical_json.encode()).hexdigest()

    # === WRITE RECEIPT ===
    receipts_dir = Path("receipts")
    task_receipt_dir = receipts_dir / task_id
    task_receipt_dir.mkdir(exist_ok=True, parents=True)

    wo09a_prime_receipt_path = task_receipt_dir / "wo09a_prime.json"
    with open(wo09a_prime_receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)

    return receipt


# STAGES registry
STAGES = {}
STAGES["9a_prime"] = run_wo09a_prime
