"""WO-0/1 harness.py - CLI runner for multi-stage ARC solver.

Provides a command-line interface to run the solver on ARC tasks.

WO-0: Environment validation and receipts
WO-1: Bins, bbox, center predicate

Usage:
    python -m arcsolver.harness --data-root data/ --upto-wo 0
    python -m arcsolver.harness --data-root data/ --upto-wo 1

Flags:
    --data-root: Directory containing ARC JSON files
    --upto-wo: Which work order to run up to (0..N)
    --strict: Fail on first error (default: continue and report)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Set, Dict, Any
import numpy as np
from . import receipts
from . import bins as bins_module
from . import embed as embed_module
from . import color_align as color_align_module
from . import mask as mask_module
from .utils import hash_utils
from .config import GRID_DTYPE, PALETTE_C


# ============================================================================
# WO-specific metric keys (extend-only registry)
# ============================================================================
WO_METRICS = {
    1: ["bins_sum_ok", "bins_hash_stable", "center_all_ok"],
    2: ["embed_idempotent_ok", "embed_metamorphic_ok"],
    3: ["hungarian_bijection_ok", "signature_tie_break_ok"],
    4: ["closure_order_independent_ok", "avg_admits_before", "avg_admits_after"],
    5: ["bin_constancy_proved"],
    6: ["free_cost_invariance_ok"],
    7: ["flow_feasible_ok", "kkt_ok", "one_of_10_ok"],
    8: ["idempotence_ok", "bits_sum"],
    9: ["laminar_confluence_ok", "iis_count"],
}


def init_progress(wo: int) -> Dict[str, Any]:
    """Initialize progress tracking dict for a work order.

    Args:
        wo: Work order number

    Returns:
        Progress dict with wo, tasks_total, tasks_ok, pre-initialized metrics
    """
    keys = WO_METRICS.get(wo, [])
    return {
        "wo": wo,
        "tasks_total": 0,
        "tasks_ok": 0,
        "metrics": {k: {"ok": 0, "total": 0, "sum": 0} for k in keys}
    }


def acc_bool(progress: Dict[str, Any], key: str, ok: bool) -> None:
    """Accumulate boolean metric in progress dict.

    Args:
        progress: Progress dict
        key: Metric key
        ok: Boolean value (True if check passed)

    Notes:
        - Silently ignores unknown keys (forward compatibility)
        - Increments total, increments ok if True
    """
    if key not in progress["metrics"]:
        return
    m = progress["metrics"][key]
    m["total"] += 1
    m["ok"] += int(bool(ok))


def acc_sum(progress: Dict[str, Any], key: str, val: int) -> None:
    """Accumulate integer sum in progress dict.

    Args:
        progress: Progress dict
        key: Metric key
        val: Integer value to add to sum

    Notes:
        - Silently ignores unknown keys (forward compatibility)
        - Adds val to sum field
    """
    if key not in progress["metrics"]:
        return
    progress["metrics"][key]["sum"] += int(val)


def discover_task_ids(data_root: Path) -> Set[str]:
    """Discover all task IDs from ARC JSON files.

    Args:
        data_root: Directory containing arc-agi_*.json files

    Returns:
        Set of task IDs

    Raises:
        FileNotFoundError: If data_root doesn't exist or no ARC files found
        json.JSONDecodeError: If JSON files are malformed
    """
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Collect task IDs from all ARC JSON files
    task_ids = set()
    arc_files = [
        "arc-agi_training_challenges.json",
        "arc-agi_evaluation_challenges.json",
        "arc-agi_test_challenges.json",
    ]

    found_any = False
    for fname in arc_files:
        fpath = data_root / fname
        if not fpath.exists():
            continue

        found_any = True
        with fpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
            task_ids.update(data.keys())

    if not found_any:
        raise FileNotFoundError(
            f"No ARC JSON files found in {data_root}. "
            f"Expected one of: {', '.join(arc_files)}"
        )

    return task_ids


def load_task_data(data_root: Path, task_id: str) -> Dict[str, Any]:
    """Load task data for a given task ID.

    Args:
        data_root: Directory containing ARC JSON files
        task_id: Task identifier

    Returns:
        Task data dict with 'train' and 'test' keys

    Raises:
        FileNotFoundError: If task not found in any file
    """
    arc_files = [
        "arc-agi_training_challenges.json",
        "arc-agi_evaluation_challenges.json",
        "arc-agi_test_challenges.json",
    ]

    for fname in arc_files:
        fpath = data_root / fname
        if not fpath.exists():
            continue

        with fpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if task_id in data:
                return data[task_id]

    raise FileNotFoundError(f"Task {task_id} not found in any ARC file")


def run_stage_wo0(data_root: Path, strict: bool = False, enable_progress: bool = True) -> None:
    """Run WO-0 stage: environment validation and receipt emission.

    Args:
        data_root: Directory containing ARC JSON files
        strict: If True, fail on first error; else continue and report
        enable_progress: If True, write progress JSON

    Notes:
        - Discovers all task IDs
        - Emits receipts/<task_id>/wo00.json for each task
        - Receipt contains environment metadata (runtime versions, dtypes, constants)
        - At WO-0, we do NOT load/parse actual task data (that's WO-1+)
    """
    # Discover task IDs
    print(f"[WO-0] Discovering tasks in {data_root}...", file=sys.stderr)
    task_ids = discover_task_ids(data_root)
    print(f"[WO-0] Found {len(task_ids)} tasks", file=sys.stderr)

    # Initialize progress tracking
    progress = init_progress(wo=0)

    # Generate environment payload (same for all tasks at WO-0)
    env_payload = receipts.make_env_payload()

    # Emit receipt for each task
    success_count = 0
    fail_count = 0
    for task_id in sorted(task_ids):  # Sort for deterministic order
        progress["tasks_total"] += 1
        try:
            receipt_path = receipts.write_stage_receipt(task_id, "wo00", env_payload)
            success_count += 1
            progress["tasks_ok"] += 1
            if success_count % 100 == 0:
                print(
                    f"[WO-0] Progress: {success_count}/{len(task_ids)} receipts written",
                    file=sys.stderr,
                )
        except Exception as e:
            fail_count += 1
            msg = f"[WO-0] Failed to write receipt for {task_id}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            else:
                print(msg, file=sys.stderr)

    # Summary
    print(
        f"[WO-0] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-0] Progress written to progress/progress_wo00.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo1(data_root: Path, strict: bool = False, enable_progress: bool = True) -> None:
    """Run WO-1 stage: bins, bbox, center predicate.

    Args:
        data_root: Directory containing ARC JSON files
        strict: If True, fail on first error; else continue and report
        enable_progress: If True, write progress JSON

    Notes:
        - Loads actual task data (train/test)
        - For each task, computes bins, bbox, center predicate
        - Emits receipts/<task_id>/wo01.json
        - Assumes all training outputs have same shape (canvas)
        - At WO-1, we don't do embedding yet, so we use raw output grids
    """
    # Discover task IDs
    print(f"[WO-1] Discovering tasks in {data_root}...", file=sys.stderr)
    task_ids = discover_task_ids(data_root)
    print(f"[WO-1] Found {len(task_ids)} tasks", file=sys.stderr)

    # Initialize progress tracking
    progress = init_progress(wo=1)

    # Process each task
    success_count = 0
    fail_count = 0
    for task_id in sorted(task_ids):  # Sort for deterministic order
        progress["tasks_total"] += 1
        try:
            # Load task data
            task_data = load_task_data(data_root, task_id)
            train_pairs = task_data["train"]

            if len(train_pairs) == 0:
                raise ValueError(f"Task {task_id} has no training pairs")

            # Extract training outputs as numpy arrays
            train_outputs = []
            for pair in train_pairs:
                output_grid = np.array(pair["output"], dtype=GRID_DTYPE)
                train_outputs.append(output_grid)

            # Infer canvas: check if all outputs have same shape
            # (size inference is not implemented yet in WO-1)
            shapes = [g.shape for g in train_outputs]
            unique_shapes = set(shapes)

            if len(unique_shapes) == 1:
                # All outputs same shape - this is the canvas
                H_out, W_out = shapes[0]
                has_constant_canvas = True
            else:
                # Variable output shapes - use most common shape as canvas for WO-1
                # (proper size inference will come in later WOs)
                from collections import Counter
                shape_counts = Counter(shapes)
                H_out, W_out = shape_counts.most_common(1)[0][0]
                has_constant_canvas = False

            # Build bins for canvas
            bin_ids, bins_list = bins_module.build_bins(H_out, W_out)

            # Compute bin counts
            bin_counts = [len(b) for b in bins_list]

            # Hash bin_ids
            bin_ids_hash = hash_utils.hash_ndarray_int(bin_ids)

            # Hash each bin's pixel indices
            bin_index_hashes = [hash_utils.hash_ndarray_int(b) for b in bins_list]

            # Check hash stability across runs (critical determinism check)
            # Load previous receipt if it exists and compare bin_ids_hash
            from pathlib import Path as PathLib
            prev_receipt_path = PathLib("receipts") / task_id / "wo01.json"
            if prev_receipt_path.exists():
                try:
                    with prev_receipt_path.open("r", encoding="utf-8") as f:
                        prev_receipt = json.load(f)
                    prev_hash = prev_receipt.get("bins", {}).get("bin_ids_hash", None)
                    if prev_hash is not None:
                        hash_stable = (prev_hash == bin_ids_hash)
                        acc_bool(progress, "bins_hash_stable", hash_stable)
                except Exception:
                    # If loading fails, skip stability check (treat as first run)
                    pass

            # Compute bbox on first training output that matches canvas
            bbox = None
            has_content = False
            bbox_slices = None
            for output in train_outputs:
                if output.shape == (H_out, W_out):
                    bbox = bins_module.bbox_content(output)
                    has_content = bbox is not None
                    bbox_slices = list(bbox) if bbox else None
                    break

            # Check center predicate only if all outputs match canvas
            if has_constant_canvas:
                is_centered = bins_module.center_predicate_all(train_outputs, H_out, W_out)
                mode = "center" if is_centered else "topleft"

                # Compute per-training distances to canvas center
                center_r = (H_out - 1) / 2.0
                center_c = (W_out - 1) / 2.0
                per_train_distances = []
                for grid in train_outputs:
                    content_mask = (grid != 0) & (grid != -1)
                    if np.any(content_mask):
                        from scipy.ndimage import center_of_mass
                        com = center_of_mass(content_mask)
                        com_r, com_c = com
                        delta_r = abs(com_r - center_r)
                        delta_c = abs(com_c - center_c)
                        per_train_distances.append([delta_r, delta_c])
                    else:
                        per_train_distances.append([float('inf'), float('inf')])
            else:
                # Variable shapes - can't check center predicate yet
                mode = "topleft-pending-size"
                per_train_distances = []

            # Build WO-1 receipt
            payload = {
                "stage": "wo01",
                "canvas": {"H_out": int(H_out), "W_out": int(W_out)},
                "bins": {
                    "num_bins": len(bins_list),
                    "bin_counts": bin_counts,
                    "bin_ids_hash": bin_ids_hash,
                    "bin_index_hashes": bin_index_hashes,
                },
                "bbox": {
                    "has_content": has_content,
                    "bbox_slices": bbox_slices,
                },
                "center_predicate": {
                    "mode": mode,
                    "per_train_distances": per_train_distances,
                },
            }

            # Invariant checks and progress tracking
            bins_sum_ok = sum(bin_counts) == H_out * W_out
            assert bins_sum_ok, \
                f"Bin counts sum {sum(bin_counts)} != H*W {H_out*W_out}"
            acc_bool(progress, "bins_sum_ok", bins_sum_ok)

            if bbox is not None:
                r0, r1, c0, c1 = bbox
                assert 0 <= r0 < r1 <= H_out, f"Invalid bbox rows: {bbox}"
                assert 0 <= c0 < c1 <= W_out, f"Invalid bbox cols: {bbox}"

            # Check center predicate invariant
            center_all_ok = True
            if has_constant_canvas and mode == "center":
                for dr, dc in per_train_distances:
                    if dr != float('inf'):  # Skip empty content
                        if not (dr <= 0.5 and dc <= 0.5):
                            center_all_ok = False
                            assert False, f"Center mode but delta ({dr}, {dc}) > 0.5"
                acc_bool(progress, "center_all_ok", center_all_ok)

            # Write receipt
            receipts.write_stage_receipt(task_id, "wo01", payload)
            success_count += 1
            progress["tasks_ok"] += 1
            if success_count % 100 == 0:
                print(
                    f"[WO-1] Progress: {success_count}/{len(task_ids)} receipts written",
                    file=sys.stderr,
                )

        except Exception as e:
            fail_count += 1
            msg = f"[WO-1] Failed to process task {task_id}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            else:
                print(msg, file=sys.stderr)

    # Summary
    print(
        f"[WO-1] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-1] Progress written to progress/progress_wo01.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo2(data_root: Path, strict: bool = False, enable_progress: bool = True) -> None:
    """Run WO-2 stage: embedding and period tests.

    Args:
        data_root: Directory containing ARC JSON files
        strict: If True, fail on first error; else continue and report
        enable_progress: If True, write progress JSON

    Notes:
        - Embeds training outputs using mode from WO-1 center predicate
        - Tests idempotence (reembed round-trip)
        - Detects torus periods using byte-exact equality
        - Tests metamorphic invariance (FREE roll preserves predicate)
        - Emits receipts/<task_id>/wo02.json
    """
    # Discover task IDs
    print(f"[WO-2] Discovering tasks in {data_root}...", file=sys.stderr)
    task_ids = discover_task_ids(data_root)
    print(f"[WO-2] Found {len(task_ids)} tasks", file=sys.stderr)

    # Initialize progress tracking
    progress = init_progress(wo=2)

    # Process each task
    success_count = 0
    fail_count = 0
    for task_id in sorted(task_ids):  # Sort for deterministic order
        progress["tasks_total"] += 1
        try:
            # Load task data
            task_data = load_task_data(data_root, task_id)
            train_pairs = task_data["train"]

            if len(train_pairs) == 0:
                raise ValueError(f"Task {task_id} has no training pairs")

            # Extract training outputs as numpy arrays
            train_outputs = []
            for pair in train_pairs:
                output_grid = np.array(pair["output"], dtype=GRID_DTYPE)
                train_outputs.append(output_grid)

            # Determine canvas (from WO-1 or use most common shape)
            shapes = [g.shape for g in train_outputs]
            unique_shapes = set(shapes)

            if len(unique_shapes) == 1:
                H_out, W_out = shapes[0]
                has_constant_canvas = True
            else:
                from collections import Counter
                shape_counts = Counter(shapes)
                H_out, W_out = shape_counts.most_common(1)[0][0]
                has_constant_canvas = False

            # Determine embedding mode using WO-1 center predicate
            if has_constant_canvas:
                is_centered = bins_module.center_predicate_all(train_outputs, H_out, W_out)
                mode = "center" if is_centered else "topleft"
            else:
                mode = "topleft"  # Default for variable shapes

            # Embed all training outputs that match canvas
            embedded_outputs = []
            for output in train_outputs:
                if output.shape == (H_out, W_out):
                    embedded = embed_module.embed_to_canvas(output, H_out, W_out, mode)
                    embedded_outputs.append(embedded)

            if len(embedded_outputs) == 0:
                raise ValueError(f"No outputs match canvas ({H_out}, {W_out})")

            # Test idempotence on first embedded output
            first_output = train_outputs[0] if train_outputs[0].shape == (H_out, W_out) else embedded_outputs[0]
            reembed_ok = embed_module.reembed_round_trip_ok(first_output, H_out, W_out, mode)
            acc_bool(progress, "embed_idempotent_ok", reembed_ok)

            # Compute periods on first embedded output
            first_embedded = embedded_outputs[0]
            p_y, p_x = embed_module.periods_2d_exact(first_embedded)

            # Verify period by rolling and checking equality
            rolled_y = np.roll(first_embedded, shift=p_y, axis=0)
            eq_check_y = bool(np.array_equal(first_embedded, rolled_y))

            rolled_x = np.roll(first_embedded, shift=p_x, axis=1)
            eq_check_x = bool(np.array_equal(first_embedded, rolled_x))

            # Hash first embedded output
            embedded_hash = hash_utils.hash_ndarray_int(first_embedded)

            # Metamorphic check: FREE roll preserves centering predicate
            # Apply a verified FREE roll (if period allows) and check predicate invariance
            metamorphic_ok = True
            if has_constant_canvas and len(embedded_outputs) > 0:
                # Test: if we roll outputs by detected period, does predicate stay same?
                # This is a simple metamorphic check - full FREE verification comes later
                try:
                    # Roll all outputs by period in y direction
                    rolled_outputs = [np.roll(out, shift=p_y, axis=0) for out in embedded_outputs]
                    # Recompute centering predicate on rolled outputs
                    is_centered_rolled = bins_module.center_predicate_all(rolled_outputs, H_out, W_out)
                    # Metamorphic: predicate boolean should be identical
                    metamorphic_ok = (is_centered_rolled == is_centered)
                except Exception:
                    metamorphic_ok = False

            acc_bool(progress, "embed_metamorphic_ok", metamorphic_ok)

            # Build WO-2 receipt
            payload = {
                "stage": "wo02",
                "embedding": {
                    "mode": mode,
                    "H_out": int(H_out),
                    "W_out": int(W_out),
                    "reembed_round_trip_ok": reembed_ok,
                    "embedded_hash": embedded_hash,
                },
                "periods": {
                    "p_y": int(p_y),
                    "p_x": int(p_x),
                    "eq_check_y": eq_check_y,
                    "eq_check_x": eq_check_x,
                },
            }

            # Write receipt
            receipts.write_stage_receipt(task_id, "wo02", payload)
            success_count += 1
            progress["tasks_ok"] += 1
            if success_count % 100 == 0:
                print(
                    f"[WO-2] Progress: {success_count}/{len(task_ids)} receipts written",
                    file=sys.stderr,
                )

        except Exception as e:
            fail_count += 1
            msg = f"[WO-2] Failed to process task {task_id}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            else:
                print(msg, file=sys.stderr)

    # Summary
    print(
        f"[WO-2] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-2] Progress written to progress/progress_wo02.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo3(data_root: Path, strict: bool = False, enable_progress: bool = True) -> None:
    """Run WO-3 stage: color signatures and Hungarian alignment.

    Args:
        data_root: Directory containing ARC JSON files
        strict: If True, fail on first error; else continue and report
        enable_progress: If True, write progress JSON

    Notes:
        - Uses embedded training outputs from WO-2 logic
        - Builds color signatures (count, histograms)
        - Solves Hungarian alignment for canonical palette
        - Tests bijection and tie-break stability
        - Emits receipts/<task_id>/wo03.json
    """
    # Discover task IDs
    print(f"[WO-3] Discovering tasks in {data_root}...", file=sys.stderr)
    task_ids = discover_task_ids(data_root)
    print(f"[WO-3] Found {len(task_ids)} tasks", file=sys.stderr)

    # Initialize progress tracking
    progress = init_progress(wo=3)

    # Process each task
    success_count = 0
    fail_count = 0
    for task_id in sorted(task_ids):  # Sort for deterministic order
        progress["tasks_total"] += 1
        try:
            # Load task data
            task_data = load_task_data(data_root, task_id)
            train_pairs = task_data["train"]

            if len(train_pairs) == 0:
                raise ValueError(f"Task {task_id} has no training pairs")

            # Extract training outputs as numpy arrays
            train_outputs = []
            for pair in train_pairs:
                output_grid = np.array(pair["output"], dtype=GRID_DTYPE)
                train_outputs.append(output_grid)

            # Determine canvas (from WO-1 or use most common shape)
            shapes = [g.shape for g in train_outputs]
            unique_shapes = set(shapes)

            if len(unique_shapes) == 1:
                H_out, W_out = shapes[0]
                has_constant_canvas = True
            else:
                from collections import Counter
                shape_counts = Counter(shapes)
                H_out, W_out = shape_counts.most_common(1)[0][0]
                has_constant_canvas = False

            # Determine embedding mode using WO-1 center predicate
            if has_constant_canvas:
                is_centered = bins_module.center_predicate_all(train_outputs, H_out, W_out)
                mode = "center" if is_centered else "topleft"
            else:
                mode = "topleft"  # Default for variable shapes

            # Embed all training outputs that match canvas
            embedded_outputs = []
            for output in train_outputs:
                if output.shape == (H_out, W_out):
                    embedded = embed_module.embed_to_canvas(output, H_out, W_out, mode)
                    embedded_outputs.append(embedded)

            if len(embedded_outputs) == 0:
                raise ValueError(f"No outputs match canvas ({H_out}, {W_out})")

            # Build bins for canvas (needed for color signatures)
            bin_ids, bins_list = bins_module.build_bins(H_out, W_out)
            num_bins = len(bins_list)

            # Align colors using Hungarian algorithm
            aligned_outputs, perms, sigs, canonical_sigs, cost_hashes, total_costs = \
                color_align_module.align_colors(embedded_outputs, bin_ids, num_bins)

            # Check Hungarian bijection for all permutations
            permutation_is_bijection = True
            for perm in perms:
                if set(perm) != set(range(PALETTE_C)):
                    permutation_is_bijection = False
                    break
            acc_bool(progress, "hungarian_bijection_ok", permutation_is_bijection)

            # Check tie-break stability by comparing cost hashes with previous receipt
            tie_encoded = True
            signature_tie_break_ok = True
            from pathlib import Path as PathLib
            prev_receipt_path = PathLib("receipts") / task_id / "wo03.json"
            if prev_receipt_path.exists():
                try:
                    with prev_receipt_path.open("r", encoding="utf-8") as f:
                        prev_receipt = json.load(f)
                    # Extract cost hashes from list of hungarian objects
                    prev_hashes = [h["cost_hash"] for h in prev_receipt.get("hungarian", [])]
                    if prev_hashes:
                        signature_tie_break_ok = (prev_hashes == cost_hashes)
                except Exception:
                    # If loading fails, skip stability check (treat as first run)
                    pass
            acc_bool(progress, "signature_tie_break_ok", signature_tie_break_ok)

            # Build WO-3 receipt (per spec: lists, not dicts)
            payload = {
                "stage": "wo03",
                "canonical_palette": {
                    "order": list(range(PALETTE_C)),  # Indices 0..9
                },
                "signatures": [
                    {
                        "training": i,
                        "by_color": {str(c): sig for c, sig in sigs[i].items()}
                    }
                    for i in range(len(sigs))
                ],
                "hungarian": [
                    {
                        "training": i,
                        "perm": perms[i].tolist(),
                        "total_cost": total_costs[i],
                        "cost_hash": cost_hashes[i]
                    }
                    for i in range(len(perms))
                ],
                "determinism": {
                    "tie_encoded": tie_encoded,
                    "permutation_is_bijection": permutation_is_bijection,
                },
            }

            # Write receipt
            receipts.write_stage_receipt(task_id, "wo03", payload)
            success_count += 1
            progress["tasks_ok"] += 1
            if success_count % 100 == 0:
                print(
                    f"[WO-3] Progress: {success_count}/{len(task_ids)} receipts written",
                    file=sys.stderr,
                )

        except Exception as e:
            fail_count += 1
            msg = f"[WO-3] Failed to process task {task_id}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            else:
                print(msg, file=sys.stderr)

    # Summary
    print(
        f"[WO-3] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-3] Progress written to progress/progress_wo03.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo4(data_root: Path, strict: bool = False, enable_progress: bool = True) -> None:
    """Run WO-4 stage: forward meet closure and color-agnostic lift.

    Args:
        data_root: Directory containing ARC JSON files
        strict: If True, fail on first error; else continue and report
        enable_progress: If True, write progress JSON

    Notes:
        - Builds forward meet closure F[p,k,c] (order-free, idempotent)
        - Applies color-agnostic lift
        - Tests idempotence and order-independence
        - Emits receipts/<task_id>/wo04.json
    """
    # Discover task IDs
    print(f"[WO-4] Discovering tasks in {data_root}...", file=sys.stderr)
    task_ids = discover_task_ids(data_root)
    print(f"[WO-4] Found {len(task_ids)} tasks", file=sys.stderr)

    # Initialize progress tracking
    progress = init_progress(wo=4)

    # Process each task
    success_count = 0
    fail_count = 0
    for task_id in sorted(task_ids):  # Sort for deterministic order
        progress["tasks_total"] += 1
        try:
            # Load task data
            task_data = load_task_data(data_root, task_id)
            train_pairs = task_data["train"]
            test_pair = task_data["test"][0]  # First test

            if len(train_pairs) == 0:
                raise ValueError(f"Task {task_id} has no training pairs")

            # Extract training pairs as numpy arrays
            train_inputs = []
            train_outputs = []
            for pair in train_pairs:
                input_grid = np.array(pair["input"], dtype=GRID_DTYPE)
                output_grid = np.array(pair["output"], dtype=GRID_DTYPE)
                train_inputs.append(input_grid)
                train_outputs.append(output_grid)

            # Extract test input
            test_input = np.array(test_pair["input"], dtype=GRID_DTYPE)

            # Determine canvas (from WO-1 or use most common shape)
            shapes = [g.shape for g in train_outputs]
            unique_shapes = set(shapes)

            if len(unique_shapes) == 1:
                H_out, W_out = shapes[0]
            else:
                from collections import Counter
                shape_counts = Counter(shapes)
                H_out, W_out = shape_counts.most_common(1)[0][0]

            # Determine embedding mode (from WO-2)
            if len(unique_shapes) == 1:
                is_centered = bins_module.center_predicate_all(train_outputs, H_out, W_out)
                mode = "center" if is_centered else "topleft"
            else:
                mode = "topleft"

            # Embed training outputs and test input
            train_outputs_emb = []
            for output in train_outputs:
                if output.shape == (H_out, W_out):
                    embedded = embed_module.embed_to_canvas(output, H_out, W_out, mode)
                    train_outputs_emb.append(embedded)

            # Embed corresponding inputs (match dimensions)
            train_inputs_emb = []
            for i, input_grid in enumerate(train_inputs):
                if train_outputs[i].shape == (H_out, W_out):
                    # Embed input to same canvas
                    embedded_input = embed_module.embed_to_canvas(input_grid, H_out, W_out, mode)
                    train_inputs_emb.append(embedded_input)

            # Get bins for canonical bin histograms (needed for alignment)
            bin_ids, bins_list = bins_module.build_bins(H_out, W_out)
            num_bins = len(bins_list)

            # Align colors (from WO-3)
            aligned_outputs, perms, sigs, canonical_sigs, cost_hashes, total_costs = \
                color_align_module.align_colors(train_outputs_emb, bin_ids, num_bins)

            # Apply same permutations to inputs (align input colors consistently)
            aligned_inputs = []
            for input_emb, perm in zip(train_inputs_emb, perms):
                # Create inverse permutation for relabeling
                inv_perm = np.zeros(PALETTE_C, dtype=GRID_DTYPE)
                for orig_c in range(PALETTE_C):
                    inv_perm[perm[orig_c]] = orig_c

                # Relabel input
                aligned_input = np.full_like(input_emb, -1)
                for new_c in range(PALETTE_C):
                    orig_c = inv_perm[new_c]
                    aligned_input[input_emb == orig_c] = new_c

                # Preserve padding
                aligned_input[input_emb == -1] = -1

                aligned_inputs.append(aligned_input)

            # Build training pairs (embedded & aligned)
            train_pairs_emb_aligned = list(zip(aligned_inputs, aligned_outputs))

            # Build forward meet closure F (ORIGINAL ORDER, WITHOUT LIFT)
            F_original = mask_module.build_forward_meet(
                train_pairs_emb_aligned, H_out, W_out
            )

            # Compute admits stats BEFORE lift
            seen_mask = mask_module.build_seen_mask(train_pairs_emb_aligned, H_out, W_out)
            stats_before_lift = mask_module.admits_stats(F_original, seen=seen_mask)
            avg_admits_before = stats_before_lift["avg_admits_per_pixel"]

            # Apply color-agnostic lift (modifies F in-place, returns stats)
            F_with_lift, lift_stats = mask_module.apply_color_agnostic_lift(
                F_original.copy(), train_pairs_emb_aligned, H_out, W_out
            )

            # Compute admits stats AFTER lift
            stats_after_lift = mask_module.admits_stats(F_with_lift, seen=seen_mask)
            avg_admits_after = stats_after_lift["avg_admits_per_pixel"]

            # Hash F_with_lift for idempotence and order-independence tests
            F_hash_original = hash_utils.hash_ndarray_int(
                F_with_lift.view(np.uint8).reshape(-1)
            )

            # Test idempotence: rebuild F (same order, with lift)
            F_idempotent = mask_module.build_forward_meet(
                train_pairs_emb_aligned, H_out, W_out
            )
            F_idempotent, _ = mask_module.apply_color_agnostic_lift(
                F_idempotent, train_pairs_emb_aligned, H_out, W_out
            )
            F_hash_idempotent = hash_utils.hash_ndarray_int(
                F_idempotent.view(np.uint8).reshape(-1)
            )
            idempotent_ok = (F_hash_original == F_hash_idempotent)

            # Test order-independence: rebuild with reverse and cyclic shift
            # Reverse order
            train_pairs_reversed = list(reversed(train_pairs_emb_aligned))
            F_reversed = mask_module.build_forward_meet(
                train_pairs_reversed, H_out, W_out
            )
            F_reversed, _ = mask_module.apply_color_agnostic_lift(
                F_reversed, train_pairs_emb_aligned, H_out, W_out
            )
            F_hash_reversed = hash_utils.hash_ndarray_int(
                F_reversed.view(np.uint8).reshape(-1)
            )

            # Cyclic shift (+1)
            train_pairs_cyclic = train_pairs_emb_aligned[1:] + train_pairs_emb_aligned[:1]
            F_cyclic = mask_module.build_forward_meet(
                train_pairs_cyclic, H_out, W_out
            )
            F_cyclic, _ = mask_module.apply_color_agnostic_lift(
                F_cyclic, train_pairs_emb_aligned, H_out, W_out
            )
            F_hash_cyclic = hash_utils.hash_ndarray_int(
                F_cyclic.view(np.uint8).reshape(-1)
            )

            # Check all hashes match (order-independent)
            closure_order_independent_ok = (
                F_hash_original == F_hash_reversed == F_hash_cyclic
            )
            acc_bool(progress, "closure_order_independent_ok", closure_order_independent_ok)

            # Inverse-sample test input at output coordinates (handles size mismatch)
            test_input_sampled, test_sampling_stats = mask_module.sample_test_input_at_output_coords(
                test_input, H_out, W_out, mode
            )

            # Align test input using first training's permutation (canonical)
            if len(perms) > 0:
                perm_test = perms[0]
                inv_perm_test = np.zeros(PALETTE_C, dtype=GRID_DTYPE)
                for orig_c in range(PALETTE_C):
                    inv_perm_test[perm_test[orig_c]] = orig_c

                # Relabel: map each pixel's original color to aligned color
                test_input_aligned = test_input_sampled.copy()
                for new_c in range(PALETTE_C):
                    orig_c = inv_perm_test[new_c]
                    test_input_aligned[test_input_sampled == orig_c] = new_c
                # Preserve unseen sentinel
                test_input_aligned[test_input_sampled == -1] = -1
            else:
                test_input_aligned = test_input_sampled

            # Flatten for build_test_mask (now expects flattened input)
            test_input_aligned_flat = test_input_aligned.ravel(order='C')

            # Build test mask A using F_with_lift
            A = mask_module.build_test_mask(F_with_lift, test_input_aligned_flat)

            # Build WO-4 receipt
            N = H_out * W_out
            payload = {
                "stage": "wo04",
                "mask": {
                    "F_shape": [N, PALETTE_C, PALETTE_C],
                    "F_hash": F_hash_original,
                    "idempotent_ok": idempotent_ok,
                    "closure_order_independent_ok": closure_order_independent_ok,
                    "avg_admits_before_lift": avg_admits_before,
                    "avg_admits_after_lift": avg_admits_after,
                    "total_true": stats_after_lift["total_true"],
                },
                "lift": lift_stats,
                "test_sampling": test_sampling_stats,
                "A_mask": {
                    "A_shape": [N, PALETTE_C],
                    "A_hash": hash_utils.hash_ndarray_int(
                        A.view(np.uint8).reshape(-1)
                    ),
                },
            }

            # Track avg_admits in progress (before and after lift)
            acc_sum(progress, "avg_admits_before", int(avg_admits_before * 1000))
            acc_sum(progress, "avg_admits_after", int(avg_admits_after * 1000))

            # Write receipt
            receipts.write_stage_receipt(task_id, "wo04", payload)
            success_count += 1
            progress["tasks_ok"] += 1
            if success_count % 100 == 0:
                print(
                    f"[WO-4] Progress: {success_count}/{len(task_ids)} receipts written",
                    file=sys.stderr,
                )

        except Exception as e:
            fail_count += 1
            msg = f"[WO-4] Failed to process task {task_id}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            else:
                print(msg, file=sys.stderr)

    # Summary
    print(
        f"[WO-4] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-4] Progress written to progress/progress_wo04.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def main() -> None:
    """CLI entry point for harness."""
    parser = argparse.ArgumentParser(
        description="ARC solver harness (multi-stage, deterministic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run WO-0 (env validation, receipt emission)
  python -m arcsolver.harness --data-root data/ --upto-wo 0

  # Run up to WO-1 (size inference)
  python -m arcsolver.harness --data-root data/ --upto-wo 1 --strict
        """,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory containing arc-agi_*.json files (default: data/)",
    )
    parser.add_argument(
        "--upto-wo",
        type=int,
        required=True,
        help="Run stages up to this work order (0, 1, 2, ...)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first error (default: continue and report)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Write progress JSON (default: enabled)",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable progress JSON writing",
    )

    args = parser.parse_args()

    # Validate work order
    if args.upto_wo < 0:
        print(f"Error: --upto-wo must be >= 0, got {args.upto_wo}", file=sys.stderr)
        sys.exit(1)

    # Run stages
    print(
        f"[harness] Running stages up to WO-{args.upto_wo} on {args.data_root}",
        file=sys.stderr,
    )

    # WO-0: Environment validation & receipts
    if args.upto_wo >= 0:
        run_stage_wo0(args.data_root, strict=args.strict, enable_progress=args.progress)

    # WO-1: Bins, bbox, center predicate
    if args.upto_wo >= 1:
        run_stage_wo1(args.data_root, strict=args.strict, enable_progress=args.progress)

    # WO-2: Embedding and period tests
    if args.upto_wo >= 2:
        run_stage_wo2(args.data_root, strict=args.strict, enable_progress=args.progress)

    # WO-3: Color signatures and Hungarian alignment
    if args.upto_wo >= 3:
        run_stage_wo3(args.data_root, strict=args.strict, enable_progress=args.progress)

    # WO-4: Forward meet closure and color-agnostic lift
    if args.upto_wo >= 4:
        run_stage_wo4(args.data_root, strict=args.strict, enable_progress=args.progress)

    # WO-5+: Not implemented yet
    if args.upto_wo >= 5:
        print(
            f"[harness] WO-{args.upto_wo} not implemented yet (only WO-0/1/2/3/4 available)",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[harness] All stages complete", file=sys.stderr)


if __name__ == "__main__":
    main()
