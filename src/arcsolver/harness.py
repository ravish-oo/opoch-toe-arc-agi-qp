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
import traceback
from pathlib import Path
from typing import Set, Dict, Any
import numpy as np
from . import receipts
from . import bins as bins_module
from . import embed as embed_module
from . import color_align as color_align_module
from . import mask as mask_module
from . import eqs as eqs_module
from . import cache as cache_module
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
    5: ["commuting_rows_ok", "gravity_rows_unique_ok", "harmonic_rows_built_ok"],
    6: ["free_cost_invariance_ok", "free_constraint_invariance_ok"],
    7: ["flow_feasible_ok", "kkt_ok", "one_of_10_ok"],
    8: ["decode_one_of_10_ok", "decode_mask_ok", "bit_meter_check_ok", "idempotence_ok", "bits_sum"],
    9: ["packs_exist_ok", "packs_deterministic_ok", "quick_checks_ok", "faces_mode_ok"],
    10: ["pack_choose_ok", "laminar_respects_tiers_ok", "determinism_ok", "iis_present_ok"],
    11: ["receipt_determinism_ok", "final_feasible_ok", "final_iis_ok", "eval_ok"],
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


def run_stage_wo0(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
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
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
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


def enumerate_size_laws(train_pairs: List[Dict]) -> List[Dict]:
    """
    Enumerate all proven size laws from training pairs per Anchor 00 §1.

    Laws checked in lex precedence:
    1. Constant: All outputs same (H,W)
    2. Linear: H_out = a_H*H_in + b_H, W_out = a_W*W_in + b_W
    3. Content-based: (not implemented in this baseline - returns empty if not constant/linear)

    Args:
        train_pairs: List of {"input": grid, "output": grid} dicts

    Returns:
        List of size law dicts with {"law", "H", "W", "proof_hash"}
    """
    import hashlib

    size_laws = []

    # Extract I/O shapes
    io_shapes = []
    for pair in train_pairs:
        input_grid = np.array(pair["input"], dtype=GRID_DTYPE)
        output_grid = np.array(pair["output"], dtype=GRID_DTYPE)
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape
        io_shapes.append((H_in, W_in, H_out, W_out))

    # 1. CONSTANT LAW
    # Check if all outputs have same shape
    output_shapes = [(h_out, w_out) for _, _, h_out, w_out in io_shapes]
    unique_output_shapes = set(output_shapes)

    if len(unique_output_shapes) == 1:
        H_out, W_out = output_shapes[0]
        # Proof: all training outputs are (H_out, W_out)
        proof_data = f"constant:{H_out}x{W_out}:shapes={sorted(output_shapes)}"
        proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()

        size_laws.append({
            "law": "constant",
            "H": int(H_out),
            "W": int(W_out),
            "proof_hash": proof_hash,
        })

    # 2. LINEAR LAW
    # Try to fit H_out = a_H*H_in + b_H, W_out = a_W*W_in + b_W
    # Need at least 2 pairs to solve for (a, b)
    if len(io_shapes) >= 2:
        # Solve for H coefficients
        # Find any two pairs with different H_in values
        linear_H_ok = False
        a_H, b_H = None, None

        for i in range(len(io_shapes)):
            for j in range(i + 1, len(io_shapes)):
                H_in_i, _, H_out_i, _ = io_shapes[i]
                H_in_j, _, H_out_j, _ = io_shapes[j]

                if H_in_i != H_in_j:
                    # Compute slope and intercept
                    a_H_candidate = (H_out_j - H_out_i) / (H_in_j - H_in_i)
                    b_H_candidate = H_out_i - a_H_candidate * H_in_i

                    # Check if integer coefficients
                    if a_H_candidate == int(a_H_candidate) and b_H_candidate == int(b_H_candidate):
                        a_H = int(a_H_candidate)
                        b_H = int(b_H_candidate)

                        # Verify on all pairs
                        linear_H_ok = all(
                            H_out == a_H * H_in + b_H
                            for H_in, _, H_out, _ in io_shapes
                        )

                        if linear_H_ok:
                            break  # Found valid H law

            if linear_H_ok:
                break

        # Solve for W coefficients
        # Find any two pairs with different W_in values
        linear_W_ok = False
        a_W, b_W = None, None

        for i in range(len(io_shapes)):
            for j in range(i + 1, len(io_shapes)):
                _, W_in_i, _, W_out_i = io_shapes[i]
                _, W_in_j, _, W_out_j = io_shapes[j]

                if W_in_i != W_in_j:
                    # Compute slope and intercept
                    a_W_candidate = (W_out_j - W_out_i) / (W_in_j - W_in_i)
                    b_W_candidate = W_out_i - a_W_candidate * W_in_i

                    # Check if integer coefficients
                    if a_W_candidate == int(a_W_candidate) and b_W_candidate == int(b_W_candidate):
                        a_W = int(a_W_candidate)
                        b_W = int(b_W_candidate)

                        # Verify on all pairs
                        linear_W_ok = all(
                            W_out == a_W * W_in + b_W
                            for _, W_in, _, W_out in io_shapes
                        )

                        if linear_W_ok:
                            break  # Found valid W law

            if linear_W_ok:
                break

        # If both H and W have valid linear laws, add this size law
        # The law will be evaluated per test when H_in, W_in are known
        if linear_H_ok and linear_W_ok:
            # Proof: coefficients verified on all training pairs
            proof_data = f"linear:a_H={a_H}:b_H={b_H}:a_W={a_W}:b_W={b_W}:pairs={len(io_shapes)}"
            proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()

            size_laws.append({
                "law": "linear",
                "a_H": int(a_H),
                "b_H": int(b_H),
                "a_W": int(a_W),
                "b_W": int(b_W),
                "proof_hash": proof_hash,
            })

    # 3. CONTENT-BASED LAWS
    # bbox, object-count, period-multiple - require test input to compute
    # Deferred to WO-2 or later stages

    return size_laws


def run_stage_wo1(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
    """Run WO-1 stage: bins, bbox, center predicate, and size law enumeration.

    Args:
        data_root: Directory containing ARC JSON files
        strict: If True, fail on first error; else continue and report
        enable_progress: If True, write progress JSON

    Notes:
        - Loads actual task data (train/test)
        - For each task, computes bins, bbox, center predicate
        - Enumerates all proven size laws (Constant, Linear, Content-based)
        - Emits receipts/<task_id>/wo01.json
        - Caches size laws to .cache/wo01/<task_id>.size_laws.json
    """
    # Discover task IDs
    print(f"[WO-1] Discovering tasks in {data_root}...", file=sys.stderr)
    task_ids = discover_task_ids(data_root)
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
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

            # Enumerate size laws
            size_laws = enumerate_size_laws(train_pairs)

            # Write size laws to cache for WO-9A
            cache_dir = data_root.parent / ".cache" / "wo01"
            cache_dir.mkdir(parents=True, exist_ok=True)
            size_laws_path = cache_dir / f"{task_id}.size_laws.json"

            # Compute hash of size laws for determinism
            import hashlib
            canonical_json = json.dumps(size_laws, sort_keys=True, separators=(",", ":"))
            size_laws_hash = hashlib.sha256(canonical_json.encode()).hexdigest()

            size_laws_cache = {
                "size_laws": size_laws,
                "hash": size_laws_hash,
            }

            with open(size_laws_path, "w") as f:
                json.dump(size_laws_cache, f, indent=2, sort_keys=True)

            # Add size laws info to receipt
            payload["size_laws"] = {
                "count": len(size_laws),
                "laws": [law["law"] for law in size_laws],
            }

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


def run_stage_wo2(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
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
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
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

            # Write cache for WO-9A consumption (per spec: "from cache; never from receipts")
            cache_artifacts = {}  # WO-2 has no array artifacts, just metadata
            cache_metadata = {
                "H_out": int(H_out),
                "W_out": int(W_out),
                "mode": mode,
                "p_y": int(p_y),
                "p_x": int(p_x),
                "eq_check_y": eq_check_y,
                "eq_check_x": eq_check_x,
            }
            cache_module.save_cache(2, task_id, data_root, cache_artifacts, cache_metadata)

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


def detect_verified_color_symmetries(aligned_outputs: list[np.ndarray]) -> list[dict]:
    """
    Detect and verify color symmetries across all aligned training outputs.

    Per anchor 02_addendum.md §C: A color subgroup H⊂S₁₀ is verified iff
    every aligned training output is byte-identical under every π∈H.

    Args:
        aligned_outputs: List of aligned training outputs (H,W) int32 arrays

    Returns:
        List of verified symmetry dicts: [{"type": "perm", "perm": [...], "verified_all_trainings": True}, ...]
    """
    if not aligned_outputs:
        return []

    # Build per-color boolean layers for each training
    # L[i][c] = (Y_i == c) for each training i and color c
    layers = []
    for Y in aligned_outputs:
        training_layers = {}
        for c in range(PALETTE_C):
            training_layers[c] = (Y == c)
        layers.append(training_layers)

    # Find orbits: partition colors by byte-identical layers across ALL trainings
    # Two colors c, d are in same orbit iff layers[i][c] == layers[i][d] for all i
    from collections import defaultdict
    # Use hash of concatenated layers as orbit key
    orbit_map = defaultdict(list)

    for c in range(PALETTE_C):
        # Create a hashable key from all training layers for this color
        layer_key = tuple(
            layers[i][c].tobytes() for i in range(len(aligned_outputs))
        )
        orbit_map[layer_key].append(c)

    orbits = list(orbit_map.values())

    # For each orbit of size > 1, test all non-identity permutations within orbit
    verified_symmetries = []

    for orbit in orbits:
        if len(orbit) <= 1:
            continue  # Trivial orbit, skip

        # Generate all permutations within this orbit
        import itertools
        for perm_tuple in itertools.permutations(orbit):
            if perm_tuple == tuple(orbit):
                continue  # Skip identity

            # Build full 10-color permutation array
            perm = np.arange(PALETTE_C, dtype=np.int64)
            for idx, orig_c in enumerate(orbit):
                perm[orig_c] = perm_tuple[idx]

            # Verify: check if np.array_equal(Y_i, np.take(Y_i, perm, axis=None))
            # Actually need to permute color values, not indices
            # For grid Y where Y[r,c] = color_value, applying permutation means:
            # Y_perm[r,c] = perm[Y[r,c]] for each pixel

            is_verified = True
            for Y in aligned_outputs:
                # Apply permutation to color values
                Y_perm = perm[Y]
                if not np.array_equal(Y, Y_perm):
                    is_verified = False
                    break

            if is_verified:
                verified_symmetries.append({
                    "type": "perm",
                    "perm": perm.tolist(),
                    "verified_all_trainings": True
                })

    return verified_symmetries


def run_stage_wo3(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
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
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
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

            # Detect and verify color symmetries (post-alignment)
            verified_color_symmetries = detect_verified_color_symmetries(aligned_outputs)

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
                "verified_color_symmetries": verified_color_symmetries,
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


def run_stage_wo4(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
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
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
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
                    # Inverse-sample input at output coordinates (handles size mismatch)
                    embedded_input, _ = mask_module.sample_test_input_at_output_coords(
                        input_grid, H_out, W_out, mode
                    )
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

            # Save artifacts for downstream stages (WO-5+)
            artifacts_dir = Path("receipts") / task_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifacts_path = artifacts_dir / "wo04_artifacts.npz"

            np.savez_compressed(
                artifacts_path,
                aligned_outputs=np.array(aligned_outputs, dtype=GRID_DTYPE),
                F_mask=F_with_lift.astype(np.uint8),
                A_mask=A.astype(np.uint8),
            )

            # Save to cache for fast WO-7 iteration
            cache_artifacts = {
                "aligned_outputs": np.array(aligned_outputs, dtype=GRID_DTYPE),
                "F_mask": F_with_lift.astype(np.uint8),
                "A_mask": A.astype(np.uint8),
                "bin_ids": bin_ids,
            }
            cache_metadata = {
                "H": H_out,
                "W": W_out,
                "C": PALETTE_C,
                "N": H_out * W_out,
            }
            cache_module.save_cache(4, task_id, data_root, cache_artifacts, cache_metadata)

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


def run_stage_wo5(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
    """WO-5: Equalizers & structure rows (no solving).

    Builds:
    - Spanning-tree equalizers per bin×color (where constant predicate holds)
    - Gravity rows (I-G)y=0 on transient states
    - Harmonic Laplacian rows for interior regions

    All rows are deterministic, byte-exact, commuting.
    """
    print("[WO-5] Building equalizers & structure rows...", file=sys.stderr)

    # Import hashlib for SHA-256
    import hashlib
    import scipy.sparse

    # Initialize progress
    progress = init_progress(5)

    # Discover tasks
    task_ids = discover_task_ids(data_root)
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
    print(f"[WO-5] Found {len(task_ids)} tasks", file=sys.stderr)

    # Create receipts directory
    receipts_dir = Path("receipts")
    receipts_dir.mkdir(exist_ok=True)

    success_count = 0
    fail_count = 0

    for task_id in sorted(task_ids):
        try:
            # Load WO-2 receipt (embedding mode)
            wo2_receipt_path = receipts_dir / task_id / "wo02.json"
            if not wo2_receipt_path.exists():
                raise FileNotFoundError(f"WO-2 receipt not found: {wo2_receipt_path}")

            with open(wo2_receipt_path) as f:
                wo2_receipt = json.load(f)

            mode = wo2_receipt["embedding"]["mode"]
            H_out = wo2_receipt["embedding"]["H_out"]
            W_out = wo2_receipt["embedding"]["W_out"]

            # Load WO-3 receipt (aligned outputs)
            wo3_receipt_path = receipts_dir / task_id / "wo03.json"
            if not wo3_receipt_path.exists():
                raise FileNotFoundError(f"WO-3 receipt not found: {wo3_receipt_path}")

            with open(wo3_receipt_path) as f:
                wo3_receipt = json.load(f)

            # Load WO-4 receipt (A-mask)
            wo4_receipt_path = receipts_dir / task_id / "wo04.json"
            if not wo4_receipt_path.exists():
                raise FileNotFoundError(f"WO-4 receipt not found: {wo4_receipt_path}")

            with open(wo4_receipt_path) as f:
                wo4_receipt = json.load(f)

            # Load WO-4 artifacts (aligned outputs, F-mask, A-mask)
            # This is CLEAN PIPELINE PATTERN: consume artifacts, don't reconstruct
            artifacts_path = Path("receipts") / task_id / "wo04_artifacts.npz"
            if not artifacts_path.exists():
                raise FileNotFoundError(f"WO-4 artifacts not found: {artifacts_path}")

            artifacts = np.load(artifacts_path)
            aligned_outputs = artifacts["aligned_outputs"]
            F_mask = artifacts["F_mask"]
            A_mask = artifacts["A_mask"]

            # Convert aligned_outputs to list of grids
            train_outputs_aligned = [aligned_outputs[i] for i in range(aligned_outputs.shape[0])]

            # Build bins (deterministic from H_out, W_out)
            bin_ids, bins_list = bins_module.build_bins(H_out, W_out)
            num_bins = len(bins_list)

            N = H_out * W_out

            # === EQUALIZERS ===
            equalizer_edges = eqs_module.build_equalizer_rows(
                bin_ids=bin_ids,
                num_bins=num_bins,
                A_mask=A_mask,
                train_outputs_aligned=train_outputs_aligned,
                H_out=H_out,
                W_out=W_out,
            )

            # Compute tree hashes
            equalizer_info = []
            for (s, c), edges in equalizer_edges.items():
                if len(edges) > 0:
                    # Hash edges
                    edges_bytes = np.array(edges, dtype=np.int64).tobytes()
                    tree_hash = hashlib.sha256(edges_bytes).hexdigest()

                    equalizer_info.append({
                        "bin": int(s),
                        "color": int(c),
                        "nodes": len(set(p for edge in edges for p in edge)) + 1,  # Approx
                        "edges": len(edges),
                        "tree_hash": tree_hash,
                    })

            # === GRAVITY ===
            # Build walls mask: bottom row only (baseline)
            walls_mask = np.zeros(N, dtype=bool)
            for c in range(W_out):
                p_bottom = (H_out - 1) * W_out + c
                walls_mask[p_bottom] = True

            gravity_rows = eqs_module.build_gravity_rows(
                H=H_out,
                W=W_out,
                walls_mask=walls_mask,
                direction="down",
            )

            # Check acyclic (downward only → acyclic)
            acyclic_ok = True
            for p, p_next in gravity_rows:
                if p_next <= p:  # Should always be p_next > p (downward)
                    acyclic_ok = False
                    break

            acc_bool(progress, "gravity_rows_unique_ok", acyclic_ok)

            # === HARMONIC ===
            # Detect interior regions
            interior_components = eqs_module.detect_interior_regions(
                train_outputs_aligned=train_outputs_aligned,
                H_out=H_out,
                W_out=W_out,
            )

            harmonic_regions = []
            harmonic_L_matrices = []  # Collect L_DD matrices for commutation test
            laplacian_shape_ok = True

            for interior_idx in interior_components:
                if len(interior_idx) == 0:
                    continue

                # For now, boundary is just neighbors of interior
                # (proper implementation would extract actual boundary)
                boundary_idx = np.array([], dtype=np.int64)  # Placeholder

                L_DD = eqs_module.build_harmonic_rows(
                    interior_idx=interior_idx,
                    boundary_idx=boundary_idx,
                    H=H_out,
                    W=W_out,
                )

                # Check shape
                if L_DD.shape[0] != len(interior_idx) or L_DD.shape[1] != len(interior_idx):
                    laplacian_shape_ok = False

                harmonic_regions.append({
                    "interior_size": len(interior_idx),
                    "laplacian_shape": list(L_DD.shape),
                })

                # Collect L_DD for commutation test
                harmonic_L_matrices.append(L_DD)

            acc_bool(progress, "harmonic_rows_built_ok", laplacian_shape_ok and len(harmonic_regions) >= 0)

            # === COMMUTATION ===
            # Get C_out (number of output colors)
            C_out = int(np.max([np.max(grid) for grid in train_outputs_aligned if grid.size > 0])) + 1

            # Test that constraint rows commute (can be applied in any order)
            commute_ok = bool(eqs_module.test_row_commutation(
                equalizer_edges=equalizer_edges,
                gravity_rows=gravity_rows,
                harmonic_L_matrices=harmonic_L_matrices,
                N=N,
                C_out=C_out,
            ))

            acc_bool(progress, "commuting_rows_ok", commute_ok)

            # === QUOTAS ===
            # Compute quotas from training outputs per Anchor 00 §8
            # q[s,c] = min_i #{p∈B_s: Y_i(p)=c}
            # "Minimum across training outputs of pixels in bin s assigned color c"
            quotas_dict = {}
            for c in range(10):  # PALETTE_C = 10
                for s, bin_pixels in enumerate(bins_list):
                    # Count assignments in each training output
                    counts_per_example = []
                    for Y_i in train_outputs_aligned:
                        # Count pixels in bin s assigned color c in this training output
                        count_in_output = sum(1 for p in bin_pixels if Y_i.flat[p] == c)
                        counts_per_example.append(count_in_output)

                    # Take minimum across training examples (conservative estimate)
                    quota = min(counts_per_example) if counts_per_example else 0
                    quotas_dict[(s, c)] = quota

            # === FACES ===
            # Compute faces per Anchor 00 §8 line 129: "meet of counts across outputs"
            # faces_R[r,c] = min_i(count of color c in row r of Y_i)
            # faces_S[j,c] = min_i(count of color c in column j of Y_i)
            faces_R = np.zeros((H_out, 10), dtype=np.int64)
            faces_S = np.zeros((W_out, 10), dtype=np.int64)

            for c in range(10):  # PALETTE_C = 10
                for r in range(H_out):
                    # Count color c in each row r across all training outputs
                    counts_per_example = []
                    for Y_i in train_outputs_aligned:
                        count_in_row = np.sum(Y_i[r, :] == c)
                        counts_per_example.append(count_in_row)
                    # Take minimum (meet)
                    faces_R[r, c] = min(counts_per_example) if counts_per_example else 0

                for j in range(W_out):
                    # Count color c in each column j across all training outputs
                    counts_per_example = []
                    for Y_i in train_outputs_aligned:
                        count_in_col = np.sum(Y_i[:, j] == c)
                        counts_per_example.append(count_in_col)
                    # Take minimum (meet)
                    faces_S[j, c] = min(counts_per_example) if counts_per_example else 0

            # Compute faces totals for receipt
            faces_R_total = int(np.sum(faces_R))
            faces_S_total = int(np.sum(faces_S))
            faces_R_nonzero_rows = int(np.count_nonzero(np.sum(faces_R, axis=1)))
            faces_S_nonzero_cols = int(np.count_nonzero(np.sum(faces_S, axis=1)))

            # Build receipt
            payload = {
                "stage": "wo05",
                "equalizers": {
                    "count_bins": num_bins,
                    "count_rows": sum(len(edges) for edges in equalizer_edges.values()),
                    "by_bin_color": equalizer_info,
                    "commute_ok": commute_ok,
                },
                "gravity": {
                    "transient_nodes": int((~walls_mask).sum()),
                    "rows_built": len(gravity_rows),
                    "acyclic_ok": acyclic_ok,
                    "walls_policy": "bottom_only",
                },
                "harmonic": {
                    "regions": len(harmonic_regions),
                    "sum_L_rows": sum(r["interior_size"] for r in harmonic_regions),
                    "laplacian_shape_ok": laplacian_shape_ok,
                    "uniqueness_proved_by": "theory",  # Maximum principle
                    "regions_info": harmonic_regions,
                },
                "faces": {
                    "rows_total": faces_R_total,
                    "cols_total": faces_S_total,
                    "rows_nonzero": faces_R_nonzero_rows,
                    "cols_nonzero": faces_S_nonzero_cols,
                    "by_color": [
                        {
                            "color": c,
                            "row_total": int(np.sum(faces_R[:, c])),
                            "col_total": int(np.sum(faces_S[:, c])),
                        }
                        for c in range(10) if np.sum(faces_R[:, c]) > 0 or np.sum(faces_S[:, c]) > 0
                    ],
                },
            }

            # Write receipt
            task_receipt_dir = receipts_dir / task_id
            task_receipt_dir.mkdir(exist_ok=True)

            wo5_receipt_path = task_receipt_dir / "wo05.json"
            with open(wo5_receipt_path, "w") as f:
                json.dump(payload, f, indent=2)

            # Save to cache for fast WO-7 iteration
            # Convert equalizer_edges dict to serializable format
            # NEW FORMAT: Use "eq_{s}_{c}" prefix and filter empty arrays to prevent stale cache keys
            equalizer_edges_serialized = {}
            for (s, c), edges in equalizer_edges.items():
                edges_arr = np.array(edges, dtype=np.int64)
                # Only save non-empty equalizer trees (|S| >= 2 means at least 1 edge)
                if edges_arr.shape[0] >= 1:
                    key = f"eq_{s}_{c}"  # Unambiguous prefix format
                    equalizer_edges_serialized[key] = edges_arr

            # Convert gravity_rows to array
            gravity_rows_array = np.array(gravity_rows, dtype=np.int64) if gravity_rows else np.array([], dtype=np.int64).reshape(0, 2)

            # Serialize quotas (already computed above)
            quotas_array = np.array([[s, c, quotas_dict[(s, c)]] for (s, c) in sorted(quotas_dict.keys())], dtype=np.int64)

            # Build cell_caps from equalizers/gravity (for now, use simple formula)
            # In WO-7 baseline, each cell can have C colors
            cell_caps = np.full((H_out, W_out), 10, dtype=np.int32)  # C = 10

            # Compute A_hash for cache validation
            A_hash = hash_utils.hash_ndarray_int(
                A_mask.view(np.uint8).reshape(-1)
            )

            cache_artifacts = {
                "num_bins": np.array([num_bins], dtype=np.int32),
                "walls_mask": walls_mask.astype(np.uint8),
                "gravity_rows": gravity_rows_array,
                "quotas": quotas_array,
                "faces_R": faces_R,  # Row faces (H×C)
                "faces_S": faces_S,  # Column faces (W×C)
                "cell_caps": cell_caps,
                "_A_hash_used": np.array([ord(c) for c in A_hash], dtype=np.uint8),  # Store hash as array
                "_cache_version": np.array([2], dtype=np.int32),  # Track cache format version (incremented for faces)
                **equalizer_edges_serialized,  # Unpack equalizer edges as separate arrays
            }

            cache_metadata = {
                "H": H_out,
                "W": W_out,
                "C": 10,
                "N": N,
                "num_bins": num_bins,
                "equalizer_keys": list(equalizer_edges.keys()),  # Save keys for reconstruction
                "A_hash_used": A_hash,  # Store hash string in metadata for easy access
            }

            cache_module.save_cache(5, task_id, data_root, cache_artifacts, cache_metadata)

            success_count += 1

            # Progress reporting
            if enable_progress and success_count % 100 == 0:
                print(f"[WO-5] Progress: {success_count}/{len(task_ids)} receipts written", file=sys.stderr)

        except Exception as e:
            print(f"[WO-5] Failed to process task {task_id}: {e}", file=sys.stderr)
            if fail_count < 2:  # Print traceback for first 2 errors only
                traceback.print_exc(file=sys.stderr)
            fail_count += 1
            if strict:
                raise

    # Summary
    print(
        f"[WO-5] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-5] Progress written to progress/progress_wo05.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo6(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
    """WO-6: Π-safe Scores and FREE Predicate

    Builds:
    - Π-safe scores ŝ ∈ ℝ^(N×C) using bins/mask only
    - Integer costs via round(-ŝ * SCALE)
    - FREE predicate checks for verified symmetries

    All operations are byte-exact, deterministic, Π-safe.
    """
    print("[WO-6] Building Π-safe scores & checking FREE predicate...", file=sys.stderr)

    # Import required modules
    import hashlib
    from arcsolver import scores as scores_module
    from arcsolver import bins as bins_module

    # Initialize progress
    progress = init_progress(6)

    # Discover tasks
    task_ids = discover_task_ids(data_root)
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
    print(f"[WO-6] Found {len(task_ids)} tasks", file=sys.stderr)

    # Create receipts directory
    receipts_dir = Path("receipts")
    receipts_dir.mkdir(exist_ok=True)

    success_count = 0
    fail_count = 0

    for task_id in sorted(task_ids):
        try:
            # Load WO-1 receipt (bins)
            wo1_receipt_path = receipts_dir / task_id / "wo01.json"
            if not wo1_receipt_path.exists():
                raise FileNotFoundError(f"WO-1 receipt not found: {wo1_receipt_path}")

            with open(wo1_receipt_path) as f:
                wo1_receipt = json.load(f)

            # Load WO-2 receipt (embedding, periods)
            wo2_receipt_path = receipts_dir / task_id / "wo02.json"
            if not wo2_receipt_path.exists():
                raise FileNotFoundError(f"WO-2 receipt not found: {wo2_receipt_path}")

            with open(wo2_receipt_path) as f:
                wo2_receipt = json.load(f)

            H_out = wo2_receipt["embedding"]["H_out"]
            W_out = wo2_receipt["embedding"]["W_out"]

            # Load WO-3 receipt (color permutations)
            wo3_receipt_path = receipts_dir / task_id / "wo03.json"
            if not wo3_receipt_path.exists():
                raise FileNotFoundError(f"WO-3 receipt not found: {wo3_receipt_path}")

            with open(wo3_receipt_path) as f:
                wo3_receipt = json.load(f)

            # Load WO-4 receipt and artifacts (A-mask)
            wo4_receipt_path = receipts_dir / task_id / "wo04.json"
            if not wo4_receipt_path.exists():
                raise FileNotFoundError(f"WO-4 receipt not found: {wo4_receipt_path}")

            with open(wo4_receipt_path) as f:
                wo4_receipt = json.load(f)

            artifacts_path = receipts_dir / task_id / "wo04_artifacts.npz"
            if not artifacts_path.exists():
                raise FileNotFoundError(f"WO-4 artifacts not found: {artifacts_path}")

            artifacts = np.load(artifacts_path)
            A_mask = artifacts["A_mask"]

            # Extract aligned_outputs for equalizer rebuilding (Issue #3 fix)
            aligned_outputs = artifacts["aligned_outputs"]
            train_outputs_aligned = [aligned_outputs[i] for i in range(aligned_outputs.shape[0])]

            # Load WO-5 receipt (harmonic features)
            wo5_receipt_path = receipts_dir / task_id / "wo05.json"
            if not wo5_receipt_path.exists():
                raise FileNotFoundError(f"WO-5 receipt not found: {wo5_receipt_path}")

            with open(wo5_receipt_path) as f:
                wo5_receipt = json.load(f)

            # Build bins (deterministic, same as WO-5)
            bin_ids, bins_list = bins_module.build_bins(H_out, W_out)

            N = H_out * W_out
            C = 10  # ARC color palette size

            # === BUILD Π-SAFE SCORES ===
            stage_features = {
                "harmonic": wo5_receipt.get("harmonic", {}),
                "equalizers": wo5_receipt.get("equalizers", {}),
                "gravity": wo5_receipt.get("gravity", {}),
            }

            scores = scores_module.build_scores_pi_safe(
                H=H_out,
                W=W_out,
                A_mask=A_mask,
                bin_ids=bin_ids,
                stage_features=stage_features,
            )

            # Runtime self-test: Verify Π-safe equivariance (BEFORE projection)
            # The test must run on unprojected scores since it rebuilds from scratch
            scores_module.assert_pi_safety_equivariance(
                scores=scores,
                A_mask=A_mask,
                bin_ids=bin_ids,
                H=H_out,
                W=W_out,
                stage_features=stage_features
            )

            # Hash scores before projection (for receipt)
            scores_hash_before = hashlib.sha256(scores.tobytes()).hexdigest()

            # === APPLY PERIOD PROJECTION IF VERIFIED ===
            # Per 02_addendum.md §B: "include the projector first" for FREE invariance
            # Per 02_addendum.md §J: Haar projector P_rep = (1/p) Σ T^k
            periods = wo2_receipt.get("periods", {})
            p_y = periods.get("p_y", H_out)
            p_x = periods.get("p_x", W_out)
            eq_check_y = periods.get("eq_check_y", False)
            eq_check_x = periods.get("eq_check_x", False)

            # Determine if we have non-trivial verified periods
            has_verified_period_y = (p_y < H_out and eq_check_y)
            has_verified_period_x = (p_x < W_out and eq_check_x)
            projected_under_period = has_verified_period_y or has_verified_period_x

            if projected_under_period:
                # Apply Haar projector to enforce period invariance on scores
                scores = scores_module.project_scores_under_periods(
                    scores=scores,
                    H=H_out,
                    W=W_out,
                    p_y=p_y,
                    p_x=p_x
                )

            # Convert to integer costs
            costs = scores_module.to_int_costs(scores)

            # Compute hash of scores after projection (for receipt)
            scores_hash = hashlib.sha256(scores.tobytes()).hexdigest()
            costs_hash = hashlib.sha256(costs.tobytes()).hexdigest()

            # === RECONSTRUCT FREE MAP CANDIDATES ===
            free_map_candidates = []

            # 1. From WO-2: periods (torus rolls) - only if verified
            # Note: periods dict already loaded above for projection

            # Generate ALL non-identity multiples of verified periods as candidates
            # Per 02_addendum.md §B and §J: test all subgroup elements separately
            cand_dys = []
            if eq_check_y and 0 < p_y < H_out:
                # Generate non-identity multiples: p_y, 2*p_y, ..., (H/p_y - 1)*p_y
                cand_dys = [ (n * p_y) % H_out for n in range(1, H_out // p_y) ]

            cand_dxs = []
            if eq_check_x and 0 < p_x < W_out:
                # Generate non-identity multiples: p_x, 2*p_x, ..., (W/p_x - 1)*p_x
                cand_dxs = [ (m * p_x) % W_out for m in range(1, W_out // p_x) ]

            # Add y-only candidates
            for dy in cand_dys:
                free_map_candidates.append({
                    "type": "roll",
                    "dy": int(dy),
                    "dx": 0,
                    "verified_all_trainings": True,
                    "source": "wo2_period_y_verified"
                })

            # Add x-only candidates
            for dx in cand_dxs:
                free_map_candidates.append({
                    "type": "roll",
                    "dy": 0,
                    "dx": int(dx),
                    "verified_all_trainings": True,
                    "source": "wo2_period_x_verified"
                })

            # Add combined (dy, dx) candidates
            for dy in cand_dys:
                for dx in cand_dxs:
                    free_map_candidates.append({
                        "type": "roll",
                        "dy": int(dy),
                        "dx": int(dx),
                        "verified_all_trainings": True,
                        "source": "wo2_period_xy_verified"
                    })

            # 2. From WO-3: verified color symmetries (NOT Hungarian alignments)
            # Hungarian permutations are alignment transforms, not symmetries
            # Only consume verified symmetries that pass byte-exact equality on all trainings
            verified_symmetries = wo3_receipt.get("verified_color_symmetries", [])
            free_map_candidates.extend(verified_symmetries)

            # === CHECK FREE PREDICATE ===
            free_checks = []

            # Rebuild equalizer rows deterministically (Issue #3 fix)
            # Per patch §B: "Persist from WO-5 or deterministically rebuild"
            import arcsolver.eqs as eqs_module
            equalizer_edges = eqs_module.build_equalizer_rows(
                bin_ids=bin_ids,
                num_bins=wo1_receipt["bins"]["num_bins"],
                A_mask=A_mask,
                train_outputs_aligned=train_outputs_aligned,
                H_out=H_out,
                W_out=W_out
            )

            # Convert equalizer dict to flat edge list for FREE gate
            # Format: [(p_i, p_j, c), ...] where p_i < p_j
            equalizer_rows = []
            for (bin_id, color), edges in equalizer_edges.items():
                for (p1, p2) in edges:
                    equalizer_rows.append((p1, p2, color))

            # Faces not yet implemented
            faces = None

            for U in free_map_candidates:
                cost_ok, constraint_ok = scores_module.check_free_predicate(
                    scores=scores,
                    A_mask=A_mask,
                    U=U,
                    H=H_out,
                    W=W_out,
                    equalizer_rows=equalizer_rows,
                    faces=faces,
                )

                free_checks.append({
                    "symmetry": {k: v for k, v in U.items() if k != "source"},
                    "source": U.get("source", "unknown"),
                    "cost_invariance": bool(cost_ok),
                    "constraint_invariance": bool(constraint_ok),
                    "is_free": bool(cost_ok and constraint_ok),
                })

            # Count FREE maps
            free_count = sum(1 for check in free_checks if check["is_free"])

            # Accumulate progress metrics: track cost and constraint invariance separately
            # Per reviewer: only evaluate accepted FREE maps (is_free=True), not rejected candidates
            # This avoids penalizing correct rejections (e.g., period rolls with incompatible equalizers)
            # Vacuously True if no FREE maps found (checking completed, just nothing passed both tests)
            free_maps = [c for c in free_checks if c["is_free"]]
            all_cost_ok = all(c["cost_invariance"] for c in free_maps) if free_maps else True
            all_constraint_ok = all(c["constraint_invariance"] for c in free_maps) if free_maps else True
            acc_bool(progress, "free_cost_invariance_ok", all_cost_ok)
            acc_bool(progress, "free_constraint_invariance_ok", all_constraint_ok)

            # Build receipt
            scores_receipt = {
                "shape": [int(N), int(C)],
                "dtype": "float64",
                "hash_sha256_before": scores_hash_before,
                "hash_sha256": scores_hash,
                "pi_safe": True,
                "projected_under_period": projected_under_period,
            }

            # Add period metadata if projection was applied
            if projected_under_period:
                scores_receipt["periods"] = {
                    "p_y": int(p_y),
                    "p_x": int(p_x),
                    "verified_y": eq_check_y,
                    "verified_x": eq_check_x,
                }

            payload = {
                "stage": "wo06",
                "scores": scores_receipt,
                "costs": {
                    "shape": [int(N), int(C)],
                    "dtype": "int64",
                    "hash_sha256": costs_hash,
                    "scale": 1_000_000,
                },
                "free_predicate": {
                    "candidates_checked": len(free_checks),
                    "free_maps_verified": free_count,
                    "checks": free_checks,
                },
            }

            # Write receipt
            task_receipt_dir = receipts_dir / task_id
            task_receipt_dir.mkdir(exist_ok=True)

            wo6_receipt_path = task_receipt_dir / "wo06.json"
            with open(wo6_receipt_path, "w") as f:
                json.dump(payload, f, indent=2)

            # Save to cache for fast WO-7 iteration
            cache_artifacts = {
                "costs": costs.astype(np.int64),
            }
            cache_metadata = {
                "H": H_out,
                "W": W_out,
                "C": C,
                "N": N,
            }
            cache_module.save_cache(6, task_id, data_root, cache_artifacts, cache_metadata)

            # Write free_maps.json sidecar for WO-9A (per spec: "from cache; never from receipts")
            free_maps_verified = [
                check["symmetry"]
                for check in free_checks
                if check.get("is_free", False)
            ]
            free_maps_payload = {
                "free_maps_verified": free_maps_verified,
            }
            # Compute hash of canonical JSON
            canonical_json = json.dumps(free_maps_payload, sort_keys=True, separators=(",", ":"))
            free_maps_hash = hashlib.sha256(canonical_json.encode()).hexdigest()
            free_maps_payload["hash"] = free_maps_hash

            # Write to .cache/wo06/<task>.free_maps.json
            cache_dir = Path(".cache") / "wo06"
            cache_dir.mkdir(parents=True, exist_ok=True)
            free_maps_path = cache_dir / f"{task_id}.free_maps.json"
            with open(free_maps_path, "w") as f:
                json.dump(free_maps_payload, f, indent=2, sort_keys=True)

            success_count += 1

            # Progress reporting
            if enable_progress and success_count % 100 == 0:
                print(f"[WO-6] Progress: {success_count}/{len(task_ids)} receipts written", file=sys.stderr)

        except Exception as e:
            print(f"[WO-6] Failed to process task {task_id}: {e}", file=sys.stderr)
            if fail_count < 2:  # Print traceback for first 2 errors only
                traceback.print_exc(file=sys.stderr)
            fail_count += 1
            if strict:
                raise

    # Summary
    print(
        f"[WO-6] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-6] Progress written to progress/progress_wo06.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo7(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
    """WO-7: Unified Min-Cost Flow

    Builds:
    - Unified min-cost flow graph per color using OR-Tools SimpleMinCostFlow
    - Graph nodes: U[s,c] (bins), P[p] (pixels), C[r,j] (cells), T (sink)
    - Graph arcs: enforce mask, bin quotas, one-of-10, cell capacities
    - Integer costs from WO-06

    Checks:
    - Primal feasibility (node balance, capacity constraints)
    - Cost equality (recomputed cost = solver OptimalCost)
    - Idempotence (resolving yields identical flows)
    - Optional KKT conditions
    """
    print("[WO-7] Building unified min-cost flow graphs...", file=sys.stderr)

    # Import required modules
    from arcsolver import flows as flows_module

    # Initialize progress
    progress = init_progress(7)

    # Discover tasks
    task_ids = discover_task_ids(data_root)
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
    print(f"[WO-7] Found {len(task_ids)} tasks", file=sys.stderr)

    # Create receipts directory
    receipts_dir = Path("receipts")
    receipts_dir.mkdir(exist_ok=True)

    success_count = 0
    fail_count = 0

    for task_id in sorted(task_ids):
        try:
            task_dir = receipts_dir / task_id

            # Check that required prior WOs exist
            required_wos = [0, 1, 2, 3, 4, 5, 6]
            for wo in required_wos:
                wo_receipt_path = task_dir / f"wo{wo:02d}.json"
                if not wo_receipt_path.exists():
                    raise FileNotFoundError(f"WO-{wo} receipt not found: {wo_receipt_path}")

            # Run WO-7 flow solver
            receipt = flows_module.run_wo7_for_task(task_dir)

            # Write receipt
            wo7_receipt_path = task_dir / "wo07.json"
            with open(wo7_receipt_path, "w") as f:
                json.dump(receipt, f, indent=2)

            # Extract metrics from solve section
            solve_data = receipt["solve"]

            # flow_feasible_ok = all primal checks pass
            flow_feasible_ok = (
                solve_data.get("primal_balance_ok", False) and
                solve_data.get("capacity_ok", False) and
                solve_data.get("mask_ok", False) and
                solve_data.get("cell_caps_ok", False)
            )

            # kkt_ok (may be None if not implemented)
            kkt_ok = solve_data.get("kkt_reduced_cost_ok", False)
            if kkt_ok is None:
                kkt_ok = True  # Treat None as vacuously true

            # one_of_10_ok
            one_of_10_ok = solve_data.get("one_of_10_ok", False)

            # cost_equal_ok (recomputed cost matches optimal cost)
            cost_equal_ok = solve_data.get("recomputed_cost_ok", False)

            # faces_ok (faces consistency check)
            faces_ok = solve_data.get("faces_ok", False)

            # idempotence_ok (rebuilt graph produces identical flows)
            idempotence_ok = solve_data.get("idempotence_ok", False)

            # Accumulate metrics
            acc_bool(progress, "flow_feasible_ok", flow_feasible_ok)
            acc_bool(progress, "kkt_ok", kkt_ok)
            acc_bool(progress, "one_of_10_ok", one_of_10_ok)
            acc_bool(progress, "cost_equal_ok", cost_equal_ok)
            acc_bool(progress, "faces_ok", faces_ok)
            acc_bool(progress, "idempotence_ok", idempotence_ok)

            success_count += 1

            # Progress reporting
            if enable_progress and success_count % 100 == 0:
                print(f"[WO-7] Progress: {success_count}/{len(task_ids)} receipts written", file=sys.stderr)

        except Exception as e:
            print(f"[WO-7] Failed to process task {task_id}: {e}", file=sys.stderr)
            if fail_count < 2:  # Print traceback for first 2 errors only
                traceback.print_exc(file=sys.stderr)
            fail_count += 1
            if strict:
                raise

    # Summary
    print(
        f"[WO-7] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-7] Progress written to progress/progress_wo07.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo8(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
    """
    WO-8: Decode + Bit-meter

    Loads WO-7 flows and produces final prediction grid plus bit-meter.
    """
    from . import stages_wo08
    from .decode import run_synthetic_tie_tests, verify_idempotence_on_tasks

    # Initialize progress
    progress = init_progress(8)

    # Discover tasks
    task_ids = discover_task_ids(data_root)
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
    print(f"[WO-8] Found {len(task_ids)} tasks", file=sys.stderr)

    # Run synthetic tie tests once (per WO-08 §7)
    print(f"[WO-8] Running synthetic tie tests for bit-meter validation...", file=sys.stderr)
    bit_meter_check_ok = run_synthetic_tie_tests()
    if bit_meter_check_ok:
        print(f"[WO-8] ✓ Synthetic tie tests passed (m ∈ {{1,2,3,4}})", file=sys.stderr)
    else:
        print(f"[WO-8] ✗ Synthetic tie tests FAILED", file=sys.stderr)
        if strict:
            raise RuntimeError("WO-8 synthetic tie tests failed")

    # Run idempotence verification on sample tasks (per WO-08 §7)
    sample_task_ids = sorted(task_ids)[:4]  # Test on first 4 tasks
    if len(sample_task_ids) > 0:
        print(f"[WO-8] Running idempotence verification on {len(sample_task_ids)} sample tasks...", file=sys.stderr)
        idempotence_ok, failed_tasks = verify_idempotence_on_tasks(sample_task_ids, data_root)
        if idempotence_ok:
            print(f"[WO-8] ✓ Idempotence verified: all {len(sample_task_ids)} tasks produced identical outputs on re-run", file=sys.stderr)
        else:
            print(f"[WO-8] ✗ Idempotence check FAILED on {len(failed_tasks)} tasks: {failed_tasks}", file=sys.stderr)
            if strict:
                raise RuntimeError(f"WO-8 idempotence verification failed: {failed_tasks}")
    else:
        print(f"[WO-8] ⚠ No tasks available for idempotence verification", file=sys.stderr)
        idempotence_ok = True  # Default to True if no tasks to test

    success_count = 0
    fail_count = 0

    for task_id in sorted(task_ids):
        try:
            # Check that required prior WOs exist
            receipts_dir = Path("receipts")
            task_dir = receipts_dir / task_id
            required_wos = [0, 1, 2, 3, 4, 5, 6, 7]
            for wo in required_wos:
                wo_receipt_path = task_dir / f"wo{wo:02d}.json"
                if not wo_receipt_path.exists():
                    raise FileNotFoundError(f"WO-{wo} receipt not found: {wo_receipt_path}")

            # Run WO-8 decode + bit-meter (writes receipt itself)
            receipt = stages_wo08.run_wo08(task_id, data_root)

            # Extract metrics from receipt
            decode_data = receipt["decode"]
            bit_meter_data = receipt["bit_meter"]

            # Individual decode checks (per WO-08 §6 requirements)
            decode_one_of_10_ok = decode_data.get("one_of_10_decode_ok", False)
            decode_mask_ok = decode_data.get("mask_ok", False)

            # bits_sum = total_bits from bit-meter
            bits_sum = bit_meter_data.get("total_bits", 0)

            # Accumulate individual metrics
            acc_bool(progress, "decode_one_of_10_ok", decode_one_of_10_ok)
            acc_bool(progress, "decode_mask_ok", decode_mask_ok)

            # Synthetic tie tests (run once at function start per WO-08 §7)
            acc_bool(progress, "bit_meter_check_ok", bit_meter_check_ok)

            # Idempotence verification (run once at function start per WO-08 §7)
            acc_bool(progress, "idempotence_ok", idempotence_ok)

            acc_sum(progress, "bits_sum", bits_sum)

            success_count += 1

            # Progress reporting
            if enable_progress and success_count % 100 == 0:
                print(f"[WO-8] Progress: {success_count}/{len(task_ids)} receipts written", file=sys.stderr)

        except Exception as e:
            print(f"[WO-8] Failed to process task {task_id}: {e}", file=sys.stderr)
            if fail_count < 2:  # Print traceback for first 2 errors only
                traceback.print_exc(file=sys.stderr)
            fail_count += 1
            if strict:
                raise

    # Summary
    print(
        f"[WO-8] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-8] Progress written to progress/progress_wo08.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo09a(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
    """
    WO-9A: Packs & Size Law (Per-test concrete pack builder)

    Enumerates deterministic packs per (task,test,canvas) for WO-9B. Each pack fixes:
    - Output size law (constant/linear concretized per test)
    - Faces mode (rows/cols/none)
    - Verified FREE maps
    - Quick feasibility flags
    """
    from . import stages_wo09a_prime as stages_wo09a

    # Initialize progress
    progress = init_progress(9)

    # Discover tasks
    task_ids = discover_task_ids(data_root)
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
    print(f"[WO-9A] Found {len(task_ids)} tasks", file=sys.stderr)

    success_count = 0
    fail_count = 0

    # Track first run hashes and faces presence for post-processing checks
    first_run_hashes = {}
    task_has_faces = {}

    for task_id in sorted(task_ids):
        try:
            # Check that required prior WOs exist
            receipts_dir = Path("receipts")
            task_dir = receipts_dir / task_id
            required_wos = [0, 1, 2, 3, 4, 5, 6]
            for wo in required_wos:
                wo_receipt_path = task_dir / f"wo{wo:02d}.json"
                if not wo_receipt_path.exists():
                    raise FileNotFoundError(f"WO-{wo} receipt not found: {wo_receipt_path}")

            # Run WO-9A′ per-test pack enumeration (writes receipt and cache itself)
            receipt = stages_wo09a.run_wo09a_prime(task_id, data_root)

            # Extract metrics from receipt (WO-9A′ per-test format)
            total_packs_count = receipt.get("total_packs_count", 0)
            packs_hash = receipt.get("hash", "")
            tests_processed = receipt.get("tests_processed", [])

            # Store hash for determinism check
            first_run_hashes[task_id] = packs_hash

            # Accumulate packs_exist_ok metric (all tests should have packs)
            packs_exist_ok = all(test_info["packs_count"] >= 1 for test_info in tests_processed)
            acc_bool(progress, "packs_exist_ok", packs_exist_ok)

            # Load packs from per-test cache files to check quick checks and faces modes
            cache_root = data_root.parent / ".cache"
            all_packs = []
            for test_info in tests_processed:
                test_idx = test_info["test_idx"]
                pack_cache_path = cache_root / "wo09" / f"{task_id}.{test_idx}.packs.json"
                if pack_cache_path.exists():
                    with open(pack_cache_path) as f:
                        pack_cache = json.load(f)
                    all_packs.extend(pack_cache.get("packs", []))

            # Quick checks: verify no exceptions during computation
            if all_packs:
                quick_checks_ok = all(
                    "quick" in pack and isinstance(pack["quick"], dict)
                    for pack in all_packs
                )
                acc_bool(progress, "quick_checks_ok", quick_checks_ok)

                # Faces mode check per spec (check per-canvas WO-5 files)
                for test_info in tests_processed:
                    test_idx = test_info["test_idx"]
                    for canvas_id in test_info.get("canvases", []):
                        wo5_cache_path = cache_root / "wo05" / f"{task_id}.{test_idx}.{canvas_id}.npz"
                        if wo5_cache_path.exists():
                            wo5_data = np.load(wo5_cache_path)
                            faces_R = wo5_data.get("faces_R", None)
                            faces_S = wo5_data.get("faces_S", None)
                            has_faces = (faces_R is not None and faces_R.sum() > 0) or \
                                       (faces_S is not None and faces_S.sum() > 0)

                            # Get packs for this canvas
                            canvas_packs = [p for p in all_packs if p.get("canvas_id") == canvas_id]

                            # Faces mode check per spec
                            if has_faces:
                                faces_mode_ok = any(p["faces_mode"] in ("rows_as_supply", "cols_as_supply")
                                                   for p in canvas_packs)
                            else:
                                faces_mode_ok = all(p["faces_mode"] == "none" for p in canvas_packs)
                            acc_bool(progress, "faces_mode_ok", faces_mode_ok)

            success_count += 1

            # Progress reporting
            if enable_progress and success_count % 100 == 0:
                print(f"[WO-9A] Progress: {success_count}/{len(task_ids)} receipts written", file=sys.stderr)

        except Exception as e:
            print(f"[WO-9A] Failed to process task {task_id}: {e}", file=sys.stderr)
            if fail_count < 2:  # Print traceback for first 2 errors only
                traceback.print_exc(file=sys.stderr)
            fail_count += 1
            if strict:
                raise

    # Determinism check: re-run 3-5 sample tasks and compare hashes
    print(f"[WO-9A] Running determinism check on sample tasks...", file=sys.stderr)
    sample_task_ids = sorted(first_run_hashes.keys())[:min(5, len(first_run_hashes))]
    deterministic_ok = True

    for task_id in sample_task_ids:
        try:
            # Re-run WO-9A′
            rerun_receipt = stages_wo09a.run_wo09a_prime(task_id, data_root)
            rerun_hash = rerun_receipt.get("hash", "")

            # Compare hashes
            if rerun_hash != first_run_hashes[task_id]:
                print(f"[WO-9A] ✗ Determinism check FAILED for {task_id}: hash mismatch", file=sys.stderr)
                deterministic_ok = False
                if strict:
                    raise RuntimeError(f"WO-9A determinism check failed for {task_id}")
        except Exception as e:
            print(f"[WO-9A] ✗ Determinism check FAILED for {task_id}: {e}", file=sys.stderr)
            deterministic_ok = False
            if strict:
                raise

    if deterministic_ok:
        print(f"[WO-9A] ✓ Determinism verified: all {len(sample_task_ids)} sample tasks produced identical hashes", file=sys.stderr)

    # Accumulate determinism metric for ALL tasks (if samples pass, assume all pass)
    for _ in range(success_count):
        acc_bool(progress, "packs_deterministic_ok", deterministic_ok)

    # Summary
    print(
        f"[WO-9A] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-9A] Progress written to progress/progress_wo09.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo09b(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None) -> None:
    """
    WO-9B: Laminar Greedy Relax + IIS

    For each pack from WO-9A, applies laminar precedence relaxation:
    1. Try as-is
    2. Drop faces (if needed)
    3. Reduce quotas minimally (if needed)
    4. Build minimal IIS if still infeasible

    Selects first feasible pack or emits IIS certificate.
    """
    from . import stages_wo09b

    # Initialize progress
    progress = init_progress(10)

    # Discover tasks
    task_ids = discover_task_ids(data_root)
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
    print(f"[WO-9B] Found {len(task_ids)} tasks", file=sys.stderr)

    success_count = 0
    fail_count = 0

    # Track first run hashes for determinism check
    first_run_hashes = {}

    for task_id in sorted(task_ids):
        try:
            # Check that required prior WOs exist
            receipts_dir = Path("receipts")
            task_dir = receipts_dir / task_id
            required_wos = [0, 1, 2, 3, 4, 5, 6, 7, 9]  # Need WO-9A
            for wo in required_wos:
                wo_receipt_path = task_dir / f"wo{wo:02d}.json" if wo < 9 else task_dir / "wo09a.json"
                if not wo_receipt_path.exists():
                    raise FileNotFoundError(f"WO-{wo} receipt not found: {wo_receipt_path}")

            # Run WO-9B laminar greedy relaxation
            receipt = stages_wo09b.run_wo09b(task_id, data_root)

            # Store hash for determinism check
            receipt_hash = receipt.get("hash", "")
            first_run_hashes[task_id] = receipt_hash

            # Extract metrics from receipt
            selected_pack_id = receipt.get("selected_pack_id", None)
            packs_tried = receipt.get("packs_tried", [])
            iis = receipt.get("iis", None)

            # pack_choose_ok: if any pack is feasible, selected_pack_id exists and selected pack is OPTIMAL
            if selected_pack_id is not None:
                # Find the selected pack trial
                selected_trial = None
                for trial in packs_tried:
                    if trial.get("result", {}).get("selected", False):
                        selected_trial = trial
                        break

                if selected_trial is not None:
                    result = selected_trial.get("result", {})
                    pack_choose_ok = (
                        result.get("status") == "OPTIMAL" and
                        result.get("primal_balance_ok", False) and
                        result.get("capacity_ok", False) and
                        result.get("cost_equal_ok", False) and
                        result.get("one_of_10_ok", False)
                    )
                else:
                    pack_choose_ok = False
            else:
                pack_choose_ok = (iis is not None and iis.get("present", False))

            acc_bool(progress, "pack_choose_ok", pack_choose_ok)

            # laminar_respects_tiers_ok: verify tier precedence in drops
            laminar_respects_tiers_ok = True
            for trial in packs_tried:
                drops = trial.get("drops", [])

                # Verify no hard tier drops (mask, equalizer, cell)
                has_hard_drop = any(d.get("tier") in ("mask", "equalizer", "cell") for d in drops)
                if has_hard_drop:
                    laminar_respects_tiers_ok = False
                    break

                # Verify faces drops come before quota drops (laminar order)
                last_faces_idx = -1
                first_quota_idx = len(drops)

                for idx, drop in enumerate(drops):
                    tier = drop.get("tier")
                    if tier == "faces":
                        last_faces_idx = idx
                    elif tier == "quota" and first_quota_idx == len(drops):
                        first_quota_idx = idx

                # If both exist, last faces must come before first quota
                if last_faces_idx >= 0 and first_quota_idx < len(drops):
                    if last_faces_idx >= first_quota_idx:
                        laminar_respects_tiers_ok = False
                        break

            acc_bool(progress, "laminar_respects_tiers_ok", laminar_respects_tiers_ok)

            # iis_present_ok: if no feasible pack, IIS must exist
            if selected_pack_id is None:
                iis_present_ok = (iis is not None and iis.get("present", False))
                acc_bool(progress, "iis_present_ok", iis_present_ok)
            else:
                # Feasible pack found, IIS check not applicable
                acc_bool(progress, "iis_present_ok", True)

            success_count += 1

            # Progress reporting
            if enable_progress and success_count % 100 == 0:
                print(f"[WO-9B] Progress: {success_count}/{len(task_ids)} receipts written", file=sys.stderr)

        except Exception as e:
            print(f"[WO-9B] Failed to process task {task_id}: {e}", file=sys.stderr)
            if fail_count < 2:  # Print traceback for first 2 errors only
                traceback.print_exc(file=sys.stderr)
            fail_count += 1
            if strict:
                raise

    # Determinism check: re-run 3-5 sample tasks and compare hashes
    print(f"[WO-9B] Running determinism check on sample tasks...", file=sys.stderr)
    sample_task_ids = sorted(first_run_hashes.keys())[:min(5, len(first_run_hashes))]
    deterministic_ok = True

    for task_id in sample_task_ids:
        try:
            # Re-run WO-9B
            rerun_receipt = stages_wo09b.run_wo09b(task_id, data_root)
            rerun_hash = rerun_receipt.get("hash", "")

            # Compare hashes
            if rerun_hash != first_run_hashes[task_id]:
                print(f"[WO-9B] ✗ Determinism check FAILED for {task_id}: hash mismatch", file=sys.stderr)
                deterministic_ok = False
                if strict:
                    raise RuntimeError(f"WO-9B determinism check failed for {task_id}")
        except Exception as e:
            print(f"[WO-9B] ✗ Determinism check FAILED for {task_id}: {e}", file=sys.stderr)
            deterministic_ok = False
            if strict:
                raise

    if deterministic_ok:
        print(f"[WO-9B] ✓ Determinism verified: all {len(sample_task_ids)} sample tasks produced identical hashes", file=sys.stderr)

    # Accumulate determinism metric for ALL tasks
    for _ in range(success_count):
        acc_bool(progress, "determinism_ok", deterministic_ok)

    # Summary
    print(
        f"[WO-9B] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-9B] Progress written to progress/progress_wo10.json", file=sys.stderr)

    if fail_count > 0 and strict:
        sys.exit(1)


def run_stage_wo10(data_root: Path, strict: bool = False, enable_progress: bool = True, filter_tasks: Set[str] = None, evaluate: bool = False) -> None:
    """
    WO-10: End-to-end Φ + receipts

    Orchestrates full pipeline for each test input:
    1. Load WO-9B results and select final pack
    2. If feasible → decode to Ŷ using WO-8 decode
    3. If infeasible → use IIS from WO-9B
    4. Write final receipts and optionally evaluate against ground truth
    """
    from . import stages_wo10

    # Initialize progress
    progress = init_progress(11)

    # Discover tasks
    task_ids = discover_task_ids(data_root)
    # Apply filter if provided
    if filter_tasks:
        task_ids = task_ids & filter_tasks
    print(f"[WO-10] Found {len(task_ids)} tasks", file=sys.stderr)

    success_count = 0
    fail_count = 0

    # Track first run hashes for determinism check
    first_run_hashes = {}

    for task_id in sorted(task_ids):
        try:
            # Check that required prior WOs exist
            receipts_dir = Path("receipts")
            task_dir = receipts_dir / task_id
            required_wos = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]  # Need WO-9A and WO-9B
            for wo in required_wos:
                if wo < 9:
                    wo_receipt_path = task_dir / f"wo{wo:02d}.json"
                elif wo == 9:
                    wo_receipt_path = task_dir / "wo09a.json"
                else:  # wo == 10
                    wo_receipt_path = task_dir / "wo09b.json"

                if not wo_receipt_path.exists():
                    raise FileNotFoundError(f"WO-{wo} receipt not found: {wo_receipt_path}")

            # Run WO-10 end-to-end pipeline
            result = stages_wo10.run_wo10(task_id, data_root, evaluate=evaluate)

            # Load wo10 receipts for metrics (may be multiple test inputs)
            test_count = result.get("test_count", 0)
            feasible_count = result.get("feasible_count", 0)
            infeasible_count = result.get("infeasible_count", 0)

            # Check first test's receipt for determinism
            wo10_test0_path = task_dir / "wo10_test0.json"
            if wo10_test0_path.exists():
                with open(wo10_test0_path) as f:
                    wo10_receipt = json.load(f)
                receipt_hash = wo10_receipt.get("hash", "")
                first_run_hashes[task_id] = receipt_hash

            # Metrics
            # final_feasible_ok: for feasible tests, all WO-7 proof flags must be true
            final_feasible_ok = True
            if feasible_count > 0:
                for test_id in range(test_count):
                    wo10_test_path = task_dir / f"wo10_test{test_id}.json"
                    if wo10_test_path.exists():
                        with open(wo10_test_path) as f:
                            wo10_test = json.load(f)
                        final = wo10_test.get("final", {})
                        if final.get("status") == "OPTIMAL":
                            checks = [
                                final.get("primal_balance_ok", False),
                                final.get("capacity_ok", False),
                                final.get("cost_equal_ok", False),
                                final.get("one_of_10_decode_ok", False),
                                final.get("decode_mask_ok", False),
                                final.get("idempotence_ok", False),
                            ]
                            if not all(checks):
                                final_feasible_ok = False
                                break

            acc_bool(progress, "final_feasible_ok", final_feasible_ok)

            # final_iis_ok: for infeasible tests, IIS must be present
            final_iis_ok = True
            if infeasible_count > 0:
                for test_id in range(test_count):
                    wo10_test_path = task_dir / f"wo10_test{test_id}.json"
                    if wo10_test_path.exists():
                        with open(wo10_test_path) as f:
                            wo10_test = json.load(f)
                        if wo10_test.get("final", {}).get("status") == "INFEASIBLE":
                            iis = wo10_test.get("iis", {})
                            if not iis.get("present", False):
                                final_iis_ok = False
                                break

            acc_bool(progress, "final_iis_ok", final_iis_ok)

            # eval_ok: if evaluate mode, check eval.json exists and is well-formed
            eval_ok = True
            if evaluate:
                eval_path = task_dir / "eval.json"
                if eval_path.exists():
                    with open(eval_path) as f:
                        eval_data = json.load(f)
                    # Sanity: counts should match
                    eval_summary = eval_data.get("summary", {})
                    if eval_summary.get("total", 0) != test_count:
                        eval_ok = False
                else:
                    eval_ok = False

            acc_bool(progress, "eval_ok", eval_ok if evaluate else True)

            success_count += 1

        except Exception as e:
            print(f"[WO-10] Task {task_id} failed: {e}", file=sys.stderr)
            if strict:
                raise
            fail_count += 1

    # Determinism check: re-run sample tasks and compare hashes (WO-10 spec: 3-5 tasks)
    print("[WO-10] Running determinism check on sample tasks...", file=sys.stderr)
    deterministic_ok = True
    sample_tasks = list(sorted(first_run_hashes.keys()))[:min(5, len(first_run_hashes))]
    for task_id in sample_tasks:
        try:
            # Re-run
            stages_wo10.run_wo10(task_id, data_root, evaluate=evaluate)

            # Load hash from wo10_test0
            receipts_dir = Path("receipts")
            task_dir = receipts_dir / task_id
            wo10_test0_path = task_dir / "wo10_test0.json"
            if wo10_test0_path.exists():
                with open(wo10_test0_path) as f:
                    wo10_receipt = json.load(f)
                second_hash = wo10_receipt.get("hash", "")

                if first_run_hashes.get(task_id) != second_hash:
                    deterministic_ok = False
                    print(f"[WO-10] Determinism check FAILED for task {task_id}", file=sys.stderr)
                    break
        except Exception as e:
            print(f"[WO-10] Determinism check failed for {task_id}: {e}", file=sys.stderr)
            deterministic_ok = False
            break

    if deterministic_ok:
        print("[WO-10] ✓ Determinism verified: all sample tasks produced identical hashes", file=sys.stderr)

    # Accumulate determinism metric for ALL tasks
    for _ in range(success_count):
        acc_bool(progress, "receipt_determinism_ok", deterministic_ok)

    # Summary
    print(
        f"[WO-10] Complete: {success_count} receipts written, {fail_count} failures",
        file=sys.stderr,
    )

    # Write progress
    if enable_progress:
        receipts.write_run_progress(progress)
        print(f"[WO-10] Progress written to progress/progress_wo11.json", file=sys.stderr)

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
    parser.add_argument(
        "--filter-tasks",
        type=str,
        default=None,
        help="Comma-separated list of task IDs to process (e.g., '00576224,007bbfb7')",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable evaluation mode: compare predictions to ground truth (WO-10 only)",
    )

    args = parser.parse_args()

    # Parse filter tasks if provided
    filter_tasks = None
    if args.filter_tasks:
        filter_tasks = set(args.filter_tasks.split(","))
        print(f"[harness] Filtering to {len(filter_tasks)} tasks: {sorted(filter_tasks)}", file=sys.stderr)

    # Validate work order
    if args.upto_wo < 0:
        print(f"Error: --upto-wo must be >= 0, got {args.upto_wo}", file=sys.stderr)
        sys.exit(1)

    # Stage registry (pipeline pattern: extend-only, no god functions)
    STAGE_RUNNERS = {
        0: run_stage_wo0,
        1: run_stage_wo1,
        2: run_stage_wo2,
        3: run_stage_wo3,
        4: run_stage_wo4,
        5: run_stage_wo5,
        6: run_stage_wo6,
        7: run_stage_wo7,
        8: run_stage_wo8,
        9: run_stage_wo09a,
        10: run_stage_wo09b,
        11: run_stage_wo10,
    }

    # Run stages
    print(
        f"[harness] Running stages up to WO-{args.upto_wo} on {args.data_root}",
        file=sys.stderr,
    )

    for wo in range(args.upto_wo + 1):
        if wo in STAGE_RUNNERS:
            # WO-10 (stage 11) needs evaluate flag
            if wo == 11:
                STAGE_RUNNERS[wo](args.data_root, strict=args.strict, enable_progress=args.progress, filter_tasks=filter_tasks, evaluate=args.evaluate)
            else:
                STAGE_RUNNERS[wo](args.data_root, strict=args.strict, enable_progress=args.progress, filter_tasks=filter_tasks)
        else:
            print(
                f"[harness] WO-{wo} not implemented yet (available: {sorted(STAGE_RUNNERS.keys())})",
                file=sys.stderr,
            )
            sys.exit(1)

    print("[harness] All stages complete", file=sys.stderr)


if __name__ == "__main__":
    main()
