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
from .utils import hash_utils
from .config import GRID_DTYPE


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

    # WO-2+: Not implemented yet
    if args.upto_wo >= 2:
        print(
            f"[harness] WO-{args.upto_wo} not implemented yet (only WO-0/1 available)",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[harness] All stages complete", file=sys.stderr)


if __name__ == "__main__":
    main()
