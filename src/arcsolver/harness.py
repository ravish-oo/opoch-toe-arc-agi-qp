"""WO-0 harness.py - CLI runner for multi-stage ARC solver.

Provides a command-line interface to run the solver on ARC tasks.

At WO-0, we only:
- Validate environment & versions
- Emit per-task receipts proving compliance with anchors/contracts
- No actual solving (stages WO-1+)

Usage:
    python -m arcsolver.harness --data-root data/ --upto-wo 0

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
from typing import Set
from . import receipts


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


def run_stage_wo0(data_root: Path, strict: bool = False) -> None:
    """Run WO-0 stage: environment validation and receipt emission.

    Args:
        data_root: Directory containing ARC JSON files
        strict: If True, fail on first error; else continue and report

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

    # Generate environment payload (same for all tasks at WO-0)
    env_payload = receipts.make_env_payload()

    # Emit receipt for each task
    success_count = 0
    fail_count = 0
    for task_id in sorted(task_ids):  # Sort for deterministic order
        try:
            receipt_path = receipts.write_stage_receipt(task_id, "wo00", env_payload)
            success_count += 1
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
        run_stage_wo0(args.data_root, strict=args.strict)

    # WO-1+: Not implemented yet
    if args.upto_wo >= 1:
        print(
            f"[harness] WO-{args.upto_wo} not implemented yet (only WO-0 available)",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[harness] All stages complete", file=sys.stderr)


if __name__ == "__main__":
    main()
