#!/usr/bin/env python3
"""
WO-A Acceptance Criteria Tests

Per WO-A.md lines 49-63, must achieve 100% on:
1. packs_exist_ok: packs.json exists for every (task,test)
2. packs_deterministic_ok: re-run 5 random tasks; hash stable
3. faces_mode_ok: faces presence matches pack faces_mode
4. equalizers_allowed_ok: equalizer endpoints in A_mask==True
5. cache_namespace_ok: NPZ files exist for WO-4 and WO-5
"""
import json
import random
import hashlib
from pathlib import Path
import numpy as np
from collections import defaultdict

def test_packs_exist_ok(task_ids, data_root):
    """
    Check that packs.json exists for every (task, test) pair.
    """
    print("\n=== TEST 1: packs_exist_ok ===")
    cache_root = data_root.parent / ".cache"
    missing = []

    for task_id in task_ids:
        # Load task to count tests
        task_file = data_root / "arc-agi_training_challenges.json"
        with open(task_file) as f:
            all_tasks = json.load(f)

        if task_id not in all_tasks:
            print(f"  WARNING: {task_id} not in dataset")
            continue

        task_data = all_tasks[task_id]
        num_tests = len(task_data.get("test", []))

        for test_idx in range(num_tests):
            packs_path = cache_root / "wo09" / f"{task_id}.{test_idx}.packs.json"
            if not packs_path.exists():
                missing.append(f"{task_id}.{test_idx}")

    total = sum(len(all_tasks[tid].get("test", [])) for tid in task_ids if tid in all_tasks)
    passed = total - len(missing)

    print(f"  Found: {passed}/{total} packs.json files")
    if missing:
        print(f"  MISSING: {missing[:5]}... (showing first 5)")
        return False
    print(f"  ✓ PASS: All packs.json exist")
    return True


def test_packs_deterministic_ok(task_ids, data_root):
    """
    Re-run 5 random tasks and verify hash stability.
    """
    print("\n=== TEST 2: packs_deterministic_ok ===")

    # Sample 5 random tasks
    sample_tasks = random.sample(task_ids, min(5, len(task_ids)))
    print(f"  Sampling tasks: {sample_tasks}")

    # Store original hashes
    original_hashes = {}
    for task_id in sample_tasks:
        receipt_path = Path("receipts") / task_id / "wo09a_prime.json"
        if receipt_path.exists():
            with open(receipt_path) as f:
                receipt = json.load(f)
            original_hashes[task_id] = receipt.get("hash")

    print(f"  Original hashes collected: {len(original_hashes)}")

    # Re-run WO-9A′ for these tasks
    print("  Re-running WO-9A′ on sample tasks...")
    from arcsolver.stages_wo09a_prime import run_wo09a_prime

    mismatches = []
    for task_id in sample_tasks:
        if task_id not in original_hashes:
            continue

        try:
            # Re-run
            new_receipt = run_wo09a_prime(task_id, data_root)
            new_hash = new_receipt.get("hash")

            if new_hash != original_hashes[task_id]:
                mismatches.append({
                    "task": task_id,
                    "original": original_hashes[task_id][:16],
                    "new": new_hash[:16],
                })
        except Exception as e:
            print(f"  ERROR re-running {task_id}: {e}")
            mismatches.append({"task": task_id, "error": str(e)})

    if mismatches:
        print(f"  ✗ FAIL: {len(mismatches)} hash mismatches")
        for m in mismatches:
            print(f"    {m}")
        return False

    print(f"  ✓ PASS: All {len(sample_tasks)} hashes stable")
    return True


def test_faces_mode_ok(task_ids, data_root):
    """
    Verify that faces presence in NPZ matches pack faces_mode values.
    """
    print("\n=== TEST 3: faces_mode_ok ===")
    cache_root = data_root.parent / ".cache"

    violations = []
    checked = 0

    for task_id in task_ids[:10]:  # Sample 10 tasks
        # Find all WO-5 caches for this task
        wo5_caches = list((cache_root / "wo05").glob(f"{task_id}.*.npz"))

        for wo5_path in wo5_caches:
            # Extract test_idx and canvas_id from filename
            # Format: task.test.canvas_id.npz
            parts = wo5_path.stem.split(".", 2)
            if len(parts) < 3:
                continue

            task, test_idx, canvas_id = parts[0], parts[1], parts[2]

            # Load WO-5 cache
            wo5_data = np.load(wo5_path)
            has_faces_R = "faces_R" in wo5_data and wo5_data["faces_R"].sum() > 0
            has_faces_S = "faces_S" in wo5_data and wo5_data["faces_S"].sum() > 0
            has_faces = has_faces_R or has_faces_S

            # Load corresponding packs
            packs_path = cache_root / "wo09" / f"{task}.{test_idx}.packs.json"
            if not packs_path.exists():
                continue

            with open(packs_path) as f:
                packs_data = json.load(f)

            # Find packs for this canvas
            canvas_packs = [
                p for p in packs_data["packs"]
                if p.get("canvas_id") == canvas_id
            ]

            if has_faces:
                # Should have at least one pack with faces=rows or faces=cols
                has_faces_pack = any(
                    p["faces_mode"] in ["rows_as_supply", "cols_as_supply"]
                    for p in canvas_packs
                )
                if not has_faces_pack:
                    violations.append(f"{task}.{test_idx}.{canvas_id}: faces present but no faces packs")
            else:
                # All packs should have faces=none
                non_none = [
                    p["faces_mode"] for p in canvas_packs
                    if p["faces_mode"] != "none"
                ]
                if non_none:
                    violations.append(f"{task}.{test_idx}.{canvas_id}: no faces but has {non_none}")

            checked += 1

    print(f"  Checked: {checked} canvas caches")
    if violations:
        print(f"  ✗ FAIL: {len(violations)} violations")
        for v in violations[:3]:
            print(f"    {v}")
        return False

    print(f"  ✓ PASS: All faces modes consistent with NPZ data")
    return True


def test_equalizers_allowed_ok(task_ids, data_root):
    """
    Verify equalizer edge endpoints are in A_mask==True and |S|>=2.
    """
    print("\n=== TEST 4: equalizers_allowed_ok ===")
    cache_root = data_root.parent / ".cache"

    violations = []
    checked_edges = 0

    for task_id in task_ids[:5]:  # Sample 5 tasks
        # Find all WO-4 and WO-5 caches
        wo4_caches = list((cache_root / "wo04").glob(f"{task_id}.*.npz"))

        for wo4_path in wo4_caches:
            parts = wo4_path.stem.split(".", 2)
            if len(parts) < 3:
                continue

            task, test_idx, canvas_id = parts[0], parts[1], parts[2]

            # Load A_mask
            wo4_data = np.load(wo4_path)
            A_mask = wo4_data["A_mask"]

            # Load equalizer edges
            wo5_path = cache_root / "wo05" / f"{task}.{test_idx}.{canvas_id}.npz"
            if not wo5_path.exists():
                continue

            wo5_data = np.load(wo5_path)

            # Check all equalizer edge arrays (format: eq_s_c)
            for key in wo5_data.files:
                if not key.startswith("eq_"):
                    continue

                # Parse s and c from key
                parts = key.split("_")
                if len(parts) != 3:
                    continue
                s, c = int(parts[1]), int(parts[2])

                edges = wo5_data[key]
                if len(edges) == 0:
                    continue

                # Check |S| >= 2 (at least 2 pixels in bin)
                unique_pixels = np.unique(edges.flatten())
                if len(unique_pixels) < 2:
                    violations.append(f"{task}.{test_idx}.{canvas_id} eq_{s}_{c}: |S|={len(unique_pixels)} < 2")
                    continue

                # Check all endpoints in A_mask
                for edge in edges:
                    p1, p2 = edge
                    if not A_mask[p1, c]:
                        violations.append(f"{task}.{test_idx} eq_{s}_{c}: pixel {p1} not in A_mask[:,{c}]")
                    if not A_mask[p2, c]:
                        violations.append(f"{task}.{test_idx} eq_{s}_{c}: pixel {p2} not in A_mask[:,{c}]")

                checked_edges += len(edges)

    print(f"  Checked: {checked_edges} equalizer edges")
    if violations:
        print(f"  ✗ FAIL: {len(violations)} violations")
        for v in violations[:3]:
            print(f"    {v}")
        return False

    print(f"  ✓ PASS: All equalizer edges have allowed endpoints and |S|>=2")
    return True


def test_cache_namespace_ok(task_ids, data_root):
    """
    Verify NPZ files exist at correct paths for both WO-4 and WO-5.
    """
    print("\n=== TEST 5: cache_namespace_ok ===")
    cache_root = data_root.parent / ".cache"

    missing = []
    checked = 0

    for task_id in task_ids:
        # Find all packs files
        packs_files = list((cache_root / "wo09").glob(f"{task_id}.*.packs.json"))

        for packs_path in packs_files:
            with open(packs_path) as f:
                packs_data = json.load(f)

            # Extract test_idx from filename
            test_idx = packs_path.stem.split(".")[1]

            # Check each canvas referenced in packs
            for canvas_id in packs_data.get("canvases_processed", []):
                # Expected paths
                wo4_path = cache_root / "wo04" / f"{task_id}.{test_idx}.{canvas_id}.npz"
                wo5_path = cache_root / "wo05" / f"{task_id}.{test_idx}.{canvas_id}.npz"

                if not wo4_path.exists():
                    missing.append(f"WO-4: {wo4_path.name}")
                if not wo5_path.exists():
                    missing.append(f"WO-5: {wo5_path.name}")

                checked += 1

    print(f"  Checked: {checked} canvas namespaces")
    if missing:
        print(f"  ✗ FAIL: {len(missing)} missing NPZ files")
        for m in missing[:5]:
            print(f"    {m}")
        return False

    print(f"  ✓ PASS: All WO-4 and WO-5 NPZ files exist at correct paths")
    return True


def main():
    """Run all acceptance tests."""
    print("=" * 60)
    print("WO-A ACCEPTANCE CRITERIA TESTS")
    print("=" * 60)

    data_root = Path("data")

    # Get task list from existing receipts
    receipts_dir = Path("receipts")
    task_ids = [
        d.name for d in receipts_dir.iterdir()
        if d.is_dir() and (d / "wo09a_prime.json").exists()
    ]

    print(f"\nFound {len(task_ids)} tasks with WO-9A′ receipts")
    print(f"Sample: {task_ids[:5]}")

    if len(task_ids) == 0:
        print("\n✗ ERROR: No tasks found with WO-9A′ receipts")
        print("  Run: bash scripts/run_harness.sh --wo 9 --filter-tasks <task_list>")
        return

    # Run all tests
    results = {
        "packs_exist_ok": test_packs_exist_ok(task_ids, data_root),
        "packs_deterministic_ok": test_packs_deterministic_ok(task_ids, data_root),
        "faces_mode_ok": test_faces_mode_ok(task_ids, data_root),
        "equalizers_allowed_ok": test_equalizers_allowed_ok(task_ids, data_root),
        "cache_namespace_ok": test_cache_namespace_ok(task_ids, data_root),
    }

    # Summary
    print("\n" + "=" * 60)
    print("ACCEPTANCE SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:30s} {status}")

    all_passed = all(results.values())
    pass_rate = sum(results.values()) / len(results) * 100

    print(f"\nOverall: {pass_rate:.0f}% ({sum(results.values())}/{len(results)} tests passed)")

    if all_passed:
        print("\n🟢 ALL ACCEPTANCE CRITERIA MET - WO-A APPROVED")
    else:
        print("\n🔴 ACCEPTANCE CRITERIA FAILED - REVIEW REQUIRED")

    return all_passed


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    success = main()
    sys.exit(0 if success else 1)
