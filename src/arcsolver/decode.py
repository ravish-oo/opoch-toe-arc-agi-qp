"""
WO-08: Decode + Bit-meter

Pure integer decode from WO-7 flow decisions and bit-meter calculation
from multiway ties in integer costs. All operations are byte-exact per Annex A.1-A.3.
"""

import math
import numpy as np
from typing import Tuple, Dict


def build_xpc_from_arcs(arcs_p: np.ndarray, arcs_c: np.ndarray, arcs_flow: np.ndarray,
                        N: int, C: int) -> np.ndarray:
    """
    Build x_pc:(N,C) uint8 from bin_to_pixel arc flows.

    Args:
        arcs_p: pixel indices (int32)
        arcs_c: color indices (uint8)
        arcs_flow: flow values 0/1 (uint8)
        N: number of pixels
        C: number of colors

    Returns:
        x_pc: (N,C) uint8 with x[p,c]=1 if flow from bin to pixel p with color c
    """
    x_pc = np.zeros((N, C), dtype=np.uint8)

    for i in range(len(arcs_p)):
        p = int(arcs_p[i])
        c = int(arcs_c[i])
        flow = int(arcs_flow[i])
        if flow > 0:
            x_pc[p, c] = 1

    return x_pc


def decode_from_xpc(x_pc: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, bool, bool]:
    """
    Decode Y_hat from x_pc with deterministic tie-breaking.

    Args:
        x_pc: (N,C) uint8 with one-of-10 guaranteed by WO-7
        H, W: canvas dimensions

    Returns:
        Y_hat: (H,W) int32 decoded grid
        one_of_10_ok: True if all pixels have <=1 color
        is_binary_ok: True if x_pc contains only 0/1 values
    """
    # Sanity checks (byte-exact per Annex A.1)
    sel_per_pixel = x_pc.sum(axis=1)
    one_of_10_ok = (sel_per_pixel <= 1).all()
    is_binary_ok = np.array_equal(x_pc, (x_pc > 0).astype(np.uint8))

    # Decode using argmax (lex tie-break: if multiple 1s, picks lowest color index)
    chosen = np.argmax(x_pc, axis=1).astype(np.int32)
    Y_hat = chosen.reshape(H, W)

    return Y_hat, one_of_10_ok, is_binary_ok


def bitmeter_from_costs(costs: np.ndarray, A_mask: np.ndarray) -> Tuple[np.ndarray, int, Dict]:
    """
    Compute bit-meter from integer costs and mask per Annex §F.

    For each pixel, counts colors tied at minimum cost among allowed channels.
    Bits minted = ceil(log2(|tie_set|)) per pixel.

    Args:
        costs: (N,C) int64 costs = round(-ŝ*SCALE) from WO-6
        A_mask: (N,C) bool from WO-4

    Returns:
        bits: (N,) int64 bits minted per pixel
        total_bits: int sum of bits
        tie_histogram: dict {tie_size: count} for receipt
    """
    N, C = costs.shape
    big = np.iinfo(costs.dtype).max

    # Mask disallowed channels with huge cost (integer-safe per Annex A.2)
    masked_costs = np.where(A_mask, costs, big)

    # Find best cost per pixel (integer min, byte-exact)
    best = masked_costs.min(axis=1)

    # Count ties: how many colors share the min cost
    ties = (masked_costs == best[:, None]).sum(axis=1)

    # Bits = ceil(log2(m)) for m-way ties; 0 if m<=1
    bits = np.fromiter(
        (0 if m <= 1 else math.ceil(math.log2(int(m))) for m in ties),
        count=N,
        dtype=np.int64
    )

    total_bits = int(bits.sum())

    # Build tie histogram for receipt (optional)
    uniq, counts = np.unique(ties, return_counts=True)
    tie_histogram = {int(k): int(v) for k, v in zip(uniq, counts)}

    return bits, total_bits, tie_histogram


def check_mask_compliance(x_pc: np.ndarray, A_mask: np.ndarray) -> bool:
    """
    Check that decoded colors respect A_mask: x[p,c]==1 => A[p,c]==True.

    Args:
        x_pc: (N,C) uint8 decoded decisions
        A_mask: (N,C) bool allowed channels

    Returns:
        True if all used colors are allowed
    """
    # Where x_pc==1, A_mask must also be True
    violations = (x_pc == 1) & (~A_mask)
    return not violations.any()


def run_synthetic_tie_tests() -> bool:
    """
    Run synthetic tie tests for bit-meter validation per WO-08 §7.

    Tests that bitmeter_from_costs() correctly computes ceil(log2(m)) bits
    for m-way ties where m ∈ {1,2,3,4}.

    Returns:
        True if all synthetic tests pass
    """
    # Test parameters
    C = 10  # Standard ARC colors
    test_cases = [
        (1, 0),  # m=1 (no tie) → 0 bits
        (2, 1),  # m=2 (2-way tie) → 1 bit
        (3, 2),  # m=3 (3-way tie) → 2 bits
        (4, 2),  # m=4 (4-way tie) → 2 bits
    ]

    for m, expected_bits in test_cases:
        # Create synthetic costs for single pixel with m-way tie
        costs = np.array([[100] * C], dtype=np.int64)  # (1, C)

        # Set first m colors to minimum cost (creating m-way tie)
        for c in range(m):
            costs[0, c] = 50

        # Allow all colors
        A_mask = np.ones((1, C), dtype=bool)

        # Run bit-meter
        bits, total_bits, tie_histogram = bitmeter_from_costs(costs, A_mask)

        # Verify results
        if bits[0] != expected_bits:
            return False
        if total_bits != expected_bits:
            return False
        if tie_histogram.get(m, 0) != 1:  # Should have exactly one pixel with m-way tie
            return False

    return True


def verify_idempotence_on_tasks(task_ids: list, data_root) -> Tuple[bool, list]:
    """
    Verify idempotence by re-running decode on sample tasks and comparing outputs.

    Tests that decode produces identical Y_hat and total_bits on repeated runs,
    confirming deterministic behavior (argmax with lex tie-break, no randomness).

    Args:
        task_ids: List of task IDs to test (typically 3-4 tasks)
        data_root: Path to data directory

    Returns:
        Tuple of (all_passed, failed_tasks) where all_passed is True if all tasks
        produced identical outputs on both runs
    """
    from pathlib import Path

    failed_tasks = []

    for task_id in task_ids:
        cache_root = data_root.parent / ".cache"

        # Load WO-07 cache (bin_to_pixel arcs)
        wo07_path = cache_root / "wo07" / f"{task_id}.npz"
        if not wo07_path.exists():
            failed_tasks.append(f"{task_id}:no_wo07_cache")
            continue

        wo07_data = np.load(wo07_path)
        arcs_p = wo07_data["arcs_p"]
        arcs_c = wo07_data["arcs_c"]
        arcs_flow = wo07_data["arcs_flow"]
        N = int(wo07_data["N"])
        C = int(wo07_data["C"])
        H = int(wo07_data["H"])
        W = int(wo07_data["W"])

        # Load WO-06 cache (costs)
        wo06_path = cache_root / "wo06" / f"{task_id}.npz"
        if not wo06_path.exists():
            failed_tasks.append(f"{task_id}:no_wo06_cache")
            continue

        wo06_data = np.load(wo06_path)
        costs = wo06_data["costs"]

        # Load WO-04 cache (A_mask)
        wo04_path = cache_root / "wo04" / f"{task_id}.npz"
        if not wo04_path.exists():
            failed_tasks.append(f"{task_id}:no_wo04_cache")
            continue

        wo04_data = np.load(wo04_path)
        A_mask = wo04_data["A_mask"]

        # First run
        x_pc_1 = build_xpc_from_arcs(arcs_p, arcs_c, arcs_flow, N, C)
        Y_hat_1, _, _ = decode_from_xpc(x_pc_1, H, W)
        bits_1, total_bits_1, _ = bitmeter_from_costs(costs, A_mask)

        # Second run (identical inputs)
        x_pc_2 = build_xpc_from_arcs(arcs_p, arcs_c, arcs_flow, N, C)
        Y_hat_2, _, _ = decode_from_xpc(x_pc_2, H, W)
        bits_2, total_bits_2, _ = bitmeter_from_costs(costs, A_mask)

        # Verify byte-exact equality
        if not np.array_equal(Y_hat_1, Y_hat_2):
            failed_tasks.append(f"{task_id}:Y_hat_mismatch")
            continue

        if total_bits_1 != total_bits_2:
            failed_tasks.append(f"{task_id}:total_bits_mismatch")
            continue

        if not np.array_equal(bits_1, bits_2):
            failed_tasks.append(f"{task_id}:bits_per_pixel_mismatch")
            continue

    all_passed = (len(failed_tasks) == 0)
    return all_passed, failed_tasks
