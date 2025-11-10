"""
WO-08: Decode + Bit-meter stage

Consumes WO-7 flow decisions and produces final prediction grid plus bit-meter.
All operations are byte-exact per Annex A.1-A.3.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict

from . import decode


def run_wo08(task_id: str, data_root: Path) -> Dict:
    """
    Run WO-08 for a single task: decode Y_hat from WO-7 flows and compute bit-meter.

    Args:
        task_id: Task identifier
        data_root: Path to data directory

    Returns:
        Receipt dict with decode and bit-meter stats
    """
    cache_root = data_root.parent / ".cache"

    # Load WO-07 cache (bin_to_pixel arcs)
    wo07_path = cache_root / "wo07" / f"{task_id}.npz"
    if not wo07_path.exists():
        raise FileNotFoundError(f"WO-07 cache not found: {wo07_path}")

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
        raise FileNotFoundError(f"WO-06 cache not found: {wo06_path}")

    wo06_data = np.load(wo06_path)
    costs = wo06_data["costs"]  # (N,C) int64

    # Load WO-04 cache (A_mask)
    wo04_path = cache_root / "wo04" / f"{task_id}.npz"
    if not wo04_path.exists():
        raise FileNotFoundError(f"WO-04 cache not found: {wo04_path}")

    wo04_data = np.load(wo04_path)
    A_mask = wo04_data["A_mask"]  # (N,C) bool

    # Build x_pc from arc data
    x_pc = decode.build_xpc_from_arcs(arcs_p, arcs_c, arcs_flow, N, C)

    # Decode Y_hat
    Y_hat, one_of_10_ok, is_binary_ok = decode.decode_from_xpc(x_pc, H, W)

    # Check mask compliance
    mask_ok = decode.check_mask_compliance(x_pc, A_mask)

    # Compute bit-meter
    bits_per_pixel, total_bits, tie_histogram = decode.bitmeter_from_costs(costs, A_mask)

    # Prepare sample ties for receipt (up to 10 examples)
    sample_ties = []
    tie_pixels = np.where(bits_per_pixel > 0)[0]
    if len(tie_pixels) > 0:
        # Take up to 10 samples
        sample_indices = tie_pixels[:min(10, len(tie_pixels))]
        for p in sample_indices:
            # Count ties at pixel p
            masked_costs = np.where(A_mask[p], costs[p], np.iinfo(costs.dtype).max)
            best = masked_costs.min()
            m = (masked_costs == best).sum()
            sample_ties.append({"p": int(p), "m": int(m)})

    # Build receipt per WO-08 spec
    receipt = {
        "stage": "wo08",
        "decode": {
            "mode": "xpc",
            "one_of_10_decode_ok": bool(one_of_10_ok),
            "mask_ok": bool(mask_ok),
            "is_binary_ok": bool(is_binary_ok),
        },
        "bit_meter": {
            "total_bits": int(total_bits),
            "tie_histogram": tie_histogram,
            "sample_ties": sample_ties,
        },
    }

    # Write receipt to disk
    receipt_dir = data_root.parent / "receipts" / task_id
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / "wo08.json"

    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)

    # Cache Y_hat and bits for later stages (optional)
    cache_dir = cache_root / "wo08"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{task_id}.npz"

    np.savez_compressed(
        cache_path,
        Y_hat=Y_hat,
        bits_per_pixel=bits_per_pixel,
        total_bits=np.array([total_bits], dtype=np.int64),
    )

    return receipt
