"""WO-0 receipts.py - Always-on receipts per task/stage.

Receipts are JSON files written to receipts/<task_id>/<stage>.json
Each stage emits a receipt proving its decisions and validations.

At WO-0, we only emit environment/contract validation receipts.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any
from . import config


def write_stage_receipt(
    task_id: str, stage: str, payload: Dict[str, Any], out_dir: str = "receipts"
) -> Path:
    """Write stage receipt for a task.

    Args:
        task_id: Task identifier
        stage: Stage name (e.g., 'wo00', 'wo01', 'size_infer', etc.)
        payload: JSON-serializable receipt data
        out_dir: Output directory (default: 'receipts')

    Returns:
        Path to written receipt file

    Notes:
        - Creates task directory if needed
        - Writes with sorted keys, Unix newlines
        - Overwrites existing receipt for same task/stage
    """
    # Create task directory
    task_dir = Path(out_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Write receipt
    receipt_path = task_dir / f"{stage}.json"
    with receipt_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            payload,
            f,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            indent=2,
        )
        f.write("\n")  # Trailing newline for Unix convention

    return receipt_path


def make_env_payload() -> Dict[str, Any]:
    """Create environment/contract validation payload for WO-0.

    Returns:
        Dict with runtime versions, dtypes, env vars, constants

    Notes:
        This payload proves the runtime satisfies:
        - Anchors 03_annex.md A.1-A.3
        - Contracts 05_contracts.md / Global
    """
    import numpy
    import scipy

    # Check OR-Tools is present
    try:
        from ortools.graph.python import min_cost_flow  # noqa: F401

        ortools_status = "ok"
    except ImportError:
        ortools_status = "MISSING"

    return {
        "stage": "wo00",
        "description": "Environment and contract validation (WO-0)",
        "runtime": {
            "python": f"{config.REQUIRED_VERSIONS['python_major_minor'][0]}.{config.REQUIRED_VERSIONS['python_major_minor'][1]}",
            "python_full": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "ortools": ortools_status,
        },
        "dtypes": {
            "GRID_DTYPE": str(config.GRID_DTYPE),
            "INT_DTYPE": str(config.INT_DTYPE),
            "SCORE_DTYPE": str(config.SCORE_DTYPE),
            "SCALE": config.SCALE,
            "MAX_ABS_SCORE": config.MAX_ABS_SCORE,
        },
        "constants": {
            "PALETTE_C": config.PALETTE_C,
            "BACKGROUND": int(config.BACKGROUND),
            "PAD_SENTINEL": int(config.PAD_SENTINEL),
        },
        "lex_orders": {
            "PALETTE_ORDER": list(config.PALETTE_ORDER),
            "PIXEL_LEX": config.PIXEL_LEX,
            "PERIOD_LEX": config.PERIOD_LEX,
            "CANVAS_LEX": config.CANVAS_LEX,
            "SIG_LEX_FIELDS": list(config.SIG_LEX_FIELDS),
        },
        "env": {
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS"),
            "PYTHONHASHSEED": os.getenv("PYTHONHASHSEED"),
        },
    }


def write_run_progress(progress: Dict[str, Any], out_dir: str = "progress") -> Path:
    """Write run-level progress JSON for a work order.

    Args:
        progress: Progress dict with wo, tasks_total, tasks_ok, metrics
        out_dir: Output directory (default: 'progress')

    Returns:
        Path to written progress file

    Notes:
        - Creates progress directory if needed
        - Writes with sorted keys, Unix newlines
        - Filename: progress_woXX.json where XX is zero-padded WO number
        - Overwrites existing progress for same WO
    """
    # Create progress directory
    progress_dir = Path(out_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)

    # Write progress
    wo_num = progress["wo"]
    progress_path = progress_dir / f"progress_wo{wo_num:02d}.json"
    with progress_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            progress,
            f,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            indent=2,
        )
        f.write("\n")  # Trailing newline for Unix convention

    return progress_path
