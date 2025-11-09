"""
WO Cache System - Materialized artifact storage for fast iteration.

Receipts = audit trail (hashes, stats, shapes)
Caches = build artifacts (full NPZ arrays)

Cache structure:
    .cache/wo04/<task_id>.<hash>.npz
    .cache/wo05/<task_id>.<hash>.npz
    .cache/wo06/<task_id>.<hash>.npz

Each cache contains:
- NPZ file with all arrays needed by downstream stages
- manifest.json with metadata and cache keys

Cache keys (for determinism):
- input_hash: SHA256 of task JSON
- code_hash: SHA256 of relevant module sources
- H, W, C: grid dimensions for sanity checks
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


def compute_input_hash(task_id: str, data_root: Path) -> str:
    """
    Compute SHA256 hash of task JSON (canonical form).

    Args:
        task_id: Task identifier
        data_root: Root data directory

    Returns:
        Hex string of SHA256 hash
    """
    # Find task JSON in training/evaluation/test
    for split in ["training", "evaluation", "test"]:
        json_path = data_root / "arc" / split / f"{task_id}.json"
        if json_path.exists():
            with open(json_path, "rb") as f:
                # Use binary read for exact hash
                content = f.read()
            return hashlib.sha256(content).hexdigest()[:16]

    raise FileNotFoundError(f"Task JSON not found: {task_id}")


def compute_code_hash(wo_num: int, modules: Optional[List[str]] = None) -> str:
    """
    Compute SHA256 hash of relevant module sources for a WO stage.

    Args:
        wo_num: WO stage number (4, 5, 6)
        modules: Optional list of module names to hash. If None, uses defaults.

    Returns:
        Hex string of SHA256 hash (truncated to 16 chars)
    """
    # Default modules per WO stage
    default_modules = {
        4: ["mask.py", "bins.py"],
        5: ["eqs.py", "cell_caps.py", "faces.py"],
        6: ["scores.py", "free_maps.py"],
    }

    if modules is None:
        modules = default_modules.get(wo_num, [])

    # Compute combined hash of all module sources
    hasher = hashlib.sha256()

    src_dir = Path(__file__).parent
    for module_name in sorted(modules):
        module_path = src_dir / module_name
        if module_path.exists():
            with open(module_path, "rb") as f:
                hasher.update(f.read())

    # Add WO number to hash to differentiate stages
    hasher.update(f"wo{wo_num:02d}".encode())

    return hasher.hexdigest()[:16]


def get_cache_path(wo_num: int, task_id: str, input_hash: str = "", code_hash: str = "") -> Path:
    """
    Get cache file path for a WO stage and task.

    Args:
        wo_num: WO stage number
        task_id: Task identifier
        input_hash: Input data hash (optional, simplified for now)
        code_hash: Code hash (optional, simplified for now)

    Returns:
        Path to cache NPZ file
    """
    cache_dir = Path(".cache") / f"wo{wo_num:02d}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Simplified cache key = just task_id for now
    # TODO: Add input_hash and code_hash validation later
    cache_key = f"{task_id}"
    return cache_dir / f"{cache_key}.npz"


def load_cache(wo_num: int, task_id: str, data_root: Path,
               expected_code_hash: Optional[str] = None) -> Optional[Dict]:
    """
    Load cached artifacts for a WO stage (simplified - no hash validation for now).

    Args:
        wo_num: WO stage number
        task_id: Task identifier
        data_root: Root data directory (unused for now)
        expected_code_hash: Expected code hash (unused for now)

    Returns:
        Dict of cached artifacts, or None if cache miss
    """
    try:
        # Simplified: just check if cache file exists
        cache_path = get_cache_path(wo_num, task_id)

        if not cache_path.exists():
            return None

        # Load NPZ
        data = np.load(cache_path, allow_pickle=True)

        # Load manifest if it exists
        manifest_path = cache_path.with_suffix(".json")
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

        # Convert NPZ to dict
        artifacts = {key: data[key] for key in data.files}

        # Add manifest metadata
        artifacts["_manifest"] = manifest

        return artifacts

    except Exception:
        return None


def save_cache(wo_num: int, task_id: str, data_root: Path,
               artifacts: Dict, metadata: Optional[Dict] = None) -> Path:
    """
    Save artifacts to cache (simplified - no hash computation for now).

    Args:
        wo_num: WO stage number
        task_id: Task identifier
        data_root: Root data directory (unused for now)
        artifacts: Dict of arrays to cache (must be numpy arrays or serializable)
        metadata: Optional additional metadata for manifest

    Returns:
        Path to saved cache file
    """
    # Simplified: just save to task-based path
    cache_path = get_cache_path(wo_num, task_id)

    # Build manifest
    manifest = {
        "wo": wo_num,
        "task_id": task_id,
    }

    # Add optional metadata
    if metadata:
        manifest.update(metadata)

    # Save NPZ (compress for space efficiency)
    # Filter out non-array items for NPZ
    arrays_to_save = {k: v for k, v in artifacts.items()
                      if isinstance(v, np.ndarray)}

    np.savez_compressed(cache_path, **arrays_to_save)

    # Save manifest
    manifest_path = cache_path.with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return cache_path


def invalidate_cache(wo_num: int, task_id: Optional[str] = None) -> int:
    """
    Invalidate (delete) cached artifacts.

    Args:
        wo_num: WO stage number
        task_id: Optional task ID. If None, invalidates all tasks for this WO.

    Returns:
        Number of cache files deleted
    """
    cache_dir = Path(".cache") / f"wo{wo_num:02d}"

    if not cache_dir.exists():
        return 0

    count = 0

    if task_id:
        # Delete specific task caches
        pattern = f"{task_id}.*.npz"
        for cache_file in cache_dir.glob(pattern):
            cache_file.unlink()
            manifest_file = cache_file.with_suffix(".json")
            if manifest_file.exists():
                manifest_file.unlink()
            count += 1
    else:
        # Delete all caches for this WO
        for cache_file in cache_dir.glob("*.npz"):
            cache_file.unlink()
            manifest_file = cache_file.with_suffix(".json")
            if manifest_file.exists():
                manifest_file.unlink()
            count += 1

    return count


def get_cache_stats() -> Dict:
    """
    Get statistics about cache usage.

    Returns:
        Dict with cache stats per WO stage
    """
    cache_root = Path(".cache")

    if not cache_root.exists():
        return {}

    stats = {}

    for wo_dir in sorted(cache_root.glob("wo*")):
        wo_num = int(wo_dir.name[2:])

        npz_files = list(wo_dir.glob("*.npz"))
        total_size = sum(f.stat().st_size for f in npz_files)

        stats[f"wo{wo_num:02d}"] = {
            "num_tasks": len(npz_files),
            "total_size_mb": total_size / (1024 * 1024),
        }

    return stats
