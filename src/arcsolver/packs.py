#!/usr/bin/env python3
"""
WO-9A: Packs & Size Law

Pure functions to enumerate deterministic "packs" for WO-9B.
Each pack fixes: size law, faces mode, FREE maps, and quick feasibility flags.

All operations are Π-safe, byte-exact, deterministic using stdlib/NumPy only.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class SizeLaw:
    """Output size law per 00 §1"""
    law: str  # "constant" | "linear" | "content"
    H: int = 0  # Concrete H for constant laws (0 for parameterized laws)
    W: int = 0  # Concrete W for constant laws (0 for parameterized laws)
    # Linear law coefficients (optional, used when law="linear")
    a_H: Optional[int] = None
    b_H: Optional[int] = None
    a_W: Optional[int] = None
    b_W: Optional[int] = None
    proof_hash: str = ""  # sha256 of proof evidence


@dataclass
class FreeMap:
    """Verified FREE symmetry from WO-2/WO-3"""
    type: str  # "roll" | "perm"
    # For rolls:
    dy: Optional[int] = None
    dx: Optional[int] = None
    # For perms:
    perm: Optional[List[int]] = None
    verified_all_trainings: bool = True


@dataclass
class QuickChecks:
    """O(NC) feasibility flags for WO-9B gating"""
    capacity_ok: bool
    faces_conflict: bool
    capacity_conflicts: List[Dict]
    trivial_period: bool = False  # (dy%H==0) OR (dx%W==0) per spec line 111-112


@dataclass
class Pack:
    """A deterministic pack configuration"""
    pack_id: str
    size_law: SizeLaw
    faces_mode: str  # "none" | "rows_as_supply" | "cols_as_supply"
    free_maps: List[FreeMap]
    quick: QuickChecks


def build_pack_id(size_law: SizeLaw, faces_mode: str, free_maps: List[FreeMap]) -> str:
    """
    Build stable pack ID string for receipts and logs.

    Format: "size=HxW|faces=mode|free=[map1;map2;...]"

    Args:
        size_law: Output size law
        faces_mode: Faces mode string
        free_maps: List of verified FREE maps

    Returns:
        Stable string ID
    """
    # Size component
    size_str = f"size={size_law.H}x{size_law.W}"

    # Faces component
    faces_str = f"faces={faces_mode}"

    # FREE maps component (canonical ordering)
    free_strs = []
    for fm in sorted(free_maps, key=lambda m: _free_map_sort_key(m)):
        if fm.type == "roll":
            free_strs.append(f"roll({fm.dy},{fm.dx})")
        elif fm.type == "perm":
            perm_str = ",".join(str(p) for p in fm.perm)
            free_strs.append(f"perm({perm_str})")

    free_str = f"free=[{';'.join(free_strs)}]" if free_strs else "free=[]"

    return f"{size_str}|{faces_str}|{free_str}"


def _free_map_sort_key(fm: FreeMap) -> Tuple:
    """Canonical sort key for FREE maps: rolls before perms, then by values"""
    if fm.type == "roll":
        return (0, fm.dy or 0, fm.dx or 0)
    elif fm.type == "perm":
        return (1, tuple(fm.perm) if fm.perm else ())
    return (2,)


def compute_quick_checks(
    quotas: np.ndarray,
    A_mask: np.ndarray,
    bin_ids: np.ndarray,
    faces_R: Optional[np.ndarray],
    faces_S: Optional[np.ndarray],
    faces_mode: str,
    free_maps: List[FreeMap],
    H: int,
    W: int,
) -> QuickChecks:
    """
    Compute O(NC) quick feasibility checks per WO-9A spec.

    Args:
        quotas: (num_bins, C) int64 quota targets
        A_mask: (N, C) bool allowed channels
        bin_ids: (N,) int64 bin assignments
        faces_R: Optional (H, C) int64 row totals
        faces_S: Optional (W, C) int64 col totals
        faces_mode: "none" | "rows_as_supply" | "cols_as_supply"
        free_maps: List of FREE maps for trivial period check
        H, W: Canvas dimensions

    Returns:
        QuickChecks with capacity_ok, faces_conflict, capacity_conflicts, trivial_period
    """
    num_bins, C = quotas.shape
    N = H * W

    # === CAPACITY CHECK ===
    # For each (bin, color), count allowed pixels: |{p ∈ B_s : A[p,c]==1}|
    # Check if q[s,c] > allowed[s,c] → capacity conflict
    capacity_conflicts = []

    for s in range(num_bins):
        bin_mask = (bin_ids == s)  # pixels in bin s
        for c in range(C):
            q_sc = int(quotas[s, c])
            # Count allowed pixels in this bin for this color
            allowed_count = int(np.sum(A_mask[bin_mask, c]))

            if q_sc > allowed_count:
                capacity_conflicts.append({
                    "bin": int(s),
                    "color": int(c),
                    "q": q_sc,
                    "allowed": allowed_count,
                })

    capacity_ok = (len(capacity_conflicts) == 0)

    # === FACES CONSISTENCY ===
    faces_conflict = False
    if faces_mode != "none" and (faces_R is not None or faces_S is not None):
        # Check sum_s q[s,c] == sum_r R[r,c] for rows_as_supply
        # or sum_s q[s,c] == sum_j S[j,c] for cols_as_supply
        for c in range(C):
            quota_sum = int(np.sum(quotas[:, c]))

            if faces_mode == "rows_as_supply" and faces_R is not None:
                face_sum = int(np.sum(faces_R[:, c]))
                if quota_sum != face_sum:
                    faces_conflict = True
                    break
            elif faces_mode == "cols_as_supply" and faces_S is not None:
                face_sum = int(np.sum(faces_S[:, c]))
                if quota_sum != face_sum:
                    faces_conflict = True
                    break

    # === TRIVIAL PERIODS ===
    # Per spec: "If a roll is (p_y==H or p_x==W) (identity), flag quick.trivial_period=true"
    trivial_period = False

    for fm in free_maps:
        if fm.type == "roll":
            dy = fm.dy if fm.dy is not None else 0
            dx = fm.dx if fm.dx is not None else 0

            # Check if either axis is trivial (modulo dimension) - OR logic per spec
            trivial_y = (dy % H == 0) if H > 0 else True
            trivial_x = (dx % W == 0) if W > 0 else True

            if trivial_y or trivial_x:
                trivial_period = True
                break

    return QuickChecks(
        capacity_ok=capacity_ok,
        faces_conflict=faces_conflict,
        capacity_conflicts=capacity_conflicts,
        trivial_period=trivial_period,
    )


def enumerate_packs(
    size_laws: List[SizeLaw],
    faces_R: Optional[np.ndarray],
    faces_S: Optional[np.ndarray],
    free_maps_verified: List[Dict],
    quotas: np.ndarray,
    A_mask: np.ndarray,
    bin_ids: np.ndarray,
) -> List[Pack]:
    """
    Enumerate all deterministic packs for WO-9B.

    Packs are ordered lexicographically by (law, H, W, faces_mode, free_maps_lex).
    Uses stable sort to preserve input order for equal keys.

    Args:
        size_laws: List of proven size laws (from WO-1/WO-2)
        faces_R: Optional (H, C) row totals from WO-5
        faces_S: Optional (W, C) col totals from WO-5
        free_maps_verified: List of verified FREE maps from WO-6
        quotas: (num_bins, C) quotas from WO-5
        A_mask: (N, C) mask from WO-4
        bin_ids: (N,) bin assignments from WO-4

    Returns:
        List of Pack objects, lexicographically sorted
    """
    packs = []

    # Convert free maps from dict to FreeMap objects
    free_maps_objects = []
    for fm_dict in free_maps_verified:
        if fm_dict["type"] == "roll":
            free_maps_objects.append(FreeMap(
                type="roll",
                dy=fm_dict.get("dy"),
                dx=fm_dict.get("dx"),
                verified_all_trainings=fm_dict.get("verified_all_trainings", True),
            ))
        elif fm_dict["type"] == "perm":
            free_maps_objects.append(FreeMap(
                type="perm",
                perm=list(fm_dict.get("perm", [])),
                verified_all_trainings=fm_dict.get("verified_all_trainings", True),
            ))

    # Determine faces modes
    has_faces = (faces_R is not None) or (faces_S is not None)
    if has_faces:
        faces_modes = ["rows_as_supply", "cols_as_supply", "none"]
    else:
        faces_modes = ["none"]

    # Enumerate packs: Cartesian product of (size_laws × faces_modes)
    for size_law in size_laws:
        H, W = size_law.H, size_law.W

        for faces_mode in faces_modes:
            # Build pack ID
            pack_id = build_pack_id(size_law, faces_mode, free_maps_objects)

            # Compute quick checks
            quick = compute_quick_checks(
                quotas=quotas,
                A_mask=A_mask,
                bin_ids=bin_ids,
                faces_R=faces_R,
                faces_S=faces_S,
                faces_mode=faces_mode,
                free_maps=free_maps_objects,
                H=H,
                W=W,
            )

            # Create pack
            pack = Pack(
                pack_id=pack_id,
                size_law=size_law,
                faces_mode=faces_mode,
                free_maps=free_maps_objects,
                quick=quick,
            )
            packs.append(pack)

    # Sort packs lexicographically with stable sort
    packs.sort(key=lambda p: (
        p.size_law.law,
        p.size_law.H,
        p.size_law.W,
        p.faces_mode,
        p.pack_id,  # pack_id already includes canonical free_maps ordering
    ))

    return packs


def pack_to_dict(pack: Pack) -> Dict:
    """Convert Pack to dict for JSON serialization"""
    # Convert FreeMap objects to dicts
    free_maps_dicts = []
    for fm in pack.free_maps:
        fm_dict = {"type": fm.type, "verified_all_trainings": fm.verified_all_trainings}
        if fm.type == "roll":
            fm_dict["dy"] = fm.dy
            fm_dict["dx"] = fm.dx
        elif fm.type == "perm":
            fm_dict["perm"] = fm.perm
        free_maps_dicts.append(fm_dict)

    # Build size_law dict with optional linear coefficients
    size_law_dict = {
        "law": pack.size_law.law,
        "H": pack.size_law.H,
        "W": pack.size_law.W,
    }
    if pack.size_law.law == "linear":
        size_law_dict["a_H"] = pack.size_law.a_H
        size_law_dict["b_H"] = pack.size_law.b_H
        size_law_dict["a_W"] = pack.size_law.a_W
        size_law_dict["b_W"] = pack.size_law.b_W

    return {
        "pack_id": pack.pack_id,
        "size_law": size_law_dict,
        "faces_mode": pack.faces_mode,
        "free_maps": free_maps_dicts,
        "quick": {
            "capacity_ok": pack.quick.capacity_ok,
            "faces_conflict": pack.quick.faces_conflict,
            "capacity_conflicts": pack.quick.capacity_conflicts,
            "trivial_period": pack.quick.trivial_period,
        },
    }


def compute_packs_hash(packs: List[Pack]) -> str:
    """
    Compute deterministic SHA-256 hash of packs list.

    Uses canonical JSON serialization with sort_keys=True.

    Args:
        packs: List of Pack objects

    Returns:
        Hex digest of SHA-256 hash
    """
    packs_dicts = [pack_to_dict(p) for p in packs]
    canonical_json = json.dumps(packs_dicts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode()).hexdigest()
