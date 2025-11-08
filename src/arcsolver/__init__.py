"""arcsolver - Deterministic, search-free ARC-AGI solver.

WO-0: Core types, guards, receipts (environment validation)
WO-1+: Size inference, flow construction, solving (future work)

Architecture:
- Π-safe: All operations depend only on observable geometry
- Byte-exact: Integer dtypes for feasibility, no epsilon tolerances
- Idempotent: Re-solving gives byte-identical results (Φ∘Φ=Φ)
- Total Unimodular (TU): Integral LP solutions without rounding
- Deterministic: Single-threaded, pinned versions, fixed hash seed

Modules:
- config: Environment guards, version asserts, dtypes, constants
- types: Canonical dataclasses (Canvas, Grid, Component, Signatures)
- receipts: JSON proof artifacts per task/stage
- harness: CLI runner for multi-stage pipeline
- utils: Hash utilities (byte-stable SHA256)

Contracts (05_contracts.md):
- Python 3.11.x, numpy 2.1.x, scipy 1.13.x, ortools 9.10.x
- All threading env vars = "1", PYTHONHASHSEED=0
- GRID_DTYPE=int32, INT_DTYPE=int64, SCORE_DTYPE=float64
- SCALE=1_000_000 for cost conversion, MAX_ABS_SCORE=1_000_000
"""
from __future__ import annotations

# Version
__version__ = "0.0.1-wo0"

# Expose key types and functions at package level
from .types import Canvas, Grid, Component, Signatures
from .receipts import write_stage_receipt, make_env_payload
from . import config

__all__ = [
    "Canvas",
    "Grid",
    "Component",
    "Signatures",
    "write_stage_receipt",
    "make_env_payload",
    "config",
]
