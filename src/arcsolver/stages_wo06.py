#!/usr/bin/env python3
"""
WO-6 Stage: Π-safe Scores and FREE Predicate

This module provides the stage runner for WO-6 following the spec pattern.
The actual implementation is in harness.py for consistency with other WOs,
but this module provides the STAGES registry interface as required by the spec.
"""
from __future__ import annotations
from pathlib import Path

# Import the actual implementation from harness
from .harness import run_stage_wo6

# STAGES registry as required by WO-06.md spec
STAGES = {}


def run_wo06(data_root: Path, strict: bool = False, enable_progress: bool = True) -> None:
    """
    WO-6 stage runner: Build Π-safe scores and check FREE predicate.

    This is a wrapper that delegates to the harness implementation.

    Args:
        data_root: Path to data directory
        strict: If True, fail on first error
        enable_progress: If True, write progress metrics
    """
    return run_stage_wo6(data_root, strict=strict, enable_progress=enable_progress)


# Register WO-6 in the STAGES registry
STAGES[6] = run_wo06
