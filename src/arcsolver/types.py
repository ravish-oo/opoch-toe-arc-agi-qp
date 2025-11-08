"""WO-0 types.py - Canonical dataclasses and type shapes.

Defines immutable dataclasses used throughout the solver:
- Canvas: output canvas dimensions
- Grid: wrapper around int32 grid data
- Component: connected component representation
- Signatures: per-color channel signatures for alignment
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import numpy as np
from .config import GRID_DTYPE, INT_DTYPE


@dataclass(frozen=True)
class Canvas:
    """Output canvas dimensions.

    Attributes:
        H: Height (rows)
        W: Width (columns)
    """

    H: int
    W: int

    def __post_init__(self):
        if self.H <= 0 or self.W <= 0:
            raise ValueError(f"Canvas dimensions must be positive, got H={self.H}, W={self.W}")

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (H, W) shape tuple."""
        return (self.H, self.W)

    @property
    def size(self) -> int:
        """Return total number of pixels."""
        return self.H * self.W


@dataclass(frozen=True)
class Grid:
    """Grid wrapper with dtype validation.

    Attributes:
        data: numpy array (H, W) with dtype int32, values in {-1, 0..9}
    """

    data: np.ndarray

    def __post_init__(self):
        # Validate dtype
        if self.data.dtype != GRID_DTYPE:
            raise RuntimeError(
                f"Grid dtype must be {GRID_DTYPE}, got {self.data.dtype}"
            )
        # Validate 2D
        if self.data.ndim != 2:
            raise ValueError(f"Grid must be 2D, got shape {self.data.shape}")

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (H, W) shape tuple."""
        return self.data.shape

    @property
    def H(self) -> int:
        """Height (rows)."""
        return self.data.shape[0]

    @property
    def W(self) -> int:
        """Width (columns)."""
        return self.data.shape[1]

    @property
    def size(self) -> int:
        """Total number of pixels."""
        return self.data.size

    def assert_values_valid(self, allow_padding: bool = True) -> None:
        """Assert all values are in valid range.

        Args:
            allow_padding: If True, allow -1 (padding sentinel); else only 0..9

        Raises:
            ValueError: If any value is out of range
        """
        min_val = -1 if allow_padding else 0
        if np.any(self.data < min_val) or np.any(self.data > 9):
            raise ValueError(
                f"Grid values must be in [{min_val}, 9], "
                f"got min={self.data.min()}, max={self.data.max()}"
            )


@dataclass(frozen=True)
class Component:
    """Connected component representation.

    Attributes:
        color: Color id (0..9)
        pixels: List of (row, col) pixel coordinates
    """

    color: int
    pixels: List[Tuple[int, int]]

    def __post_init__(self):
        if not (0 <= self.color <= 9):
            raise ValueError(f"Component color must be 0..9, got {self.color}")
        if len(self.pixels) == 0:
            raise ValueError("Component must have at least one pixel")

    @property
    def area(self) -> int:
        """Number of pixels in component."""
        return len(self.pixels)


@dataclass(frozen=True)
class Signatures:
    """Per-color channel signatures for alignment.

    Attributes:
        by_color: Dict mapping color (0..9) to signature tuple
                  Each signature: (neg_count, row_hist, col_hist, bin_hist, color_id)
    """

    by_color: Dict[int, Tuple[Any, ...]]

    def __post_init__(self):
        # Validate all keys are valid colors
        for c in self.by_color.keys():
            if not (0 <= c <= 9):
                raise ValueError(f"Signature color must be 0..9, got {c}")

    def get_canonical_order(self) -> List[int]:
        """Return colors sorted by signature lex order.

        Returns:
            List of color ids in canonical (lex-sorted) order
        """
        # Sort by signature tuple (already in correct lex order per SIG_LEX_FIELDS)
        return sorted(self.by_color.keys(), key=lambda c: self.by_color[c])
