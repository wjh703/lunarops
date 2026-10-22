"""Small fixed-size vector helpers for the per-evaluation hot path."""

from __future__ import annotations

import numpy as np


def cross3(a, b) -> np.ndarray:
    """Bitwise equivalent of ``np.cross`` for 3-vectors without its axis handling."""
    a0, a1, a2 = a
    b0, b1, b2 = b
    return np.array((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0))
