"""Foundational physical constants; model-specific conventions live with their model."""

from __future__ import annotations

C = 299_792_458.0
C2 = C * C
SECONDS_PER_DAY = 86400.0

__all__ = ["C2", "SECONDS_PER_DAY", "C"]
