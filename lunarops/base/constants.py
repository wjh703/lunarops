"""Foundational physical constants shared by LunarOps.

Keep this module deliberately small: it is imported by nearly every layer.
Model-specific constants live with the model that owns their convention.
"""

from __future__ import annotations

C = 299_792_458.0
C2 = C * C
SECONDS_PER_DAY = 86400.0

__all__ = ["C2", "SECONDS_PER_DAY", "C"]
