"""Type declarations of the compiled Chebyshev ephemeris kernel.

Epochs are two-part TDB seconds past J2000: ``a`` holds whole seconds and ``f`` the remainder. Outputs are
in the links' scaled units (m, m s⁻¹, m s⁻² for SPK links built with ``scale=1000``).
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

class EphemerisCoverageError(ValueError):
    """An epoch lies outside the coverage of an ephemeris link."""

    body_index: int
    epoch_index: int
    target: int
    center: int

class ChebyshevLink:
    """Type 2 segments of one (target, centre) pair with their priority-resolved coverage."""

    target: int
    center: int

    def __init__(
        self,
        target: int,
        center: int,
        records: list[NDArray[np.float64]],
        init_s: Sequence[float],
        interval_s: Sequence[float],
        coefficient_counts: Sequence[int],
        bounds_s: NDArray[np.float64],
        owners: NDArray[np.int32],
        priorities: NDArray[np.int32],
        scale: float,
    ) -> None: ...
    @property
    def coverage_s(self) -> tuple[float, float]:
        """First and last covered epoch, TDB seconds past J2000."""

def evaluate_states(
    plans: list[list[tuple[ChebyshevLink, float]]],
    a: NDArray[np.float64],
    f: NDArray[np.float64],
    order: int,
) -> NDArray[np.float64]:
    """Return the summed link states of several bodies, shape ``(n_epochs, n_bodies, 3 * (order + 1))``."""
