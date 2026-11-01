from typing import Any

import numpy as np
from numpy.typing import NDArray

def openmp_enabled() -> bool: ...

def point_mass_symmetric(
    positions: NDArray[np.float64],
    gravitational_parameters: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def point_mass_symmetric_batch(
    positions: NDArray[np.float64],
    gravitational_parameters: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def nonspherical_gravity_accelerations(
    positions: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    gravitational_parameter: float,
    reference_radius: float,
    norm: np.ndarray,
) -> NDArray[np.float64]: ...

class LunarRhsKernel:
    def __init__(self, cfg: dict[str, Any]) -> None: ...
    @property
    def state_dimension(self) -> int: ...
    def set_epoch(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        gravitational_parameters: NDArray[np.float64],
        cache_accelerations: NDArray[np.float64],
        cache_potentials: NDArray[np.float64],
        earth_fixed2inertial: NDArray[np.float64],
        moon_fixed2inertial: NDArray[np.float64],
        earth_ephemeris_acceleration: NDArray[np.float64],
        earth_c20: float,
        earth_j2: float,
    ) -> None: ...
    def set_step(
        self,
        tide_vectors: NDArray[np.float64],
        raiser_gravitational_parameters: NDArray[np.float64],
        inertia_total: NDArray[np.float64],
        inertia_derivative: NDArray[np.float64],
        moon_coefficients: NDArray[np.float64],
    ) -> None: ...
    def derivative(self, y: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def terms(self, y: NDArray[np.float64]) -> dict[str, Any]: ...
