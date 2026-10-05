import numpy as np
from numpy.typing import NDArray

def point_mass_symmetric(
    positions: NDArray[np.float64],
    gravitational_parameters: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def point_mass_with_cache(
    positions: NDArray[np.float64],
    gravitational_parameters: NDArray[np.float64],
    integrated_count: int,
    external_accelerations: NDArray[np.float64],
    external_potentials: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def eih_correction(
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    gravitational_parameters: NDArray[np.float64],
    potentials: NDArray[np.float64],
    newtonian_accelerations: NDArray[np.float64],
    targets: NDArray[np.intp],
    speed_of_light_squared: float,
) -> NDArray[np.float64]: ...

def nonspherical_gravity_accelerations(
    positions: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    gravitational_parameter: float,
    reference_radius: float,
) -> NDArray[np.float64]: ...
