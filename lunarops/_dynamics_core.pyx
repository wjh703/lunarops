# cython: language_level=3

from libc.math cimport sqrt

import numpy as np
cimport numpy as cnp


def point_mass_symmetric(
    const double[:, ::1] positions,
    const double[::1] gravitational_parameters,
):
    """Return all Newtonian accelerations and potentials with one evaluation per pair."""
    cdef Py_ssize_t count = positions.shape[0]
    if positions.shape[1] != 3 or gravitational_parameters.shape[0] != count:
        raise ValueError("Expected positions (N,3) and gravitational parameters (N,)")

    cdef cnp.ndarray[cnp.float64_t, ndim=2] acceleration_array = np.zeros((count, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] potential_array = np.zeros(count, dtype=np.float64)
    cdef double[:, ::1] acceleration = acceleration_array
    cdef double[::1] potential = potential_array
    cdef Py_ssize_t i, j
    cdef double dx, dy, dz, distance2, inverse_distance, inverse_distance3
    cdef double mu_i, mu_j
    cdef bint coincident = False

    with nogil:
        for i in range(count - 1):
            mu_i = gravitational_parameters[i]
            for j in range(i + 1, count):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                distance2 = dx * dx + dy * dy + dz * dz
                if distance2 == 0.0:
                    coincident = True
                    continue
                inverse_distance = 1.0 / sqrt(distance2)
                inverse_distance3 = inverse_distance * inverse_distance * inverse_distance
                mu_j = gravitational_parameters[j]

                acceleration[i, 0] += mu_j * dx * inverse_distance3
                acceleration[i, 1] += mu_j * dy * inverse_distance3
                acceleration[i, 2] += mu_j * dz * inverse_distance3
                acceleration[j, 0] -= mu_i * dx * inverse_distance3
                acceleration[j, 1] -= mu_i * dy * inverse_distance3
                acceleration[j, 2] -= mu_i * dz * inverse_distance3
                potential[i] += mu_j * inverse_distance
                potential[j] += mu_i * inverse_distance

    if coincident:
        raise ValueError("Distinct bodies have coincident positions")
    return acceleration_array, potential_array
