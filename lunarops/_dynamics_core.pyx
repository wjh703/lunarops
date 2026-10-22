# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

import cython
from cython.parallel cimport prange
from libc.math cimport sqrt, sin, cos, atan2, pow

import numpy as np
cimport numpy as cnp

cdef extern from *:
    """
    static int lunarops_has_openmp(void) {
    #ifdef _OPENMP
        return 1;
    #else
        return 0;
    #endif
    }
    """
    int lunarops_has_openmp() noexcept nogil


def openmp_enabled():
    """Whether this dynamics extension was compiled with OpenMP."""
    return bool(lunarops_has_openmp())


@cython.cdivision(True)
cdef bint _point_mass_symmetric_epoch(
    const double[:, ::1] positions,
    const double[::1] gravitational_parameters,
    double[:, ::1] acceleration,
    double[::1] potential,
) noexcept nogil:
    cdef Py_ssize_t count = positions.shape[0]
    cdef Py_ssize_t i, j
    cdef double dx, dy, dz, distance2, inverse_distance, inverse_distance3
    cdef double mu_i, mu_j
    cdef bint coincident = False

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
    return coincident


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
    cdef bint coincident
    with nogil:
        coincident = _point_mass_symmetric_epoch(positions, gravitational_parameters, acceleration, potential)

    if coincident:
        raise ValueError("Distinct bodies have coincident positions")
    return acceleration_array, potential_array


@cython.cdivision(True)
def point_mass_symmetric_batch(
    const double[:, :, ::1] positions,
    const double[:, ::1] gravitational_parameters,
):
    """Evaluate source/source Newtonian terms for many epochs."""
    cdef Py_ssize_t epoch_count = positions.shape[0]
    cdef Py_ssize_t count = positions.shape[1]
    if (
        positions.shape[2] != 3
        or gravitational_parameters.shape[0] != epoch_count
        or gravitational_parameters.shape[1] != count
    ):
        raise ValueError("Expected positions (E,N,3) and gravitational parameters (E,N)")
    cdef cnp.ndarray[cnp.float64_t, ndim=3] acceleration_array = np.zeros(
        (epoch_count, count, 3), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=2] potential_array = np.zeros(
        (epoch_count, count), dtype=np.float64
    )
    cdef double[:, :, ::1] acceleration = acceleration_array
    cdef double[:, ::1] potential = potential_array
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] coincident_array = np.zeros(epoch_count, dtype=np.uint8)
    cdef cnp.uint8_t[::1] coincident = coincident_array
    cdef Py_ssize_t epoch
    with nogil:
        # Small systems do not amortize the OpenMP team launch cost.
        # Keep the exact scalar pair traversal inside each independent epoch.
        for epoch in prange(epoch_count, schedule="static", use_threads_if=epoch_count >= 8 and count >= 64):
            coincident[epoch] = _point_mass_symmetric_epoch(
                positions[epoch], gravitational_parameters[epoch], acceleration[epoch], potential[epoch]
            )
    if np.any(coincident_array):
        raise ValueError("Distinct bodies have coincident positions")
    return acceleration_array, potential_array


@cython.cdivision(True)
cdef bint _point_mass_cached_core(
    const double* positions,
    const double* gravitational_parameters,
    Py_ssize_t count,
    Py_ssize_t integrated_count,
    const double* external_accelerations,
    const double* external_potentials,
    double* acceleration,
    double* potential,
) noexcept nogil:
    """Moving-target Newtonian terms reusing the prescribed subsystem; True if bodies coincide."""
    cdef Py_ssize_t external_count = count - integrated_count
    cdef Py_ssize_t i, j, external_index
    cdef double dx, dy, dz, distance2, inverse_distance, inverse_distance3, weight
    cdef bint coincident = False

    for i in range(integrated_count):
        acceleration[3 * i] = 0.0
        acceleration[3 * i + 1] = 0.0
        acceleration[3 * i + 2] = 0.0
        potential[i] = 0.0
        for j in range(count):
            if i == j:
                continue
            dx = positions[3 * j] - positions[3 * i]
            dy = positions[3 * j + 1] - positions[3 * i + 1]
            dz = positions[3 * j + 2] - positions[3 * i + 2]
            distance2 = dx * dx + dy * dy + dz * dz
            if distance2 == 0.0:
                coincident = True
                continue
            inverse_distance = 1.0 / sqrt(distance2)
            inverse_distance3 = inverse_distance * inverse_distance * inverse_distance
            weight = gravitational_parameters[j] * inverse_distance3
            acceleration[3 * i] += weight * dx
            acceleration[3 * i + 1] += weight * dy
            acceleration[3 * i + 2] += weight * dz
            potential[i] += gravitational_parameters[j] * inverse_distance

    for external_index in range(external_count):
        j = integrated_count + external_index
        acceleration[3 * j] = external_accelerations[3 * external_index]
        acceleration[3 * j + 1] = external_accelerations[3 * external_index + 1]
        acceleration[3 * j + 2] = external_accelerations[3 * external_index + 2]
        potential[j] = external_potentials[external_index]
        for i in range(integrated_count):
            dx = positions[3 * i] - positions[3 * j]
            dy = positions[3 * i + 1] - positions[3 * j + 1]
            dz = positions[3 * i + 2] - positions[3 * j + 2]
            distance2 = dx * dx + dy * dy + dz * dz
            if distance2 == 0.0:
                coincident = True
                continue
            inverse_distance = 1.0 / sqrt(distance2)
            inverse_distance3 = inverse_distance * inverse_distance * inverse_distance
            weight = gravitational_parameters[i] * inverse_distance3
            acceleration[3 * j] += weight * dx
            acceleration[3 * j + 1] += weight * dy
            acceleration[3 * j + 2] += weight * dz
            potential[j] += gravitational_parameters[i] * inverse_distance
    return coincident


def point_mass_with_cache(
    const double[:, ::1] positions,
    const double[::1] gravitational_parameters,
    Py_ssize_t integrated_count,
    const double[:, ::1] external_accelerations,
    const double[::1] external_potentials,
):
    """Evaluate moving-target terms while reusing the prescribed subsystem."""
    cdef Py_ssize_t count = positions.shape[0]
    cdef Py_ssize_t external_count = count - integrated_count
    if (
        positions.shape[1] != 3
        or gravitational_parameters.shape[0] != count
        or integrated_count < 0
        or integrated_count > count
        or external_accelerations.shape[0] != external_count
        or external_accelerations.shape[1] != 3
        or external_potentials.shape[0] != external_count
    ):
        raise ValueError("Point-mass cache arrays have incompatible shapes")

    cdef cnp.ndarray[cnp.float64_t, ndim=2] acceleration_array = np.empty((count, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] potential_array = np.empty(count, dtype=np.float64)
    cdef double[:, ::1] acceleration = acceleration_array
    cdef double[::1] potential = potential_array
    cdef bint coincident

    with nogil:
        coincident = _point_mass_cached_core(
            &positions[0, 0], &gravitational_parameters[0], count, integrated_count,
            &external_accelerations[0, 0] if external_count > 0 else NULL,
            &external_potentials[0] if external_count > 0 else NULL,
            &acceleration[0, 0], &potential[0],
        )

    if coincident:
        raise ValueError("Distinct bodies have coincident positions")
    return acceleration_array, potential_array


@cython.cdivision(True)
cdef bint _eih_target(
    Py_ssize_t i,
    const double* positions,
    const double* velocities,
    const double* gravitational_parameters,
    const double* potentials,
    const double* newtonian_accelerations,
    Py_ssize_t count,
    double inverse_speed_of_light_squared,
    double* result,
) noexcept nogil:
    """EIH correction of one target body; True if bodies coincide."""
    cdef Py_ssize_t j
    cdef double dx, dy, dz, distance2, inverse_distance, inverse_distance3
    cdef double vi_x, vi_y, vi_z, vj_x, vj_y, vj_z
    cdef double vi2, vj2, vi_dot_vj, delta_dot_vj, delta_dot_a, radial_velocity
    cdef double vector_dot, factor, weight, common, sx, sy, sz
    cdef bint coincident = False

    vi_x = velocities[3 * i]
    vi_y = velocities[3 * i + 1]
    vi_z = velocities[3 * i + 2]
    vi2 = vi_x * vi_x + vi_y * vi_y + vi_z * vi_z
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for j in range(count):
        if i == j:
            continue
        dx = positions[3 * j] - positions[3 * i]
        dy = positions[3 * j + 1] - positions[3 * i + 1]
        dz = positions[3 * j + 2] - positions[3 * i + 2]
        distance2 = dx * dx + dy * dy + dz * dz
        if distance2 == 0.0:
            coincident = True
            continue
        inverse_distance = 1.0 / sqrt(distance2)
        inverse_distance3 = inverse_distance * inverse_distance * inverse_distance
        vj_x = velocities[3 * j]
        vj_y = velocities[3 * j + 1]
        vj_z = velocities[3 * j + 2]
        vj2 = vj_x * vj_x + vj_y * vj_y + vj_z * vj_z
        vi_dot_vj = vi_x * vj_x + vi_y * vj_y + vi_z * vj_z
        delta_dot_vj = dx * vj_x + dy * vj_y + dz * vj_z
        radial_velocity = delta_dot_vj * inverse_distance
        delta_dot_a = (
            dx * newtonian_accelerations[3 * j]
            + dy * newtonian_accelerations[3 * j + 1]
            + dz * newtonian_accelerations[3 * j + 2]
        )
        factor = (
            -4.0 * potentials[i]
            - potentials[j]
            + vi2
            + 2.0 * vj2
            - 4.0 * vi_dot_vj
            - 1.5 * radial_velocity * radial_velocity
            + 0.5 * delta_dot_a
        )
        vector_dot = (
            dx * (4.0 * vi_x - 3.0 * vj_x)
            + dy * (4.0 * vi_y - 3.0 * vj_y)
            + dz * (4.0 * vi_z - 3.0 * vj_z)
        )
        weight = gravitational_parameters[j] * inverse_distance3
        common = 3.5 * inverse_distance * gravitational_parameters[j]
        sx += weight * (dx * factor + vector_dot * (vj_x - vi_x)) + common * newtonian_accelerations[3 * j]
        sy += weight * (dy * factor + vector_dot * (vj_y - vi_y)) + common * newtonian_accelerations[3 * j + 1]
        sz += weight * (dz * factor + vector_dot * (vj_z - vi_z)) + common * newtonian_accelerations[3 * j + 2]
    result[0] = sx * inverse_speed_of_light_squared
    result[1] = sy * inverse_speed_of_light_squared
    result[2] = sz * inverse_speed_of_light_squared
    return coincident


def eih_correction(
    const double[:, ::1] positions,
    const double[:, ::1] velocities,
    const double[::1] gravitational_parameters,
    const double[::1] potentials,
    const double[:, ::1] newtonian_accelerations,
    const cnp.intp_t[::1] targets,
    double speed_of_light_squared,
):
    """Evaluate EIH corrections for selected targets without array temporaries."""
    cdef Py_ssize_t count = positions.shape[0]
    if (
        positions.shape[1] != 3
        or velocities.shape[0] != count
        or velocities.shape[1] != 3
        or gravitational_parameters.shape[0] != count
        or potentials.shape[0] != count
        or newtonian_accelerations.shape[0] != count
        or newtonian_accelerations.shape[1] != 3
        or speed_of_light_squared <= 0.0
    ):
        raise ValueError("EIH arrays have incompatible shapes")

    cdef cnp.ndarray[cnp.float64_t, ndim=2] correction_array = np.zeros((count, 3), dtype=np.float64)
    cdef double[:, ::1] correction = correction_array
    cdef Py_ssize_t target_index, i
    cdef double inverse_speed_of_light_squared
    cdef bint coincident = False

    for target_index in range(targets.shape[0]):
        i = targets[target_index]
        if i < 0 or i >= count:
            raise ValueError("EIH target index is outside the body array")
    inverse_speed_of_light_squared = 1.0 / speed_of_light_squared

    with nogil:
        for target_index in range(targets.shape[0]):
            i = targets[target_index]
            if _eih_target(
                i, &positions[0, 0], &velocities[0, 0], &gravitational_parameters[0], &potentials[0],
                &newtonian_accelerations[0, 0], count, inverse_speed_of_light_squared, &correction[i, 0],
            ):
                coincident = True

    if coincident:
        raise ValueError("Distinct bodies have coincident positions.")
    return correction_array


@cython.cdivision(True)
cdef bint _earth_tide_core(
    const double* earth_moon_position,
    const double* raiser_vectors,
    const double* raiser_gravitational_parameters,
    Py_ssize_t raiser_count,
    double earth_gravitational_parameter,
    double moon_gravitational_parameter,
    double earth_radius,
    double k20,
    double k21,
    double k22,
    double* acceleration,
) noexcept nogil:
    """Write the DE Earth-tide vector; True if the Earth-Moon vector or a raiser vector is zero."""
    cdef Py_ssize_t index
    cdef double rx, ry, rz, rho2, radius2, radius, common_x, common_y, common_z
    cdef double r0x, r0y, r0z, r1x, r1y, r1z, r2x, r2y, r2z
    cdef double rho0_2, rho2_2, norm0_2, norm1_2, norm2_2
    cdef double dot1, dot2, factor, scale, norm0_5, norm1_5, norm2_5
    cdef double t0x, t0y, t0z, t1x, t1y, t1z, t2x, t2y, t2z
    cdef const double* v
    cdef bint invalid_raiser = False

    acceleration[0] = 0.0
    acceleration[1] = 0.0
    acceleration[2] = 0.0
    rx = earth_moon_position[0]
    ry = earth_moon_position[1]
    rz = earth_moon_position[2]
    rho2 = rx * rx + ry * ry
    radius2 = rho2 + rz * rz
    if radius2 <= 0.0:
        return True
    radius = sqrt(radius2)
    common_x = rx / radius2
    common_y = ry / radius2
    common_z = rz / radius2
    scale = 1.5 * (earth_gravitational_parameter + moon_gravitational_parameter) / earth_gravitational_parameter

    for index in range(raiser_count):
        v = raiser_vectors + 9 * index
        r0x = v[0]
        r0y = v[1]
        r0z = v[2]
        r1x = v[3]
        r1y = v[4]
        r1z = v[5]
        r2x = v[6]
        r2y = v[7]
        r2z = v[8]
        rho0_2 = r0x * r0x + r0y * r0y
        rho2_2 = r2x * r2x + r2y * r2y
        norm0_2 = rho0_2 + r0z * r0z
        norm1_2 = r1x * r1x + r1y * r1y + r1z * r1z
        norm2_2 = rho2_2 + r2z * r2z
        if norm0_2 <= 0.0 or norm1_2 <= 0.0 or norm2_2 <= 0.0:
            invalid_raiser = True
            continue
        dot1 = rx * r1x + ry * r1y
        dot2 = rx * r2x + ry * r2y
        t0x = rho0_2 * rx - 5.0 * ((rz * r0z) * (rz * r0z) + 0.5 * rho2 * rho0_2) * common_x + norm0_2 * rx
        t0y = rho0_2 * ry - 5.0 * ((rz * r0z) * (rz * r0z) + 0.5 * rho2 * rho0_2) * common_y + norm0_2 * ry
        t0z = 2.0 * rz * r0z * r0z - 5.0 * ((rz * r0z) * (rz * r0z) + 0.5 * rho2 * rho0_2) * common_z + norm0_2 * rz
        t1x = 2.0 * rz * r1z * r1x - 10.0 * rz * r1z * dot1 * common_x
        t1y = 2.0 * rz * r1z * r1y - 10.0 * rz * r1z * dot1 * common_y
        t1z = 2.0 * dot1 * r1z - 10.0 * rz * r1z * dot1 * common_z
        t2x = 2.0 * dot2 * r2x - rho2_2 * rx - 5.0 * (dot2 * dot2 - 0.5 * rho2 * rho2_2) * common_x
        t2y = 2.0 * dot2 * r2y - rho2_2 * ry - 5.0 * (dot2 * dot2 - 0.5 * rho2 * rho2_2) * common_y
        t2z = -5.0 * (dot2 * dot2 - 0.5 * rho2 * rho2_2) * common_z
        factor = scale * (raiser_gravitational_parameters[index] * pow(earth_radius, 5.0) / pow(radius, 5.0))
        norm0_5 = pow(sqrt(norm0_2), 5.0)
        norm1_5 = pow(sqrt(norm1_2), 5.0)
        norm2_5 = pow(sqrt(norm2_2), 5.0)
        acceleration[0] += factor * (k20 * t0x / norm0_5 + k21 * t1x / norm1_5 + k22 * t2x / norm2_5)
        acceleration[1] += factor * (k20 * t0y / norm0_5 + k21 * t1y / norm1_5 + k22 * t2y / norm2_5)
        acceleration[2] += factor * (k20 * t0z / norm0_5 + k21 * t1z / norm1_5 + k22 * t2z / norm2_5)
    return invalid_raiser


def earth_tide_relative_acceleration(
    const double[::1] earth_moon_position,
    const double[:, :, ::1] raiser_vectors,
    const double[::1] raiser_gravitational_parameters,
    double earth_gravitational_parameter,
    double moon_gravitational_parameter,
    double earth_radius,
    double k20,
    double k21,
    double k22,
):
    """Evaluate the DE Earth-tide vector for delayed raiser vectors."""
    cdef Py_ssize_t raiser_count = raiser_vectors.shape[0]
    if (
        earth_moon_position.shape[0] != 3
        or raiser_vectors.shape[1] != 3
        or raiser_vectors.shape[2] != 3
        or raiser_gravitational_parameters.shape[0] != raiser_count
        or earth_gravitational_parameter <= 0.0
        or moon_gravitational_parameter <= 0.0
        or earth_radius <= 0.0
    ):
        raise ValueError("Earth-tide arrays or constants have invalid shapes or values")
    if (
        earth_moon_position[0] * earth_moon_position[0]
        + earth_moon_position[1] * earth_moon_position[1]
        + earth_moon_position[2] * earth_moon_position[2]
    ) <= 0.0:
        raise ValueError("Earth-tide Earth-Moon position must be nonzero")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] acceleration_array = np.zeros(3, dtype=np.float64)
    cdef double[::1] acceleration = acceleration_array
    cdef bint invalid_raiser

    with nogil:
        invalid_raiser = _earth_tide_core(
            &earth_moon_position[0],
            &raiser_vectors[0, 0, 0] if raiser_count > 0 else NULL,
            &raiser_gravitational_parameters[0] if raiser_count > 0 else NULL,
            raiser_count,
            earth_gravitational_parameter, moon_gravitational_parameter, earth_radius, k20, k21, k22,
            &acceleration[0],
        )

    if invalid_raiser:
        raise ValueError("Earth-tide raiser position must be nonzero")
    return acceleration_array


@cython.cdivision(True)
cdef void _sh_point(
    const double* position,
    const double* coefficients,
    Py_ssize_t degree,
    double gravitational_parameter,
    double reference_radius,
    const double* norm,
    double* raw,
    double* draw,
    double* cos_m,
    double* sin_m,
    double* acceleration,
) noexcept nogil:
    """One-point fully-normalized spherical-harmonic acceleration (degree 0 excluded).

    ``coefficients`` holds C then S, each ``(degree+1)**2`` row-major; ``raw``/``draw``
    are caller-owned ``(degree+1)**2`` workspaces. Degree zero is the central
    point-mass term, evaluated by the Newtonian model.
    """
    cdef Py_ssize_t stride = degree + 1
    cdef Py_ssize_t size = stride * stride
    cdef Py_ssize_t n, m, k
    cdef double sin_lat, cos_lat, lon, radius, inv_radius2, rho_n
    cdef double pmm, dpmm
    cdef double trig, dtrig, radial_sum, latitude_sum, longitude_sum, cnm, snm
    cdef double ar, alat, aeast, cos_lon, sin_lon
    cdef double eps = 1.0e-15

    radius = sqrt(position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
    if radius == 0.0:
        acceleration[0] = 0.0
        acceleration[1] = 0.0
        acceleration[2] = 0.0
        return
    sin_lat = position[2] / radius
    if sin_lat > 1.0:
        sin_lat = 1.0
    elif sin_lat < -1.0:
        sin_lat = -1.0
    cos_lat = sqrt((1.0 - sin_lat) * (1.0 + sin_lat))
    if cos_lat < eps:
        cos_lat = eps
    lon = atan2(position[1], position[0])
    cos_lon = cos(lon)
    sin_lon = sin(lon)
    cos_m[0] = 1.0
    sin_m[0] = 0.0
    for m in range(1, degree + 1):
        cos_m[m] = cos_m[m - 1] * cos_lon - sin_m[m - 1] * sin_lon
        sin_m[m] = sin_m[m - 1] * cos_lon + cos_m[m - 1] * sin_lon

    # Every raw/draw entry read below is written first, so no clearing is needed.
    radial_sum = 0.0
    latitude_sum = 0.0
    longitude_sum = 0.0
    for m in range(degree + 1):
        pmm = 1.0
        dpmm = 0.0
        for k in range(1, m + 1):
            pmm *= (2.0 * k - 1.0) * cos_lat
            dpmm = -k * sin_lat / cos_lat * pmm
        raw[m * stride + m] = pmm
        draw[m * stride + m] = dpmm
        if m < degree:
            raw[(m + 1) * stride + m] = (2.0 * m + 1.0) * sin_lat * pmm
            draw[(m + 1) * stride + m] = (2.0 * m + 1.0) * (cos_lat * pmm + sin_lat * dpmm)
        for n in range(m + 2, degree + 1):
            raw[n * stride + m] = (
                (2.0 * n - 1.0) * sin_lat * raw[(n - 1) * stride + m] - (n + m - 1.0) * raw[(n - 2) * stride + m]
            ) / (n - m)
            draw[n * stride + m] = (
                (2.0 * n - 1.0) * (cos_lat * raw[(n - 1) * stride + m] + sin_lat * draw[(n - 1) * stride + m])
                - (n + m - 1.0) * draw[(n - 2) * stride + m]
            ) / (n - m)

    for n in range(1, degree + 1):
        rho_n = pow(reference_radius / radius, n)
        for m in range(n + 1):
            cnm = coefficients[n * stride + m]
            snm = coefficients[size + n * stride + m]
            trig = cnm * cos_m[m] + snm * sin_m[m]
            dtrig = m * (-cnm * sin_m[m] + snm * cos_m[m])
            radial_sum += (n + 1.0) * rho_n * norm[n * stride + m] * raw[n * stride + m] * trig
            latitude_sum += rho_n * norm[n * stride + m] * draw[n * stride + m] * trig
            longitude_sum += rho_n * norm[n * stride + m] * raw[n * stride + m] * dtrig

    inv_radius2 = gravitational_parameter / (radius * radius)
    ar = -inv_radius2 * radial_sum
    alat = inv_radius2 * latitude_sum
    aeast = inv_radius2 * longitude_sum / cos_lat
    acceleration[0] = (ar * cos_lat - alat * sin_lat) * cos_lon - aeast * sin_lon
    acceleration[1] = (ar * cos_lat - alat * sin_lat) * sin_lon + aeast * cos_lon
    acceleration[2] = ar * sin_lat + alat * cos_lat


@cython.cdivision(True)
def nonspherical_gravity_accelerations(
    const double[:, ::1] positions,
    const double[:, :, ::1] coefficients,
    double gravitational_parameter,
    double reference_radius,
    const double[:, ::1] norm,
):
    """Evaluate fully-normalized 4pi spherical-harmonic accelerations.

    Known limit: unnormalized Legendre intermediates overflow above roughly
    degree 150. Current production fields use lower degrees.
    """
    cdef Py_ssize_t count = positions.shape[0]
    cdef Py_ssize_t degree
    if positions.shape[1] != 3 or coefficients.shape[0] != 2 or coefficients.shape[1] != coefficients.shape[2]:
        raise ValueError("Expected positions (N,3) and coefficients (2,L+1,L+1)")
    if gravitational_parameter <= 0.0 or reference_radius <= 0.0:
        raise ValueError("gravitational_parameter and reference_radius must be positive")
    degree = coefficients.shape[1] - 1
    if norm.shape[0] != degree + 1 or norm.shape[1] != degree + 1:
        raise ValueError("norm must have shape (L+1,L+1)")

    cdef cnp.ndarray[cnp.float64_t, ndim=2] acceleration_array = np.empty((count, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] raw_array = np.empty((degree + 1, degree + 1), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] draw_array = np.empty((degree + 1, degree + 1), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] cos_array = np.empty(degree + 1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sin_array = np.empty(degree + 1, dtype=np.float64)
    cdef double[:, ::1] acceleration = acceleration_array
    cdef double[:, ::1] raw = raw_array
    cdef double[:, ::1] draw = draw_array
    cdef double[::1] cos_m = cos_array
    cdef double[::1] sin_m = sin_array
    cdef Py_ssize_t i

    with nogil:
        for i in range(count):
            _sh_point(
                &positions[i, 0], &coefficients[0, 0, 0], degree, gravitational_parameter, reference_radius,
                &norm[0, 0], &raw[0, 0], &draw[0, 0], &cos_m[0], &sin_m[0], &acceleration[i, 0],
            )
    return acceleration_array


include "_lunar_rhs.pxi"
