# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

import cython
from libc.math cimport sqrt, sin, cos, atan2, exp, lgamma, pow

import numpy as np
cimport numpy as cnp


@cython.cdivision(True)
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


@cython.cdivision(True)
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
    cdef Py_ssize_t i, j, external_index
    cdef double dx, dy, dz, distance2, inverse_distance, inverse_distance3, weight
    cdef bint coincident = False

    with nogil:
        for i in range(integrated_count):
            acceleration[i, 0] = 0.0
            acceleration[i, 1] = 0.0
            acceleration[i, 2] = 0.0
            potential[i] = 0.0
            for j in range(count):
                if i == j:
                    continue
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                distance2 = dx * dx + dy * dy + dz * dz
                if distance2 == 0.0:
                    coincident = True
                    continue
                inverse_distance = 1.0 / sqrt(distance2)
                inverse_distance3 = inverse_distance * inverse_distance * inverse_distance
                weight = gravitational_parameters[j] * inverse_distance3
                acceleration[i, 0] += weight * dx
                acceleration[i, 1] += weight * dy
                acceleration[i, 2] += weight * dz
                potential[i] += gravitational_parameters[j] * inverse_distance

        for external_index in range(external_count):
            j = integrated_count + external_index
            acceleration[j, 0] = external_accelerations[external_index, 0]
            acceleration[j, 1] = external_accelerations[external_index, 1]
            acceleration[j, 2] = external_accelerations[external_index, 2]
            potential[j] = external_potentials[external_index]
            for i in range(integrated_count):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dz = positions[i, 2] - positions[j, 2]
                distance2 = dx * dx + dy * dy + dz * dz
                if distance2 == 0.0:
                    coincident = True
                    continue
                inverse_distance = 1.0 / sqrt(distance2)
                inverse_distance3 = inverse_distance * inverse_distance * inverse_distance
                weight = gravitational_parameters[i] * inverse_distance3
                acceleration[j, 0] += weight * dx
                acceleration[j, 1] += weight * dy
                acceleration[j, 2] += weight * dz
                potential[j] += gravitational_parameters[i] * inverse_distance

    if coincident:
        raise ValueError("Distinct bodies have coincident positions")
    return acceleration_array, potential_array


@cython.cdivision(True)
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
    cdef Py_ssize_t target_index, i, j
    cdef double dx, dy, dz, distance2, inverse_distance, inverse_distance3
    cdef double vi_x, vi_y, vi_z, vj_x, vj_y, vj_z
    cdef double vi2, vj2, vi_dot_vj, delta_dot_vj, delta_dot_a, radial_velocity
    cdef double vector_dot, factor, weight, common, sx, sy, sz
    cdef double ax, ay, az, inverse_speed_of_light_squared
    cdef bint coincident = False

    for target_index in range(targets.shape[0]):
        i = targets[target_index]
        if i < 0 or i >= count:
            raise ValueError("EIH target index is outside the body array")
    inverse_speed_of_light_squared = 1.0 / speed_of_light_squared

    with nogil:
        for target_index in range(targets.shape[0]):
            i = targets[target_index]
            vi_x = velocities[i, 0]
            vi_y = velocities[i, 1]
            vi_z = velocities[i, 2]
            vi2 = vi_x * vi_x + vi_y * vi_y + vi_z * vi_z
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for j in range(count):
                if i == j:
                    continue
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                distance2 = dx * dx + dy * dy + dz * dz
                if distance2 == 0.0:
                    coincident = True
                    continue
                inverse_distance = 1.0 / sqrt(distance2)
                inverse_distance3 = inverse_distance * inverse_distance * inverse_distance
                vj_x = velocities[j, 0]
                vj_y = velocities[j, 1]
                vj_z = velocities[j, 2]
                vj2 = vj_x * vj_x + vj_y * vj_y + vj_z * vj_z
                vi_dot_vj = vi_x * vj_x + vi_y * vj_y + vi_z * vj_z
                delta_dot_vj = dx * vj_x + dy * vj_y + dz * vj_z
                radial_velocity = delta_dot_vj * inverse_distance
                delta_dot_a = (
                    dx * newtonian_accelerations[j, 0]
                    + dy * newtonian_accelerations[j, 1]
                    + dz * newtonian_accelerations[j, 2]
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
                sx += weight * (dx * factor + vector_dot * (vj_x - vi_x)) + common * newtonian_accelerations[j, 0]
                sy += weight * (dy * factor + vector_dot * (vj_y - vi_y)) + common * newtonian_accelerations[j, 1]
                sz += weight * (dz * factor + vector_dot * (vj_z - vi_z)) + common * newtonian_accelerations[j, 2]
            ax = sx * inverse_speed_of_light_squared
            ay = sy * inverse_speed_of_light_squared
            az = sz * inverse_speed_of_light_squared
            correction[i, 0] = ax
            correction[i, 1] = ay
            correction[i, 2] = az

    if coincident:
        raise ValueError("Distinct bodies have coincident positions.")
    return correction_array


@cython.cdivision(True)
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

    cdef cnp.ndarray[cnp.float64_t, ndim=1] acceleration_array = np.zeros(3, dtype=np.float64)
    cdef double[::1] acceleration = acceleration_array
    cdef Py_ssize_t index
    cdef double rx, ry, rz, rho2, radius2, radius, common_x, common_y, common_z
    cdef double r0x, r0y, r0z, r1x, r1y, r1z, r2x, r2y, r2z
    cdef double rho0_2, rho2_2, norm0_2, norm1_2, norm2_2
    cdef double dot1, dot2, factor, scale
    cdef double t0x, t0y, t0z, t1x, t1y, t1z, t2x, t2y, t2z
    cdef bint invalid_raiser = False

    rx = earth_moon_position[0]
    ry = earth_moon_position[1]
    rz = earth_moon_position[2]
    rho2 = rx * rx + ry * ry
    radius2 = rho2 + rz * rz
    if radius2 <= 0.0:
        raise ValueError("Earth-tide Earth-Moon position must be nonzero")
    radius = sqrt(radius2)
    common_x = rx / radius2
    common_y = ry / radius2
    common_z = rz / radius2
    scale = 1.5 * (earth_gravitational_parameter + moon_gravitational_parameter) / earth_gravitational_parameter

    with nogil:
        for index in range(raiser_count):
            r0x = raiser_vectors[index, 0, 0]
            r0y = raiser_vectors[index, 0, 1]
            r0z = raiser_vectors[index, 0, 2]
            r1x = raiser_vectors[index, 1, 0]
            r1y = raiser_vectors[index, 1, 1]
            r1z = raiser_vectors[index, 1, 2]
            r2x = raiser_vectors[index, 2, 0]
            r2y = raiser_vectors[index, 2, 1]
            r2z = raiser_vectors[index, 2, 2]
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
            t0z = 2.0 * rz * r0z * r0z + rho0_2 * 0.0 - 5.0 * ((rz * r0z) * (rz * r0z) + 0.5 * rho2 * rho0_2) * common_z + norm0_2 * rz
            t1x = 2.0 * rz * r1z * r1x - 10.0 * rz * r1z * dot1 * common_x
            t1y = 2.0 * rz * r1z * r1y - 10.0 * rz * r1z * dot1 * common_y
            t1z = 2.0 * dot1 * r1z - 10.0 * rz * r1z * dot1 * common_z
            t2x = 2.0 * dot2 * r2x - rho2_2 * rx - 5.0 * (dot2 * dot2 - 0.5 * rho2 * rho2_2) * common_x
            t2y = 2.0 * dot2 * r2y - rho2_2 * ry - 5.0 * (dot2 * dot2 - 0.5 * rho2 * rho2_2) * common_y
            t2z = -5.0 * (dot2 * dot2 - 0.5 * rho2 * rho2_2) * common_z
            factor = scale * (raiser_gravitational_parameters[index] * pow(earth_radius, 5.0) / pow(radius, 5.0))
            acceleration[0] += factor * (k20 * t0x / pow(sqrt(norm0_2), 5.0) + k21 * t1x / pow(sqrt(norm1_2), 5.0) + k22 * t2x / pow(sqrt(norm2_2), 5.0))
            acceleration[1] += factor * (k20 * t0y / pow(sqrt(norm0_2), 5.0) + k21 * t1y / pow(sqrt(norm1_2), 5.0) + k22 * t2y / pow(sqrt(norm2_2), 5.0))
            acceleration[2] += factor * (k20 * t0z / pow(sqrt(norm0_2), 5.0) + k21 * t1z / pow(sqrt(norm1_2), 5.0) + k22 * t2z / pow(sqrt(norm2_2), 5.0))

    if invalid_raiser:
        raise ValueError("Earth-tide raiser position must be nonzero")
    return acceleration_array

@cython.cdivision(True)
def nonspherical_gravity_accelerations(
    const double[:, ::1] positions,
    const double[:, :, ::1] coefficients,
    double gravitational_parameter,
    double reference_radius,
):
    """Evaluate fully-normalized 4pi spherical-harmonic accelerations."""
    cdef Py_ssize_t count = positions.shape[0]
    cdef Py_ssize_t degree
    if positions.shape[1] != 3 or coefficients.shape[0] != 2 or coefficients.shape[1] != coefficients.shape[2]:
        raise ValueError("Expected positions (N,3) and coefficients (2,L+1,L+1)")
    if gravitational_parameter <= 0.0 or reference_radius <= 0.0:
        raise ValueError("gravitational_parameter and reference_radius must be positive")
    degree = coefficients.shape[1] - 1

    cdef cnp.ndarray[cnp.float64_t, ndim=2] acceleration_array = np.empty((count, 3), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] raw_array = np.empty((degree + 1, degree + 1), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] draw_array = np.empty((degree + 1, degree + 1), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] norm_array = np.empty((degree + 1, degree + 1), dtype=np.float64)
    cdef double[:, ::1] acceleration = acceleration_array
    cdef double[:, ::1] raw = raw_array
    cdef double[:, ::1] draw = draw_array
    cdef double[:, ::1] norm = norm_array
    cdef Py_ssize_t i, n, m, k
    cdef double sin_lat, cos_lat, lon, radius, inv_radius2, rho_n
    cdef double pmm, dpmm, pn, dpn, normalization
    cdef double trig, dtrig, radial_sum, latitude_sum, longitude_sum, cnm, snm
    cdef double ar, alat, aeast, cos_lon, sin_lon
    cdef double eps = 1.0e-15

    with nogil:
        for i in range(count):
            radius = sqrt(positions[i, 0] * positions[i, 0] + positions[i, 1] * positions[i, 1] + positions[i, 2] * positions[i, 2])
            if radius == 0.0:
                acceleration[i, 0] = 0.0
                acceleration[i, 1] = 0.0
                acceleration[i, 2] = 0.0
                continue
            sin_lat = positions[i, 2] / radius
            if sin_lat > 1.0:
                sin_lat = 1.0
            elif sin_lat < -1.0:
                sin_lat = -1.0
            cos_lat = sqrt((1.0 - sin_lat) * (1.0 + sin_lat))
            if cos_lat < eps:
                cos_lat = eps
            lon = atan2(positions[i, 1], positions[i, 0])
            cos_lon = cos(lon)
            sin_lon = sin(lon)

            for n in range(degree + 1):
                for m in range(degree + 1):
                    raw[n, m] = 0.0
                    draw[n, m] = 0.0
                    norm[n, m] = 0.0

            radial_sum = 0.0
            latitude_sum = 0.0
            longitude_sum = 0.0
            for m in range(degree + 1):
                pmm = 1.0
                dpmm = 0.0
                for k in range(1, m + 1):
                    pmm *= (2.0 * k - 1.0) * cos_lat
                    dpmm = -k * sin_lat / cos_lat * pmm
                normalization = sqrt((2.0 * m + 1.0) * 2.0 * exp(lgamma(1.0) - lgamma(2.0 * m + 1.0))) if m > 0 else 1.0
                raw[m, m] = pmm
                draw[m, m] = dpmm
                norm[m, m] = normalization
                if m < degree:
                    raw[m + 1, m] = (2.0 * m + 1.0) * sin_lat * pmm
                    draw[m + 1, m] = (2.0 * m + 1.0) * (cos_lat * pmm + sin_lat * dpmm)
                    norm[m + 1, m] = sqrt((2.0 * (m + 1.0) + 1.0) * 2.0 * exp(lgamma(2.0) - lgamma(2.0 * m + 2.0)))
                for n in range(m + 2, degree + 1):
                    raw[n, m] = ((2.0 * n - 1.0) * sin_lat * raw[n - 1, m] - (n + m - 1.0) * raw[n - 2, m]) / (n - m)
                    draw[n, m] = ((2.0 * n - 1.0) * (cos_lat * raw[n - 1, m] + sin_lat * draw[n - 1, m]) - (n + m - 1.0) * draw[n - 2, m]) / (n - m)
                    norm[n, m] = sqrt((2.0 * n + 1.0) * (2.0 if m > 0 else 1.0) * exp(lgamma(n - m + 1.0) - lgamma(n + m + 1.0)))

            # Degree zero is the central point-mass term. It is evaluated by
            # the Newtonian point-mass model, so the figure kernel starts at
            # degree one and avoids double-counting it.
            for n in range(1, degree + 1):
                rho_n = pow(reference_radius / radius, n)
                for m in range(n + 1):
                    cnm = coefficients[0, n, m]
                    snm = coefficients[1, n, m]
                    trig = cnm * cos(m * lon) + snm * sin(m * lon)
                    dtrig = m * (-cnm * sin(m * lon) + snm * cos(m * lon))
                    radial_sum += (n + 1.0) * rho_n * norm[n, m] * raw[n, m] * trig
                    latitude_sum += rho_n * norm[n, m] * draw[n, m] * trig
                    longitude_sum += rho_n * norm[n, m] * raw[n, m] * dtrig

            inv_radius2 = gravitational_parameter / (radius * radius)
            ar = -inv_radius2 * radial_sum
            alat = inv_radius2 * latitude_sum
            aeast = inv_radius2 * longitude_sum / cos_lat
            acceleration[i, 0] = (ar * cos_lat - alat * sin_lat) * cos_lon - aeast * sin_lon
            acceleration[i, 1] = (ar * cos_lat - alat * sin_lat) * sin_lon + aeast * cos_lon
            acceleration[i, 2] = ar * sin_lat + alat * cos_lat
    return acceleration_array
