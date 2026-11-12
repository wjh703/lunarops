# Included by _dynamics_core.pyx.
# Compiled forms of the per-step delayed-history evaluations: the Hermite sample of the accepted nodes
# (lunarops.classes.dynamics.history.interpolate_state_derivative) and the delayed lunar deformation
# (lunarops.classes.dynamics.deformation.LunarDeformation.evaluate with deformation_tensors). They follow
# the Python formulas term by term; the Python versions use BLAS for the small products, whose summation
# order differs, so results agree to a few ulp rather than bit for bit.

cdef enum:
    _MAX_HERMITE_NODES = 32


@cython.cdivision(True)
def hermite_sample(
    const double[:, :] states,
    const double[:, :] derivatives,
    double first_time,
    double step,
    double offset,
    double[::1] state_out,
    double[::1] derivative_out,
):
    """Evaluate the Hermite interpolant of regularly spaced nodes and its derivative.

    Same interpolant as ``interpolate_state_derivative``: it matches the states and their derivatives at
    every node. ``states`` and ``derivatives`` have shape ``(d, n)``; the nodes are
    ``first_time + step * j``. Returns the index of a node within 1e-13 grid units of ``offset`` (whose
    state and derivative are copied unchanged), else -1.
    """
    cdef Py_ssize_t dimension = states.shape[0]
    cdef Py_ssize_t count = states.shape[1]
    cdef Py_ssize_t i, j, k, middle
    cdef Py_ssize_t exact = -1
    cdef double coordinate, product, denominator, slope, inverse_sum, lagrange, lagrange_dot
    cdef double squared, square_dot, weight_term, centered, sum_h, sum_g, sum_h_dot, sum_g_dot
    cdef double distances[_MAX_HERMITE_NODES]
    cdef double h[_MAX_HERMITE_NODES]
    cdef double g[_MAX_HERMITE_NODES]
    cdef double h_dot[_MAX_HERMITE_NODES]
    cdef double g_dot[_MAX_HERMITE_NODES]
    if count < 1 or count > _MAX_HERMITE_NODES:
        raise ValueError("hermite_sample needs 1 to 32 nodes")
    if (
        derivatives.shape[0] != dimension
        or derivatives.shape[1] != count
        or state_out.shape[0] != dimension
        or derivative_out.shape[0] != dimension
    ):
        raise ValueError("hermite_sample arrays have inconsistent shapes")
    if count == 1:
        step = 1.0
    with nogil:
        coordinate = (offset - first_time) / step
        for j in range(count):
            distances[j] = coordinate - <double>j
        for j in range(count):
            if fabs(distances[j]) < 1e-13:
                exact = j
                break
        if exact >= 0:
            for i in range(dimension):
                state_out[i] = states[i, exact]
                derivative_out[i] = derivatives[i, exact]
            count = 0  # skip the interpolation below
        for j in range(count):
            product = 1.0
            denominator = 1.0
            slope = 0.0
            inverse_sum = 0.0
            for k in range(count):
                if k == j:
                    continue
                product = product * distances[k]
                denominator = denominator * <double>(j - k)
                slope = slope + 1.0 / <double>(j - k)
                inverse_sum = inverse_sum + 1.0 / distances[k]
            lagrange = product / denominator
            lagrange_dot = lagrange * inverse_sum
            squared = lagrange * lagrange
            square_dot = 2.0 * lagrange * lagrange_dot
            weight_term = 1.0 - 2.0 * distances[j] * slope
            h[j] = weight_term * squared
            g[j] = distances[j] * squared
            h_dot[j] = -2.0 * slope * squared + weight_term * square_dot
            g_dot[j] = squared + distances[j] * square_dot
        middle = count // 2
        for i in range(dimension if count else 0):
            sum_h = 0.0
            sum_g = 0.0
            sum_h_dot = 0.0
            sum_g_dot = 0.0
            for j in range(count):
                # Subtract a constant before differentiating large position values.
                centered = states[i, j] - states[i, middle]
                sum_h = sum_h + centered * h[j]
                sum_g = sum_g + derivatives[i, j] * g[j]
                sum_h_dot = sum_h_dot + centered * h_dot[j]
                sum_g_dot = sum_g_dot + derivatives[i, j] * g_dot[j]
            state_out[i] = states[i, middle] + sum_h + step * sum_g
            derivative_out[i] = sum_h_dot / step + sum_g_dot
    return exact


@cython.cdivision(True)
def lunar_deformation(
    const double[::1] state,
    const double[::1] state_derivative,
    const double[:, ::1] undistorted_inertia,
    double love_k2,
    double mean_motion_rad_s,
    bint include_tidal,
    bint include_rotational,
    double radius_m,
    double earth_mu,
    double moon_mu,
    double c21_static,
    double s21_static,
    double s22_static,
    double[:, ::1] total_out,
    double[:, ::1] derivative_out,
    double[:, ::1] tidal_out,
    double[:, ::1] rotational_out,
    double[:, ::1] degree2_out,
):
    """Delayed lunar inertia, its rate, the tidal and rotational parts and the degree-2 coefficients.

    ``state`` is the integration state at ``t - tau`` (relative position and velocity in ICRF, 3-1-3 Euler
    angles, mantle angular velocity in PA) and ``state_derivative`` its time derivative; only the angular
    acceleration (entries 9:12) of the latter is used. Folkner et al. (2014), Eqs. 40-42; Park et al.
    (2021), Eqs. 47-55; degree-2 static offsets of Pavlov et al. (2016), Eq. 16.
    """
    cdef double cf, sf, ct, st, cp, sp
    cdef double R[3][3]
    cdef double r[3]
    cdef double r_dot[3]
    cdef double omega[3]
    cdef double omega_dot[3]
    cdef double q, q_dot, distance2, distance, distance5, distance7, rr, a, b, ww, wwd, n2, value
    cdef double sqrt5 = sqrt(5.0)
    cdef double sqrt35 = sqrt(3.0 / 5.0)
    cdef double sqrt125 = sqrt(12.0 / 5.0)
    cdef double xx, yy, zz, xy, xz, yz
    cdef Py_ssize_t i, j
    if state.shape[0] < 12 or state_derivative.shape[0] < 12:
        raise ValueError("lunar_deformation needs the 12 translational and rotational state entries")
    for name, array in (
        ("undistorted_inertia", undistorted_inertia),
        ("total_out", total_out),
        ("derivative_out", derivative_out),
        ("tidal_out", tidal_out),
        ("rotational_out", rotational_out),
    ):
        if array.shape[0] != 3 or array.shape[1] != 3:
            raise ValueError(f"lunar_deformation: {name} must have shape (3, 3)")
    if degree2_out.shape[0] != 2 or degree2_out.shape[1] != 3:
        raise ValueError("lunar_deformation: degree2_out must have shape (2, 3)")
    with nogil:
        cf = cos(state[6])
        sf = sin(state[6])
        ct = cos(state[7])
        st = sin(state[7])
        cp = cos(state[8])
        sp = sin(state[8])
        R[0][0] = cf * cp - sf * ct * sp
        R[0][1] = -cf * sp - sf * ct * cp
        R[0][2] = sf * st
        R[1][0] = sf * cp + cf * ct * sp
        R[1][1] = -sf * sp + cf * ct * cp
        R[1][2] = -cf * st
        R[2][0] = st * sp
        R[2][1] = st * cp
        R[2][2] = ct
        for i in range(3):
            omega[i] = state[9 + i]
            omega_dot[i] = state_derivative[9 + i]
        # Moon-to-Earth vector in PA axes and its rate in the rotating frame.
        for i in range(3):
            r[i] = -(R[0][i] * state[0] + R[1][i] * state[1] + R[2][i] * state[2])
        for i in range(3):
            r_dot[i] = -(R[0][i] * state[3] + R[1][i] * state[4] + R[2][i] * state[5])
        r_dot[0] = r_dot[0] - (omega[1] * r[2] - omega[2] * r[1])
        r_dot[1] = r_dot[1] - (omega[2] * r[0] - omega[0] * r[2])
        r_dot[2] = r_dot[2] - (omega[0] * r[1] - omega[1] * r[0])
        distance2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
        distance = sqrt(distance2)
        distance5 = distance2 * distance2 * distance
        distance7 = distance5 * distance2
        rr = r[0] * r_dot[0] + r[1] * r_dot[1] + r[2] * r_dot[2]
        a = love_k2 * earth_mu / moon_mu * radius_m * radius_m * radius_m
        b = love_k2 * radius_m * radius_m * radius_m / (3.0 * moon_mu)
        ww = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]
        wwd = omega[0] * omega_dot[0] + omega[1] * omega_dot[1] + omega[2] * omega_dot[2]
        n2 = mean_motion_rad_s * mean_motion_rad_s
        for i in range(3):
            for j in range(3):
                q = r[i] * r[j]
                q_dot = r_dot[i] * r[j] + r[i] * r_dot[j]
                if i == j:
                    q = q - distance2 / 3.0
                    q_dot = q_dot - 2.0 * rr / 3.0
                if include_tidal:
                    tidal_out[i, j] = -a * q / distance5
                    derivative_out[i, j] = a * (5.0 * rr * q / distance7 - q_dot / distance5)
                else:
                    tidal_out[i, j] = 0.0
                    derivative_out[i, j] = 0.0
                if include_rotational:
                    value = omega[i] * omega[j]
                    q_dot = omega_dot[i] * omega[j] + omega[i] * omega_dot[j]
                    if i == j:
                        value = value - (ww - n2) / 3.0
                        q_dot = q_dot - 2.0 * wwd / 3.0
                    if i == 2 and j == 2:
                        value = value - n2
                    rotational_out[i, j] = b * value
                    derivative_out[i, j] = derivative_out[i, j] + b * q_dot
                else:
                    rotational_out[i, j] = 0.0
                total_out[i, j] = undistorted_inertia[i, j] + tidal_out[i, j] + rotational_out[i, j]
        # Fully normalised degree-2 coefficients of the inertia tensor (rows C and S, orders 0-2).
        xx = total_out[0, 0]
        yy = total_out[1, 1]
        zz = total_out[2, 2]
        xy = total_out[0, 1]
        xz = total_out[0, 2]
        yz = total_out[1, 2]
        degree2_out[0, 0] = xx / (2.0 * sqrt5) + yy / (2.0 * sqrt5) - zz / sqrt5
        degree2_out[0, 1] = -sqrt35 * xz + c21_static * sqrt35
        degree2_out[0, 2] = -sqrt125 / 4.0 * xx + sqrt125 / 4.0 * yy
        degree2_out[1, 0] = 0.0
        degree2_out[1, 1] = -sqrt35 * yz + s21_static * sqrt35
        degree2_out[1, 2] = -sqrt125 / 2.0 * xy + s22_static * sqrt125
    return None
