# Included by dynamics.pyx.
# Compiled forms of the per-step delayed-history evaluations: the Hermite sample of the accepted nodes
# (lunarops.classes.dynamics.history.interpolate_state_derivative) and the delayed lunar deformation
# (lunarops.classes.dynamics.deformation.LunarDeformation.evaluate with deformation_tensors). They follow
# the Python formulas term by term; the Python versions use BLAS for the small products, whose summation
# order differs, so results agree to a few ulp rather than bit for bit.

cdef enum:
    _MAX_HERMITE_NODES = 32


cdef Py_ssize_t _hermite_core(
    const double[:, :] states,
    const double[:, :] derivatives,
    double first_time,
    double step,
    double offset,
    double[::1] state_out,
    double[::1] derivative_out,
) noexcept nogil:
    """Body of hermite_sample, after its checks (``step`` already 1 for one node)."""
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
    return _hermite_core(states, derivatives, first_time, step, offset, state_out, derivative_out)


cdef void _lunar_deformation_parts(
    const double* state,
    const double* state_derivative,
    double love_k2,
    double mean_motion_rad_s,
    bint include_tidal,
    bint include_rotational,
    double radius_m,
    double earth_mu,
    double moon_mu,
    double* tidal_out,
    double* rotational_out,
    double* derivative_out,
) noexcept nogil:
    """Tidal and rotational deformation and their rate, row-major 3 x 3 each (see lunar_deformation)."""
    cdef double cf, sf, ct, st, cp, sp
    cdef double R[3][3]
    cdef double r[3]
    cdef double r_dot[3]
    cdef double omega[3]
    cdef double omega_dot[3]
    cdef double q, q_dot, distance2, distance, distance5, distance7, rr, a, b, ww, wwd, n2, value
    cdef Py_ssize_t i, j
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
                tidal_out[3 * i + j] = -a * q / distance5
                derivative_out[3 * i + j] = a * (5.0 * rr * q / distance7 - q_dot / distance5)
            else:
                tidal_out[3 * i + j] = 0.0
                derivative_out[3 * i + j] = 0.0
            if include_rotational:
                value = omega[i] * omega[j]
                q_dot = omega_dot[i] * omega[j] + omega[i] * omega_dot[j]
                if i == j:
                    value = value - (ww - n2) / 3.0
                    q_dot = q_dot - 2.0 * wwd / 3.0
                if i == 2 and j == 2:
                    value = value - n2
                rotational_out[3 * i + j] = b * value
                derivative_out[3 * i + j] = derivative_out[3 * i + j] + b * q_dot
            else:
                rotational_out[3 * i + j] = 0.0


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
    cdef double sqrt5 = sqrt(5.0)
    cdef double sqrt35 = sqrt(3.0 / 5.0)
    cdef double sqrt125 = sqrt(12.0 / 5.0)
    cdef double xx, yy, zz, xy, xz, yz
    cdef double tidal[9]
    cdef double rotational[9]
    cdef double derivative[9]
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
        _lunar_deformation_parts(
            &state[0], &state_derivative[0], love_k2, mean_motion_rad_s, include_tidal, include_rotational,
            radius_m, earth_mu, moon_mu, tidal, rotational, derivative,
        )
        for i in range(3):
            for j in range(3):
                tidal_out[i, j] = tidal[3 * i + j]
                rotational_out[i, j] = rotational[3 * i + j]
                derivative_out[i, j] = derivative[3 * i + j]
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


def lunar_deformation_differences(
    const double[::1] state,
    const double[::1] state_derivative,
    const Py_ssize_t[::1] components,
    const unsigned char[::1] of_derivative,
    const double[::1] increments,
    double love_k2,
    double mean_motion_rad_s,
    bint include_tidal,
    bint include_rotational,
    double radius_m,
    double earth_mu,
    double moon_mu,
    double[:, ::1] deformation_out,
    double[:, ::1] derivative_out,
):
    """Central differences of the deformation (tidal plus rotational) and of its rate, one row per component.

    Row k differences ``state[components[k]]``, or ``state_derivative[components[k]]`` where
    ``of_derivative[k]``, by ``+-increments[k]``; the outputs have shape (k, 9), row-major 3 x 3.
    """
    cdef Py_ssize_t count = components.shape[0]
    cdef Py_ssize_t dimension = state.shape[0]
    cdef Py_ssize_t row, i, component
    cdef double increment
    cdef double tidal_plus[9]
    cdef double rotational_plus[9]
    cdef double rate_plus[9]
    cdef double tidal_minus[9]
    cdef double rotational_minus[9]
    cdef double rate_minus[9]
    if dimension < 12 or state_derivative.shape[0] != dimension:
        raise ValueError("lunar_deformation_differences needs two states of the same dimension, at least 12")
    if of_derivative.shape[0] != count or increments.shape[0] != count:
        raise ValueError("components, of_derivative and increments must have the same length")
    if deformation_out.shape[0] != count or deformation_out.shape[1] != 9:
        raise ValueError("deformation_out must have shape (k, 9)")
    if derivative_out.shape[0] != count or derivative_out.shape[1] != 9:
        raise ValueError("derivative_out must have shape (k, 9)")
    for row in range(count):
        if components[row] < 0 or components[row] >= dimension:
            raise ValueError("component out of range")
    cdef double[::1] work = np.empty(2 * dimension, dtype=np.float64)
    cdef double* y = &work[0]
    cdef double* y_dot = y + dimension
    with nogil:
        for row in range(count):
            for i in range(dimension):
                y[i] = state[i]
                y_dot[i] = state_derivative[i]
            component = components[row]
            increment = increments[row]
            if of_derivative[row]:
                y_dot[component] = state_derivative[component] + increment
            else:
                y[component] = state[component] + increment
            _lunar_deformation_parts(
                y, y_dot, love_k2, mean_motion_rad_s, include_tidal, include_rotational, radius_m, earth_mu,
                moon_mu, tidal_plus, rotational_plus, rate_plus,
            )
            if of_derivative[row]:
                y_dot[component] = state_derivative[component] - increment
            else:
                y[component] = state[component] - increment
            _lunar_deformation_parts(
                y, y_dot, love_k2, mean_motion_rad_s, include_tidal, include_rotational, radius_m, earth_mu,
                moon_mu, tidal_minus, rotational_minus, rate_minus,
            )
            for i in range(9):
                deformation_out[row, i] = (
                    (tidal_plus[i] + rotational_plus[i]) - (tidal_minus[i] + rotational_minus[i])
                ) / (2.0 * increment)
                derivative_out[row, i] = (rate_plus[i] - rate_minus[i]) / (2.0 * increment)
    return None


# ---------------------------------------------------------------------------------------------------------------
# Stencil selection and single-abscissa sampling of the accepted history.
#
# Compiled forms of lunarops.classes.dynamics.history.hermite_stencil and relative_state_query and of
# lunarops.base.interpolation.interpolate_regular_grid at one abscissa of column-stored samples. They repeat the
# Python arithmetic operation for operation, so their results are bit-identical: the scalar stencil arithmetic is
# the same IEEE operations, the sum of the barycentric factors is numpy's pairwise summation (eight accumulators
# from eight terms on), and the weighted sum of the stencil values -- which numpy hands to BLAS (gemv), whose
# summation order depends on the BLAS kernel -- is still taken by numpy.matmul on operands of the same layout.

from libc.math cimport ceil, rint

cdef double _OFFSET_TOLERANCE_S = 1e-9  # lunarops.classes.dynamics.trajectory.OFFSET_TOLERANCE_S
cdef double _NODE_TOLERANCE = 1e-13  # lunarops.base.interpolation.NODE_TOLERANCE
cdef Py_ssize_t _EXTRAPOLATION_NODES = 4  # history._EXTRAPOLATION_NODES
cdef double _MAX_EXTRAPOLATION_SPACINGS = 3.5  # history._MAX_EXTRAPOLATION_SPACINGS
cdef Py_ssize_t _TRANSLATION_EXTRAPOLATION_NODES = 7  # history.TRANSLATION_EXTRAPOLATION_NODES

# Sources of relative_state_query_native, the values of history.RelativeStateSource.
SOURCE_EPHEMERIS = 1
SOURCE_LAGRANGE = 2
SOURCE_HERMITE = 3

cdef object _matmul = np.matmul
cdef dict _lagrange_workspaces = {}


cdef double _pairwise_sum(const double* values, Py_ssize_t count) noexcept nogil:
    """Sum as numpy's add.reduce of a contiguous row: sequential below 8 terms, else 8 accumulators."""
    cdef double res = 0.0
    cdef double r[8]
    cdef Py_ssize_t i, j
    if count < 8:
        for i in range(count):
            res = res + values[i]
        return res
    for j in range(8):
        r[j] = values[j]
    i = 8
    while i < count - count % 8:
        for j in range(8):
            r[j] = r[j] + values[i + j]
        i += 8
    res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    while i < count:
        res = res + values[i]
        i += 1
    return res


cdef bint _is_behind(const double[:] times, Py_ssize_t size, double offset) noexcept nogil:
    cdef double direction = 1.0 if times[size - 1] >= times[0] else -1.0
    return (times[0] - offset) * direction > _OFFSET_TOLERANCE_S


cdef int _hermite_stencil(
    const double[:] times, Py_ssize_t split, double offset, Py_ssize_t order,
    Py_ssize_t* start_out, Py_ssize_t* stop_out, Py_ssize_t* stride_out,
) except -1:
    """history.hermite_stencil; writes the slice bounds into the outputs.

    Returns 1 for an extrapolation stencil (whose slice carries its stride), else 0.
    """
    cdef Py_ssize_t size = times.shape[0]
    cdef Py_ssize_t base = 0, first = 0, length, count, stride, right, start
    cdef double step, ahead_steps
    if size == 0:
        raise IndexError("index -1 is out of bounds for axis 0 with size 0")
    if _is_behind(times, size, offset):
        raise ValueError("Delayed integration sample is outside the accepted history (before the oldest node)")
    length = size
    if times[size - 1] >= times[0]:
        if offset > times[size - 1] + _OFFSET_TOLERANCE_S:
            raise ValueError("Delayed integration sample is outside the accepted history (beyond the latest node)")
        if offset < 0:
            length = min(max(split, 0), size)
        else:
            first = min(max(split, 0), size)
            length = size - first
            base = first
        if length == 0:
            raise IndexError("index -1 is out of bounds for axis 0 with size 0")
    count = min((order + 1) // 2, length)
    step = (times[first + length - 1] - times[first]) / <double>(length - 1) if length > 1 else 1.0
    if (offset - times[first + length - 1]) * step > _OFFSET_TOLERANCE_S:
        ahead_steps = (offset - times[first + length - 1]) / step
        stride = max(1, <Py_ssize_t>ceil(ahead_steps / _MAX_EXTRAPOLATION_SPACINGS))
        count = min(_EXTRAPOLATION_NODES, (length - 1) // stride + 1)
        start_out[0] = base + length - 1 - stride * (count - 1)
        stop_out[0] = base + length
        stride_out[0] = stride
        return 1
    right = <Py_ssize_t>ceil((offset - times[first]) / step)
    start = max(0, min(right - count // 2, length - count))
    start_out[0] = base + start
    stop_out[0] = base + start + count
    stride_out[0] = 1
    return 0


def hermite_stencil(const double[:] times, Py_ssize_t split, double offset, Py_ssize_t order):
    """Return the nodes of the Hermite sample at ``offset`` as a slice of ``times`` (history.hermite_stencil)."""
    cdef Py_ssize_t start, stop, stride
    if _hermite_stencil(times, split, offset, order, &start, &stop, &stride):
        return slice(start, stop, stride)
    return slice(start, stop)


def relative_state_query_native(
    const double[:] times, Py_ssize_t split, double offset, Py_ssize_t order, bint has_deformation
):
    """Return ``(source, ahead_s, order, extrapolate, stencil)`` of history.relative_state_query.

    ``source`` is one of ``SOURCE_EPHEMERIS``, ``SOURCE_LAGRANGE`` and ``SOURCE_HERMITE``; ``stencil`` is the
    Hermite slice or ``None``.
    """
    cdef Py_ssize_t size = times.shape[0]
    cdef Py_ssize_t start, stop, stride
    cdef double ahead
    if size == 0:
        raise IndexError("index -1 is out of bounds for axis 0 with size 0")
    ahead = (offset - times[size - 1]) if times[size - 1] >= times[0] else (times[size - 1] - offset)
    if ahead > _OFFSET_TOLERANCE_S:
        return SOURCE_LAGRANGE, ahead, min(order, _TRANSLATION_EXTRAPOLATION_NODES - 1), True, None
    if has_deformation:
        if _hermite_stencil(times, split, offset, order, &start, &stop, &stride):
            return SOURCE_HERMITE, ahead, 0, False, slice(start, stop, stride)
        return SOURCE_HERMITE, ahead, 0, False, slice(start, stop)
    if _is_behind(times, size, offset):
        return SOURCE_EPHEMERIS, ahead, 0, False, None
    return SOURCE_LAGRANGE, ahead, order, False, None


def hermite_evaluate_native(const double[:] grid, const double[:, :] states, const double[:, :] derivatives, double offset):
    """history.hermite_evaluate for 1 to 32 nodes: the Hermite interpolant and its derivative at ``offset``."""
    cdef Py_ssize_t count = grid.shape[0]
    cdef Py_ssize_t dimension = states.shape[0]
    if count < 1 or count > _MAX_HERMITE_NODES:
        raise ValueError("hermite_evaluate_native needs 1 to 32 nodes")
    if derivatives.shape[0] != dimension or derivatives.shape[1] != count or states.shape[1] != count:
        raise ValueError("hermite_evaluate_native arrays have inconsistent shapes")
    state = np.empty(dimension)
    derivative = np.empty(dimension)
    _hermite_core(states, derivatives, grid[0], grid[1] - grid[0] if count > 1 else 1.0, offset, state, derivative)
    return state, derivative


def lagrange_column_sample(const double[:] nodes, values, double x, Py_ssize_t count, bint extrapolate):
    """trajectory.interpolate_regular with ``check_grid=False``: one abscissa of column-stored samples.

    ``values`` has shape ``(m, n)`` (any strides) and ``nodes`` shape ``(n,)``; ``count`` is ``order + 1``.
    Returns a new array of shape ``(m,)``, bit-identical to the Python routine.
    """
    cdef const double[:, :] samples = values
    cdef Py_ssize_t rows = nodes.shape[0]
    cdef Py_ssize_t m = samples.shape[0]
    cdef Py_ssize_t low, high, start, i, j, nearest_index
    cdef double step, startf, s0, local_step, local, nearest, total
    cdef double weight
    cdef double[::1] result_view
    cdef double[::1] factor_view
    cdef double[:, :, ::1] centred_view
    cdef const double[:, :, :] product_view
    if samples.shape[1] != rows:
        raise ValueError("Invalid regular interpolation arrays")
    if not rows:
        raise ValueError("Regular interpolation requires at least one node")
    result = np.empty(m)
    result_view = result
    if not extrapolate:
        if nodes[0] <= nodes[rows - 1]:
            low = 0
            high = rows - 1
        else:
            low = rows - 1
            high = 0
        if x < nodes[low] - _OFFSET_TOLERANCE_S or x > nodes[high] + _OFFSET_TOLERANCE_S:
            raise ValueError("Epoch is outside propagated trajectory coverage")
        if x <= nodes[low] or x >= nodes[high]:
            j = low if x <= nodes[low] else high
            for i in range(m):
                result_view[i] = samples[i, j]
            return result
    if not isfinite(x):
        raise ValueError("Interpolation abscissae must be finite")
    if count < 1:
        raise ValueError("Invalid regular interpolation arrays")
    count = min(count, rows)
    if count == 1:
        step = (nodes[rows - 1] - nodes[0]) / <double>(rows - 1) if rows > 1 else 1.0
        nearest = min(max(rint((x - nodes[0]) / step), 0.0), <double>(rows - 1))
        j = <Py_ssize_t>nearest
        for i in range(m):
            result_view[i] = samples[i, j]
        return result
    step = (nodes[rows - 1] - nodes[0]) / <double>(rows - 1)
    if step == 0.0 or not isfinite(step):
        raise ValueError("Interpolation nodes must form a regular grid")
    startf = ceil((x - nodes[0]) / step - 0.5 * <double>count)
    startf = min(max(startf, 0.0), <double>(rows - count))
    start = <Py_ssize_t>startf
    s0 = nodes[start]
    local_step = (nodes[start + count - 1] - s0) / <double>(count - 1)
    local = (x - s0) / local_step
    nearest = rint(local)
    if fabs(local - nearest) < _NODE_TOLERANCE and nearest >= 0 and nearest < count:
        nearest_index = start + <Py_ssize_t>nearest
        for i in range(m):
            result_view[i] = samples[i, nearest_index]
        return result
    workspace = _lagrange_workspaces.get((count, m))
    if workspace is None:
        factors = np.empty((1, count))
        centred = np.empty((1, count, m))
        workspace = _lagrange_workspaces[(count, m)] = (factors, factors[:, None, :], centred)
    factors, factor_rows, centred = workspace
    factor_view = factors.reshape(count)
    centred_view = centred
    # Barycentric weights (-1)^j C(count - 1, j), exact in double.
    weight = 1.0
    for j in range(count):
        factor_view[j] = weight / (local - <double>j)
        weight = -weight * <double>(count - 1 - j) / <double>(j + 1)
    for j in range(count):
        for i in range(m):
            centred_view[0, j, i] = samples[i, start + j] - samples[i, start]
    product_view = _matmul(factor_rows, centred)
    total = _pairwise_sum(&factor_view[0], count)
    for i in range(m):
        result_view[i] = samples[i, start] + product_view[0, 0, i] / total
    return result
