# Included by dynamics.pyx.
# Input shifts of the directional kernel differences of the variational equations
# (lunarops.classes.dynamics.variational). They build the same arrays as the numpy expressions they replace,
# element by element with the same operations, so the results are bit-identical; the products of matrices, which
# numpy hands to BLAS, stay with the callers.


def tide_vector_shifts(
    double increment,
    const double[:] raiser_share,
    lagged,
    Py_ssize_t size,
):
    """Return the tide-vector shifts of ``LocalJacobian.tide_jacobians``, shape ``(3, 3, size)``.

    Row ``(j, a)`` moves the order-``j`` vectors of every raiser ``r`` (flat index ``(r * 3 + j) * 3 + k``)
    by ``(increment * raiser_share[r]) * lagged[j][k, a]``: a step of the delayed relative position along
    ICRF axis ``a``, in the lagged frame of order ``j``. ``size`` is the size of the tide payload,
    ``9 * len(raiser_share)``.
    """
    cdef Py_ssize_t raisers = raiser_share.shape[0]
    cdef Py_ssize_t order, axis, r, k
    cdef double scaled
    cdef const double[:, :] frame
    if size != 9 * raisers or len(lagged) != 3:
        raise ValueError("tide_vector_shifts needs three lagged frames and a payload of 9 values per raiser")
    shifts = np.zeros((3, 3, size))
    cdef double[:, :, ::1] out = shifts
    for order in range(3):
        frame = lagged[order]
        if frame.shape[0] != 3 or frame.shape[1] != 3:
            raise ValueError("lagged frames must be 3 x 3")
        for axis in range(3):
            for r in range(raisers):
                scaled = increment * raiser_share[r]
                for k in range(3):
                    out[order, axis, (r * 3 + order) * 3 + k] = scaled * frame[k, axis]
    return shifts


def tide_lag_shifts(
    const double[:, :, :] payload,
    Py_ssize_t order,
    double cos_plus,
    double sin_plus,
    double cos_minus,
    double sin_minus,
    Py_ssize_t size,
    Py_ssize_t segment_start,
):
    """Return the input shifts ``(plus, minus)`` of a tide-lag column, each of shape ``(size,)``.

    The order-``order`` vectors ``v`` of the payload (shape ``(raisers, 3, 3)``) are rotated about ``z`` by the
    angle of ``(cos, sin)``; the shift ``((c - 1) v_x - s v_y, s v_x + (c - 1) v_y, 0)`` lands at the flat payload
    index within the tide segment, which starts at ``segment_start``.
    """
    cdef Py_ssize_t raisers = payload.shape[0]
    cdef Py_ssize_t r, index
    cdef double vx, vy, c1
    if payload.shape[1] != 3 or payload.shape[2] != 3 or not 0 <= order < 3:
        raise ValueError("tide_lag_shifts needs a payload of shape (raisers, 3, 3) and an order 0 to 2")
    if segment_start < 0 or segment_start + 9 * raisers > size:
        raise ValueError("The tide segment exceeds the input size")
    plus = np.zeros(size)
    minus = np.zeros(size)
    cdef double[::1] out_plus = plus
    cdef double[::1] out_minus = minus
    for r in range(raisers):
        vx = payload[r, order, 0]
        vy = payload[r, order, 1]
        index = segment_start + (r * 3 + order) * 3
        c1 = cos_plus - 1.0
        out_plus[index] = c1 * vx - sin_plus * vy
        out_plus[index + 1] = sin_plus * vx + c1 * vy
        c1 = cos_minus - 1.0
        out_minus[index] = c1 * vx - sin_minus * vy
        out_minus[index + 1] = sin_minus * vx + c1 * vy
    return plus, minus
