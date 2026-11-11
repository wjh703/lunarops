# Double-double arithmetic, included by _dynamics_core.pyx.
#
# A value is the unevaluated sum hi + lo of two doubles with |lo| <= ulp(hi)/2,
# which carries about 32 significant decimal digits. The formulas are the
# error-free transformations of Dekker (1971) and the accurate double-word
# algorithms of Joldes, Muller & Popescu (2017), ACM TOMS 44(2). They rely on
# IEEE-754 evaluation without fused multiply-add contraction, which is why the
# extension is compiled with -ffp-contract=off (/fp:strict on Windows).


cdef struct dd_t:
    double hi
    double lo


cdef inline dd_t _fast_two_sum(double a, double b) noexcept nogil:
    """Exact a + b for |a| >= |b|."""
    cdef dd_t r
    r.hi = a + b
    r.lo = b - (r.hi - a)
    return r


cdef inline dd_t _two_sum(double a, double b) noexcept nogil:
    """Exact a + b."""
    cdef dd_t r
    cdef double bb
    r.hi = a + b
    bb = r.hi - a
    r.lo = (a - (r.hi - bb)) + (b - bb)
    return r


cdef inline dd_t _two_prod(double a, double b) noexcept nogil:
    """Exact a * b by Veltkamp splitting."""
    cdef dd_t r
    cdef double t, a1, a2, b1, b2
    r.hi = a * b
    t = 134217729.0 * a
    a1 = t - (t - a)
    a2 = a - a1
    t = 134217729.0 * b
    b1 = t - (t - b)
    b2 = b - b1
    r.lo = ((a1 * b1 - r.hi) + a1 * b2 + a2 * b1) + a2 * b2
    return r


cdef inline dd_t dd_add(dd_t x, dd_t y) noexcept nogil:
    cdef dd_t s = _two_sum(x.hi, y.hi)
    cdef dd_t t = _two_sum(x.lo, y.lo)
    cdef dd_t v = _fast_two_sum(s.hi, s.lo + t.hi)
    return _fast_two_sum(v.hi, t.lo + v.lo)


cdef inline dd_t dd_add_d(dd_t x, double y) noexcept nogil:
    cdef dd_t s = _two_sum(x.hi, y)
    return _fast_two_sum(s.hi, x.lo + s.lo)


cdef inline dd_t dd_neg(dd_t x) noexcept nogil:
    cdef dd_t r
    r.hi = -x.hi
    r.lo = -x.lo
    return r


cdef inline dd_t dd_mul(dd_t x, dd_t y) noexcept nogil:
    cdef dd_t c = _two_prod(x.hi, y.hi)
    return _fast_two_sum(c.hi, c.lo + (x.hi * y.lo + x.lo * y.hi))


cdef inline dd_t dd_mul_d(dd_t x, double y) noexcept nogil:
    cdef dd_t c = _two_prod(x.hi, y)
    return _fast_two_sum(c.hi, c.lo + x.lo * y)


@cython.cdivision(True)
cdef inline dd_t dd_div(dd_t x, dd_t y) noexcept nogil:
    cdef double q1, q2, q3
    cdef dd_t r
    q1 = x.hi / y.hi
    r = dd_add(x, dd_neg(dd_mul_d(y, q1)))
    q2 = r.hi / y.hi
    r = dd_add(r, dd_neg(dd_mul_d(y, q2)))
    q3 = r.hi / y.hi
    return dd_add_d(_fast_two_sum(q1, q2), q3)


@cython.cdivision(True)
cdef inline dd_t dd_sqrt(dd_t x) noexcept nogil:
    """Square root of a positive value by one Heron correction."""
    cdef double root = sqrt(x.hi)
    cdef dd_t residual = dd_add(x, dd_neg(_two_prod(root, root)))
    return _fast_two_sum(root, residual.hi / (2.0 * root))


def adams_accumulate(
    double step,
    const double[:] state,
    const double[:] state_low,
    const double[:, :] derivative,
    const double[:, :] derivative_low,
    const double[:] weight,
    const double[:] weight_low,
):
    """Return ``y + step * sum_j weight[j] * f[:, j]`` in double-double arithmetic.

    Parameters
    ----------
    step : float
        Signed step in s.
    state, state_low : numpy.ndarray
        High and low parts of ``y``, shape ``(d,)``.
    derivative, derivative_low : numpy.ndarray
        High and low parts of ``f``, shape ``(d, m)``.
    weight, weight_low : numpy.ndarray
        High and low parts of the weights, shape ``(m,)``.

    Returns
    -------
    tuple of numpy.ndarray
        High and low parts of the result, each of shape ``(d,)``.
    """
    cdef Py_ssize_t dimension = state.shape[0]
    cdef Py_ssize_t count = weight.shape[0]
    if (
        state_low.shape[0] != dimension
        or derivative.shape[0] != dimension
        or derivative_low.shape[0] != dimension
        or derivative.shape[1] != count
        or derivative_low.shape[1] != count
        or weight_low.shape[0] != count
    ):
        raise ValueError("Inconsistent shapes in adams_accumulate")
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result_array = np.empty(dimension, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result_low_array = np.empty(dimension, dtype=np.float64)
    cdef double[::1] result = result_array
    cdef double[::1] result_low = result_low_array
    cdef Py_ssize_t i, j
    cdef dd_t total, term, node, beta
    with nogil:
        for i in range(dimension):
            total.hi = 0.0
            total.lo = 0.0
            for j in range(count):
                beta.hi = weight[j]
                beta.lo = weight_low[j]
                node.hi = derivative[i, j]
                node.lo = derivative_low[i, j]
                term = dd_mul(beta, node)
                total = dd_add(total, term)
            node.hi = state[i]
            node.lo = state_low[i]
            total = dd_add(node, dd_mul_d(total, step))
            result[i] = total.hi
            result_low[i] = total.lo
    return result_array, result_low_array
