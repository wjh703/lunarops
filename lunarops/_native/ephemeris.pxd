# cython: language_level=3
"""C-level interface of the Chebyshev ephemeris kernel.

Other Cython modules can ``cimport`` :class:`ChebyshevLink` and call :meth:`ChebyshevLink.evaluate` (or
:func:`link_evaluate` on its ``data``) without the GIL, for example from a compiled right-hand side. A link
is read-only after construction, so several threads may evaluate it at once, each with its own ``hint``.
"""

cimport cython


ctypedef struct Segment:
    const double* records  # normalised records [MID, RADIUS, position coefficients, velocity coefficients]
    double init_s
    double interval_s
    Py_ssize_t record_size
    Py_ssize_t record_count
    int position_count
    int velocity_count  # 0: the velocity is the derivative of the position expansion


ctypedef struct LinkData:
    Segment* segments
    Py_ssize_t segment_count
    const double* bounds
    const int* owners
    const int* priorities
    Py_ssize_t interval_count
    double scale


# Add sign times the position (order 0), velocity (order >= 1) and acceleration (order 2) at the epoch
# a + f (TDB seconds past J2000, a whole seconds) to out[0:3], out[3:6], out[6:9]. Return 0, or -1 when
# the epoch lies outside the link's coverage. hint carries the last interval between calls.
cdef int link_evaluate(const LinkData* link, double a, double f, int order, Py_ssize_t* hint, double sign,
                       double* out) noexcept nogil


@cython.final
cdef class ChebyshevLink:
    cdef LinkData data
    cdef readonly int target
    cdef readonly int center
    cdef object _keep

    # link_evaluate on this link's data.
    cdef int evaluate(self, double a, double f, int order, Py_ssize_t* hint, double sign,
                      double* out) noexcept nogil
