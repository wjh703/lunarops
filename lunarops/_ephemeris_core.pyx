# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Chebyshev evaluation of SPK type 2 segments for :mod:`lunarops.classes.ephemerides.chebyshev`.

Epochs are passed as two parts, ``a`` (whole seconds, exact) and ``f`` (a fraction of at most a few days, in
seconds), both TDB seconds past J2000, so that the record offset ``(a - MID) + f`` keeps sub-nanosecond
resolution over the whole kernel span. A :class:`ChebyshevLink` holds the segments of one (target,
centre) pair; coverage is pre-partitioned into elementary intervals, each owned by the highest-priority
segment covering it. All evaluation is read-only and ``nogil``.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport floor
from libc.stdlib cimport free, malloc

cnp.import_array()

cdef enum:
    MAX_COEFFICIENTS = 64


class EphemerisCoverageError(ValueError):
    """An epoch lies outside the coverage of an ephemeris link.

    Attributes
    ----------
    body_index : int
        Position of the body in the requested body list.
    epoch_index : int
        Position of the epoch in the requested epoch arrays.
    target, center : int
        NAIF IDs of the link without coverage.
    """

    def __init__(self, message, body_index, epoch_index, target, center):
        super().__init__(message)
        self.body_index = body_index
        self.epoch_index = epoch_index
        self.target = target
        self.center = center


ctypedef struct Segment:
    const double* records
    double init_s
    double interval_s
    Py_ssize_t record_size
    Py_ssize_t record_count
    int coefficient_count


cdef class ChebyshevLink:
    """Type 2 segments of one (target, centre) pair with their priority-resolved coverage.

    Parameters
    ----------
    target, center : int
        NAIF IDs of the pair.
    records : list of ndarray
        Record matrix of each segment, shape ``(record_count, record_size)``, C-contiguous float64.
    init_s, interval_s : sequence of float
        ``INIT`` and ``INTLEN`` of each segment, in seconds.
    coefficient_counts : sequence of int
        Chebyshev coefficients per component of each segment.
    bounds_s : ndarray, shape (n + 1,)
        Increasing boundaries of the elementary coverage intervals, TDB seconds past J2000.
    owners : ndarray of int32, shape (n,)
        Segment owning each interval, -1 for a gap.
    priorities : ndarray of int32, shape (n,)
        Priority of each interval's owner (larger wins at a shared boundary).
    scale : float
        Factor applied to every output (1000 converts km to m).
    """

    cdef Segment* _segments
    cdef Py_ssize_t _segment_count
    cdef const double* _bounds
    cdef const int* _owners
    cdef const int* _priorities
    cdef Py_ssize_t _interval_count
    cdef double _scale
    cdef readonly int target
    cdef readonly int center
    cdef object _keep

    def __cinit__(self):
        self._segments = NULL

    def __init__(self, int target, int center, list records, init_s, interval_s, coefficient_counts,
                 cnp.ndarray bounds_s, cnp.ndarray owners, cnp.ndarray priorities, double scale):
        cdef Py_ssize_t n = len(records), i
        cdef const double[:, ::1] matrix
        bounds = np.ascontiguousarray(bounds_s, dtype=np.float64)
        owner_array = np.ascontiguousarray(owners, dtype=np.int32)
        priority_array = np.ascontiguousarray(priorities, dtype=np.int32)
        if bounds.ndim != 1 or bounds.shape[0] < 2 or owner_array.shape[0] != bounds.shape[0] - 1:
            raise ValueError("bounds_s must have one more entry than owners")
        if priority_array.shape[0] != owner_array.shape[0]:
            raise ValueError("priorities and owners must have the same length")
        if np.any(np.diff(bounds) <= 0.0):
            raise ValueError("bounds_s must be strictly increasing")
        if np.any((owner_array < -1) | (owner_array >= n)):
            raise ValueError("owner index out of range")
        arrays = [np.ascontiguousarray(matrix_in, dtype=np.float64) for matrix_in in records]
        self._segments = <Segment*> malloc(max(n, 1) * sizeof(Segment))
        if self._segments == NULL:
            raise MemoryError()
        self._segment_count = n
        for i in range(n):
            matrix = arrays[i]
            k = int(coefficient_counts[i])
            if k < 1 or k > MAX_COEFFICIENTS or matrix.shape[1] != 2 + 3 * k or matrix.shape[0] < 1:
                raise ValueError(f"unsupported record layout for segment {i}")
            self._segments[i].records = &matrix[0, 0]
            self._segments[i].init_s = float(init_s[i])
            self._segments[i].interval_s = float(interval_s[i])
            self._segments[i].record_size = matrix.shape[1]
            self._segments[i].record_count = matrix.shape[0]
            self._segments[i].coefficient_count = k
        self._interval_count = owner_array.shape[0]
        self._bounds = <const double*> cnp.PyArray_DATA(bounds)
        self._owners = <const int*> cnp.PyArray_DATA(owner_array)
        self._priorities = <const int*> cnp.PyArray_DATA(priority_array)
        self._scale = scale
        self.target = target
        self.center = center
        # The record pointers and interval tables point into these arrays (memory maps included).
        self._keep = (arrays, bounds, owner_array, priority_array)

    def __dealloc__(self):
        free(self._segments)

    @property
    def coverage_s(self):
        """First and last covered epoch, TDB seconds past J2000."""
        return float(self._bounds[0]), float(self._bounds[self._interval_count])

    cdef Py_ssize_t _interval(self, double t, Py_ssize_t hint) noexcept nogil:
        # Highest-priority interval containing t (both ends included), or -1.
        cdef const double* b = self._bounds
        cdef Py_ssize_t n = self._interval_count, lo, hi, mid, j
        if 0 <= hint < n and b[hint] < t < b[hint + 1] and self._owners[hint] >= 0:
            return hint
        if t < b[0] or t > b[n]:
            return -1
        lo = 0
        hi = n - 1
        while lo < hi:  # largest j with b[j] <= t
            mid = (lo + hi + 1) // 2
            if b[mid] <= t:
                lo = mid
            else:
                hi = mid - 1
        j = lo
        if t == b[j] and j > 0 and self._owners[j - 1] >= 0:
            if self._owners[j] < 0 or self._priorities[j - 1] > self._priorities[j]:
                j -= 1
        if self._owners[j] < 0:
            return -1
        return j

    cdef int evaluate(self, double a, double f, int order, Py_ssize_t* hint, double sign,
                      double* out) noexcept nogil:
        """Add ``sign`` times the state (order 0..2) at epoch ``a + f`` to ``out``; return -1 without coverage."""
        cdef Py_ssize_t j = self._interval(a + f, hint[0]), record, k
        cdef const Segment* s
        cdef const double* c
        cdef double t[MAX_COEFFICIENTS]
        cdef double d[MAX_COEFFICIENTS]
        cdef double dd[MAX_COEFFICIENTS]
        cdef double tau, two_tau, radius, value, scale
        cdef int n, comp
        if j < 0:
            return -1
        hint[0] = j
        s = &self._segments[self._owners[j]]
        n = s.coefficient_count
        record = <Py_ssize_t> floor(((a - s.init_s) + f) / s.interval_s)
        if record < 0:
            record = 0
        elif record >= s.record_count:
            record = s.record_count - 1  # the last record includes the segment end (SPKR02)
        c = s.records + record * s.record_size
        radius = c[1]
        tau = ((a - c[0]) + f) / radius
        two_tau = 2.0 * tau
        t[0] = 1.0
        if n > 1:
            t[1] = tau
        for k in range(2, n):
            t[k] = two_tau * t[k - 1] - t[k - 2]
        if order >= 1:
            d[0] = 0.0
            if n > 1:
                d[1] = 1.0
            for k in range(2, n):
                d[k] = 2.0 * t[k - 1] + two_tau * d[k - 1] - d[k - 2]
        if order >= 2:
            dd[0] = 0.0
            if n > 1:
                dd[1] = 0.0
            for k in range(2, n):
                dd[k] = 4.0 * d[k - 1] + two_tau * dd[k - 1] - dd[k - 2]
        scale = sign * self._scale
        for comp in range(3):
            # Sum from the highest degree down: the small terms are accumulated first.
            value = 0.0
            for k in range(n - 1, -1, -1):
                value += c[2 + comp * n + k] * t[k]
            out[comp] += scale * value
            if order >= 1:
                value = 0.0
                for k in range(n - 1, 0, -1):
                    value += c[2 + comp * n + k] * d[k]
                out[3 + comp] += scale * (value / radius)
            if order >= 2:
                value = 0.0
                for k in range(n - 1, 1, -1):
                    value += c[2 + comp * n + k] * dd[k]
                out[6 + comp] += scale * (value / (radius * radius))
        return 0


def evaluate_states(list plans, const double[::1] a, const double[::1] f, int order):
    """Return the summed link states of several bodies at several epochs.

    Parameters
    ----------
    plans : list of list of (ChebyshevLink, float)
        For each body, the links whose signed states add up to the body state.
    a, f : ndarray of float64, shape (n_epochs,)
        Two-part epochs: whole TDB seconds past J2000 and the remaining seconds.
    order : {0, 1, 2}
        Highest time derivative returned.

    Returns
    -------
    ndarray, shape (n_epochs, n_bodies, 3 * (order + 1))
        Positions, then velocities, then accelerations, in the links' scaled units.

    Raises
    ------
    EphemerisCoverageError
        If an epoch lies outside a link's coverage.
    """
    if order < 0 or order > 2:
        raise ValueError("order must be 0, 1 or 2")
    if a.shape[0] != f.shape[0]:
        raise ValueError("a and f must have the same length")
    cdef Py_ssize_t n_epochs = a.shape[0], n_bodies = len(plans), width = 3 * (order + 1)
    cdef Py_ssize_t body, epoch, failed
    cdef Py_ssize_t hint
    cdef ChebyshevLink link
    cdef double sign
    result = np.zeros((n_epochs, n_bodies, width), dtype=np.float64)
    cdef double[:, :, ::1] out = result
    for body in range(n_bodies):
        for item in plans[body]:
            link = <ChebyshevLink> item[0]
            sign = float(item[1])
            hint = 0
            failed = -1
            with nogil:
                for epoch in range(n_epochs):
                    if link.evaluate(a[epoch], f[epoch], order, &hint, sign, &out[epoch, body, 0]) != 0:
                        failed = epoch
                        break
            if failed >= 0:
                first, last = link.coverage_s
                raise EphemerisCoverageError(
                    f"epoch {a[failed] + f[failed]!r} s TDB past J2000 is outside the coverage "
                    f"[{first!r}, {last!r}] s of NAIF {link.target} relative to {link.center}",
                    body, failed, link.target, link.center,
                )
    return result
