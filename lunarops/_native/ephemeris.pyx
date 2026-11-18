# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Chebyshev evaluation of SPK (types 2, 3, 20) and binary PCK (types 2, 20) segments.

Records arrive normalised by :func:`lunarops.fileio.daf.chebyshev_layout`.

Epochs are passed as two parts, ``a`` (whole seconds, exact) and ``f`` (a fraction of at most a few days, in
seconds), both TDB seconds past J2000, so that the record offset ``(a - MID) + f`` keeps sub-nanosecond
resolution over the whole kernel span. A :class:`ChebyshevLink` holds the segments of one (target,
centre) pair; coverage is pre-partitioned into elementary intervals, each owned by the highest-priority
segment covering it. :class:`StatePlan` flattens the links of several bodies so that
:func:`evaluate_states` runs them in one ``nogil`` pass. All evaluation is read-only and ``nogil``. The
C-level declarations are in ``ephemeris.pxd``.
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


cdef class ChebyshevLink:
    """Chebyshev segments of one (target, centre) pair with their priority-resolved coverage.

    Parameters
    ----------
    target, center : int
        NAIF IDs of the pair (for an orientation link, the frame class ID and the reference frame).
    records : list of ndarray
        Normalised record matrix of each segment (see :class:`lunarops.fileio.daf.ChebyshevLayout`), shape
        ``(record_count, 2 + 3 * (position_count + velocity_count))``, float64.
    init_s, interval_s : sequence of float
        ``INIT`` and ``INTLEN`` of each segment, in seconds.
    position_counts, velocity_counts : sequence of int
        Chebyshev coefficients per position and per velocity component of each segment; a velocity count
        of 0 takes the velocity as the derivative of the position expansion.
    bounds_s : ndarray, shape (n + 1,)
        Increasing boundaries of the elementary coverage intervals, TDB seconds past J2000.
    owners : ndarray of int32, shape (n,)
        Segment owning each interval, -1 for a gap.
    priorities : ndarray of int32, shape (n,)
        Priority of each interval's owner (larger wins at a shared boundary).
    scale : float
        Factor applied to every output (1000 converts km to m).
    """

    def __cinit__(self):
        self.data.segments = NULL
        self.data.segment_count = 0
        self.data.interval_count = 0

    def __init__(self, int target, int center, list records, init_s, interval_s, position_counts,
                 velocity_counts, cnp.ndarray bounds_s, cnp.ndarray owners, cnp.ndarray priorities, double scale):
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
        self.data.segments = <Segment*> malloc(max(n, 1) * sizeof(Segment))
        if self.data.segments == NULL:
            raise MemoryError()
        self.data.segment_count = n
        for i in range(n):
            matrix = arrays[i]
            n_position = int(position_counts[i])
            n_velocity = int(velocity_counts[i])
            if (
                n_position < 1 or n_position > MAX_COEFFICIENTS or n_velocity < 0 or n_velocity > MAX_COEFFICIENTS
                or matrix.shape[1] != 2 + 3 * (n_position + n_velocity) or matrix.shape[0] < 1
            ):
                raise ValueError(f"unsupported record layout for segment {i}")
            if float(interval_s[i]) <= 0.0:
                raise ValueError(f"non-positive record interval for segment {i}")
            self.data.segments[i].records = &matrix[0, 0]
            self.data.segments[i].init_s = float(init_s[i])
            self.data.segments[i].interval_s = float(interval_s[i])
            self.data.segments[i].record_size = matrix.shape[1]
            self.data.segments[i].record_count = matrix.shape[0]
            self.data.segments[i].position_count = n_position
            self.data.segments[i].velocity_count = n_velocity
        self.data.interval_count = owner_array.shape[0]
        self.data.bounds = <const double*> cnp.PyArray_DATA(bounds)
        self.data.owners = <const int*> cnp.PyArray_DATA(owner_array)
        self.data.priorities = <const int*> cnp.PyArray_DATA(priority_array)
        self.data.scale = scale
        self.target = target
        self.center = center
        # The record pointers and interval tables point into these arrays (memory maps included).
        self._keep = (arrays, bounds, owner_array, priority_array)

    def __dealloc__(self):
        free(self.data.segments)

    @property
    def coverage_s(self):
        """First and last covered epoch, TDB seconds past J2000."""
        return float(self.data.bounds[0]), float(self.data.bounds[self.data.interval_count])

    cdef int evaluate(self, double a, double f, int order, Py_ssize_t* hint, double sign,
                      double* out) noexcept nogil:
        """Add ``sign`` times the state (order 0..2) at epoch ``a + f`` to ``out``; return -1 without coverage."""
        return link_evaluate(&self.data, a, f, order, hint, sign, out)


cdef inline Py_ssize_t _link_interval(const LinkData* link, double t, Py_ssize_t hint) noexcept nogil:
    # Highest-priority interval containing t (both ends included), or -1.
    cdef const double* b = link.bounds
    cdef const int* owners = link.owners
    cdef Py_ssize_t n = link.interval_count, lo, hi, mid, j
    if 0 <= hint < n and b[hint] < t < b[hint + 1] and owners[hint] >= 0:
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
    if t == b[j] and j > 0 and owners[j - 1] >= 0:
        if owners[j] < 0 or link.priorities[j - 1] > link.priorities[j]:
            j -= 1
    if owners[j] < 0:
        return -1
    return j


cdef int link_evaluate(const LinkData* link, double a, double f, int order, Py_ssize_t* hint, double sign,
                       double* out) noexcept nogil:
    cdef Py_ssize_t j = _link_interval(link, a + f, hint[0]), record, k
    cdef const Segment* s
    cdef const double* c
    cdef const double* cv
    cdef double t[MAX_COEFFICIENTS]
    cdef double d[MAX_COEFFICIENTS]
    cdef double dd[MAX_COEFFICIENTS]
    cdef double tau, two_tau, radius, value, scale
    cdef int n, nv, nt, comp, need_d, need_dd
    if j < 0:
        return -1
    hint[0] = j
    s = &link.segments[link.owners[j]]
    n = s.position_count
    nv = s.velocity_count
    record = <Py_ssize_t> floor(((a - s.init_s) + f) / s.interval_s)
    if record < 0:
        record = 0
    elif record >= s.record_count:
        record = s.record_count - 1  # the last record includes the segment end (SPKR02)
    c = s.records + record * s.record_size
    cv = c + 2 + 3 * n
    radius = c[1]
    tau = ((a - c[0]) + f) / radius
    two_tau = 2.0 * tau
    nt = n if n > nv else nv
    if nv == 0:  # velocity and acceleration from the position expansion
        need_d = order >= 1
        need_dd = order >= 2
    else:  # velocity from its own expansion, acceleration from its derivative
        need_d = order >= 2
        need_dd = 0
    t[0] = 1.0
    if nt > 1:
        t[1] = tau
    for k in range(2, nt):
        t[k] = two_tau * t[k - 1] - t[k - 2]
    if need_d:
        d[0] = 0.0
        if nt > 1:
            d[1] = 1.0
        for k in range(2, nt):
            d[k] = 2.0 * t[k - 1] + two_tau * d[k - 1] - d[k - 2]
    if need_dd:
        dd[0] = 0.0
        if nt > 1:
            dd[1] = 0.0
        for k in range(2, nt):
            dd[k] = 4.0 * d[k - 1] + two_tau * dd[k - 1] - dd[k - 2]
    scale = sign * link.scale
    for comp in range(3):
        # Sum from the highest degree down: the small terms are accumulated first.
        value = 0.0
        for k in range(n - 1, -1, -1):
            value += c[2 + comp * n + k] * t[k]
        out[comp] += scale * value
        if order >= 1:
            value = 0.0
            if nv == 0:
                for k in range(n - 1, 0, -1):
                    value += c[2 + comp * n + k] * d[k]
                value /= radius
            else:
                for k in range(nv - 1, -1, -1):
                    value += cv[comp * nv + k] * t[k]
            out[3 + comp] += scale * value
        if order >= 2:
            value = 0.0
            if nv == 0:
                for k in range(n - 1, 1, -1):
                    value += c[2 + comp * n + k] * dd[k]
                value /= radius * radius
            else:
                for k in range(nv - 1, 0, -1):
                    value += cv[comp * nv + k] * d[k]
                value /= radius
            out[6 + comp] += scale * value
    return 0


cdef class StatePlan:
    """The link plans of several bodies, flattened for :func:`evaluate_states`.

    Parameters
    ----------
    plans : sequence of sequence of (ChebyshevLink, float)
        For each body, the links whose signed states add up to the body state.
    """

    cdef const LinkData** _links
    cdef double* _signs
    cdef Py_ssize_t* _bodies
    cdef Py_ssize_t _entry_count
    cdef readonly Py_ssize_t body_count
    cdef list _link_objects

    def __cinit__(self):
        self._links = NULL
        self._signs = NULL
        self._bodies = NULL
        self._entry_count = 0

    def __init__(self, plans):
        cdef ChebyshevLink link
        cdef Py_ssize_t body, entry = 0
        items = [(body, item) for body, plan in enumerate(plans) for item in plan]
        self.body_count = len(plans)
        self._entry_count = len(items)
        self._links = <const LinkData**> malloc(max(self._entry_count, 1) * sizeof(LinkData*))
        self._signs = <double*> malloc(max(self._entry_count, 1) * sizeof(double))
        self._bodies = <Py_ssize_t*> malloc(max(self._entry_count, 1) * sizeof(Py_ssize_t))
        if self._links == NULL or self._signs == NULL or self._bodies == NULL:
            raise MemoryError()
        self._link_objects = []
        for body, item in items:
            link = <ChebyshevLink?> item[0]
            self._link_objects.append(link)  # keeps the LinkData of every entry alive
            self._links[entry] = &link.data
            self._signs[entry] = float(item[1])
            self._bodies[entry] = body
            entry += 1

    def __dealloc__(self):
        free(self._links)
        free(self._signs)
        free(self._bodies)


def evaluate_states(plans, const double[::1] a, const double[::1] f, int order):
    """Return the summed link states of several bodies at several epochs.

    Parameters
    ----------
    plans : StatePlan or list of list of (ChebyshevLink, float)
        For each body, the links whose signed states add up to the body state; pass a :class:`StatePlan`
        to reuse the flattened form across calls.
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
    cdef StatePlan plan = plans if isinstance(plans, StatePlan) else StatePlan(plans)
    cdef Py_ssize_t n_epochs = a.shape[0], n_bodies = plan.body_count, width = 3 * (order + 1)
    cdef Py_ssize_t entry, epoch, failed = -1, failed_entry = -1, hint, row = n_bodies * width
    cdef double* base
    cdef ChebyshevLink link
    result = np.zeros((n_epochs, n_bodies, width), dtype=np.float64)
    cdef double[:, :, ::1] out = result
    if n_epochs == 0 or n_bodies == 0:
        return result
    base = &out[0, 0, 0]
    with nogil:
        # Entries run body by body, each body's links in plan order: the order of the sums is fixed.
        for entry in range(plan._entry_count):
            hint = 0
            for epoch in range(n_epochs):
                if link_evaluate(
                    plan._links[entry], a[epoch], f[epoch], order, &hint, plan._signs[entry],
                    base + epoch * row + plan._bodies[entry] * width,
                ) != 0:
                    failed = epoch
                    break
            if failed >= 0:
                failed_entry = entry
                break
    if failed >= 0:
        link = plan._link_objects[failed_entry]
        first, last = link.coverage_s
        raise EphemerisCoverageError(
            f"epoch {a[failed] + f[failed]!r} s TDB past J2000 is outside the coverage "
            f"[{first!r}, {last!r}] s of NAIF {link.target} relative to {link.center}",
            plan._bodies[failed_entry], failed, link.target, link.center,
        )
    return result
