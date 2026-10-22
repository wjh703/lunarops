# Fused lunar right-hand side, included by _dynamics_core.pyx.
#
# One call evaluates the Moon-relative translation (point mass, EIH, solar J2,
# Earth/Moon figure, Earth tide, Lense-Thirring, solar radiation pressure) and,
# optionally, the mantle/core rotation (figure torques, Earth J2 torque,
# geodetic precession, core-mantle coupling). Every formula follows the Python
# reference implementation in lunarops/classes/dynamics term by term, including
# the order of operations, so the two paths agree to rounding.

from libc.math cimport fabs, isfinite


cdef inline void _cross3(const double* a, const double* b, double* out) noexcept nogil:
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]


cdef inline double _dot3(const double* a, const double* b) noexcept nogil:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


cdef inline void _matvec(const double* m, const double* v, double* out) noexcept nogil:
    out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2]
    out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2]
    out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2]


cdef inline void _mat_t_vec(const double* m, const double* v, double* out) noexcept nogil:
    out[0] = m[0] * v[0] + m[3] * v[1] + m[6] * v[2]
    out[1] = m[1] * v[0] + m[4] * v[1] + m[7] * v[2]
    out[2] = m[2] * v[0] + m[5] * v[1] + m[8] * v[2]


@cython.cdivision(True)
cdef bint _quaternion_matrix(const double* q, double* r) noexcept nogil:
    """Active rotation of a scalar-first Hamilton quaternion, normalized here only."""
    cdef double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    cdef double a, x, y, z
    if norm == 0.0 or not isfinite(norm):
        return True
    a = q[0] / norm
    x = q[1] / norm
    y = q[2] / norm
    z = q[3] / norm
    r[0] = 1 - 2 * (y * y + z * z)
    r[1] = 2 * (x * y - a * z)
    r[2] = 2 * (x * z + a * y)
    r[3] = 2 * (x * y + a * z)
    r[4] = 1 - 2 * (x * x + z * z)
    r[5] = 2 * (y * z - a * x)
    r[6] = 2 * (x * z - a * y)
    r[7] = 2 * (y * z + a * x)
    r[8] = 1 - 2 * (x * x + y * y)
    return False


@cython.cdivision(True)
cdef bint _solve3(const double* a_in, const double* b_in, double* x) noexcept nogil:
    """Solve a 3x3 system by Gaussian elimination with partial pivoting; True if singular."""
    cdef double a[9]
    cdef double b[3]
    cdef Py_ssize_t i, j, k, pivot
    cdef double factor, swap, best
    for i in range(9):
        a[i] = a_in[i]
    for i in range(3):
        b[i] = b_in[i]
    for k in range(3):
        pivot = k
        best = fabs(a[3 * k + k])
        for i in range(k + 1, 3):
            if fabs(a[3 * i + k]) > best:
                best = fabs(a[3 * i + k])
                pivot = i
        if best == 0.0:
            return True
        if pivot != k:
            for j in range(3):
                swap = a[3 * k + j]
                a[3 * k + j] = a[3 * pivot + j]
                a[3 * pivot + j] = swap
            swap = b[k]
            b[k] = b[pivot]
            b[pivot] = swap
        for i in range(k + 1, 3):
            factor = a[3 * i + k] / a[3 * k + k]
            for j in range(k, 3):
                a[3 * i + j] -= factor * a[3 * k + j]
            b[i] -= factor * b[k]
    for i in range(2, -1, -1):
        swap = b[i]
        for j in range(i + 1, 3):
            swap -= a[3 * i + j] * x[j]
        x[i] = swap / a[3 * i + i]
    return False


cdef class LunarRhsKernel:
    """Fused Moon-relative translation and optional mantle/core rotation derivative.

    ``set_epoch`` loads the prescribed bodies of one epoch, ``set_step`` loads the
    candidate-independent delayed data of one ABM step, and ``derivative`` then
    evaluates any number of candidate states without touching the GIL.

    The kernel owns its workspaces, so one instance must not be called from several
    threads at once; use one kernel per thread or process.
    """

    cdef Py_ssize_t n, sun, n_targets, n_force, n_torque, n_raisers
    cdef Py_ssize_t deg_earth, deg_moon, deg_sun
    cdef bint has_rotation, has_core, geodetic
    cdef bint use_pm, use_eih, use_solar_j2, use_earth_figure, use_moon_figure, use_tide, use_lt, use_srp
    cdef double c2
    # spherical-harmonic fields
    cdef double[::1] earth_coeffs, earth_norm, moon_coeffs, moon_norm, sun_coeffs, sun_norm
    cdef double earth_field_gm, earth_field_radius, moon_field_gm, moon_field_radius
    cdef double sun_field_gm, sun_field_radius, moon_radius2
    cdef double sun_matrix[9]
    cdef Py_ssize_t earth_c20_index
    # Moon figure targets
    cdef Py_ssize_t[::1] targets, force_slots, torque_kind, torque_arg
    # tide, Lense-Thirring, radiation pressure
    cdef double tide_mu_earth, tide_mu_moon, tide_radius, k20, k21, k22, tide_scale
    cdef double lt_angular_momentum[3]
    cdef double lt_gamma, lt_gm, srp_epsilon, srp_gm
    # rotation constants
    cdef double core_diagonal[3]
    cdef double core_viscosity
    cdef double torque_earth_radius, torque_earth_j2
    # per-epoch buffers
    cdef double[::1] pos, vel, gm, ext_acc, ext_pot, nacc, npot
    cdef double earth_matrix[9]
    cdef double epoch_moon_matrix[9]
    cdef double earth_eph_acc[3]
    # per-step buffers
    cdef double[::1] tide_vectors, raiser_gm
    cdef double inertia[9]
    cdef double inertia_dot[9]
    # workspaces
    cdef double[::1] raw, draw, cos_m, sin_m, rel_all, a_all

    def __init__(self, dict cfg):
        cdef Py_ssize_t i, max_degree, stride
        self.n = cfg["n_bodies"]
        self.sun = cfg["sun_index"]
        self.c2 = cfg["speed_of_light_squared"]
        self.use_pm = cfg["point_mass"]
        self.use_eih = cfg["eih"]
        self.use_solar_j2 = cfg["solar_j2"] is not None
        self.use_earth_figure = cfg["earth_figure"] is not None
        self.use_moon_figure = cfg["moon_figure_slots"] is not None
        self.use_tide = cfg["tide"] is not None
        self.use_lt = cfg["lense_thirring"] is not None
        self.use_srp = cfg["radiation_pressure"] is not None
        self.has_rotation = cfg["rotation"] is not None
        max_degree = 1
        # Spherical-harmonic fields: (coefficients, norm, field gm, field radius)
        empty = np.zeros(2, dtype=np.float64)
        self.earth_coeffs = self.earth_norm = self.moon_coeffs = self.moon_norm = empty
        self.sun_coeffs = self.sun_norm = empty
        self.deg_earth = self.deg_moon = self.deg_sun = 0
        if self.use_earth_figure:
            f = cfg["earth_figure"]
            self.earth_coeffs = np.ascontiguousarray(f["coefficients"], dtype=np.float64).ravel().copy()
            self.earth_norm = np.ascontiguousarray(f["norm"], dtype=np.float64).ravel().copy()
            self.earth_field_gm = f["gm"]
            self.earth_field_radius = f["radius"]
            self.deg_earth = f["degree"]
            self.earth_c20_index = 2 * (self.deg_earth + 1)
            max_degree = max(max_degree, self.deg_earth)
        if self.use_solar_j2:
            f = cfg["solar_j2"]
            self.sun_coeffs = np.ascontiguousarray(f["coefficients"], dtype=np.float64).ravel().copy()
            self.sun_norm = np.ascontiguousarray(f["norm"], dtype=np.float64).ravel().copy()
            self.sun_field_gm = f["gm"]
            self.sun_field_radius = f["radius"]
            self.deg_sun = f["degree"]
            sun_matrix = np.ascontiguousarray(f["matrix"], dtype=np.float64).ravel()
            for i in range(9):
                self.sun_matrix[i] = sun_matrix[i]
            max_degree = max(max_degree, self.deg_sun)
        targets = np.ascontiguousarray(cfg["moon_targets"], dtype=np.intp)
        self.targets = targets if len(targets) else np.zeros(1, dtype=np.intp)
        self.n_targets = len(targets)
        slots = cfg["moon_figure_slots"]
        force_slots = np.ascontiguousarray([] if slots is None else slots, dtype=np.intp)
        self.force_slots = force_slots if len(force_slots) else np.zeros(1, dtype=np.intp)
        self.n_force = len(force_slots)
        torque_kind = np.ascontiguousarray(cfg["torque_kind"], dtype=np.intp)
        torque_arg = np.ascontiguousarray(cfg["torque_arg"], dtype=np.intp)
        self.torque_kind = torque_kind if len(torque_kind) else np.zeros(1, dtype=np.intp)
        self.torque_arg = torque_arg if len(torque_arg) else np.zeros(1, dtype=np.intp)
        self.n_torque = len(torque_kind)
        if self.n_targets:
            f = cfg["moon_field"]
            self.moon_coeffs = np.ascontiguousarray(f["coefficients"], dtype=np.float64).ravel().copy()
            self.moon_norm = np.ascontiguousarray(f["norm"], dtype=np.float64).ravel().copy()
            self.moon_field_gm = f["gm"]
            self.moon_field_radius = f["radius"]
            self.moon_radius2 = f["radius"] ** 2
            self.deg_moon = f["degree"]
            max_degree = max(max_degree, self.deg_moon)
        if self.use_tide:
            t = cfg["tide"]
            self.tide_mu_earth = t["mu_earth"]
            self.tide_mu_moon = t["mu_moon"]
            self.tide_radius = t["radius"]
            self.k20 = t["k20"]
            self.k21 = t["k21"]
            self.k22 = t["k22"]
            self.tide_scale = t["acceleration_scale"]
            self.n_raisers = t["raisers"]
        if self.use_lt:
            t = cfg["lense_thirring"]
            for i in range(3):
                self.lt_angular_momentum[i] = t["angular_momentum"][i]
            self.lt_gamma = t["gamma"]
            self.lt_gm = t["gm"]
        if self.use_srp:
            t = cfg["radiation_pressure"]
            self.srp_epsilon = t["epsilon"]
            self.srp_gm = t["gm"]
        if self.has_rotation:
            r = cfg["rotation"]
            self.has_core = r["core"] is not None
            self.geodetic = r["geodetic"]
            if self.has_core:
                for i in range(3):
                    self.core_diagonal[i] = r["core"]["diagonal"][i]
                self.core_viscosity = r["core"]["viscosity"]
            if r["earth_torque_radius"] is not None:
                self.torque_earth_radius = r["earth_torque_radius"]
        stride = max_degree + 1
        self.raw = np.zeros(stride * stride, dtype=np.float64)
        self.draw = np.zeros(stride * stride, dtype=np.float64)
        self.cos_m = np.zeros(stride, dtype=np.float64)
        self.sin_m = np.zeros(stride, dtype=np.float64)
        self.rel_all = np.zeros(3 * max(self.n_targets, 1), dtype=np.float64)
        self.a_all = np.zeros(3 * max(self.n_targets, 1), dtype=np.float64)
        self.pos = np.zeros(3 * self.n, dtype=np.float64)
        self.vel = np.zeros(3 * self.n, dtype=np.float64)
        self.gm = np.zeros(self.n, dtype=np.float64)
        self.ext_acc = np.zeros(3 * max(self.n - 2, 1), dtype=np.float64)
        self.ext_pot = np.zeros(max(self.n - 2, 1), dtype=np.float64)
        self.nacc = np.zeros(3 * self.n, dtype=np.float64)
        self.npot = np.zeros(self.n, dtype=np.float64)
        self.tide_vectors = np.zeros(9 * max(self.n_raisers, 1), dtype=np.float64)
        self.raiser_gm = np.zeros(max(self.n_raisers, 1), dtype=np.float64)

    @property
    def state_dimension(self):
        if not self.has_rotation:
            return 6
        return 16 if self.has_core else 13

    def set_epoch(
        self,
        const double[:, ::1] positions,
        const double[:, ::1] velocities,
        const double[::1] gravitational_parameters,
        const double[:, ::1] cache_accelerations,
        const double[::1] cache_potentials,
        const double[:, ::1] earth_fixed2inertial,
        const double[:, ::1] moon_fixed2inertial,
        const double[::1] earth_ephemeris_acceleration,
        double earth_c20,
        double earth_j2,
    ):
        cdef Py_ssize_t i, j
        if positions.shape[0] != self.n or velocities.shape[0] != self.n or gravitational_parameters.shape[0] != self.n:
            raise ValueError("Epoch arrays do not match the kernel body count")
        if cache_accelerations.shape[0] != self.n - 2 or cache_potentials.shape[0] != self.n - 2:
            raise ValueError("Point-mass cache does not match the kernel body count")
        for i in range(self.n):
            self.gm[i] = gravitational_parameters[i]
            for j in range(3):
                self.pos[3 * i + j] = positions[i, j]
                self.vel[3 * i + j] = velocities[i, j]
        for i in range(self.n - 2):
            self.ext_pot[i] = cache_potentials[i]
            for j in range(3):
                self.ext_acc[3 * i + j] = cache_accelerations[i, j]
        for i in range(3):
            self.earth_eph_acc[i] = earth_ephemeris_acceleration[i]
            for j in range(3):
                self.earth_matrix[3 * i + j] = earth_fixed2inertial[i, j]
                self.epoch_moon_matrix[3 * i + j] = moon_fixed2inertial[i, j]
        if self.use_earth_figure:
            self.earth_coeffs[self.earth_c20_index] = earth_c20
        self.torque_earth_j2 = earth_j2

    def set_step(
        self,
        const double[:, :, ::1] tide_vectors,
        const double[::1] raiser_gravitational_parameters,
        const double[:, ::1] inertia_total,
        const double[:, ::1] inertia_derivative,
        const double[:, :, ::1] moon_coefficients,
    ):
        cdef Py_ssize_t i, j, k, ell
        if self.use_tide:
            if tide_vectors.shape[0] != self.n_raisers:
                raise ValueError("Tide payload does not match the kernel raiser count")
            for i in range(self.n_raisers):
                self.raiser_gm[i] = raiser_gravitational_parameters[i]
                for j in range(3):
                    for k in range(3):
                        self.tide_vectors[9 * i + 3 * j + k] = tide_vectors[i, j, k]
        for i in range(3):
            for j in range(3):
                self.inertia[3 * i + j] = inertia_total[i, j]
                self.inertia_dot[3 * i + j] = inertia_derivative[i, j]
        if self.n_targets:
            k = 0
            for i in range(2):
                for j in range(moon_coefficients.shape[1]):
                    for ell in range(moon_coefficients.shape[2]):
                        self.moon_coeffs[k] = moon_coefficients[i, j, ell]
                        k += 1

    def derivative(self, const double[::1] y):
        cdef Py_ssize_t dimension = self.state_dimension
        cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(dimension, dtype=np.float64)
        cdef double[::1] out = result
        cdef int status
        if y.shape[0] != dimension:
            raise ValueError("State vector does not match the kernel state dimension")
        with nogil:
            status = self._eval(&y[0], &out[0])
        if status == 1:
            raise ValueError("A finite nonzero attitude quaternion is required")
        if status == 2:
            raise ValueError("Distinct bodies have coincident positions")
        if status == 3:
            raise ValueError("Earth-tide Earth-Moon or raiser position must be nonzero")
        if status == 4:
            raise ValueError("Mantle inertia is singular")
        return result

    @cython.cdivision(True)
    cdef int _eval(self, const double* y, double* out) noexcept nogil:
        cdef double* pos = &self.pos[0]
        cdef double* vel = &self.vel[0]
        cdef double* gm = &self.gm[0]
        cdef double* nacc = &self.nacc[0]
        cdef double* npot = &self.npot[0]
        cdef double* raw = &self.raw[0]
        cdef double* draw = &self.draw[0]
        cdef double* cos_m = &self.cos_m[0]
        cdef double* sin_m = &self.sin_m[0]
        cdef double* rel_all = &self.rel_all[0]
        cdef double* a_all = &self.a_all[0]
        cdef double rm[9]
        cdef double d[3]
        cdef double rel[3]
        cdef double a[3]
        cdef double tmp[3]
        cdef double tmp2[3]
        cdef double total[3]
        cdef double fsum[3]
        cdef double w
        cdef double scale, dist, dist2, dist3, coeff, dot
        cdef Py_ssize_t k, slot, body, fs, raiser
        cdef bint failed

        # Moon state: Earth plus the relative state.
        for k in range(3):
            pos[3 + k] = pos[k] + y[k]
            vel[3 + k] = vel[k] + y[3 + k]
        if self.has_rotation:
            if _quaternion_matrix(y + 6, rm):
                return 1
        else:
            for k in range(9):
                rm[k] = self.epoch_moon_matrix[k]

        # Newtonian point masses (also supply the EIH auxiliaries).
        total[0] = 0.0
        total[1] = 0.0
        total[2] = 0.0
        if self.use_pm:
            if _point_mass_cached_core(
                pos, gm, self.n, 2, &self.ext_acc[0], &self.ext_pot[0], nacc, npot
            ):
                return 2
            for k in range(3):
                total[k] = nacc[3 + k]
        if self.use_eih:
            if _eih_target(1, pos, vel, gm, npot, nacc, self.n, 1.0 / self.c2, tmp):
                return 2
            for k in range(3):
                total[k] += tmp[k]

        if self.use_solar_j2:
            for k in range(3):
                d[k] = pos[3 + k] - pos[3 * self.sun + k]
            _matvec(self.sun_matrix, d, rel)
            _sh_point(
                rel, &self.sun_coeffs[0], self.deg_sun, self.sun_field_gm, self.sun_field_radius,
                &self.sun_norm[0], raw, draw, cos_m, sin_m, a,
            )
            _mat_t_vec(self.sun_matrix, a, tmp)
            scale = gm[self.sun] / self.sun_field_gm
            for k in range(3):
                total[k] += scale * tmp[k]

        if self.use_earth_figure:
            for k in range(3):
                d[k] = pos[3 + k] - pos[k]
            _mat_t_vec(self.earth_matrix, d, rel)
            _sh_point(
                rel, &self.earth_coeffs[0], self.deg_earth, self.earth_field_gm, self.earth_field_radius,
                &self.earth_norm[0], raw, draw, cos_m, sin_m, a,
            )
            _matvec(self.earth_matrix, a, tmp)
            for k in range(3):
                total[k] += tmp[k]

        # Moon-fixed positions and direct accelerations of every figure partner and torque source.
        for slot in range(self.n_targets):
            body = self.targets[slot]
            for k in range(3):
                d[k] = pos[3 * body + k] - pos[3 + k]
            _mat_t_vec(rm, d, rel_all + 3 * slot)
            _sh_point(
                rel_all + 3 * slot, &self.moon_coeffs[0], self.deg_moon, self.moon_field_gm,
                self.moon_field_radius, &self.moon_norm[0], raw, draw, cos_m, sin_m, a_all + 3 * slot,
            )
        if self.use_moon_figure:
            fsum[0] = 0.0
            fsum[1] = 0.0
            fsum[2] = 0.0
            for fs in range(self.n_force):
                slot = self.force_slots[fs]
                body = self.targets[slot]
                _matvec(rm, a_all + 3 * slot, tmp)
                w = gm[body] / gm[1]
                for k in range(3):
                    fsum[k] += w * tmp[k]
            for k in range(3):
                total[k] += -fsum[k]

        if self.use_tide:
            for k in range(3):
                d[k] = pos[3 + k] - pos[k]
            _mat_t_vec(self.earth_matrix, d, rel)
            if _earth_tide_core(
                rel, &self.tide_vectors[0], &self.raiser_gm[0], self.n_raisers,
                self.tide_mu_earth, self.tide_mu_moon, self.tide_radius, self.k20, self.k21, self.k22, a,
            ):
                return 3
            _matvec(self.earth_matrix, a, tmp)
            for k in range(3):
                total[k] += self.tide_scale * tmp[k]

        if self.use_lt:
            for k in range(3):
                d[k] = pos[3 + k] - pos[3 * self.sun + k]
                rel[k] = vel[3 + k] - vel[3 * self.sun + k]
            dist = sqrt(_dot3(d, d))
            dist2 = pow(dist, 2.0)
            dist3 = pow(dist, 3.0)
            dot = _dot3(self.lt_angular_momentum, d)
            coeff = (1.0 + self.lt_gamma) / (2.0 * self.c2 * dist3)
            for k in range(3):
                tmp[k] = coeff * (-self.lt_angular_momentum[k] + 3.0 * dot * d[k] / dist2)
            _cross3(tmp, rel, a)
            scale = gm[self.sun] / self.lt_gm
            for k in range(3):
                total[k] += (2.0 * a[k]) * scale

        if self.use_srp:
            for k in range(3):
                d[k] = pos[3 + k] - pos[3 * self.sun + k]
            dist = sqrt(_dot3(d, d))
            dist3 = pow(dist, 3.0)
            scale = gm[self.sun] / self.srp_gm
            for k in range(3):
                total[k] += (self.srp_epsilon * self.srp_gm * d[k] / dist3) * scale

        for k in range(3):
            out[k] = y[3 + k]
            out[3 + k] = total[k] - self.earth_eph_acc[k]
        if not self.has_rotation:
            return 0
        return self._rotation(y, rm, out)

    @cython.cdivision(True)
    cdef int _rotation(self, const double* y, const double* rm, double* out) noexcept nogil:
        cdef double* pos = &self.pos[0]
        cdef double* vel = &self.vel[0]
        cdef double* gm = &self.gm[0]
        cdef double* rel_all = &self.rel_all[0]
        cdef double* a_all = &self.a_all[0]
        cdef const double* wm = y + 10
        cdef const double* wc = y + 13
        cdef double torque[3]
        cdef double nu[3]
        cdef double reaction[3]
        cdef double tmp[3]
        cdef double tmp2[3]
        cdef double tmp3[3]
        cdef double cmb[3]
        cdef double viscous[3]
        cdef double pressure[3]
        cdef double mantle[9]
        cdef double rhs[3]
        cdef double mdot[3]
        cdef double r_em[3]
        cdef double v_em[3]
        cdef double r_sm[3]
        cdef double v_sm[3]
        cdef double u[3]
        cdef double p[3]
        cdef double ku[3]
        cdef double kp[3]
        cdef double e2[3]
        cdef double c1[3]
        cdef double c2v[3]
        cdef double c3[3]
        cdef double c4[3]
        cdef double c5[3]
        cdef double terms[3]
        cdef double distance, s, spin, coeff, norm_em, norm_sm, j2
        cdef Py_ssize_t k, i, slot, body

        torque[0] = 0.0
        torque[1] = 0.0
        torque[2] = 0.0
        for k in range(self.n_torque):
            if self.torque_kind[k] == 0:
                # DE430 Eq. 43 / DE440 Eq. 56: figure torque from the shared Moon field.
                slot = self.torque_arg[k]
                body = self.targets[slot]
                coeff = (-gm[body]) / gm[1]
                for i in range(3):
                    reaction[i] = coeff * a_all[3 * slot + i]
                _cross3(rel_all + 3 * slot, reaction, tmp)
                for i in range(3):
                    torque[i] += tmp[i] / self.moon_radius2
            elif self.torque_kind[k] == 1:
                # Quadrupole-only point-mass torque with the current whole-Moon inertia.
                body = self.torque_arg[k]
                for i in range(3):
                    tmp[i] = pos[3 * body + i] - pos[3 + i]
                _mat_t_vec(rm, tmp, u)
                distance = sqrt(_dot3(u, u))
                for i in range(3):
                    u[i] = u[i] / distance
                _matvec(self.inertia, u, ku)
                _cross3(u, ku, c1)
                coeff = 3 * gm[body] / pow(distance, 3.0)
                for i in range(3):
                    torque[i] += coeff * c1[i]
            else:
                # DE440 Eq. 57: Earth J2 figure-figure torque.
                for i in range(3):
                    tmp[i] = pos[i] - pos[3 + i]
                _mat_t_vec(rm, tmp, u)
                distance = sqrt(_dot3(u, u))
                for i in range(3):
                    u[i] = u[i] / distance
                e2[0] = self.earth_matrix[2]
                e2[1] = self.earth_matrix[5]
                e2[2] = self.earth_matrix[8]
                _mat_t_vec(rm, e2, p)
                s = _dot3(u, p)
                _matvec(self.inertia, u, ku)
                _matvec(self.inertia, p, kp)
                _cross3(u, ku, c1)
                _cross3(u, kp, c2v)
                _cross3(p, ku, c3)
                _cross3(p, kp, c4)
                coeff = 7.5 * gm[0] * self.torque_earth_radius ** 2 * self.torque_earth_j2 / pow(distance, 5.0)
                for i in range(3):
                    terms[i] = (1 - 7 * s * s) * c1[i] + 2 * s * (c2v[i] + c3[i]) - 0.4 * c4[i]
                    torque[i] += coeff * terms[i]

        nu[0] = 0.0
        nu[1] = 0.0
        nu[2] = 0.0
        if self.geodetic:
            # de Sitter rate 3/(2c^2) sum_j GM_j (r_jM x v_jM)/r_jM^3 for j = Earth, Sun.
            for i in range(3):
                r_em[i] = pos[3 + i] - pos[i]
                v_em[i] = vel[3 + i] - vel[i]
                r_sm[i] = pos[3 + i] - pos[3 * self.sun + i]
                v_sm[i] = vel[3 + i] - vel[3 * self.sun + i]
            norm_em = sqrt(_dot3(r_em, r_em))
            norm_sm = sqrt(_dot3(r_sm, r_sm))
            _cross3(r_em, v_em, tmp)
            _cross3(r_sm, v_sm, tmp2)
            for i in range(3):
                tmp[i] = 1.5 / self.c2 * (
                    gm[0] * tmp[i] / pow(norm_em, 3.0) + gm[self.sun] * tmp2[i] / pow(norm_sm, 3.0)
                )
            _mat_t_vec(rm, tmp, nu)

        for i in range(3):
            viscous[i] = 0.0
            pressure[i] = 0.0
        if self.has_core:
            for i in range(3):
                viscous[i] = self.core_viscosity * (wc[i] - wm[i])
            coeff = (self.core_diagonal[2] - self.core_diagonal[0]) * wc[2]
            pressure[0] = coeff * (-wc[1])
            pressure[1] = coeff * wc[0]
            pressure[2] = coeff * 0.0
        for i in range(3):
            cmb[i] = viscous[i] + pressure[i]

        for i in range(9):
            mantle[i] = self.inertia[i]
        if self.has_core:
            mantle[0] -= self.core_diagonal[0]
            mantle[4] -= self.core_diagonal[1]
            mantle[8] -= self.core_diagonal[2]
        for i in range(3):
            tmp[i] = wm[i] - nu[i]
        _matvec(mantle, wm, tmp2)
        _cross3(tmp, tmp2, tmp3)
        _matvec(self.inertia_dot, wm, c1)
        for i in range(3):
            rhs[i] = torque[i] + cmb[i] - tmp3[i] - c1[i]
        if _solve3(mantle, rhs, mdot):
            return 4

        # Quaternion kinematics from the raw (unnormalized) integration state.
        out[6] = 0.5 * (-(y[7] * wm[0] + y[8] * wm[1] + y[9] * wm[2]))
        tmp[0] = y[7]
        tmp[1] = y[8]
        tmp[2] = y[9]
        _cross3(tmp, wm, tmp2)
        for i in range(3):
            out[7 + i] = 0.5 * (y[6] * wm[i] + tmp2[i])
        for i in range(3):
            out[10 + i] = mdot[i]
        if self.has_core:
            for i in range(3):
                ku[i] = self.core_diagonal[i] * wc[i]
                tmp[i] = wm[i] - nu[i]
            _cross3(tmp, ku, tmp2)
            for i in range(3):
                out[13 + i] = (-cmb[i] - tmp2[i]) / self.core_diagonal[i]
        return 0
