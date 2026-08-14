# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Cython implementation of the LunarOps IERS 2010 numerical backend.

The algorithms and coefficient tables are derived from the pinned IERS
Conventions v1.3.0 software.  Routine names and interfaces are LunarOps
specific, and calendar/time-scale work is intentionally delegated to ERFA in
``lunarops._iers2010``. See ``docs/IERS_CYTHON_MIGRATION.md`` for provenance
and adaptation details.
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport atan2, atan2f, cos, cosf, fabs, fabsf, fmod, sin, sinf, sqrt, sqrtf

cnp.import_array()

include "_iers2010_tables.pxi"

cdef double PI = 3.1415926535897932384626433
cdef double TWO_PI = 6.283185307179586476925287
cdef double DEG2RAD = 0.017453292519943295769236907684886
cdef double RAD2DEG = 57.295779513082320876798154814105

cdef const double[:, ::1] ORTHOW = np.asarray(_ORTHOW, dtype=np.float64)
cdef const double[:, ::1] CNMTX_SP = np.asarray(_CNMTX_SP, dtype=np.float64)
cdef const double[:, ::1] CNMTX_LINES = np.asarray(_CNMTX_LINES, dtype=np.float64)
# These DATA statements use default-real literals assigned to DOUBLE PRECISION
# arrays, so the reference compiler rounds them through float32 first.
cdef const double[:, ::1] PMS_TERMS = np.asarray(_PMS_TERMS, dtype=np.float32).astype(np.float64)
cdef const double[:, ::1] UTLIBR_TERMS = np.asarray(_UTLIBR_TERMS, dtype=np.float32).astype(np.float64)
cdef const double[:, ::1] STEP2DIU_TERMS = np.asarray(_STEP2DIU_TERMS, dtype=np.float64)
cdef const double[:, ::1] STEP2LON_TERMS = np.asarray(_STEP2LON_TERMS, dtype=np.float64)
cdef const cnp.int32_t[:, ::1] HARDISP_INPUT_DOODSON = np.asarray(
    _HARDISP_INPUT_DOODSON, dtype=np.int32
)
cdef const float[::1] HARDISP_TAMP = np.asarray(_HARDISP_TAMP, dtype=np.float32)
cdef const cnp.int32_t[:, ::1] HARDISP_DOODSON = np.asarray(_HARDISP_DOODSON, dtype=np.int32)


cpdef double lunarops_fcul_a(double latitude, double height_m, double temperature_k, double elevation_deg):
    cdef double sine = sin(elevation_deg * DEG2RAD)
    cdef double temperature_c = temperature_k - 273.15
    cdef double cosphi = cos(latitude * DEG2RAD)
    cdef double a1 = 0.00121008 + 0.0000017295 * temperature_c + 0.00003191 * cosphi - 0.000000018478 * height_m
    cdef double a2 = 0.00304965 + 0.000002346 * temperature_c - 0.0001035 * cosphi - 0.00000001856 * height_m
    cdef double a3 = 0.068777 + 0.00001972 * temperature_c - 0.003458 * cosphi + 0.0000001060 * height_m
    cdef double map_zen = 1.0 + a1 / (1.0 + a2 / (1.0 + a3))
    return map_zen / (sine + a1 / (sine + a2 / (sine + a3)))


cpdef tuple lunarops_fculzd_hpa(
    double latitude,
    double ellipsoidal_height_m,
    double pressure_hpa,
    double water_vapor_pressure_hpa,
    double wavelength_um,
):
    cdef double sigma = 1.0 / wavelength_um
    cdef double sigma2 = sigma * sigma
    cdef double f = 1.0 - (<float>0.00266) * cos(2.0 * DEG2RAD * latitude) - 0.00000028 * ellipsoidal_height_m
    cdef double corr = 1.0 + 0.000000534 * (375.0 - 450.0)
    cdef double fh = 0.01 * corr * (
        19990.975 * (238.0185 + sigma2) / ((238.0185 - sigma2) ** 2)
        + 579.55174 * (57.362 + sigma2) / ((57.362 - sigma2) ** 2)
    )
    cdef double zhd = 0.002416579 * fh * pressure_hpa / f
    cdef double fnh = 0.003101 * (
        295.235 + 3.0 * 2.6422 * sigma2 + 5.0 * -0.032380 * sigma2 * sigma2
        + 7.0 * 0.004028 * sigma2 * sigma2 * sigma2
    )
    cdef double zwd = 0.0001 * (5.316 * fnh - (<float>3.759) * fh) * water_vapor_pressure_hpa / f
    return zhd + zwd, zhd, zwd


cdef void lunarops_cnmtx(double dmjd, double[::1] h) noexcept:
    cdef double anm[3][3][3]
    cdef double bnm[3][3][3]
    cdef double p[3][3]
    cdef double q[3][3]
    cdef double alpha, dt60, pinm, ap, am, bp, bm
    cdef int i, j, k, m, n, idx
    for n in range(3):
        for m in range(3):
            for k in range(3):
                anm[n][m][k] = 0.0
                bnm[n][m][k] = 0.0
    for k in range(3):
        dt60 = dmjd - (k - 1) * 2.0 - 37076.5
        for j in range(71):
            n = <int>CNMTX_LINES[j, 0]
            m = <int>CNMTX_LINES[j, 1]
            pinm = ((n + m) % 2) * TWO_PI / 4.0
            alpha = fmod(CNMTX_LINES[j, 3] - pinm, TWO_PI) + fmod(CNMTX_LINES[j, 4] * dt60, TWO_PI)
            anm[n][m][k] += CNMTX_LINES[j, 2] * cos(alpha)
            bnm[n][m][k] -= CNMTX_LINES[j, 2] * sin(alpha)
    for m in range(1, 3):
        ap = anm[2][m][2] + anm[2][m][0]
        am = anm[2][m][2] - anm[2][m][0]
        bp = bnm[2][m][2] + bnm[2][m][0]
        bm = bnm[2][m][2] - bnm[2][m][0]
        p[0][m] = CNMTX_SP[m - 1, 0] * anm[2][m][1]
        p[1][m] = CNMTX_SP[m - 1, 1] * anm[2][m][1] - CNMTX_SP[m - 1, 2] * ap
        p[2][m] = CNMTX_SP[m - 1, 3] * anm[2][m][1] - CNMTX_SP[m - 1, 4] * ap + CNMTX_SP[m - 1, 5] * bm
        q[0][m] = CNMTX_SP[m - 1, 0] * bnm[2][m][1]
        q[1][m] = CNMTX_SP[m - 1, 1] * bnm[2][m][1] - CNMTX_SP[m - 1, 2] * bp
        q[2][m] = CNMTX_SP[m - 1, 3] * bnm[2][m][1] - CNMTX_SP[m - 1, 4] * bp - CNMTX_SP[m - 1, 5] * am
        for k in range(3):
            anm[2][m][k] = p[k][m]
            bnm[2][m][k] = q[k][m]
    idx = 0
    for m in range(1, 3):
        for k in range(3):
            h[idx] = anm[2][m][k]
            h[idx + 1] = bnm[2][m][k]
            idx += 2


cpdef cnp.ndarray lunarops_ortho_eop(double time):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] h_arr = np.empty(12, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.zeros(3, dtype=np.float64)
    cdef double[::1] h = h_arr
    cdef int i, j
    lunarops_cnmtx(time, h)
    for i in range(3):
        for j in range(12):
            result[i] += h[j] * ORTHOW[i, j]
    return result


cdef inline double lunarops_mod(double value, double divisor) noexcept:
    return fmod(value, divisor)


cpdef cnp.ndarray lunarops_pmsdnut2(
    double rmjd, double l, double lp, double f, double d, double om
):
    cdef double t = (rmjd - 51544.5) / 36525.0
    cdef double gmst = lunarops_mod(
        67310.54841 + t * ((8640184.812866 + 3155760000.0) + t * (0.093104 + t * -0.0000062)),
        86400.0,
    )
    cdef double args[6]
    cdef double angle
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.zeros(2, dtype=np.float64)
    cdef int i, j
    args[0] = lunarops_mod(gmst / (86400.0 / TWO_PI) + PI, TWO_PI)
    args[1], args[2], args[3], args[4], args[5] = l, lp, f, d, om
    for j in range(15, 25):
        angle = 0.0
        for i in range(6):
            angle += PMS_TERMS[j, i] * args[i]
        angle = lunarops_mod(angle, TWO_PI)
        result[0] += PMS_TERMS[j, 7] * sin(angle) + PMS_TERMS[j, 8] * cos(angle)
        result[1] += PMS_TERMS[j, 9] * sin(angle) + PMS_TERMS[j, 10] * cos(angle)
    return result


cpdef tuple lunarops_utlibr(
    double rmjd, double l, double lp, double f, double d, double om
):
    cdef double t = (rmjd - 51544.5) / 36525.0
    cdef double gmst = lunarops_mod(
        67310.54841 + t * ((8640184.812866 + 3155760000.0) + t * (0.093104 + t * -0.0000062)),
        86400.0,
    )
    cdef double args[6]
    cdef double angle, dut1 = 0.0, dlod = 0.0
    cdef int i, j
    args[0] = lunarops_mod(gmst / (86400.0 / TWO_PI) + PI, TWO_PI)
    args[1], args[2], args[3], args[4], args[5] = l, lp, f, d, om
    for j in range(11):
        angle = 0.0
        for i in range(6):
            angle += UTLIBR_TERMS[j, i] * args[i]
        angle = lunarops_mod(angle, TWO_PI)
        dut1 += UTLIBR_TERMS[j, 7] * sin(angle) + UTLIBR_TERMS[j, 8] * cos(angle)
        dlod += UTLIBR_TERMS[j, 9] * sin(angle) + UTLIBR_TERMS[j, 10] * cos(angle)
    return dut1, dlod


cdef inline double lunarops_norm3(const double[::1] value) noexcept:
    return sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


cdef void lunarops_st1idiu(
    const double[::1] xsta, const double[::1] xsun, const double[::1] xmon,
    double fac2sun, double fac2mon, double[::1] out
) noexcept:
    cdef double rsta = lunarops_norm3(xsta), rsun = lunarops_norm3(xsun), rmon = lunarops_norm3(xmon)
    cdef double dhi = -0.0025, dli = -0.0007
    cdef double sinphi = xsta[2] / rsta
    cdef double cosphi = sqrt(xsta[0] * xsta[0] + xsta[1] * xsta[1]) / rsta
    cdef double cos2phi = cosphi * cosphi - sinphi * sinphi
    cdef double sinla = xsta[1] / (cosphi * rsta)
    cdef double cosla = xsta[0] / (cosphi * rsta)
    cdef double drsun = -3.0 * dhi * sinphi * cosphi * fac2sun * xsun[2] * (xsun[0] * sinla - xsun[1] * cosla) / (rsun * rsun)
    cdef double drmon = -3.0 * dhi * sinphi * cosphi * fac2mon * xmon[2] * (xmon[0] * sinla - xmon[1] * cosla) / (rmon * rmon)
    cdef double dnsun = -3.0 * dli * cos2phi * fac2sun * xsun[2] * (xsun[0] * sinla - xsun[1] * cosla) / (rsun * rsun)
    cdef double dnmon = -3.0 * dli * cos2phi * fac2mon * xmon[2] * (xmon[0] * sinla - xmon[1] * cosla) / (rmon * rmon)
    cdef double desun = -3.0 * dli * sinphi * fac2sun * xsun[2] * (xsun[0] * cosla + xsun[1] * sinla) / (rsun * rsun)
    cdef double demon = -3.0 * dli * sinphi * fac2mon * xmon[2] * (xmon[0] * cosla + xmon[1] * sinla) / (rmon * rmon)
    cdef double dr = drsun + drmon, dn = dnsun + dnmon, de = desun + demon
    out[0] = dr * cosla * cosphi - de * sinla - dn * sinphi * cosla
    out[1] = dr * sinla * cosphi + de * cosla - dn * sinphi * sinla
    out[2] = dr * sinphi + dn * cosphi


cdef void lunarops_st1isem(
    const double[::1] xsta, const double[::1] xsun, const double[::1] xmon,
    double fac2sun, double fac2mon, double[::1] out
) noexcept:
    cdef double rsta = lunarops_norm3(xsta), rsun = lunarops_norm3(xsun), rmon = lunarops_norm3(xmon)
    cdef double dhi = -0.0022, dli = -0.0007
    cdef double sinphi = xsta[2] / rsta
    cdef double cosphi = sqrt(xsta[0] * xsta[0] + xsta[1] * xsta[1]) / rsta
    cdef double sinla = xsta[1] / (cosphi * rsta), cosla = xsta[0] / (cosphi * rsta)
    cdef double costwola = cosla * cosla - sinla * sinla, sintwola = 2.0 * cosla * sinla
    cdef double drsun = -0.75 * dhi * cosphi * cosphi * fac2sun * (
        (xsun[0] * xsun[0] - xsun[1] * xsun[1]) * sintwola - 2.0 * xsun[0] * xsun[1] * costwola
    ) / (rsun * rsun)
    cdef double drmon = -0.75 * dhi * cosphi * cosphi * fac2mon * (
        (xmon[0] * xmon[0] - xmon[1] * xmon[1]) * sintwola - 2.0 * xmon[0] * xmon[1] * costwola
    ) / (rmon * rmon)
    cdef double dnsun = 1.5 * dli * sinphi * cosphi * fac2sun * (
        (xsun[0] * xsun[0] - xsun[1] * xsun[1]) * sintwola - 2.0 * xsun[0] * xsun[1] * costwola
    ) / (rsun * rsun)
    cdef double dnmon = 1.5 * dli * sinphi * cosphi * fac2mon * (
        (xmon[0] * xmon[0] - xmon[1] * xmon[1]) * sintwola - 2.0 * xmon[0] * xmon[1] * costwola
    ) / (rmon * rmon)
    cdef double desun = -1.5 * dli * cosphi * fac2sun * (
        (xsun[0] * xsun[0] - xsun[1] * xsun[1]) * costwola + 2.0 * xsun[0] * xsun[1] * sintwola
    ) / (rsun * rsun)
    cdef double demon = -1.5 * dli * cosphi * fac2mon * (
        (xmon[0] * xmon[0] - xmon[1] * xmon[1]) * costwola + 2.0 * xmon[0] * xmon[1] * sintwola
    ) / (rmon * rmon)
    cdef double dr = drsun + drmon, dn = dnsun + dnmon, de = desun + demon
    out[0] = dr * cosla * cosphi - de * sinla - dn * sinphi * cosla
    out[1] = dr * sinla * cosphi + de * cosla - dn * sinphi * sinla
    out[2] = dr * sinphi + dn * cosphi


cdef void lunarops_st1l1(
    const double[::1] xsta, const double[::1] xsun, const double[::1] xmon,
    double fac2sun, double fac2mon, double[::1] out
) noexcept:
    cdef double rsta = lunarops_norm3(xsta), rsun = lunarops_norm3(xsun), rmon = lunarops_norm3(xmon)
    cdef double l1d = 0.0012, l1sd = 0.0024
    cdef double sinphi = xsta[2] / rsta
    cdef double cosphi = sqrt(xsta[0] * xsta[0] + xsta[1] * xsta[1]) / rsta
    cdef double sinla = xsta[1] / (cosphi * rsta), cosla = xsta[0] / (cosphi * rsta)
    cdef double dnsun = -l1d * sinphi * sinphi * fac2sun * xsun[2] * (xsun[0] * cosla + xsun[1] * sinla) / (rsun * rsun)
    cdef double dnmon = -l1d * sinphi * sinphi * fac2mon * xmon[2] * (xmon[0] * cosla + xmon[1] * sinla) / (rmon * rmon)
    cdef double desun = l1d * sinphi * (cosphi * cosphi - sinphi * sinphi) * fac2sun * xsun[2] * (xsun[0] * sinla - xsun[1] * cosla) / (rsun * rsun)
    cdef double demon = l1d * sinphi * (cosphi * cosphi - sinphi * sinphi) * fac2mon * xmon[2] * (xmon[0] * sinla - xmon[1] * cosla) / (rmon * rmon)
    cdef double de = 3.0 * (desun + demon), dn = 3.0 * (dnsun + dnmon)
    out[0] = -de * sinla - dn * sinphi * cosla
    out[1] = de * cosla - dn * sinphi * sinla
    out[2] = dn * cosphi
    dnsun = -0.5 * l1sd * sinphi * cosphi * fac2sun * ((xsun[0] * xsun[0] - xsun[1] * xsun[1]) * cos(2.0 * atan2(xsta[1], xsta[0])) + 2.0 * xsun[0] * xsun[1] * sin(2.0 * atan2(xsta[1], xsta[0]))) / (rsun * rsun)
    dnmon = -0.5 * l1sd * sinphi * cosphi * fac2mon * ((xmon[0] * xmon[0] - xmon[1] * xmon[1]) * cos(2.0 * atan2(xsta[1], xsta[0])) + 2.0 * xmon[0] * xmon[1] * sin(2.0 * atan2(xsta[1], xsta[0]))) / (rmon * rmon)
    desun = -0.5 * l1sd * sinphi * sinphi * cosphi * fac2sun * ((xsun[0] * xsun[0] - xsun[1] * xsun[1]) * sin(2.0 * atan2(xsta[1], xsta[0])) - 2.0 * xsun[0] * xsun[1] * cos(2.0 * atan2(xsta[1], xsta[0]))) / (rsun * rsun)
    demon = -0.5 * l1sd * sinphi * sinphi * cosphi * fac2mon * ((xmon[0] * xmon[0] - xmon[1] * xmon[1]) * sin(2.0 * atan2(xsta[1], xsta[0])) - 2.0 * xmon[0] * xmon[1] * cos(2.0 * atan2(xsta[1], xsta[0]))) / (rmon * rmon)
    de = 3.0 * (desun + demon)
    dn = 3.0 * (dnsun + dnmon)
    out[0] += -de * sinla - dn * sinphi * cosla
    out[1] += de * cosla - dn * sinphi * sinla
    out[2] += dn * cosphi


cdef void lunarops_step2diu(
    const double[::1] xsta, double fractional_hour, double t, double[::1] out
) noexcept:
    cdef double s, tau, pr, h, p, zns, ps, thetaf, dr, dn, de
    cdef double rsta = lunarops_norm3(xsta)
    cdef double sinphi = xsta[2] / rsta
    cdef double cosphi = sqrt(xsta[0] * xsta[0] + xsta[1] * xsta[1]) / rsta
    cdef double zla = atan2(xsta[1], xsta[0])
    cdef int j
    s = fmod(218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * t) * t) * t, 360.0)
    tau = fractional_hour * 15.0 + 280.4606184 + (36000.7700536 + (0.00038793 - 0.0000000258 * t) * t) * t - s
    pr = (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007 * t) * t) * t) * t
    s += pr
    h = 280.46645 + (36000.7697489 + (0.00030322222 + (0.000000020 - 0.00000000654 * t) * t) * t) * t
    p = 83.35324312 + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263 * t) * t) * t) * t
    zns = 234.95544499 + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650 * t) * t) * t) * t
    ps = 282.93734098 + (1.71945766667 + (0.00045688889 + (-0.00000001778 - 0.00000000334 * t) * t) * t) * t
    s=fmod(s,360.0); tau=fmod(tau,360.0); h=fmod(h,360.0); p=fmod(p,360.0); zns=fmod(zns,360.0); ps=fmod(ps,360.0)
    out[0] = out[1] = out[2] = 0.0
    for j in range(31):
        thetaf = (tau + STEP2DIU_TERMS[j, 0] * s + STEP2DIU_TERMS[j, 1] * h + STEP2DIU_TERMS[j, 2] * p + STEP2DIU_TERMS[j, 3] * zns + STEP2DIU_TERMS[j, 4] * ps) * DEG2RAD
        dr = STEP2DIU_TERMS[j, 5] * 2.0 * sinphi * cosphi * sin(thetaf + zla) + STEP2DIU_TERMS[j, 6] * 2.0 * sinphi * cosphi * cos(thetaf + zla)
        dn = STEP2DIU_TERMS[j, 7] * (cosphi * cosphi - sinphi * sinphi) * sin(thetaf + zla) + STEP2DIU_TERMS[j, 8] * (cosphi * cosphi - sinphi * sinphi) * cos(thetaf + zla)
        de = STEP2DIU_TERMS[j, 7] * sinphi * cos(thetaf + zla) - STEP2DIU_TERMS[j, 8] * sinphi * sin(thetaf + zla)
        out[0] += dr * cosphi * cos(zla) - de * sin(zla) - dn * sinphi * cos(zla)
        out[1] += dr * cosphi * sin(zla) + de * cos(zla) - dn * sinphi * sin(zla)
        out[2] += dr * sinphi + dn * cosphi
    out[0] /= 1000.0; out[1] /= 1000.0; out[2] /= 1000.0


cdef void lunarops_step2lon(const double[::1] xsta, double t, double[::1] out) noexcept:
    cdef double s, pr, h, p, zns, ps, thetaf, dr, dn
    cdef double rsta = lunarops_norm3(xsta)
    cdef double sinphi = xsta[2] / rsta
    cdef double cosphi = sqrt(xsta[0] * xsta[0] + xsta[1] * xsta[1]) / rsta
    cdef double zla = atan2(xsta[1], xsta[0])
    cdef int j
    s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * t) * t) * t
    pr = (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007 * t) * t) * t) * t
    s += pr
    h = 280.46645 + (36000.7697489 + (0.00030322222 + (0.000000020 + -0.00000000654 * t) * t) * t) * t
    p = 83.35324312 + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263 * t) * t) * t) * t
    zns = 234.95544499 + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650 * t) * t) * t) * t
    ps = 282.93734098 + (1.71945766667 + (0.00045688889 + (-0.00000001778 + -0.00000000334 * t) * t) * t) * t
    s=fmod(s,360.0); h=fmod(h,360.0); p=fmod(p,360.0); zns=fmod(zns,360.0); ps=fmod(ps,360.0)
    out[0] = out[1] = out[2] = 0.0
    for j in range(5):
        thetaf = (STEP2LON_TERMS[j, 0] * s + STEP2LON_TERMS[j, 1] * h + STEP2LON_TERMS[j, 2] * p + STEP2LON_TERMS[j, 3] * zns + STEP2LON_TERMS[j, 4] * ps) * DEG2RAD
        dr = STEP2LON_TERMS[j, 5] * (3.0 * sinphi * sinphi - 1.0) / 2.0 * cos(thetaf) + STEP2LON_TERMS[j, 7] * (3.0 * sinphi * sinphi - 1.0) / 2.0 * sin(thetaf)
        dn = STEP2LON_TERMS[j, 6] * 2.0 * cosphi * sinphi * cos(thetaf) + STEP2LON_TERMS[j, 8] * 2.0 * cosphi * sinphi * sin(thetaf)
        out[0] += dr * cosphi * cos(zla) - dn * sinphi * cos(zla)
        out[1] += dr * cosphi * sin(zla) - dn * sinphi * sin(zla)
        out[2] += dr * sinphi + dn * cosphi
    out[0] /= 1000.0; out[1] /= 1000.0; out[2] /= 1000.0


cpdef cnp.ndarray lunarops_dehanttideinel(
    object xsta_obj, double fractional_hour, object xsun_obj, object xmon_obj, double tt_centuries
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] xsta_arr = np.ascontiguousarray(xsta_obj, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] xsun_arr = np.ascontiguousarray(xsun_obj, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] xmon_arr = np.ascontiguousarray(xmon_obj, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] correction = np.zeros(3, dtype=np.float64)
    cdef const double[::1] xsta = xsta_arr, xsun = xsun_arr, xmon = xmon_arr
    cdef double[::1] out = result, cor = correction
    cdef double scs = xsta[0]*xsun[0]+xsta[1]*xsun[1]+xsta[2]*xsun[2]
    cdef double scm = xsta[0]*xmon[0]+xsta[1]*xmon[1]+xsta[2]*xmon[2]
    cdef double rsta=lunarops_norm3(xsta), rsun=lunarops_norm3(xsun), rmon=lunarops_norm3(xmon)
    cdef double scsun=scs/rsta/rsun, scmon=scm/rsta/rmon
    cdef double cosphi=sqrt(xsta[0]*xsta[0]+xsta[1]*xsta[1])/rsta
    cdef double h2=0.6078-0.0006*(1.0-1.5*cosphi*cosphi)
    cdef double l2=0.0847+0.0002*(1.0-1.5*cosphi*cosphi)
    cdef double p2sun=3.0*(h2/2.0-l2)*scsun*scsun-h2/2.0
    cdef double p2mon=3.0*(h2/2.0-l2)*scmon*scmon-h2/2.0
    cdef double p3sun=2.5*(0.292-3.0*0.015)*scsun*scsun*scsun+1.5*(0.015-0.292)*scsun
    cdef double p3mon=2.5*(0.292-3.0*0.015)*scmon*scmon*scmon+1.5*(0.015-0.292)*scmon
    cdef double x2sun=3.0*l2*scsun, x2mon=3.0*l2*scmon
    cdef double x3sun=1.5*0.015*(5.0*scsun*scsun-1.0), x3mon=1.5*0.015*(5.0*scmon*scmon-1.0)
    cdef double re=6378136.6
    cdef double fac2sun=332946.0482*(re/rsun)**3*re
    cdef double fac2mon=0.0123000371*(re/rmon)**3*re
    cdef double fac3sun=fac2sun*(re/rsun), fac3mon=fac2mon*(re/rmon)
    cdef int i
    for i in range(3):
        out[i]=fac2sun*(x2sun*xsun[i]/rsun+p2sun*xsta[i]/rsta)+fac2mon*(x2mon*xmon[i]/rmon+p2mon*xsta[i]/rsta)+fac3sun*(x3sun*xsun[i]/rsun+p3sun*xsta[i]/rsta)+fac3mon*(x3mon*xmon[i]/rmon+p3mon*xsta[i]/rsta)
    lunarops_st1idiu(xsta,xsun,xmon,fac2sun,fac2mon,cor)
    for i in range(3): out[i]+=cor[i]
    lunarops_st1isem(xsta,xsun,xmon,fac2sun,fac2mon,cor)
    for i in range(3): out[i]+=cor[i]
    lunarops_st1l1(xsta,xsun,xmon,fac2sun,fac2mon,cor)
    for i in range(3): out[i]+=cor[i]
    lunarops_step2diu(xsta,fractional_hour,tt_centuries,cor)
    for i in range(3): out[i]+=cor[i]
    lunarops_step2lon(xsta,tt_centuries,cor)
    for i in range(3): out[i]+=cor[i]
    return result


cdef void lunarops_hardisp_doodson_state(
    double tt_centuries, double utc_day_fraction,
    double l, double lp, double f, double d, double om,
    double* doodson_phase, double* doodson_frequency
) noexcept:
    cdef double angles[6]
    cdef double rates[6]
    cdef double fd1, fd2, fd3, fd4, fd5, value
    cdef int i, j
    angles[0] = 360.0 * utc_day_fraction - d * RAD2DEG
    angles[1] = (f + om) * RAD2DEG
    angles[2] = angles[1] - d * RAD2DEG
    angles[3] = angles[1] - l * RAD2DEG
    angles[4] = -om * RAD2DEG
    angles[5] = angles[2] - lp * RAD2DEG
    fd1 = 0.0362916471 + 0.0000000013 * tt_centuries
    fd2 = 0.0027377786
    fd3 = 0.0367481951 - 0.0000000005 * tt_centuries
    fd4 = 0.0338631920 - 0.0000000003 * tt_centuries
    fd5 = -0.0001470938 + 0.0000000003 * tt_centuries
    rates[0]=1.0-fd4; rates[1]=fd3+fd5; rates[2]=rates[1]-fd4
    rates[3]=rates[1]-fd1; rates[4]=-fd5; rates[5]=rates[2]-fd2
    for j in range(342):
        value = 0.0
        for i in range(6): value += HARDISP_DOODSON[j,i]*angles[i]
        value = fmod(value,360.0)
        if value < 0.0: value += 360.0
        doodson_phase[j] = value
        value = 0.0
        for i in range(6): value += HARDISP_DOODSON[j,i]*rates[i]
        doodson_frequency[j] = value


cdef void lunarops_spline(
    int n, int offset, const float* x, const float* u,
    float* s, float* work
) noexcept:
    cdef int i, j, idx, n1
    cdef float q1, qn, c
    if n <= 3:
        for i in range(n): s[offset+i]=0.0
        return
    q1=((u[offset+1]-u[offset])/(x[offset+1]-x[offset])**2-(u[offset+2]-u[offset])/(x[offset+2]-x[offset])**2)/(1.0/(x[offset+1]-x[offset])-1.0/(x[offset+2]-x[offset]))
    qn=((u[offset+n-2]-u[offset+n-1])/(x[offset+n-2]-x[offset+n-1])**2-(u[offset+n-3]-u[offset+n-1])/(x[offset+n-3]-x[offset+n-1])**2)/(1.0/(x[offset+n-2]-x[offset+n-1])-1.0/(x[offset+n-3]-x[offset+n-1]))
    s[offset]=6.0*((u[offset+1]-u[offset])/(x[offset+1]-x[offset])-q1)
    n1=n-1
    for i in range(1,n1):
        idx=offset+i
        s[idx]=(u[idx-1]/(x[idx]-x[idx-1])-u[idx]*(1.0/(x[idx]-x[idx-1])+1.0/(x[idx+1]-x[idx]))+u[idx+1]/(x[idx+1]-x[idx]))*6.0
    s[offset+n-1]=6.0*(qn+(u[offset+n-2]-u[offset+n-1])/(x[offset+n-1]-x[offset+n-2]))
    work[0]=2.0*(x[offset+1]-x[offset]); work[1]=1.5*(x[offset+1]-x[offset])+2.0*(x[offset+2]-x[offset+1]); s[offset+1]-=0.5*s[offset]
    for i in range(2,n1):
        idx=offset+i
        c=(x[idx]-x[idx-1])/work[i-1]; work[i]=2.0*(x[idx+1]-x[idx-1])-c*(x[idx]-x[idx-1]); s[idx]-=c*s[idx-1]
    c=(x[offset+n-1]-x[offset+n-2])/work[n-2]; work[n-1]=(2.0-c)*(x[offset+n-1]-x[offset+n-2]); s[offset+n-1]=(s[offset+n-1]-c*s[offset+n-2])/work[n-1]
    for j in range(n1):
        i=n-2-j; idx=offset+i; s[idx]=(s[idx]-(x[idx+1]-x[idx])*s[idx+1])/work[i]


cdef float lunarops_spline_eval(
    float y, int n, int offset, const float* x,
    const float* u, const float* s
) noexcept:
    cdef int k, k1=offset, k2=offset+1
    cdef float dy, dy1, dk
    if y<=x[offset]: return u[offset]
    if y>=x[offset+n-1]: return u[offset+n-1]
    for k in range(offset+1,offset+n):
        if x[k-1]<y and x[k]>=y: k1=k-1; k2=k
    dy=x[k2]-y; dy1=y-x[k1]; dk=x[k2]-x[k1]
    return (s[k1]*dy*dy*dy+s[k2]*dy1*dy1*dy1)/(6.0*dk)+dy1*(u[k2]/dk-s[k2]*dk/6.0)+dy*(u[k1]/dk-s[k1]*dk/6.0)


cdef int lunarops_hardisp_admittance(
    const float* input_amp, const float* input_phase,
    const double* all_phase, const double* all_freq,
    float* out_amp, double* out_freq, double* out_phase
) noexcept:
    cdef float rf[11]
    cdef float rl[11]
    cdef float ai[11]
    cdef float zr[11]
    cdef float zi[11]
    cdef float work[11]
    cdef int i,j,m,k=0,ii,diff,nlp=0,ndi=0,nsd=0,band,start,count
    cdef float re,im,sf,temp,dtr=<float>0.01745329252
    for i in range(11):
        for j in range(342):
            ii = 0
            for m in range(6):
                diff = HARDISP_DOODSON[j,m] - HARDISP_INPUT_DOODSON[i,m]
                ii += -diff if diff < 0 else diff
            if ii==0: break
        if ii==0:
            rl[k]=input_amp[i]*cosf(-dtr*input_phase[i])/fabsf(HARDISP_TAMP[j])
            ai[k]=input_amp[i]*sinf(-dtr*input_phase[i])/fabsf(HARDISP_TAMP[j])
            rf[k]=all_freq[j]; k+=1
    # Stable ordering by frequency reproduces SHELLS for these distinct inputs.
    for i in range(k-1):
        for j in range(i+1,k):
            if rf[j]<rf[i]:
                temp=rf[i];rf[i]=rf[j];rf[j]=temp
                temp=rl[i];rl[i]=rl[j];rl[j]=temp
                temp=ai[i];ai[i]=ai[j];ai[j]=temp
    for i in range(k):
        if rf[i]<0.5:nlp+=1
        elif rf[i]<1.5:ndi+=1
        elif rf[i]<2.5:nsd+=1
    if nlp: lunarops_spline(nlp,0,rf,rl,zr,work); lunarops_spline(nlp,0,rf,ai,zi,work)
    lunarops_spline(ndi,nlp,rf,rl,zr,work)
    lunarops_spline(ndi,nlp,rf,ai,zi,work)
    lunarops_spline(nsd,nlp+ndi,rf,rl,zr,work)
    lunarops_spline(nsd,nlp+ndi,rf,ai,zi,work)
    count=0
    for i in range(342):
        band=HARDISP_DOODSON[i,0]
        if band==0 and nlp==0: continue
        sf=all_freq[i]
        if band==0: start=0; re=lunarops_spline_eval(sf,nlp,start,rf,rl,zr); im=lunarops_spline_eval(sf,nlp,start,rf,ai,zi)
        elif band==1: start=nlp; re=lunarops_spline_eval(sf,ndi,start,rf,rl,zr); im=lunarops_spline_eval(sf,ndi,start,rf,ai,zi)
        else: start=nlp+ndi; re=lunarops_spline_eval(sf,nsd,start,rf,rl,zr); im=lunarops_spline_eval(sf,nsd,start,rf,ai,zi)
        out_freq[count]=all_freq[i]
        out_phase[count]=all_phase[i]+(180.0 if band==0 else 90.0 if band==1 else 0.0)+atan2f(im,re)/dtr
        if out_phase[count]>180.0: out_phase[count]-=360.0
        out_amp[count]=HARDISP_TAMP[i]*sqrtf(re*re+im*im); count+=1
    return count


cdef void lunarops_hardisp_evaluate_axis(
    int n, double sample_seconds, int count,
    const float* amplitudes, double* frequencies, double* phases,
    double* output,
) noexcept:
    cdef float hc[684]
    cdef float omega[342]
    cdef double scratch[1026]
    cdef double previous
    cdef float value
    cdef int i,j,start=0,block,remaining
    for j in range(count):
        phases[j] *= 0.01745329252
        frequencies[j] = sample_seconds*PI*frequencies[j]/43200.0
        omega[j] = <float>frequencies[j]
    while start<n:
        remaining=n-start
        block=600 if remaining>600 else remaining
        for j in range(count):
            hc[2*j]=<float>(amplitudes[j]*cos(phases[j]))
            hc[2*j+1]=<float>(-amplitudes[j]*sin(phases[j]))
            scratch[3*j]=hc[2*j]
            scratch[3*j+1]=<double>(hc[2*j]*cosf(omega[j])-hc[2*j+1]*sinf(omega[j]))
            scratch[3*j+2]=2.0*cos(<double>omega[j])
        for i in range(block):
            value=0.0
            for j in range(count):
                value=<float>(value+scratch[3*j])
                previous=scratch[3*j]
                scratch[3*j]=scratch[3*j+2]*previous-scratch[3*j+1]
                scratch[3*j+1]=previous
            output[start+i]=value
        start+=block
        if start<n:
            for j in range(count):
                phases[j]=fmod(phases[j]+block*frequencies[j],TWO_PI)


cpdef tuple lunarops_hardisp(
    double tt_centuries, double utc_day_fraction,
    double l, double lp, double f, double d, double om,
    int n, double sample_seconds, object blq_amp_obj, object blq_phase_obj
):
    cdef cnp.ndarray[cnp.float32_t,ndim=2] amp_in=np.ascontiguousarray(blq_amp_obj,dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t,ndim=2] phase_in=np.ascontiguousarray(blq_phase_obj,dtype=np.float32)
    cdef cnp.ndarray[cnp.float64_t,ndim=2] result=np.empty((3,n),np.float64)
    cdef int axis,nout
    cdef double allp[342]
    cdef double allf[342]
    cdef float oa[342]
    cdef double of[342]
    cdef double op[342]
    lunarops_hardisp_doodson_state(tt_centuries,utc_day_fraction,l,lp,f,d,om,allp,allf)
    for axis in range(3):
        nout=lunarops_hardisp_admittance(&amp_in[axis,0],&phase_in[axis,0],allp,allf,oa,of,op)
        lunarops_hardisp_evaluate_axis(n,sample_seconds,nout,oa,of,op,&result[axis,0])
    # BLQ rows are Up, West, South; the legacy/public API is Up, South, West.
    return result[0].copy(),result[2].copy(),result[1].copy()
