# Time and reference frames

LunarOps states the time scale, frame, axes and unit of every quantity at
every interface. This page collects the conventions.

## Time

### Epochs

All times are `Epoch` values: a two-part Julian date `(jd1, jd2)` with an
explicit scale, `UTC`, `TT` or `TDB`. The two-part form keeps sub-nanosecond
resolution over centuries. An operation that needs a particular scale checks
it and fails on any other; nothing is converted silently.

| Where | Scale |
|---|---|
| Normal-point transmit epochs, catalogs, EOP tables, configuration times | UTC |
| Station clock intervals (the observed round-trip time) | TT, realized by the station UTC clock |
| Ephemeris queries, light-time events, dynamics | TDB |

Civil parsing and formatting, UTC↔TAI↔TT, leap seconds and UTC elapsed-time
arithmetic come from ERFA. TT↔TDB uses ERFA `dtdb`; for an observer on the
Earth the station-dependent part is added,
`dtdb(TDB, UT1, λ, u, v) − dtdb(TDB, 0, 0, 0, 0)`, with the station longitude
λ and its distances `u` from the spin axis and `v` from the equatorial plane.
No time conversion uses an ephemeris time series.

### Local time

Observation programs accept `utcOffsetHours`, a fixed offset in whole minutes.
Naive timestamps in the configuration are read as civil time with that offset
and converted to UTC before use. Outputs keep the UTC columns and add separate
local-time columns with an explicit ISO offset, for example
`2024-09-18T08:00:00+08:00`. The model itself always runs in UTC/TT/TDB.

### Intervals

Configuration intervals are half-open, `[start, endExclusive)`, and `null`
means unbounded. Interval keys that are dates (variance components, bias
intervals, range-bias rows) compare against the UTC date of the transmit
epoch.

## Frames

| Frame | Used for | Axes |
|---|---|---|
| ITRF | station coordinates, station displacements | terrestrial, realized by the station catalog |
| GCRS | geocentric positions in the light-time solution | ICRS-aligned, geocentric, TT-compatible scale |
| BCRS | ephemeris states, light propagation | ICRS-aligned, barycentric, TDB-compatible scale |
| LCRS | selenocentric positions of the reflectors | ICRS-aligned, selenocentric |
| Moon PA | reflector catalog coordinates, lunar gravity field | lunar principal axes of the selected ephemeris |

### ITRF ↔ GCRS

The terrestrial-to-celestial rotation follows the IERS Conventions (2010)
CIO-based procedure with ERFA: the IAU 2006/2000A CIP coordinates `X, Y` and
CIO locator `s` (`xys06a`), corrected by the observed celestial pole offsets
`dX, dY`; the Earth rotation angle from UT1 (`era00`); and polar motion with
the TIO locator `s'` (`pom00`, `sp00`).

The Earth-orientation parameters come from a native EOP table (see
[Input data](input-data.md#earth-orientation)). LunarOps interpolates
`xp, yp, UT1−UTC, dX, dY` linearly in UTC MJD, handling UT1−UTC across leap
seconds through UT1−TAI. It then adds the diurnal and semidiurnal variations
of polar motion and UT1: the ocean-tide terms (IERS Conventions 2010, Ch. 8)
and the libration terms (Ch. 5, Tables 5.1a and 5.1b). The C04 values
are used as observed: no zonal-tide term is restored and no free-core-nutation
model is added to `dX, dY`. An epoch outside the table is an error.

### BCRS ↔ GCRS and BCRS ↔ LCRS

Positions move between the barycentric and the geocentric or selenocentric
systems with the relativistic transformation of the IERS Conventions (2010),
Ch. 11 (Eqs. 11.18–11.19). It includes the external gravitational potential at
the Earth or Moon from the Sun, the planets and the other body, and the scale
difference between the coordinate times: `L_B − L_G` for the GCRS. For the LCRS
the scale difference depends on the ephemeris, set by
`lunarRelativisticScaleConvention` of the `calceph` ephemeris:

`tdbCompatibleLunarSurface`
:   The lunar coordinates of the kernel are TDB-compatible; LunarOps applies
    `L_B − L_L` with `L_L` for the lunar surface.

`alreadyScaled`
:   The kernel's lunar coordinates already carry their native scale; no scale
    difference is applied.

### Moon PA ↔ LCRS

The lunar orientation is read from the ephemeris' `.bpc` kernel as three Euler
angles and turned into the PA→LCRS rotation matrix. Reflector coordinates in
the catalog and the lunar gravity field are expressed in these PA axes, so
both must belong to the same ephemeris. The optional
`longitudeLibrationCorrection: inpop21a` adds a three-term periodic correction
to the longitude libration for INPOP21a kernels.

## Units and signs

- SI units everywhere internally: metres, seconds, radians, m³/s² for GM.
  Configuration and file fields that use other units carry the unit in their
  name (`Deg`, `Arcsec`, `Mas`, `Cm`, `Hpa`, `Nm`).
- Residuals are one-way: `oc_one_way_m = 0.5 c (observed − computed)` of the
  round-trip time.
- Normal-point uncertainty is two-way time; the one-way range sigma is
  `0.5 c uncertainty_two_way_s`.
- Station longitude is east positive, latitude north positive, height
  ellipsoidal (WGS84).
- Ocean-loading displacements are ordered (Up, West, South), the axis order of
  the IERS HARDISP routine; they are converted to ITRF through the local
  East-North-Up frame as `ENU = (−West, −South, Up)`.
- Lunar attitude is 3-1-3 Euler angles `(φ, θ, ψ)` in radians, rotating PA axes to
  inertial axes as `Rz(φ) Rx(θ) Rz(ψ)`.
