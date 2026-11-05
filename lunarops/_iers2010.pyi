"""Type declarations of the IERS Conventions (2010) routines.

The routines implement the algorithms of the IERS Conventions (2010)
software; calendar and time-scale handling is done with ERFA.
"""

from numpy import float64
from numpy.typing import ArrayLike, NDArray

HARDISP_MIN_UTC: tuple[int, int, int, int, int, int]
"""Earliest supported UTC start ``(year, month, day, hour, minute, second)`` of :func:`hardisp`."""

HARDISP_VALID_UNTIL_UTC_EXCLUSIVE: tuple[int, int, int, int, int, int]
"""Exclusive upper bound of the UTC epochs of a :func:`hardisp` series."""

def fcul_a(latitude: float, height_m: float, t_k: float, elev_deg: float) -> float:
    """Return the Mendes-Pavlis mean mapping function for optical ranging.

    Parameters
    ----------
    latitude : float
        Geodetic latitude of the station in degrees.
    height_m : float
        Station height above the ellipsoid in m.
    t_k : float
        Surface temperature in K.
    elev_deg : float
        Elevation angle of the target in degrees.

    Returns
    -------
    float
        Ratio of slant to zenith delay (dimensionless).

    References
    ----------
    IERS Conventions (2010), Ch. 9, Sec. 9.1.
    """

def fculzd_hpa(
    latitude: float,
    ellip_ht: float,
    pressure: float,
    wvp: float,
    lambda_um: float,
) -> tuple[float, float, float]:
    """Return the Mendes-Pavlis zenith delays for optical ranging.

    Parameters
    ----------
    latitude : float
        Geodetic latitude of the station in degrees.
    ellip_ht : float
        Station height above the ellipsoid in m.
    pressure : float
        Surface pressure in hPa.
    wvp : float
        Surface water-vapour pressure in hPa.
    lambda_um : float
        Laser wavelength in µm.

    Returns
    -------
    total : float
        Zenith total delay in m.
    hydrostatic : float
        Zenith hydrostatic delay in m.
    wet : float
        Zenith non-hydrostatic (wet) delay in m.

    References
    ----------
    IERS Conventions (2010), Ch. 9, Sec. 9.1.
    """

def ortho_eop(time: float) -> NDArray[float64]:
    """Return the diurnal and semidiurnal EOP variations due to ocean tides.

    Parameters
    ----------
    time : float
        Epoch as a Modified Julian Date.

    Returns
    -------
    numpy.ndarray
        ``(Δx_p, Δy_p, ΔUT1)`` with polar motion in µas and UT1 in µs,
        shape ``(3,)``.

    References
    ----------
    IERS Conventions (2010), Ch. 8, Sec. 8.2.
    """

def pmsdnut2(rmjd: float) -> NDArray[float64]:
    """Return the polar-motion libration with periods below two days.

    Parameters
    ----------
    rmjd : float
        Epoch as a Modified Julian Date, used as TT for the fundamental
        arguments and as UT1 for GMST.

    Returns
    -------
    numpy.ndarray
        Polar-motion corrections ``(Δx_p, Δy_p)`` in µas, shape ``(2,)``.

    References
    ----------
    IERS Conventions (2010), Ch. 5, Table 5.1a.
    """

def utlibr(rmjd: float) -> tuple[float, float]:
    """Return the subdiurnal libration in UT1 and LOD.

    Parameters
    ----------
    rmjd : float
        Epoch as a Modified Julian Date, used as TT for the fundamental
        arguments and as UT1 for GMST.

    Returns
    -------
    dut1 : float
        UT1 correction in µs.
    dlod : float
        Length-of-day correction in µs per day.

    References
    ----------
    IERS Conventions (2010), Ch. 5, Table 5.1b.
    """

def fundarg(t: float) -> tuple[float, float, float, float, float]:
    """Return the Delaunay fundamental arguments of lunisolar nutation.

    Parameters
    ----------
    t : float
        TT Julian centuries since J2000.0.

    Returns
    -------
    tuple of float
        ``(l, l', F, D, Ω)`` in rad.

    References
    ----------
    IERS Conventions (2010), Ch. 5, Eq. 5.43.
    """

def dehanttideinel(
    xsta: ArrayLike,
    yr: int,
    month: int,
    day: int,
    fhr: float,
    xsun: ArrayLike,
    xmon: ArrayLike,
) -> NDArray[float64]:
    """Return the solid Earth tide displacement of a station.

    Parameters
    ----------
    xsta : array_like
        Geocentric station position in the ITRF, m, shape ``(3,)``.
    yr, month, day : int
        UTC calendar date.
    fhr : float
        UTC hour of day in ``[0, 24)``.
    xsun, xmon : array_like
        Geocentric positions of the Sun and the Moon in the ITRF, m,
        shape ``(3,)``.

    Returns
    -------
    numpy.ndarray
        Station displacement in the ITRF, m, shape ``(3,)``.

    Raises
    ------
    ValueError
        If ``fhr`` is not finite or outside ``[0, 24)``.

    References
    ----------
    IERS Conventions (2010), Ch. 7, Sec. 7.1.1.
    """

def hardisp(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    n: int,
    sample: float,
    blq_amp: ArrayLike,
    blq_phase: ArrayLike,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Return an ocean-loading displacement series at regular UTC epochs.

    Parameters
    ----------
    year, month, day, hour, minute, second : int
        UTC start epoch; ``second`` must not be 60.
    n : int
        Number of samples.
    sample : float
        Sampling interval in s.
    blq_amp : array_like
        BLQ amplitudes in m, shape ``(3, 11)`` with rows Up, West, South.
    blq_phase : array_like
        BLQ phases in degrees (Greenwich lags), shape ``(3, 11)``.

    Returns
    -------
    up, west, south : numpy.ndarray
        Displacements in m in BLQ axis order, each of shape ``(n,)``.

    Raises
    ------
    ValueError
        If the epochs, sample count, interval or BLQ arrays are invalid.

    References
    ----------
    IERS Conventions (2010), Ch. 7, Sec. 7.1.2.
    """
