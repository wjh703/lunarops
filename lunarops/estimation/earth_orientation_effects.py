"""Partials of the one-way range with respect to corrections of the tabulated Earth-orientation parameters.

The ITRS-to-GCRS rotation is ``Q R3(-ERA) W`` with ``W = R3(-s') R2(x_p) R1(y_p)`` (IERS Conventions 2010, Ch. 5).
To first order a correction of UT1 or of the polar motion moves the station in the GCRS as if its ITRS position
``r = (x, y, z)`` were displaced by

* ``dERA (z x r)`` for ``dUT1``, with ``dERA = 2 pi 1.00273781191135448 dUT1 / 86400 s``;
* ``dx_p (-z, 0, x)`` for ``dx_p`` (``R2``);
* ``dy_p (0, z, -y)`` for ``dy_p`` (``R1``),

so each partial is the station-position partial ``d rho / d r`` (ITRS) times that displacement (``W`` is taken as the
identity, a 1e-6 relative error). Units: UT1 corrections in milliseconds of time, polar motion in milliarcseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Earth rotation angle rate with respect to UT1, in radians per second of UT1 (IERS 2010, Eq. 5.15).
ERA_RATE_RAD_PER_UT1_S = 2.0 * np.pi * 1.00273781191135448 / 86400.0
_MAS_TO_RAD = np.deg2rad(1.0 / 3_600_000.0)
EARTH_ORIENTATION_COMPONENTS = ("ut1Ms", "xpMas", "ypMas")


def earth_orientation_partials(station_partials: np.ndarray, station_itrf_m: np.ndarray) -> np.ndarray:
    """Return the partials of the one-way range with respect to ``dUT1`` (ms), ``dx_p`` and ``dy_p`` (mas).

    Parameters
    ----------
    station_partials : ndarray, shape (n, 3)
        Partials of the one-way range with respect to the station ITRS coordinates, in m/m.
    station_itrf_m : ndarray, shape (n, 3)
        Station ITRS positions, in metres.

    Returns
    -------
    ndarray, shape (n, 3)
        Partials in metres per millisecond (UT1) and metres per milliarcsecond (x_p, y_p).
    """
    x, y, z = np.asarray(station_itrf_m, dtype=float).T
    zero = np.zeros_like(x)
    ut1 = np.stack([-y, x, zero], axis=1) * (ERA_RATE_RAD_PER_UT1_S * 1.0e-3)
    xp = np.stack([-z, zero, x], axis=1) * _MAS_TO_RAD
    yp = np.stack([zero, z, -y], axis=1) * _MAS_TO_RAD
    partials = np.asarray(station_partials, dtype=float)
    return np.stack([np.einsum("ij,ij->i", partials, d) for d in (ut1, xp, yp)], axis=1)


@dataclass(frozen=True, eq=False)
class EarthOrientationUncertainty:
    """Tabulated uncertainties of UT1 (ms) and of the polar motion (mas), interpolated linearly in time.

    Attributes
    ----------
    mjd_utc : ndarray, shape (k,)
        Epochs of the table, UTC Modified Julian Dates, increasing.
    sigmas : ndarray, shape (k, 3)
        Sigmas of UT1 (ms), x_p and y_p (mas) at the epochs.
    """

    mjd_utc: np.ndarray
    sigmas: np.ndarray

    @classmethod
    def from_c04(cls, path: str | Path) -> EarthOrientationUncertainty:
        """Read the ``x Er``, ``y Er`` and ``UT1-UTC Er`` columns of an IERS 20 C04 file.

        Raises
        ------
        ValueError
            If the file has no data rows, a row has fewer than 16 columns, or an uncertainty is not positive.
        """
        epochs, values = [], []
        for line in Path(path).expanduser().read_text(encoding="ascii").splitlines():
            if not line.strip() or line.lstrip().startswith(("#", "%")):
                continue
            fields = line.split()
            if len(fields) < 16:
                raise ValueError(f"{path}: C04 row with {len(fields)} columns, expected the 20 C04 layout.")
            epochs.append(float(fields[4]))
            values.append((float(fields[15]) * 1.0e3, float(fields[13]) * 1.0e3, float(fields[14]) * 1.0e3))
        if not epochs:
            raise ValueError(f"{path}: no C04 rows.")
        sigmas = np.asarray(values)
        if not np.all(sigmas > 0.0):
            raise ValueError(f"{path}: C04 uncertainties must be positive.")
        return cls(np.asarray(epochs), sigmas)

    def at(self, epoch_jd: np.ndarray) -> np.ndarray:
        """Return the sigmas ``(UT1 ms, x_p mas, y_p mas)`` at UTC Julian dates, shape ``(n, 3)``.

        Raises
        ------
        ValueError
            If an epoch lies outside the table.
        """
        mjd = np.asarray(epoch_jd, dtype=float) - 2_400_000.5
        if np.any(mjd < self.mjd_utc[0]) or np.any(mjd > self.mjd_utc[-1]):
            raise ValueError("Epoch outside the Earth-orientation uncertainty table.")
        return np.stack([np.interp(mjd, self.mjd_utc, self.sigmas[:, k]) for k in range(3)], axis=1)


__all__ = [
    "EARTH_ORIENTATION_COMPONENTS",
    "ERA_RATE_RAD_PER_UT1_S",
    "EarthOrientationUncertainty",
    "earth_orientation_partials",
]
