"""Partials of the one-way range with respect to corrections of the tabulated Earth-orientation parameters.

The ITRS-to-GCRS rotation is ``Q R3(-ERA) W`` with ``W = R3(-s') R2(x_p) R1(y_p)`` (IERS Conventions 2010, Ch. 5).
To first order a correction of UT1 or of the polar motion moves the station in the GCRS as if its ITRS position
``r = (x, y, z)`` were displaced by

* ``dERA (z x r)`` for ``dUT1``, with ``dERA = 2 pi 1.00273781191135448 dUT1 / 86400 s``;
* ``dx_p (-z, 0, x)`` for ``dx_p`` (``R2``);
* ``dy_p (0, z, -y)`` for ``dy_p`` (``R1``),

so each partial is the station-position partial ``d rho / d r`` (ITRS) times that displacement (``W`` is taken as the
identity, a 1e-6 relative error). Units: UT1 corrections in milliseconds of time, polar motion in milliarcseconds.

What one station determines
    All three displacements are perpendicular to ``r``: they lie in the station's tangent plane, so the partials of
    one station have rank two. A station night determines the east and north displacement of the station, i.e. the
    classical ``UT0`` and variation of latitude ``VOL`` (as in the nightly UT0/VOL determinations from LLR, e.g. of
    the IfE Hannover and JPL),

    ``dUT0 = dUT1 + tan(phi) (sin(lambda) dx_p + cos(lambda) dy_p) / (dERA/dUT1)``,
    ``dVOL = cos(lambda) dx_p - sin(lambda) dy_p``,

    with geocentric latitude ``phi`` and longitude ``lambda``, and not UT1 and the polar motion separately. The
    corrections are estimated as random effects with the independent tabulated sigmas of UT1, x_p and y_p as prior;
    this gives the same fit as UT0 and VOL with that prior propagated to them, and the split of a night's correction
    into UT1 and polar motion is set by the prior alone. They are therefore reported as ``dUT0`` and ``dVOL``
    (:func:`rotation_combinations`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lunarops.classes.observation.equations import EquationBatch
from lunarops.estimation.lsq.blocks import RowBlocks
from lunarops.estimation.lsq.random_effects import RandomEffectElimination, RandomEffectLayout

#: Earth rotation angle rate with respect to UT1, in radians per second of UT1 (IERS 2010, Eq. 5.15).
ERA_RATE_RAD_PER_UT1_S = 2.0 * np.pi * 1.00273781191135448 / 86400.0
_MAS_TO_RAD = np.deg2rad(1.0 / 3_600_000.0)
_UT1_MS_TO_RAD = ERA_RATE_RAD_PER_UT1_S * 1.0e-3
#: The combinations of the corrections one station determines: UT0 (ms) and the variation of latitude (mas).
ROTATION_COMBINATIONS = ("ut0Ms", "volMas")


def _displacements(station_itrf_m: np.ndarray) -> np.ndarray:
    """Return the ITRS displacement of each station per unit ``dUT1`` (ms), ``dx_p``, ``dy_p`` (mas), shape (n, 3, 3).

    Column ``k`` of each 3x3 block is the displacement of effect ``k``, in metres.
    """
    x, y, z = np.atleast_2d(np.asarray(station_itrf_m, dtype=float)).T
    zero = np.zeros_like(x)
    ut1 = np.stack([-y, x, zero], axis=1) * _UT1_MS_TO_RAD
    xp = np.stack([-z, zero, x], axis=1) * _MAS_TO_RAD
    yp = np.stack([zero, z, -y], axis=1) * _MAS_TO_RAD
    return np.stack([ut1, xp, yp], axis=2)


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
    return np.einsum("ij,ijk->ik", np.asarray(station_partials, dtype=float), _displacements(station_itrf_m))


def rotation_combinations(station_itrf_m: np.ndarray) -> np.ndarray:
    """Return the matrices mapping ``(dUT1 ms, dx_p mas, dy_p mas)`` to ``(dUT0 ms, dVOL mas)``, shape (n, 2, 3).

    ``dUT0`` is the east displacement of the station expressed as a UT1 correction, ``dVOL`` its north displacement
    expressed as a latitude change (geocentric east and north at the station); see the module docstring.

    Parameters
    ----------
    station_itrf_m : ndarray, shape (n, 3) or (3,)
        Station ITRS positions, in metres, off the rotation axis.
    """
    position = np.atleast_2d(np.asarray(station_itrf_m, dtype=float))
    x, y, z = position.T
    axis_distance = np.hypot(x, y)
    radius = np.linalg.norm(position, axis=1)
    east = np.stack([-y, x, np.zeros_like(x)], axis=1) / axis_distance[:, None]
    north = np.stack([-z * x, -z * y, axis_distance**2], axis=1) / (axis_distance * radius)[:, None]
    displacements = _displacements(position)
    ut0 = np.einsum("ij,ijk->ik", east, displacements) / (axis_distance * _UT1_MS_TO_RAD)[:, None]
    vol = np.einsum("ij,ijk->ik", north, displacements) / (radius * _MAS_TO_RAD)[:, None]
    return np.stack([ut0, vol], axis=1)


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


@dataclass(frozen=True, eq=False)
class StationNightCorrections:
    """Random corrections of UT1 and the polar motion per station night, with the tabulated sigmas as prior.

    Attributes
    ----------
    layout : RandomEffectLayout
        Station-night blocks; the corrections occupy its ``partial_slots``, the offsets of the night sessions (if
        any) the other slots.
    prior_precision : ndarray, shape (B, 3)
        Prior precision of the corrections of every station night, in 1/ms^2 (UT1) and 1/mas^2 (x_p, y_p).
    first_rows : ndarray of int, shape (B,)
        First row of every station night.
    positions : ndarray, shape (B, 3)
        ITRS position of the station of every night, in metres.
    """

    layout: RandomEffectLayout
    prior_precision: np.ndarray
    first_rows: np.ndarray
    positions: np.ndarray

    @classmethod
    def build(
        cls,
        equations: EquationBatch,
        nights: RowBlocks,
        station_positions: Mapping[str, np.ndarray],
        uncertainty: EarthOrientationUncertainty,
        sigma_scale: float,
        sessions: RowBlocks | None = None,
    ) -> StationNightCorrections:
        """Return the corrections of the station ``nights`` of ``equations``, plus the offsets of ``sessions``.

        The prior sigmas are the tabulated uncertainties at the first epoch of each station night, times
        ``sigma_scale``.

        Raises
        ------
        ValueError
            If the equations have no station position partials or a station has no position.
        """
        if equations.station_position_itrf.shape[1] != 3:
            raise ValueError("earthOrientationCorrection needs the station position partials.")
        stations = np.asarray(equations.station, dtype=object)
        missing = set(stations.tolist()) - set(station_positions)
        if missing:
            raise ValueError(f"No station positions for the Earth-orientation partials of {sorted(missing)}.")
        positions = np.asarray([station_positions[key] for key in stations.tolist()])
        partials = earth_orientation_partials(equations.station_position_itrf, positions)
        first = nights.order[nights.starts]
        sigmas = uncertainty.at((equations.epoch_jd1 + equations.epoch_jd2)[first])
        layout = RandomEffectLayout.with_partials(nights, partials, sessions)
        return cls(layout, 1.0 / (sigma_scale * sigmas) ** 2, first, positions[first])

    def records(
        self,
        equations: EquationBatch,
        elimination: RandomEffectElimination,
        effects: np.ndarray,
        cofactor: np.ndarray,
    ) -> list[dict[str, object]]:
        """Per station night: station, first epoch, count, and the corrections of UT0 and VOL with their sigmas.

        A station night determines only UT0 (ms) and the variation of latitude VOL (mas), not UT1 and the polar
        motion separately (see the module docstring), so the estimated corrections and their posterior and prior
        covariances are mapped to these two. The posterior sigmas are those of the weighted system, not scaled by
        ``sigma0_post``: the prior sigmas are absolute, so a combination the data do not determine keeps its prior
        sigma.
        """
        layout = self.layout
        slots = list(layout.partial_slots)
        stations = [str(station) for station in layout.blocks.key]
        mapping = rotation_combinations(self.positions)
        values = np.einsum("bjk,bk->bj", mapping, effects[:, slots])
        posterior = np.einsum("bjk,bkl,bjl->bj", mapping, elimination.posterior_covariance(cofactor, slots), mapping)
        prior = np.einsum("bjk,bk,bjk->bj", mapping, 1.0 / self.prior_precision, mapping)
        counts = np.bincount(layout.blocks.index, minlength=layout.blocks.count)
        records = []
        for block, station in enumerate(stations):
            record: dict[str, object] = {
                "station": station,
                "firstEpochUtc": equations.epoch(int(self.first_rows[block])).isot(precision=3),
                "observationCount": int(counts[block]),
            }
            for k, name in enumerate(ROTATION_COMBINATIONS):
                record[name] = float(values[block, k])
                record[f"{name}Sigma"] = float(np.sqrt(max(posterior[block, k], 0.0)))
                record[f"{name}PriorSigma"] = float(np.sqrt(prior[block, k]))
            records.append(record)
        return records


__all__ = [
    "ERA_RATE_RAD_PER_UT1_S",
    "ROTATION_COMBINATIONS",
    "EarthOrientationUncertainty",
    "StationNightCorrections",
    "earth_orientation_partials",
    "rotation_combinations",
]
