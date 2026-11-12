"""Lunar body-fixed orientation and optional longitude-libration correction."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Protocol, cast

import erfa
import numpy as np

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.classes.time import Epoch, require_tdb_epoch

_MAS_TO_RAD = np.deg2rad(1.0 / 3_600_000.0)
_JULIAN_CENTURY_DAYS = 36_525.0


class LongitudeLibrationCorrection(StrEnum):
    """Periodic correction added to the kernel's lunar longitude libration: ``none`` or ``inpop21a``."""

    NONE = "none"
    INPOP21A = "inpop21a"


class LunarOrientationProvider(Protocol):
    """Interface of a source of lunar principal-axis (PA) orientation at TDB epochs."""

    def pa_to_lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        """Return the rotation from Moon PA axes to LCRS (ICRF-aligned) axes, shape (3, 3)."""
        ...

    def longitude_libration_correction_rad(self, epoch_tdb: Epoch) -> float:
        """Return the correction added to the longitude-libration angle, in radians."""
        ...

    @property
    def correction(self) -> LongitudeLibrationCorrection:
        """Longitude-libration correction applied by this provider."""
        ...


def normalize_longitude_libration_correction(
    value: LongitudeLibrationCorrection | str | None,
) -> LongitudeLibrationCorrection:
    """Return the longitude-libration correction named by ``value``.

    Parameters
    ----------
    value : LongitudeLibrationCorrection or str or None
        Correction or its case-insensitive name; ``None`` means ``"none"``.

    Returns
    -------
    LongitudeLibrationCorrection
        Parsed correction.

    Raises
    ------
    TypeError
        If ``value`` is not a string, enum member or ``None``.
    ValueError
        If the name is not a supported correction.
    """
    if isinstance(value, LongitudeLibrationCorrection):
        return value
    if value is None:
        return LongitudeLibrationCorrection.NONE
    if not isinstance(value, str):
        raise TypeError("longitude-libration correction must be a string, enum, or None")
    try:
        return LongitudeLibrationCorrection(value.strip().lower())
    except ValueError:
        allowed = ", ".join(item.value for item in LongitudeLibrationCorrection)
        raise ValueError(f"Unsupported longitude-libration correction {value!r}; expected one of: {allowed}.") from None


class EulerAngleLunarOrientation:
    """Lunar principal-axis orientation from the Euler angles of a binary PCK orientation kernel.

    The kernel (read by :class:`~lunarops.classes.ephemerides.chebyshev.ChebyshevEphemeris`)
    supplies 3-1-3 Euler angles (phi, theta, psi) of the Moon PA frame relative to ICRF and
    their rates. The ICRF-to-PA rotation is Rz(psi) Rx(theta) Rz(phi); with the ``inpop21a`` correction
    the correction angle is added to psi before the rotation is formed.

    Parameters
    ----------
    angles_at_epoch : callable
        Function of a TDB ``Epoch`` returning (phi, theta, psi, phi_dot, theta_dot, psi_dot), in rad and
        rad/s.
    angles_many_at_epochs : callable
        Function of a tuple of TDB epochs returning those values as an array, shape (n_epochs, 6).
    correction : LongitudeLibrationCorrection
        Longitude-libration correction to apply.
    j2000_epoch_tdb : Epoch or None
        J2000.0 (JD 2451545.0 TT) expressed on TDB, the origin of the correction's time argument. Required
        when ``correction`` is not ``NONE``.
    """

    def __init__(
        self,
        angles_at_epoch: Callable[[Epoch], np.ndarray],
        angles_many_at_epochs: Callable[[tuple[Epoch, ...]], np.ndarray],
        correction: LongitudeLibrationCorrection,
        j2000_epoch_tdb: Epoch | None,
    ) -> None:
        self._angles_at_epoch = angles_at_epoch
        self._angles_many_at_epochs = angles_many_at_epochs
        self._correction = correction
        self._j2000_epoch_tdb = cast(Epoch, j2000_epoch_tdb)

    @property
    def correction(self) -> LongitudeLibrationCorrection:
        """Longitude-libration correction applied by this provider."""
        return self._correction

    def longitude_libration_correction_rad(self, epoch_tdb: Epoch) -> float:
        """Return the correction added to the longitude-libration angle psi.

        For ``inpop21a`` the correction in milliarcseconds is
        4.5 cos(l') + 1.8 cos(2l - 2D) + 10.5 cos(2F - 2l), with the Delaunay arguments l, l', F, D of IERS
        Conventions (2010), Eq. 5.43, evaluated at Julian centuries of TDB since J2000.0.

        Parameters
        ----------
        epoch_tdb : Epoch
            Epoch on the TDB scale.

        Returns
        -------
        float
            Correction in radians; 0 when no correction is configured.

        Raises
        ------
        ValueError
            If ``epoch_tdb`` is not on the TDB scale.
        """
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        if self._correction is LongitudeLibrationCorrection.NONE:
            return 0.0
        centuries = (
            (epoch.jd1 - self._j2000_epoch_tdb.jd1) + (epoch.jd2 - self._j2000_epoch_tdb.jd2)
        ) / _JULIAN_CENTURY_DAYS
        lunar_anomaly, solar_anomaly, argument_latitude, elongation, _ = (
            float(value) for value in _iers2010.fundarg(float(centuries))
        )
        correction_mas = (
            4.5 * np.cos(solar_anomaly)
            + 1.8 * np.cos(2.0 * lunar_anomaly - 2.0 * elongation)
            + 10.5 * np.cos(2.0 * argument_latitude - 2.0 * lunar_anomaly)
        )
        return float(correction_mas * _MAS_TO_RAD)

    def angular_velocity_rad_s(self, epoch_tdb: Epoch) -> np.ndarray:
        """Return the lunar angular velocity on Moon PA axes.

        The angular velocity is formed from the kernel's Euler angles and rates. With the ``inpop21a``
        correction, psi and its rate include the correction and its analytic time derivative.

        Parameters
        ----------
        epoch_tdb : Epoch
            Epoch on the TDB scale.

        Returns
        -------
        ndarray, shape (3,)
            Angular velocity of the PA frame relative to ICRF, PA components, in rad/s (TDB seconds).
        """
        epoch = require_tdb_epoch(epoch_tdb)
        _phi, theta, psi, phi_dot, theta_dot, psi_dot = self._angles_at_epoch(epoch)
        if self._correction is not LongitudeLibrationCorrection.NONE:
            psi += self.longitude_libration_correction_rad(epoch)
            t = (
                (epoch.jd1 - self._j2000_epoch_tdb.jd1) + (epoch.jd2 - self._j2000_epoch_tdb.jd2)
            ) / _JULIAN_CENTURY_DAYS
            l, lp, f, d, _ = _iers2010.fundarg(t)
            # Derivatives of ERFA fal03/falp03/faf03/fad03 (IERS 2003),
            # arcsec/century converted to radians/second.
            polynomials = (
                (1717915923.2178, 31.8792, 0.051635, -0.00024470),
                (129596581.0481, -0.5532, 0.000136, -0.00001149),
                (1739527262.8478, -12.7512, -0.001037, 0.00000417),
                (1602961601.2090, -6.3706, 0.006593, -0.00003169),
            )
            rates = np.asarray([a + t * (2 * b + t * (3 * c + t * 4 * e)) for a, b, c, e in polynomials])
            dl, dlp, df, dd = rates * np.deg2rad(1 / 3600) / (_JULIAN_CENTURY_DAYS * 86400)
            psi_dot += _MAS_TO_RAD * (
                -4.5 * np.sin(lp) * dlp
                - 1.8 * np.sin(2 * l - 2 * d) * (2 * dl - 2 * dd)
                - 10.5 * np.sin(2 * f - 2 * l) * (2 * df - 2 * dl)
            )
        return np.array(
            [
                phi_dot * np.sin(theta) * np.sin(psi) + theta_dot * np.cos(psi),
                phi_dot * np.sin(theta) * np.cos(psi) - theta_dot * np.sin(psi),
                phi_dot * np.cos(theta) + psi_dot,
            ]
        )

    def pa_to_lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        """Return the rotation from Moon PA axes to LCRS (ICRF-aligned) axes.

        Parameters
        ----------
        epoch_tdb : Epoch
            Epoch on the TDB scale.

        Returns
        -------
        ndarray, shape (3, 3)
            Read-only matrix ``M`` with ``r_lcrs = M @ r_pa``.
        """
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        phi, theta, psi = np.asarray(self._angles_at_epoch(epoch), dtype=float)[:3]
        psi += self.longitude_libration_correction_rad(epoch)
        matrix = (erfa.rz(psi, np.eye(3)) @ erfa.rx(theta, np.eye(3)) @ erfa.rz(phi, np.eye(3))).T
        result = np.ascontiguousarray(matrix)
        result.setflags(write=False)
        return result

    def pa_to_lcrs_matrices(self, epochs) -> np.ndarray:
        """Return the PA-to-LCRS rotations at several TDB epochs, shape (n_epochs, 3, 3)."""
        epoch_array = tuple(epochs)
        angles = self._angles_many_at_epochs(epoch_array)
        matrices = np.empty((len(epoch_array), 3, 3), dtype=float)
        for index, (epoch, values) in enumerate(zip(epoch_array, angles, strict=True)):
            phi, theta, psi = values[:3]
            psi += self.longitude_libration_correction_rad(epoch)
            matrices[index] = (erfa.rz(psi, np.eye(3)) @ erfa.rx(theta, np.eye(3)) @ erfa.rz(phi, np.eye(3))).T
        return matrices


class FixedLunarOrientation:
    """Lunar orientation given by prescribed PA-to-LCRS matrices, without a longitude-libration correction.

    Parameters
    ----------
    matrix_at_epoch : callable
        Function of a TDB ``Epoch`` returning the PA-to-LCRS rotation, shape (3, 3).
    """

    correction = LongitudeLibrationCorrection.NONE

    def __init__(self, matrix_at_epoch: Callable[[Epoch], np.ndarray]):
        self._matrix_at_epoch = matrix_at_epoch

    def longitude_libration_correction_rad(self, epoch_tdb: Epoch) -> float:
        """Return 0 rad after checking that ``epoch_tdb`` is on the TDB scale."""
        require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        return 0.0

    def pa_to_lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        """Return the prescribed rotation from Moon PA axes to LCRS axes at a TDB epoch, shape (3, 3)."""
        return self._matrix_at_epoch(require_tdb_epoch(epoch_tdb, name="epoch_tdb"))

    def pa_to_lcrs_matrices(self, epochs) -> np.ndarray:
        """Return the prescribed PA-to-LCRS rotations at several TDB epochs, shape (n_epochs, 3, 3)."""
        return np.asarray([self.pa_to_lcrs_matrix(epoch) for epoch in epochs], dtype=float)


__all__ = [
    "EulerAngleLunarOrientation",
    "FixedLunarOrientation",
    "LongitudeLibrationCorrection",
    "LunarOrientationProvider",
    "normalize_longitude_libration_correction",
]
