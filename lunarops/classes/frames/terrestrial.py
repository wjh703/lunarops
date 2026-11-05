"""ITRF/ITRS and GCRS transforms driven by explicit EOP and ERFA."""

from __future__ import annotations

import math
from collections import OrderedDict

import erfa
import numpy as np
from numpy.typing import ArrayLike

from lunarops.classes.time import Epoch, TdbTopocentricArguments, utc2tt

from .earth_orientation import EarthOrientationProvider
from .high_frequency_eop import HighFrequencyEopCorrection, high_frequency_eop_correction, require_utc_epoch

_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)


class TerrestrialFrameTransform:
    """Rotation between the GCRS and the ITRF from Earth-orientation parameters and ERFA.

    The rotation follows the IERS 2010 CIO-based procedure: IAU 2006/2000A CIP coordinates X, Y and CIO
    locator s (ERFA ``xys06a``) corrected by the tabulated dX, dY; the Earth rotation angle from UT1
    (ERFA ``era00``); and polar motion with the TIO locator s' (ERFA ``sp00``, ``pom00``). UT1 and polar
    motion include the ocean-tide and libration high-frequency corrections. The 16 most recent matrices are
    cached by UTC epoch.

    Parameters
    ----------
    earth_orientation_provider : EarthOrientationProvider
        Source of polar motion, UT1 - UTC and celestial pole offsets.

    References
    ----------
    .. [1] Petit, G., & Luzum, B. (Eds.) (2010). IERS Conventions (2010), IERS Technical Note 36, Ch. 5.
    """

    def __init__(self, earth_orientation_provider: EarthOrientationProvider) -> None:
        self.earth_orientation_provider = earth_orientation_provider
        self._matrix_cache: OrderedDict[tuple[float, float], np.ndarray] = OrderedDict()

    def _ut1_jd_and_high_frequency(
        self,
        epoch: Epoch,
        background_dut1_s: float,
    ) -> tuple[float, float, HighFrequencyEopCorrection]:
        high_frequency = high_frequency_eop_correction(
            epoch,
            background_ut1_minus_utc_s=background_dut1_s,
        )
        ut1_jd1, ut1_jd2 = erfa.utcut1(
            epoch.jd1,
            epoch.jd2,
            background_dut1_s + high_frequency.delta_ut1_s,
        )
        return float(ut1_jd1), float(ut1_jd2), high_frequency

    def ut1_jd(self, epoch_utc: Epoch) -> tuple[float, float]:
        """Return the two-part UT1 Julian Date of a UTC epoch.

        UT1 - UTC is the interpolated table value plus the ocean-tide and libration UT1 corrections.

        Parameters
        ----------
        epoch_utc : Epoch
            Epoch on the UTC scale.

        Returns
        -------
        tuple of float
            UT1 Julian Date parts ``(jd1, jd2)``, in days.
        """
        epoch = require_utc_epoch(epoch_utc)
        eop = self.earth_orientation_provider.interpolate(epoch)
        ut1_jd1, ut1_jd2, _ = self._ut1_jd_and_high_frequency(epoch, eop.ut1_minus_utc_s)
        return ut1_jd1, ut1_jd2

    def tdb_topocentric_arguments(
        self,
        position_itrf_m: ArrayLike,
        epoch_utc: Epoch,
    ) -> TdbTopocentricArguments:
        """Return the ERFA ``dtdb`` topocentric arguments of a station.

        Longitude and the distances from the spin axis and the equatorial plane are taken from the ITRF
        Cartesian position; the UT1 fraction of day counts from 0h UT1.

        Parameters
        ----------
        position_itrf_m : array_like, shape (3,)
            Station position (X, Y, Z) in the ITRF, in metres.
        epoch_utc : Epoch
            Epoch on the UTC scale.

        Returns
        -------
        TdbTopocentricArguments
            Arguments for ``TimeScaleConverter``.
        """
        x_m, y_m, z_m = np.asarray(position_itrf_m, float)
        ut1_jd1, ut1_jd2 = self.ut1_jd(epoch_utc)
        # Julian Dates start at noon whereas dtdb's UT fraction starts at 0h.
        ut1_fraction_of_day = (math.fmod(ut1_jd1, 1.0) + math.fmod(ut1_jd2, 1.0) + 0.5) % 1.0
        return TdbTopocentricArguments(
            ut1_fraction_of_day=ut1_fraction_of_day,
            longitude_rad=float(math.atan2(y_m, x_m)),
            distance_from_spin_axis_km=float(math.hypot(x_m, y_m) / 1000.0),
            north_of_equatorial_plane_km=float(z_m / 1000.0),
        )

    def gcrs2itrf_matrix(self, epoch_utc: Epoch) -> np.ndarray:
        """Return the GCRS-to-ITRF rotation matrix at a UTC epoch.

        Parameters
        ----------
        epoch_utc : Epoch
            Epoch on the UTC scale.

        Returns
        -------
        ndarray, shape (3, 3)
            Read-only matrix ``R`` with ``r_itrf = R @ r_gcrs``.

        Raises
        ------
        ValueError
            If the epoch is not on UTC or is outside the Earth-orientation table.
        """
        epoch = require_utc_epoch(epoch_utc)
        key = (epoch.jd1, epoch.jd2)
        if key in self._matrix_cache:
            self._matrix_cache.move_to_end(key)
            return self._matrix_cache[key]
        tt = utc2tt(epoch)

        eop = self.earth_orientation_provider.interpolate(epoch)
        ut1_jd1, ut1_jd2, high_frequency = self._ut1_jd_and_high_frequency(epoch, eop.ut1_minus_utc_s)

        xp = (eop.xp_arcsec + high_frequency.delta_xp_arcsec) * _ARCSEC_TO_RAD
        yp = (eop.yp_arcsec + high_frequency.delta_yp_arcsec) * _ARCSEC_TO_RAD

        x, y, s = erfa.xys06a(tt.jd1, tt.jd2)
        x += eop.dx_arcsec * _ARCSEC_TO_RAD
        y += eop.dy_arcsec * _ARCSEC_TO_RAD
        celestial_to_intermediate = erfa.c2ixys(x, y, s)

        era = erfa.era00(ut1_jd1, ut1_jd2)
        tio_locator = erfa.sp00(tt.jd1, tt.jd2)
        polar_motion = erfa.pom00(xp, yp, tio_locator)
        matrix = erfa.c2tcio(celestial_to_intermediate, era, polar_motion)
        matrix.setflags(write=False)
        self._matrix_cache[key] = matrix
        if len(self._matrix_cache) > 16:
            self._matrix_cache.popitem(last=False)
        return matrix

    def gcrs2itrf(self, position_gcrs_m: ArrayLike, epoch_utc: Epoch) -> np.ndarray:
        """Rotate a GCRS vector, shape (3,), to the ITRF at a UTC epoch; units are preserved."""
        matrix = self.gcrs2itrf_matrix(epoch_utc)
        return matrix @ np.asarray(position_gcrs_m, float)

    def itrf2gcrs(self, position_itrf_m: ArrayLike, epoch_utc: Epoch) -> np.ndarray:
        """Rotate an ITRF vector, shape (3,), to the GCRS at a UTC epoch; units are preserved."""
        matrix = self.gcrs2itrf_matrix(epoch_utc)
        return matrix.T @ np.asarray(position_itrf_m, float)


__all__ = ["TerrestrialFrameTransform"]
