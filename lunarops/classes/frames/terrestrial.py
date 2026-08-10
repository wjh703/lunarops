"""ITRF/ITRS and GCRS transforms driven by explicit EOP and ERFA."""

from __future__ import annotations

import erfa
import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.array_validation import readonly_matrix3x3, vector3
from lunarops.classes.time import Epoch, TimeScale, utc2tt

from .earth_orientation import EarthOrientationProvider
from .high_frequency_eop import high_frequency_eop_correction

_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)


class TerrestrialFrameTransform:
    def __init__(self, earth_orientation_provider: EarthOrientationProvider) -> None:
        if not isinstance(earth_orientation_provider, EarthOrientationProvider):
            raise TypeError("earth_orientation_provider must be an EarthOrientationProvider instance.")
        self.earth_orientation_provider = earth_orientation_provider

    @staticmethod
    def _require_utc_epoch(epoch_utc: Epoch) -> Epoch:
        if not isinstance(epoch_utc, Epoch):
            raise TypeError("Frame transforms require an Epoch.")
        return epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")

    def gcrs2itrf_matrix(self, epoch_utc: Epoch) -> np.ndarray:
        """Return an IERS 2010 GCRS-to-ITRF rotation matrix."""
        epoch = self._require_utc_epoch(epoch_utc)
        tt = utc2tt(epoch)

        background_dut1_s = self.earth_orientation_provider.ut1_minus_utc_s(epoch)
        high_frequency = high_frequency_eop_correction(
            epoch,
            background_ut1_minus_utc_s=background_dut1_s,
        )
        dut1_s = background_dut1_s + high_frequency.delta_ut1_s
        ut1_jd1, ut1_jd2 = erfa.utcut1(epoch.jd1, epoch.jd2, dut1_s)

        pole = self.earth_orientation_provider.polar_motion(epoch)
        xp = (pole.xp_arcsec + high_frequency.delta_xp_arcsec) * _ARCSEC_TO_RAD
        yp = (pole.yp_arcsec + high_frequency.delta_yp_arcsec) * _ARCSEC_TO_RAD

        offsets = self.earth_orientation_provider.celestial_pole_offsets(epoch)
        x, y, s = erfa.xys06a(tt.jd1, tt.jd2)
        x += offsets.dx_arcsec * _ARCSEC_TO_RAD
        y += offsets.dy_arcsec * _ARCSEC_TO_RAD
        celestial_to_intermediate = erfa.c2ixys(x, y, s)

        era = erfa.era00(ut1_jd1, ut1_jd2)
        tio_locator = erfa.sp00(tt.jd1, tt.jd2)
        polar_motion = erfa.pom00(xp, yp, tio_locator)
        matrix = np.asarray(
            erfa.c2tcio(celestial_to_intermediate, era, polar_motion),
            dtype=float,
        )
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("ERFA returned an invalid celestial-to-terrestrial matrix.")
        return readonly_matrix3x3(matrix, name="gcrs2itrf_matrix")

    def gcrs2itrf(self, position_gcrs_m: ArrayLike, epoch_utc: Epoch) -> np.ndarray:
        matrix = self.gcrs2itrf_matrix(epoch_utc)
        return matrix @ vector3(position_gcrs_m, name="position_gcrs_m")

    def itrf2gcrs(self, position_itrf_m: ArrayLike, epoch_utc: Epoch) -> np.ndarray:
        matrix = self.gcrs2itrf_matrix(epoch_utc)
        return matrix.T @ vector3(position_itrf_m, name="position_itrf_m")


__all__ = ["TerrestrialFrameTransform"]
