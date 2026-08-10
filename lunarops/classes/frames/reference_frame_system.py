"""Facade combining time conversion and terrestrial/lunar/relativistic frames."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from lunarops.classes.time import Epoch, TimeScaleConverter
from lunarops.classes.ephemerides import Ephemeris

from .earth_orientation import EarthOrientationProvider
from .lunar import LunarFrameTransform
from .relativistic import RelativisticFrameTransform
from .terrestrial import TerrestrialFrameTransform


class ReferenceFrameSystem:
    def __init__(
        self,
        ephemeris: Ephemeris,
        earth_orientation_provider: EarthOrientationProvider,
    ) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        if not isinstance(earth_orientation_provider, EarthOrientationProvider):
            raise TypeError("earth_orientation_provider must be an EarthOrientationProvider instance.")
        self.ephemeris = ephemeris
        self.earth_orientation_provider = earth_orientation_provider
        self.time_scale_converter = TimeScaleConverter(ephemeris)
        self.terrestrial_transform = TerrestrialFrameTransform(earth_orientation_provider)
        self.lunar_transform = LunarFrameTransform(ephemeris)
        self.relativistic_transform = RelativisticFrameTransform(ephemeris)

    def itrf2gcrs(self, position_itrf_m: ArrayLike, epoch_utc: Epoch) -> np.ndarray:
        return self.terrestrial_transform.itrf2gcrs(position_itrf_m, epoch_utc)

    def gcrs2itrf(self, position_gcrs_m: ArrayLike, epoch_utc: Epoch) -> np.ndarray:
        return self.terrestrial_transform.gcrs2itrf(position_gcrs_m, epoch_utc)

    def pa2lcrs(self, position_pa_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.lunar_transform.pa2lcrs(position_pa_m, epoch_tdb)

    def lcrs2pa(self, position_lcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.lunar_transform.lcrs2pa(position_lcrs_m, epoch_tdb)

    def gcrs2bcrs(self, position_gcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.gcrs2bcrs(position_gcrs_m, epoch_tdb)

    def bcrs2gcrs(self, position_bcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.bcrs2gcrs(position_bcrs_m, epoch_tdb)

    def lcrs2bcrs(self, position_lcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.lcrs2bcrs(position_lcrs_m, epoch_tdb)

    def bcrs2lcrs(self, position_bcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.bcrs2lcrs(position_bcrs_m, epoch_tdb)

    def lcrs2gcrs(self, position_lcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.lcrs2gcrs(position_lcrs_m, epoch_tdb)

    def gcrs2lcrs(self, position_gcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.gcrs2lcrs(position_gcrs_m, epoch_tdb)

    def external_gravitational_potential_m2_s2(
        self,
        center_body_name: str,
        epoch_tdb: Epoch,
        perturbing_body_names: Iterable[str],
    ) -> float:
        return self.relativistic_transform.external_gravitational_potential_m2_s2(
            center_body_name,
            epoch_tdb,
            perturbing_body_names,
        )


__all__ = ["ReferenceFrameSystem"]
