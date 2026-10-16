"""Shared time and frame services for an observation runtime."""

from __future__ import annotations



from lunarops.classes.ephemerides.ephemeris_provider import Ephemeris
from lunarops.classes.ephemerides.lunar_orientation import LunarOrientationProvider
from lunarops.classes.frames.coordinate_scaling import LunarRelativisticScale
from lunarops.classes.time import TimeScaleConverter

from .earth_orientation import EarthOrientationProvider
from .relativistic import RelativisticFrameTransform
from .terrestrial import TerrestrialFrameTransform


class ReferenceFrameSystem:
    def __init__(
        self,
        ephemeris: Ephemeris,
        earth_orientation_provider: EarthOrientationProvider,
        lunar_orientation: LunarOrientationProvider,
        lunar_scale: LunarRelativisticScale,
    ) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        if not isinstance(earth_orientation_provider, EarthOrientationProvider):
            raise TypeError("earth_orientation_provider must be an EarthOrientationProvider instance.")
        if not isinstance(lunar_orientation, LunarOrientationProvider):
            raise TypeError("lunar_orientation must implement LunarOrientationProvider.")
        if not isinstance(lunar_scale, LunarRelativisticScale):
            raise TypeError("lunar_scale must be a LunarRelativisticScale.")
        self.ephemeris = ephemeris
        self.earth_orientation_provider = earth_orientation_provider
        self.lunar_orientation = lunar_orientation
        self.lunar_scale = lunar_scale
        self.time_scale_converter = TimeScaleConverter()
        self.terrestrial_transform = TerrestrialFrameTransform(earth_orientation_provider)
        self.relativistic_transform = RelativisticFrameTransform(ephemeris, lunar_scale)



__all__ = ["ReferenceFrameSystem"]
