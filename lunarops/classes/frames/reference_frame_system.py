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
    """Time and reference-frame services shared by the observation model of one run.

    Parameters
    ----------
    ephemeris : Ephemeris
        Barycentric body states at TDB epochs.
    earth_orientation_provider : EarthOrientationProvider
        Earth-orientation parameters at UTC epochs.
    lunar_orientation : LunarOrientationProvider
        Moon PA-to-LCRS orientation.
    lunar_scale : LunarRelativisticScale
        BCRS<->LCRS coordinate scaling.

    Attributes
    ----------
    ephemeris, earth_orientation_provider, lunar_orientation, lunar_scale
        The constructor arguments.
    time_scale_converter : TimeScaleConverter
        UTC/TT/TDB converter.
    terrestrial_transform : TerrestrialFrameTransform
        ITRF<->GCRS rotation built from ``earth_orientation_provider``.
    relativistic_transform : RelativisticFrameTransform
        BCRS<->GCRS and BCRS<->LCRS spatial transformations built from ``ephemeris`` and ``lunar_scale``.
    """

    def __init__(
        self,
        ephemeris: Ephemeris,
        earth_orientation_provider: EarthOrientationProvider,
        lunar_orientation: LunarOrientationProvider,
        lunar_scale: LunarRelativisticScale,
    ) -> None:
        self.ephemeris = ephemeris
        self.earth_orientation_provider = earth_orientation_provider
        self.lunar_orientation = lunar_orientation
        self.lunar_scale = lunar_scale
        self.time_scale_converter = TimeScaleConverter()
        self.terrestrial_transform = TerrestrialFrameTransform(earth_orientation_provider)
        self.relativistic_transform = RelativisticFrameTransform(ephemeris, lunar_scale)


__all__ = ["ReferenceFrameSystem"]
