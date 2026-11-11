"""A static synthetic solar system and the LLR observation pipeline built on it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lunarops.classes.displacement import CompositeStationDisplacement
from lunarops.classes.ephemerides import FixedLunarOrientation, LunarRelativisticScale
from lunarops.classes.frames import EarthOrientationProvider, EarthOrientationValues, ReferenceFrameSystem
from lunarops.classes.observation import (
    LightTimeSolver,
    LlrObservationModel,
    LlrObservationProcessor,
    NptRecord,
    ObservationCatalogState,
    ObservationResolver,
    ReflectorRecord,
    StationRecord,
)
from lunarops.classes.range_bias.table import RangeBiasTable
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.context import RunContext
from tests.support.synthetic import LinearEphemeris

EARTH_MOON_M = 384_400_000.0
TRANSMIT_UTC = Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC)
# Earth at the SSB, the Moon on +x and the Sun beyond it, all at rest.
STATIC_BODIES = {
    "EARTH": ((0.0, 0.0, 0.0), (0, 0, 0)),
    "MOON": ((EARTH_MOON_M, 0.0, 0.0), (0, 0, 0)),
    "SUN": ((149_597_870_700.0, 0.0, 0.0), (0, 0, 0)),
    "MERCURY BARYCENTER": ((5.0e10, 2.0e10, 0.0), (0, 0, 0)),
    "VENUS BARYCENTER": ((1.0e11, 3.0e10, 0.0), (0, 0, 0)),
    "MARS BARYCENTER": ((2.0e11, 4.0e10, 0.0), (0, 0, 0)),
    "JUPITER BARYCENTER": ((7.0e11, 5.0e10, 0.0), (0, 0, 0)),
    "SATURN BARYCENTER": ((1.4e12, 6.0e10, 0.0), (0, 0, 0)),
    "URANUS BARYCENTER": ((2.8e12, 7.0e10, 0.0), (0, 0, 0)),
    "NEPTUNE BARYCENTER": ((4.5e12, 8.0e10, 0.0), (0, 0, 0)),
}


class ZeroEarthOrientation(EarthOrientationProvider):
    """Zero polar motion, UT1 = UTC and no celestial pole offsets at every epoch."""

    @property
    def source_file_path(self) -> Path:
        return Path("zero.eop")

    def interpolate(self, epoch_utc) -> EarthOrientationValues:
        return EarthOrientationValues(0.0, 0.0, 0.0)


def static_frames() -> ReferenceFrameSystem:
    return ReferenceFrameSystem(
        LinearEphemeris(STATIC_BODIES),
        ZeroEarthOrientation(),
        FixedLunarOrientation(lambda epoch: np.eye(3)),
        LunarRelativisticScale.from_convention("alreadyScaled"),
    )


def station_facing_the_moon(frames, *, zenith_angle_deg: float = 0.0) -> np.ndarray:
    """ITRF position on the equator whose GCRS direction is ``zenith_angle_deg`` from the Moon at transmit."""
    angle = np.deg2rad(zenith_angle_deg)
    gcrs = 6_378_137.0 * np.array([np.cos(angle), np.sin(angle), 0.0])
    return frames.terrestrial_transform.gcrs2itrf(gcrs, TRANSMIT_UTC)


def npt_record(index: int = 4) -> NptRecord:
    return NptRecord(
        station_name="APOLLO",
        reflector_name="APOLLO15",
        transmit_epoch=TRANSMIT_UTC,
        round_trip_time_s=2.55,
        uncertainty_two_way_s=100.0e-12,
        pressure_hpa=900.0,
        temperature_k=285.0,
        humidity_percent=25.0,
        wavelength_nm=532.0,
        index=index,
        station_code="70610",
        reflector_code="A15",
    )


def observation_processor(
    *, frames=None, troposphere=None, station_displacement=None, station_itrf_m=(6_378_137.0, 0.0, 0.0)
) -> LlrObservationProcessor:
    frames = static_frames() if frames is None else frames
    solver = LightTimeSolver(
        frames,
        gravitational_delay_models=(),
        troposphere_delay_models=() if troposphere is None else (troposphere,),
        station_displacement_model=station_displacement or CompositeStationDisplacement(()),
        reflector_displacement_models=(),
    )
    state = ObservationCatalogState(
        {"APOLLO": StationRecord(name="Apache Point", itrf_xyz_m=tuple(station_itrf_m))},
        {"APOLLO15": ReflectorRecord(name="Apollo 15", moon_fixed_xyz_m=(1_737_400.0, 0.0, 0.0))},
    )
    return LlrObservationProcessor(ObservationResolver(state), LlrObservationModel(frames, solver, RangeBiasTable()))


class StaticKernelEphemeris(LinearEphemeris):
    """``STATIC_BODIES`` with the attributes of a kernel-directory ephemeris."""

    def __init__(self):
        super().__init__(STATIC_BODIES)
        self.lunar_orientation = FixedLunarOrientation(lambda epoch: np.eye(3))
        self.relativistic_scale = LunarRelativisticScale.from_convention("alreadyScaled")


class StaticKernelContext(RunContext):
    """Run context whose ephemeris and Earth-rotation declarations resolve to the static system."""

    def create_class(self, category, config, *, cache=True):
        if category == "ephemerides":
            return StaticKernelEphemeris()
        if category == "earthRotation":
            return ZeroEarthOrientation()
        return super().create_class(category, config, cache=cache)
