from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.frames import (
    EarthOrientationSample,
    TabulatedEarthOrientation,
    TerrestrialFrameTransform,
    earth_rotation_libration_eop_correction,
    ocean_tide_eop_correction,
)
from lunarops.classes.frames.relativistic import RelativisticFrameTransform
from lunarops.classes.relativistic.constants import GM_SUN


class _Ephemeris(Ephemeris):
    @property
    def source_file_path(self) -> Path:
        return Path("fake.eph")

    def body_state_bcrs(self, body_name: str, epoch_tdb: Epoch) -> BodyState:
        epoch_tdb.require_scale(TimeScale.TDB)
        positions = {
            "EARTH": [0.0, 0.0, 0.0],
            "SUN": [1.0e11, 0.0, 0.0],
            "MOON": [4.0e8, 0.0, 0.0],
        }
        return BodyState(
            np.asarray(positions[body_name.strip().upper()]),
            np.zeros(3),
        )

    def pa2lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        epoch_tdb.require_scale(TimeScale.TDB)
        return np.eye(3)


def test_tabulated_eop_public_names_and_validation():
    sample = EarthOrientationSample(60_000.0, 0.1, 0.2, -0.3)
    eop = TabulatedEarthOrientation((sample,), source_file_path="eop.txt")

    assert eop.source_file_path == Path("eop.txt")
    assert eop.mjd_utc_range == (60_000.0, 60_000.0)
    assert eop.ut1_minus_utc_s(Epoch(2_400_000.5, 60_000.0, TimeScale.UTC)) == pytest.approx(-0.3)
    with pytest.raises(ValueError, match="must be finite"):
        EarthOrientationSample(60_000.0, np.nan, 0.2, 0.0)


def test_high_frequency_public_api_requires_explicit_background_dut1():
    epoch_utc = Epoch(2_400_000.5, 55_227.4, TimeScale.UTC)
    with pytest.raises(TypeError):
        cast(Any, ocean_tide_eop_correction)(epoch_utc)
    correction = ocean_tide_eop_correction(
        epoch_utc,
        background_ut1_minus_utc_s=-0.177,
    )
    assert np.isfinite(correction.delta_ut1_s)
    assert np.isfinite(
        earth_rotation_libration_eop_correction(Epoch(2_400_000.5, 55_227.4, TimeScale.TDB)).delta_xp_arcsec
    )


def test_gcrs2itrf_matrix_is_read_only():
    epoch = Epoch.from_calendar(2020, 1, 1, 6, scale=TimeScale.UTC)
    eop = TabulatedEarthOrientation((EarthOrientationSample(epoch.mjd, 0.076, 0.282, -0.177),))
    matrix = TerrestrialFrameTransform(eop).gcrs2itrf_matrix(epoch)
    assert not matrix.flags.writeable
    with pytest.raises(ValueError):
        matrix[0, 0] = 0.0


def test_external_gravitational_potential_normalizes_and_deduplicates_names():
    epoch = Epoch(2_450_000.5, 0.0, TimeScale.TDB)
    transform = RelativisticFrameTransform(_Ephemeris())
    potential = transform.external_gravitational_potential_m2_s2(" earth ", epoch, (" sun ",))
    assert potential == pytest.approx(GM_SUN / 1.0e11)
    deduplicated = transform.external_gravitational_potential_m2_s2("EARTH", epoch, ("SUN", " sun "))
    assert deduplicated == pytest.approx(potential)
    with pytest.raises(ValueError, match="must not also appear"):
        transform.external_gravitational_potential_m2_s2("EARTH", epoch, ("earth",))
