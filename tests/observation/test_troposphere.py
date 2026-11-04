"""Mendes-Pavlis optical troposphere."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from lunarops import _iers2010
from lunarops.classes.delays import TroposphereInput
from lunarops.classes.delays.troposphere import Iers2010MendesPavlisTroposphere
from tests.tolerances import DISPLACEMENT_M, assert_close

WEATHER = TroposphereInput(pressure_hpa=1013.25, temperature_k=293.15, relative_humidity_percent=50.0,
                           latitude_rad=np.deg2rad(45.0), height_m=100.0, wavelength_um=0.532)


def test_slant_delay_at_thirty_degrees_reproduces_the_reference_value():
    assert_close(Iers2010MendesPavlisTroposphere().slant_delay_m(WEATHER, elevation_rad=np.deg2rad(30.0)),
                 4.882704085022898, DISPLACEMENT_M)


def test_slant_delay_is_the_fcul_zenith_delay_times_the_fcul_mapping_function():
    model = Iers2010MendesPavlisTroposphere()
    elevation = np.deg2rad(30.0)
    zenith = model.slant_delay_m(WEATHER, elevation_rad=np.pi / 2) / _iers2010.fcul_a(45.0, 100.0, 293.15, 90.0)
    mapping = _iers2010.fcul_a(45.0, 100.0, 293.15, 30.0)
    # The FCUL mapping depends only on latitude, height, temperature and elevation.
    assert_close(model.slant_delay_m(WEATHER, elevation_rad=elevation), zenith * mapping, DISPLACEMENT_M)
    # A sea-level optical zenith delay is about 2.4 m.
    assert 2.3 < zenith < 2.5


def test_delay_decreases_with_elevation():
    model = Iers2010MendesPavlisTroposphere()
    delays = [model.slant_delay_m(WEATHER, elevation_rad=np.deg2rad(e)) for e in (3.0, 10.0, 30.0, 60.0, 90.0)]

    assert all(earlier > later for earlier, later in pairwise(delays))
    # The light-time solver clamps elevations to the mapping function's 3 degree validity limit.
    assert model.elevation_floor_rad == pytest.approx(np.deg2rad(3.0))


def test_temperature_outside_the_water_vapour_conversion_domain_is_rejected():
    with pytest.raises(ValueError, match="conversion domain"):
        Iers2010MendesPavlisTroposphere().slant_delay_m(replace(WEATHER, temperature_k=1e308),
                                                        elevation_rad=np.deg2rad(30.0))
