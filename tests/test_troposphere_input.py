from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

from lunarops.classes.delays import TroposphereInput
from lunarops.classes.delays.troposphere import Iers2010MendesPavlisTroposphere


def _input(**changes):
    values = {
        "pressure_hpa": 1013.25,
        "temperature_k": 293.15,
        "relative_humidity_percent": 50.0,
        "latitude_rad": np.deg2rad(45.0),
        "height_m": 100.0,
        "wavelength_um": 0.532,
    }
    values.update(changes)
    return TroposphereInput(**values)


def test_troposphere_input_is_frozen_and_slotted():
    data = _input()

    assert not hasattr(data, "__dict__")
    with pytest.raises(FrozenInstanceError):
        cast(Any, data).pressure_hpa = 900.0


def test_troposphere_models_consume_input_object():
    data = _input()

    model = Iers2010MendesPavlisTroposphere()
    assert model.elevation_floor_rad == pytest.approx(np.deg2rad(3.0))
    delay = model.slant_delay_m(data, elevation_rad=np.deg2rad(30.0))
    assert delay == pytest.approx(4.882704085022898, rel=0.0, abs=1.0e-14)


def test_water_vapor_pressure_rejects_invalid_inputs():
    model = Iers2010MendesPavlisTroposphere()
    with pytest.raises(ValueError, match="conversion domain"):
        model._water_vapor_pressure_hpa(1.0e308, 50.0)


def test_water_vapor_pressure_conversion_is_explicit():
    actual = Iers2010MendesPavlisTroposphere._water_vapor_pressure_hpa(293.15, 50.0)

    assert actual == pytest.approx(11.686412364256276, rel=0.0, abs=1.0e-14)
