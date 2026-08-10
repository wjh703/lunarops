import numpy as np
import pytest

from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.observation.equations import ObservationEquation

_UTC_EPOCH = Epoch(2458849.5, 0.0, TimeScale.UTC)


def test_observation_equation_normalizes_and_freezes_partials():
    equation = ObservationEquation(
        observed_minus_computed_one_way_m=0.25,
        sigma_one_way_m=0.01,
        design_partials={"geometry": np.array([1.0, 2.0, 3.0])},
        observation_id=7,
        station_key="STA",
        reflector_key="REF",
        transmit_epoch_utc=_UTC_EPOCH,
        wavelength_nm=532.0,
    )

    assert equation.observed_minus_computed_one_way_m == 0.25
    assert equation.transmit_epoch_utc is _UTC_EPOCH
    assert np.allclose(equation.design_partials["geometry"], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        equation.design_partials["geometry"][0] = 9.0
