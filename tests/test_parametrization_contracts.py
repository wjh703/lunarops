import json

import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import Parametrization, ParametrizationList
from lunarops.classes.parametrization.station_range_bias import StationRangeBiasParametrization


def _eq(station="APOLLO"):
    return ObservationEquation(
        observed_minus_computed_one_way_m=0.0,
        sigma_one_way_m=1.0,
        design_partials={"station_range_bias": np.array([1.0])},
        observation_id=station,
        station_key=station,
        reflector_key="REF",
        transmit_epoch_utc=Epoch.from_isot("2008-01-01T00:00:00", scale=TimeScale.UTC),
    )


def assert_parametrization_contract(block: Parametrization, equations, context=None):
    block.setup(equations, context)
    names1 = block.parameter_names()
    names2 = block.parameter_names()
    assert names1 == names2
    for eq in equations:
        assert len(block.design_columns(eq)) == len(names1)
    assert block.reference_values() == pytest.approx(np.zeros(len(names1)))
    json.dumps(block.state(), default=str)
    with pytest.raises(ValueError):
        block.apply_update(np.zeros(len(names1) + 1))


def test_station_range_bias_parametrization_contract():
    assert_parametrization_contract(StationRangeBiasParametrization(per="station"), [_eq("APOLLO")])


class OtherParametrization(Parametrization):
    def parameter_names(self):
        return [ParameterName("other", "position.x")]

    def design_columns(self, equation):
        return np.array([1.0])

    def apply_update(self, delta):
        pass


def test_unselected_parametrization_still_reduces_observations():
    equation = _eq("APOLLO")
    bias = StationRangeBiasParametrization(per="station")
    other = OtherParametrization()
    parametrizations = ParametrizationList([bias, other])
    parametrizations.setup([equation], None)
    bias.apply_update(np.array([0.25]))

    selected = parametrizations.select_blocks((other.block_id,))
    selected.setup([equation], None)

    assert selected.parameter_names() == other.parameter_names()
    assert selected.reduced_observation(equation) == pytest.approx(-0.25)
