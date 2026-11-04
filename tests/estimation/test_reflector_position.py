"""Reflector-position parametrization on the synthetic observation pipeline."""

import numpy as np
import pytest

from lunarops.classes.observation import NptDataset, ObservationOptions, ReflectorRecord
from lunarops.classes.parametrization.reflector_position import ReflectorPositionParametrization
from lunarops.config.registry import resolve
from tests.support.observation import npt_record, observation_processor
from tests.tolerances import REL, assert_close


def _equations(processor, *, partials=True):
    return processor.equations(NptDataset([npt_record()]), options=ObservationOptions(
        min_elevation_deg=-90.0, include_reflector_position_partials=partials))


def test_update_moves_the_catalog_entry_seen_by_the_resolver():
    processor = observation_processor()
    block = ReflectorPositionParametrization(model_state=processor.model_state, reflectors=["APOLLO15"])
    original = processor.model_state.reflector_catalog["APOLLO15"]
    block.setup(_equations(processor))

    assert_close(block.x0(), [1_737_400.0, 0.0, 0.0], REL)
    block.update(np.array([1.0, 2.0, 3.0]))

    updated = processor.model_state.reflector_catalog["APOLLO15"]
    assert updated is not original
    assert_close(updated.moon_fixed_xyz_m, [1_737_401.0, 2.0, 3.0], REL)
    assert_close(block.x0(), [1_737_401.0, 2.0, 3.0], REL)
    assert processor.resolver.resolve(npt_record()).reflector is updated


@pytest.mark.parametrize(("reflectors", "error"), [
    ("APOLLO15", TypeError), (123, TypeError), ([], ValueError), (["APOLLO15", "APOLLO15"], ValueError),
])
def test_configured_selection_is_validated(reflectors, error):
    with pytest.raises(error):
        resolve("parametrization", {"type": "reflectorPosition", "reflectors": reflectors})
    with pytest.raises(ValueError, match="unknown configuration key"):
        resolve("parametrization", {"type": "reflectorPosition", "reflector": ["APOLLO15"]})


def test_unobserved_reflectors_and_missing_partials_are_reported():
    processor = observation_processor()
    processor.model_state.reflector_catalog["UNOBSERVED"] = ReflectorRecord(name="Unobserved",
                                                                            moon_fixed_xyz_m=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="have no observations"):
        ReflectorPositionParametrization(model_state=processor.model_state,
                                         reflectors=["UNOBSERVED"]).setup(_equations(processor))

    equations = _equations(processor, partials=False)
    block = ReflectorPositionParametrization(model_state=processor.model_state, reflectors=["APOLLO15"])
    block.setup(equations)
    with pytest.raises(ValueError, match="broadcast"):
        block.design(equations, np.zeros((len(equations), len(block.names))))
