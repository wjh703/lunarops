import numpy as np
import pytest

from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.observation_factory import ensure_registered
from lunarops.classes.parametrization.station_range_bias import StationRangeBiasParametrization
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.registry import instantiate, validate_class_config


def _eq(station_key, epoch, station_id=None):
    return ObservationEquation(
        observed_minus_computed_one_way_m=0.0,
        sigma_one_way_m=1.0,
        design_partials={},
        observation_id=1,
        station_key=station_key,
        reflector_key="apollo15",
        transmit_epoch_utc=epoch,
    )


def test_station_mode_one_parameter_per_station():
    eqs = [_eq("APOLLO", Epoch(2454466.5, 0.0, TimeScale.UTC)), _eq("GRASSE", Epoch(2454832.5, 0.0, TimeScale.UTC))]
    block = StationRangeBiasParametrization(per="station")
    block.setup(eqs, None)

    assert block.keys == ["APOLLO", "GRASSE"]
    assert [str(name) for name in block.parameter_names()] == [
        "APOLLO:rangeBias::",
        "GRASSE:rangeBias::",
    ]
    assert block.design_entries(eqs[0]) == [(0, 1.0)]
    assert block.reference_values() == pytest.approx([0.0, 0.0])


def test_station_interval_mode_keeps_overlapping_explicit_intervals():
    # APOLLO on 2008-01-01 is inside both the broad 2006-2010 interval and
    # the shorter 2007-12-15 to 2008-06-30 interval.
    eq = _eq("APOLLO", Epoch(2454466.5, 0.0, TimeScale.UTC), station_id="7045")
    block = StationRangeBiasParametrization(
        per="station+interval",
        intervals=[
            {
                "station": "APOLLO",
                "start": "2006-04-07",
                "endExclusive": "2010-11-01",
            },
            {
                "station": "APOLLO",
                "start": "2007-12-15",
                "endExclusive": "2008-06-30",
            },
        ],
    )
    block.setup([eq], None)

    assert block.keys == [
        "APOLLO_2006-04-07_2010-11-01",
        "APOLLO_2007-12-15_2008-06-30",
    ]
    assert block.design_entries(eq) == [(0, 1.0), (1, 1.0)]

    block.apply_update(np.array([0.2, -0.05]))
    assert block.reduce_observation(eq) == 0.15000000000000002
    assert block.reference_values() == pytest.approx([0.2, -0.05])
    assert [str(name) for name in block.parameter_names()] == [
        "APOLLO:rangeBias:interval:2006-04-07/2010-11-01",
        "APOLLO:rangeBias:interval:2007-12-15/2008-06-30",
    ]


def test_station_mode_matches_exact_canonical_observation():
    eq = _eq("APOLLO", Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC), station_id="70610")
    block = StationRangeBiasParametrization(stations=["APOLLO"], per="station")
    block.setup([eq], None)

    assert block.keys == ["APOLLO"]
    assert [str(name) for name in block.parameter_names()] == ["APOLLO:rangeBias::"]
    assert block.design_entries(eq) == [(0, 1.0)]


def test_station_range_bias_rejects_empty_station_selection():
    ensure_registered()
    with pytest.raises(ValueError, match="at least 1 item"):
        _create("parametrization", {"type": "stationRangeBias", "stations": []})


def test_station_range_bias_rejects_intervals_in_station_mode():
    with pytest.raises(ValueError, match=r"require per='station\+interval'"):
        StationRangeBiasParametrization(
            per="station",
            intervals=[
                {
                    "station": "APOLLO",
                    "start": "2020-01-01",
                    "endExclusive": None,
                }
            ],
        )


def test_station_range_bias_rejects_alias_schema():
    ensure_registered()
    with pytest.raises(TypeError, match="must be a sequence"):
        _create(
            "parametrization",
            {
                "type": "stationRangeBias",
                "per": "station+interval",
                "intervals": {"APOLLO": ["2006-04-07/2010-11-01"]},
            },
        )


def _create(category, config):
    return instantiate(category, validate_class_config(category, config))
