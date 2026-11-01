import erfa
import numpy as np
import pytest

from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.empirical_libration import (
    EmpiricalLibrationParametrization,
    _parse_term,
)
from lunarops.classes.time import Epoch, TimeScale

MAS = np.deg2rad(1.0 / 3_600_000.0)


def _eq(partial=(1.0e3, -2.0e3, 3.0e3), iso="2020-03-01T00:00:00"):
    return ObservationEquation(
        observed_minus_computed_one_way_m=0.0,
        sigma_one_way_m=0.01,
        design_partials={"lunar_orientation_pa": np.asarray(partial)},
        observation_id=1,
        station_key="S",
        reflector_key="R",
        transmit_epoch_utc=Epoch.from_isot(iso, scale=TimeScale.UTC),
    )


def _block(**extra):
    terms = [
        {"name": "lp365", "axis": "psi", "delaunay": {"lp": 1}, **extra},
        {"name": "p2190", "axis": "x", "periodDays": 2190.0},
    ]
    return EmpiricalLibrationParametrization(terms=[_parse_term({"estimate": True, "estimateSin": True, "cosMas": 0.0, "sinMas": 0.0, **t}) for t in terms])


def test_design_uses_axis_partial_and_delaunay_argument():
    block = _block()
    eq = _eq()
    entries = dict(block.design_entries(eq))
    jd = eq.transmit_epoch_utc.jd1 + eq.transmit_epoch_utc.jd2
    lp = erfa.falp03((jd - 2451545.0) / 36525.0)
    assert entries[0] == pytest.approx(3.0e3 * MAS * np.cos(lp))
    assert entries[1] == pytest.approx(3.0e3 * MAS * np.sin(lp))
    arg = 2 * np.pi * (jd - 2451545.0) / 2190.0
    assert entries[2] == pytest.approx(1.0e3 * MAS * np.cos(arg))
    assert entries[3] == pytest.approx(1.0e3 * MAS * np.sin(arg))


def test_reduction_tracks_applied_coefficients():
    block = _block()
    eq = _eq()
    assert block.reduce_observation(eq) == 0.0
    block.apply_update(np.array([2.0, -1.0, 0.5, 0.0]))
    design = dict(block.design_entries(eq))
    expected = 2.0 * design[0] - 1.0 * design[1] + 0.5 * design[2]
    assert block.reduce_observation(eq) == pytest.approx(expected)
    np.testing.assert_allclose(block.reference_values(), [2.0, -1.0, 0.5, 0.0])


def test_fixed_term_is_applied_but_not_estimated():
    terms = [
        _parse_term({"name": "fixed", "delaunay": {"lp": 1}, "cosMas": 4.5, "sinMas": 0.0, "estimate": False}),
        _parse_term({"name": "free", "periodDays": 1306.0}),
    ]
    block = EmpiricalLibrationParametrization(terms=terms)
    assert [str(n) for n in block.parameter_names()] == [
        "moon:libration.free.cos::",
        "moon:libration.free.sin::",
    ]
    eq = _eq()
    jd = eq.transmit_epoch_utc.jd1 + eq.transmit_epoch_utc.jd2
    lp = erfa.falp03((jd - 2451545.0) / 36525.0)
    assert block.reduce_observation(eq) == pytest.approx(3.0e3 * MAS * 4.5 * np.cos(lp))


def test_invalid_term_definitions_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        _parse_term({"name": "bad", "delaunay": {"l": 1}, "periodDays": 10.0})
    with pytest.raises(ValueError, match="unknown Delaunay"):
        _parse_term({"name": "bad", "delaunay": {"q": 1}})
