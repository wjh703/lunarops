import erfa
import numpy as np
import pytest

from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.empirical_libration import (
    EmpiricalLibrationParametrization,
    _parse_term,
)
from lunarops.classes.time import Epoch, TimeScale

MAS = np.deg2rad(1.0 / 3_600_000.0)


def _eq(partial=(1.0e3, -2.0e3, 3.0e3), iso="2020-03-01T00:00:00"):
    return EquationBatch.single(
        oc=0.0,
        sigma=0.01,
        lunar_orientation_pa=np.asarray(partial),
        observation_id=1,
        station="S",
        reflector="R",
        epoch=Epoch.from_isot(iso, scale=TimeScale.UTC),
        receive_epoch=Epoch.from_isot(iso, scale=TimeScale.UTC),
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
    entries = _design(block, eq)
    jd = eq.epoch(0).jd1 + eq.epoch(0).jd2
    lp = erfa.falp03((jd - 2451545.0) / 36525.0)
    assert entries[0] == pytest.approx(3.0e3 * MAS * np.cos(lp))
    assert entries[1] == pytest.approx(3.0e3 * MAS * np.sin(lp))
    arg = 2 * np.pi * (jd - 2451545.0) / 2190.0
    assert entries[2] == pytest.approx(1.0e3 * MAS * np.cos(arg))
    assert entries[3] == pytest.approx(1.0e3 * MAS * np.sin(arg))


def test_reduction_tracks_applied_coefficients():
    block = _block()
    eq = _eq()
    assert block.reduce(eq) == 0.0
    block.update(np.array([2.0, -1.0, 0.5, 0.0]))
    design = _design(block, eq)
    expected = 2.0 * design[0] - 1.0 * design[1] + 0.5 * design[2]
    assert block.reduce(eq) == pytest.approx(expected)
    np.testing.assert_allclose(block.x0(), [2.0, -1.0, 0.5, 0.0])


def test_fixed_term_is_applied_but_not_estimated():
    terms = [
        _parse_term({"name": "fixed", "delaunay": {"lp": 1}, "cosMas": 4.5, "sinMas": 0.0, "estimate": False}),
        _parse_term({"name": "free", "periodDays": 1306.0}),
    ]
    block = EmpiricalLibrationParametrization(terms=terms)
    assert [str(n) for n in block.names] == [
        "moon:libration.free.cos::",
        "moon:libration.free.sin::",
    ]
    eq = _eq()
    jd = eq.epoch(0).jd1 + eq.epoch(0).jd2
    lp = erfa.falp03((jd - 2451545.0) / 36525.0)
    assert block.reduce(eq) == pytest.approx(3.0e3 * MAS * 4.5 * np.cos(lp))


def test_invalid_term_definitions_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        _parse_term({"name": "bad", "delaunay": {"l": 1}, "periodDays": 10.0})
    with pytest.raises(ValueError, match="unknown Delaunay"):
        _parse_term({"name": "bad", "delaunay": {"q": 1}})


def _design(block, batch):
    out = np.zeros((len(batch), len(block.names)))
    block.design(batch, out)
    return out[0]


def test_batch_fundamentals_match_scalar_erfa_and_are_reused():
    batch = EquationBatch.concatenate([_eq(iso=iso) for iso in
                                       ("2012-01-01T00:00:00", "2019-02-03T12:34:56", "2020-03-01T00:00:00")])
    fundamental = batch.fundamental_arguments
    for actual, function in zip(fundamental, (erfa.fal03, erfa.falp03, erfa.faf03, erfa.fad03, erfa.faom03)):
        expected = np.asarray([function(((batch.epoch(i).jd1 + batch.epoch(i).jd2) - 2451545.0) / 36525.0)
                               for i in range(len(batch))])
        assert actual.tobytes() == expected.tobytes()
    block = _block()
    design = np.zeros((len(batch), len(block.names)))
    block.design(batch, design)
    block.reduce(batch)
    assert batch.fundamental_arguments is fundamental


def test_basis_reused_after_coefficient_update(monkeypatch):
    block, batch = _block(), _eq()
    design = _design(block, batch)

    def unexpected_trigonometry(*args):
        raise AssertionError("design and reduction must share the epoch basis")

    monkeypatch.setattr(np, "cos", unexpected_trigonometry)
    monkeypatch.setattr(np, "sin", unexpected_trigonometry)
    delta = np.array([2.0, -1.0, 0.5, 0.0])
    block.update(delta)
    expected = 2.0 * design[0] - design[1] + 0.5 * design[2]
    assert block.reduce(batch) == pytest.approx(expected)
