"""Parametrization blocks: shared contract, column layout, linear reduction and model updates."""

import json

import erfa
import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.observation.resolver import ObservationCatalogState
from lunarops.classes.parametrization.base import Parametrization, ParametrizationList
from lunarops.classes.parametrization.empirical_libration import (
    _TERM_SCHEMA,
    EmpiricalLibrationParametrization,
    _parse_term,
)
from lunarops.classes.parametrization.reflector_position import ReflectorPositionParametrization
from lunarops.classes.parametrization.station_position import StationPositionParametrization
from lunarops.classes.parametrization.station_range_bias import StationRangeBiasParametrization
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.registry import create, resolve
from lunarops.parallel.observation_spec import apply_catalog_state, snapshot_catalog_state
from tests.tolerances import REL, Tolerance, assert_close

MAS = np.deg2rad(1.0 / 3_600_000.0)


def _state():
    return ObservationCatalogState(
        {"A": StationRecord("A", [1.0e6, 2.0e6, 3.0e6], [0.01, 0.02, 0.03], "2000-01-01T00:00:00"),
         "B": StationRecord("B", [4.0e6, 5.0e6, 6.0e6], [0.0, 0.0, 0.0], "2000-01-01T00:00:00")},
        {"R": ReflectorRecord("R", [1.7e6, 0.0, 0.0])},
    )


def _eq(station="A", iso="2010-01-01T00:00:00", identity=None, **partials):
    epoch = Epoch.from_isot(iso, scale=TimeScale.UTC)
    defaults = {"station_position_itrf": (0.1, -0.2, 0.3), "reflector_position_pa": (0.5, 0.1, -0.2),
                "lunar_orientation_pa": (1.0e3, -2.0e3, 3.0e3), "lunar_ecliptic_libration": (4.0e2, 5.0e2, -6.0e2)}
    return EquationBatch.single(oc=0.0, sigma=0.01, observation_id=identity or f"{station}-{iso}", station=station,
                                reflector="R", epoch=epoch, receive_epoch=epoch,
                                **{k: np.asarray(v) for k, v in {**defaults, **partials}.items()})


def _term(item):
    return _parse_term(_TERM_SCHEMA.resolve(item, path="term"))


def _libration(**extra):
    return EmpiricalLibrationParametrization(terms=[
        _term({"name": "lp365", "axis": "psi", "delaunay": {"lp": 1}, **extra}),
        _term({"name": "p2190", "axis": "x", "periodDays": 2190.0}),
        _term({"name": "rho", "axis": "rho", "periodDays": 27.3}),
    ])


def _design(block, batch):
    out = np.zeros((len(batch), len(block.names)))
    block.design(batch, out)
    return out


BLOCKS = {
    "stationRangeBias": lambda state: StationRangeBiasParametrization(per="station"),
    "stationPosition": lambda state: StationPositionParametrization(model_state=state, stations=["A"],
                                                                    estimate_velocity=True),
    "reflectorPosition": lambda state: ReflectorPositionParametrization(model_state=state, reflectors=["R"]),
    "empiricalLibration": lambda state: _libration(),
}
BATCH = [("A", "2010-01-01T00:00:00"), ("A", "2012-06-01T12:00:00"), ("B", "2015-03-01T00:00:00")]


@pytest.mark.parametrize("kind", list(BLOCKS))
def test_every_block_honours_the_parametrization_contract(kind):
    state = _state()
    block = BLOCKS[kind](state)
    batch = EquationBatch.concatenate([_eq(*item) for item in BATCH])
    names = block.setup(batch)
    x0 = block.x0()
    dx = np.linspace(0.1, 0.3, len(names))

    assert names == block.names and len(names) == len(x0) > 0
    assert _design(block, batch).shape == (len(batch), len(names))
    json.dumps(block.state(), default=str)
    block.update(dx)
    assert_close(block.x0(), x0 + dx, REL)


@pytest.mark.parametrize("kind", ["stationRangeBias", "empiricalLibration"])
def test_linear_blocks_reduce_observations_by_the_design_times_the_update(kind):
    block = BLOCKS[kind](_state())
    batch = EquationBatch.concatenate([_eq(*item) for item in BATCH])
    block.setup(batch)
    before, design = block.reduce(batch), _design(block, batch)
    dx = np.linspace(-0.2, 0.3, len(block.names))
    block.update(dx)
    assert_close(block.reduce(batch) - before, design @ dx, Tolerance("LINEAR_REDUCTION", atol=1e-15, unit="m"))


def test_unselected_blocks_still_reduce_observations():
    class Other(Parametrization):
        TYPE = "other"

        def __init__(self):
            self.names = [ParameterName("other", "position.x")]

        def design(self, batch, out):
            out[:, 0] = 1.0

        def reduce(self, batch):
            return np.zeros(len(batch))

        def update(self, delta):
            pass

    equation = _eq()
    bias, other = StationRangeBiasParametrization(per="station"), Other()
    blocks = ParametrizationList([bias, other])
    blocks.setup(equation)
    bias.update(np.array([0.25]))
    selected = blocks.select_blocks((other.block_id,))
    selected.setup(equation)

    assert selected.names == other.names
    assert selected.reduce(equation) == pytest.approx(-0.25)


def test_station_position_maps_partials_to_columns_and_excludes_fixed_stations():
    block = StationPositionParametrization(model_state=_state(), fixed_stations=["B"])
    batch = EquationBatch.concatenate([_eq("A"), _eq("B")])
    block.setup(batch)

    assert block.names == [ParameterName("A", f"position.{axis}") for axis in "xyz"]
    assert_close(_design(block, batch), [[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]], REL)
    assert_close(block.x0(), [1.0e6, 2.0e6, 3.0e6], REL)
    with pytest.raises(ValueError, match="no estimated stations"):
        StationPositionParametrization(model_state=_state(), fixed_stations=["A", "B"]).setup(batch)


def test_station_velocity_columns_scale_the_position_partials_by_years_since_the_position_epoch():
    state = _state()
    block = StationPositionParametrization(model_state=state, stations=["A"], estimate_velocity=True)
    batch = EquationBatch.concatenate([_eq("A", "2010-01-01T00:00:00"), _eq("A", "2016-12-31T23:59:59"),
                                       _eq("A", "2017-01-01T00:00:00")])
    block.setup(batch)
    design = _design(block, batch)
    years = np.array([state.station_catalog["A"].years_since_position_epoch(batch.epoch(i)) for i in range(3)])

    assert years[0] == pytest.approx(10.0, abs=0.01)
    assert_close(design[:, :3], np.tile([0.1, -0.2, 0.3], (3, 1)), REL)
    assert_close(design[:, 3:], years[:, None] * [0.1, -0.2, 0.3], REL)
    assert_close(_design(block, batch), design, REL)  # repeated design with cached years


def test_station_update_moves_the_catalog_and_reaches_mpi_workers_through_a_snapshot():
    state = _state()
    block = StationPositionParametrization(model_state=state, stations=["A"], estimate_velocity=True)
    block.setup(_eq("A"))
    block.update(np.array([0.5, -0.25, 1.0, 0.001, 0.002, -0.003]))
    worker = type("Processor", (), {"model_state": _state()})()
    apply_catalog_state(worker, snapshot_catalog_state(state))

    assert_close(state.station_catalog["A"].itrf_xyz_m, [1.0e6 + 0.5, 2.0e6 - 0.25, 3.0e6 + 1.0], REL)
    assert_close(state.station_catalog["A"].itrf_velocity_m_per_year, [0.011, 0.022, 0.027], REL)
    assert block.max_update_norm(np.array([3.0, 4.0, 0.0, 9.0, 9.0, 9.0])) == pytest.approx(5.0)
    for name in ("itrf_xyz_m", "itrf_velocity_m_per_year"):
        np.testing.assert_array_equal(getattr(worker.model_state.station_catalog["A"], name),
                                      getattr(state.station_catalog["A"], name))


def test_missing_partials_are_reported():
    epoch = Epoch.from_isot("2010-01-01T00:00:00", scale=TimeScale.UTC)
    bare = EquationBatch.single(oc=0.0, sigma=0.01, observation_id=1, station="A", reflector="R", epoch=epoch,
                                receive_epoch=epoch)
    block = StationPositionParametrization(model_state=_state())
    block.setup(bare)
    with pytest.raises(ValueError, match="broadcast"):
        _design(block, bare)


def test_range_bias_per_station_has_one_unit_column_per_station():
    batch = EquationBatch.concatenate([_eq("APOLLO", "2008-01-01T00:00:00"), _eq("GRASSE", "2009-01-01T00:00:00")])
    block = StationRangeBiasParametrization(per="station")
    block.setup(batch)

    assert block.keys == ["APOLLO", "GRASSE"]
    assert [str(name) for name in block.names] == ["APOLLO:rangeBias::", "GRASSE:rangeBias::"]
    assert_close(_design(block, batch), np.eye(2), REL)
    selected = StationRangeBiasParametrization(stations=["APOLLO"], per="station")
    selected.setup(batch)
    assert [str(name) for name in selected.names] == ["APOLLO:rangeBias::"]


def test_range_bias_intervals_may_overlap_and_both_apply():
    equation = _eq("APOLLO", "2008-01-01T00:00:00")
    block = StationRangeBiasParametrization(per="station+interval", intervals=[
        {"station": "APOLLO", "start": "2006-04-07", "endExclusive": "2010-11-01"},
        {"station": "APOLLO", "start": "2007-12-15", "endExclusive": "2008-06-30"},
    ])
    block.setup(equation)
    block.update(np.array([0.2, -0.05]))

    assert block.keys == ["APOLLO_2006-04-07_2010-11-01", "APOLLO_2007-12-15_2008-06-30"]
    assert [str(name) for name in block.names] == ["APOLLO:rangeBias:interval:2006-04-07/2010-11-01",
                                                   "APOLLO:rangeBias:interval:2007-12-15/2008-06-30"]
    assert_close(block.reduce(equation), [0.15], REL)


def test_range_bias_configuration_is_strict():
    with pytest.raises(ValueError, match="at least 1 item"):
        create("parametrization", resolve("parametrization", {"type": "stationRangeBias", "stations": []}))
    with pytest.raises(TypeError, match="must be a sequence"):
        resolve("parametrization", {"type": "stationRangeBias", "per": "station+interval",
                                    "intervals": {"APOLLO": ["2006-04-07/2010-11-01"]}})
    with pytest.raises(ValueError, match=r"require per='station\+interval'"):
        StationRangeBiasParametrization(per="station", intervals=[{"station": "APOLLO", "start": "2020-01-01",
                                                                   "endExclusive": None}])


def test_libration_design_uses_the_axis_partial_and_its_argument():
    block = _libration()
    batch = _eq(iso="2020-03-01T00:00:00")
    block.setup(batch)
    design = _design(block, batch)[0]
    jd = batch.epoch(0).jd1 + batch.epoch(0).jd2
    lp = erfa.falp03((jd - 2451545.0) / 36525.0)
    arg = 2 * np.pi * (jd - 2451545.0) / 2190.0

    # psi uses the z PA rotation partial, x the x one, rho the first ecliptic libration partial.
    assert_close(design[:2], 3.0e3 * MAS * np.array([np.cos(lp), np.sin(lp)]), REL)
    assert_close(design[2:4], 1.0e3 * MAS * np.array([np.cos(arg), np.sin(arg)]), REL)
    rho = 2 * np.pi * (jd - 2451545.0) / 27.3
    assert_close(design[4:], 4.0e2 * MAS * np.array([np.cos(rho), np.sin(rho)]), REL)


def test_batch_fundamental_arguments_match_scalar_erfa():
    batch = EquationBatch.concatenate([_eq(iso=iso) for iso in
                                       ("2012-01-01T00:00:00", "2019-02-03T12:34:56", "2020-03-01T00:00:00")])
    for actual, function in zip(batch.fundamental_arguments,
                                (erfa.fal03, erfa.falp03, erfa.faf03, erfa.fad03, erfa.faom03), strict=True):
        expected = [function((batch.epoch(i).jd1 + batch.epoch(i).jd2 - 2451545.0) / 36525.0) for i in range(3)]
        assert_close(actual, expected, Tolerance("FUNDAMENTAL_ARGUMENT_RAD", atol=1e-13, unit="rad"))


def test_fixed_libration_terms_are_applied_but_not_estimated():
    block = EmpiricalLibrationParametrization(terms=[
        _term({"name": "fixed", "delaunay": {"lp": 1}, "cosMas": 4.5, "estimate": False}),
        _term({"name": "free", "periodDays": 1306.0}),
    ])
    batch = _eq(iso="2020-03-01T00:00:00")
    block.setup(batch)
    jd = batch.epoch(0).jd1 + batch.epoch(0).jd2

    assert [str(n) for n in block.names] == ["moon:libration.free.cos::", "moon:libration.free.sin::"]
    assert_close(block.reduce(batch), [3.0e3 * MAS * 4.5 * np.cos(erfa.falp03((jd - 2451545.0) / 36525.0))], REL)


def test_invalid_libration_terms_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        _term({"name": "bad", "delaunay": {"l": 1}, "periodDays": 10.0})
    with pytest.raises(ValueError, match="unknown Delaunay"):
        _term({"name": "bad", "delaunay": {"q": 1}})
