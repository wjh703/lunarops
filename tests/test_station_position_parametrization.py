import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.observation.resolver import ObservationCatalogState
from lunarops.classes.parametrization.station_position import StationPositionParametrization
from lunarops.classes.time import Epoch, TimeScale
from lunarops.parallel.observation_spec import apply_catalog_state, snapshot_catalog_state


def _state():
    stations = {
        "A": StationRecord("A", [1.0e6, 2.0e6, 3.0e6], [0.01, 0.02, 0.03], "2000-01-01T00:00:00"),
        "B": StationRecord("B", [4.0e6, 5.0e6, 6.0e6], [0.0, 0.0, 0.0], "2000-01-01T00:00:00"),
    }
    reflectors = {"R": ReflectorRecord("R", [1.7e6, 0.0, 0.0])}
    return ObservationCatalogState(stations, reflectors)


def _equation(station, iso="2010-01-01T00:00:00", partial=(0.1, -0.2, 0.3)):
    return EquationBatch.single(
        oc=0.0,
        sigma=0.01,
        station_position_itrf=np.asarray(partial),
        observation_id=f"{station}-{iso}",
        station=station,
        reflector="R",
        epoch=Epoch.from_isot(iso, scale=TimeScale.UTC),
        receive_epoch=Epoch.from_isot(iso, scale=TimeScale.UTC),
    )


def test_fixed_station_is_excluded_and_partials_map_to_columns():
    state = _state()
    block = StationPositionParametrization(model_state=state, fixed_stations=["B"])
    block.setup(EquationBatch.concatenate([_equation("A"), _equation("B")]))
    assert block.names == [ParameterName("A", f"position.{axis}") for axis in "xyz"]
    assert _design(block, _equation("B")) == pytest.approx([0.0, 0.0, 0.0])
    assert _design(block, _equation("A")) == pytest.approx([0.1, -0.2, 0.3])
    np.testing.assert_allclose(block.x0(), [1.0e6, 2.0e6, 3.0e6])


def test_velocity_columns_scale_with_years_since_position_epoch():
    state = _state()
    block = StationPositionParametrization(model_state=state, stations=["A"], estimate_velocity=True)
    eq = _equation("A", iso="2010-01-01T00:00:00")
    block.setup(eq)
    years = state.station_catalog["A"].years_since_position_epoch(eq.epoch(0))
    assert years == pytest.approx(10.0, abs=0.01)
    entries = _design(block, eq)
    for axis, value in enumerate((0.1, -0.2, 0.3)):
        assert entries[axis] == pytest.approx(value)
        assert entries[3 + axis] == pytest.approx(value * years)


def test_apply_update_moves_catalog_and_survives_mpi_snapshot():
    state = _state()
    block = StationPositionParametrization(model_state=state, stations=["A"], estimate_velocity=True)
    block.setup(_equation("A"))
    block.update(np.array([0.5, -0.25, 1.0, 0.001, 0.002, -0.003]))
    np.testing.assert_allclose(state.station_catalog["A"].itrf_xyz_m, [1.0e6 + 0.5, 2.0e6 - 0.25, 3.0e6 + 1.0])
    np.testing.assert_allclose(state.station_catalog["A"].itrf_velocity_m_per_year, [0.011, 0.022, 0.027])
    assert block.max_update_norm(np.array([3.0, 4.0, 0.0, 9.0, 9.0, 9.0])) == pytest.approx(5.0)

    class Processor:
        model_state = _state()

    worker = Processor()
    apply_catalog_state(worker, snapshot_catalog_state(state))
    np.testing.assert_allclose(worker.model_state.station_catalog["A"].itrf_xyz_m, state.station_catalog["A"].itrf_xyz_m)
    np.testing.assert_allclose(
        worker.model_state.station_catalog["A"].itrf_velocity_m_per_year,
        state.station_catalog["A"].itrf_velocity_m_per_year,
    )


def test_all_stations_fixed_is_rejected():
    state = _state()
    block = StationPositionParametrization(model_state=state, fixed_stations=["A", "B"])
    with pytest.raises(ValueError, match="no estimated stations"):
        block.setup(EquationBatch.concatenate([_equation("A"), _equation("B")]))


def test_missing_partial_block_is_reported():
    state = _state()
    block = StationPositionParametrization(model_state=state)
    eq = EquationBatch.single(
        oc=0.0,
        sigma=0.01,
        observation_id=1,
        station="A",
        reflector="R",
        epoch=Epoch.from_isot("2010-01-01T00:00:00", scale=TimeScale.UTC),
        receive_epoch=Epoch.from_isot("2010-01-01T00:00:00", scale=TimeScale.UTC),
    )
    block.setup(eq)
    with pytest.raises(ValueError, match="broadcast"):
        _design(block, eq)


def _design(block, batch):
    out = np.zeros((len(batch), len(block.names)))
    block.design(batch, out)
    return out[0]
