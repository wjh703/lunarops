from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from lunarops.classes.observation import ObservationResultDetail
from lunarops.classes.parametrization.station_range_bias import (
    StationRangeBiasParametrization,
)
from lunarops.classes.time import Epoch
from lunarops.config.context import RunContext
from lunarops.config.schema import ConfigSchema
from lunarops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    load_datasets,
    make_processing_options,
    output_level,
)
from lunarops.programs.specs import observation_fields


class _Dataset:
    def __init__(self, size: int) -> None:
        self.records = [SimpleNamespace(index=None) for _ in range(size)]
        self.filter_args = None

    def filter_time(self, start, end):
        self.filter_args = (start, end)
        return self

    def assign_indices(self, *, start: int) -> None:
        for offset, record in enumerate(self.records):
            record.index = start + offset


def test_load_datasets_uses_working_directory_and_assigns_global_indices(
    monkeypatch,
    tmp_path,
):
    from lunarops.fileio import normal_points
    from lunarops.fileio.formats import normal_point_sources

    source_paths = [tmp_path / "a.npt", tmp_path / "b.npt"]
    datasets_by_path = {
        source_paths[0]: _Dataset(2),
        source_paths[1]: _Dataset(1),
    }
    captured = {}

    def resolve(paths):
        captured["inputs"] = paths
        return source_paths

    def read(path):
        return datasets_by_path[Path(path)]

    monkeypatch.setattr(normal_point_sources, "resolve_normal_point_inputs", resolve)
    monkeypatch.setattr(normal_points, "read_normal_points", read)
    context = RunContext(working_dir=tmp_path)

    datasets = load_datasets(
        dict(ConfigSchema(observation_fields()).resolve({}), inputFilesNormalPoints=["input"], startTime="2020-01-01", endTime="2021-01-01"),
        context,
    )

    assert captured["inputs"] == [tmp_path / "input"]
    assert list(datasets) == ["a", "b"]
    assert [record.index for record in datasets["a"].records] == [0, 1]
    assert [record.index for record in datasets["b"].records] == [2]
    assert cast(Any, datasets["a"]).filter_args == (Epoch.from_isot("2020-01-01"), Epoch.from_isot("2021-01-01"))


def test_workflow_builds_canonical_options_and_parametrization():
    config = {
        "overrideStation": "APOLLO",
        "overrideReflector": "APOLLO15",
        "minElevationDeg": 12.5,
        "showProgress": False,
        "outputLevel": "full",
        "parametrization": [{"type": "stationRangeBias", "per": "station"}],
    }
    context = RunContext()

    options = make_processing_options(dict(ConfigSchema(observation_fields()).resolve({}), **config), include_design=True)
    blocks = build_parametrization(config, context, None)

    assert options.station_identifier == "APOLLO"
    assert options.reflector_identifier == "APOLLO15"
    assert options.min_elevation_deg == 12.5
    assert options.include_reflector_position_partials
    assert output_level(config) is ObservationResultDetail.FULL
    assert output_level(config, include_design=True) is ObservationResultDetail.FULL
    assert len(blocks.blocks) == 1
    assert isinstance(blocks.blocks[0], StationRangeBiasParametrization)


def test_serial_equation_source_reuses_processor_and_sets_iteration_progress():
    calls = []

    class Processor:
        def equations(self, dataset, *, options):
            calls.append((dataset, options))
            return dataset

    context = RunContext(runtime=None)
    datasets = {"a": _batch(1), "b": _batch(2)}
    source = build_equation_source(
        dict(ConfigSchema(observation_fields()).resolve({}), showProgress=True, parametrization=[]),
        context,
        datasets,
        Processor(),
    )

    equations = source(4)

    assert equations.observation_id.tolist() == [1, 2]
    assert [item[0] for item in calls] == list(datasets.values())
    assert all(not item[1].include_reflector_position_partials for item in calls)
    assert all(item[1].progress_description == "linearization 4" for item in calls)


def test_mpi_equation_source_snapshots_updated_state_each_iteration(monkeypatch):
    from lunarops.llr_workflow import EquationSource, MpiEquationBackend

    snapshots = []

    class State:
        position = 1.0

        def reflector_positions_pa_m(self):
            return {"R": [self.position, 0.0, 0.0]}

        def station_states_itrf_m(self):
            return {"S": ([self.position, 0.0, 0.0], [0.0, 0.0, 0.0])}

    def evaluate(runtime, spec, datasets, options, **kwargs):
        snapshots.append((kwargs["catalog_state"], options, kwargs))
        return datasets

    monkeypatch.setattr("lunarops.parallel.mpi.mpi_observations", evaluate)
    state = State()
    datasets = {"first": _batch(1), "second": _batch(2)}
    source = EquationSource(
        MpiEquationBackend(object(), {"specId": "test"}, 8),
        datasets,
        make_processing_options(dict(ConfigSchema(observation_fields()).resolve({}), showProgress=False), include_design=True),
        state,
    )
    assert source(1).observation_id.tolist() == [1, 2]
    state.position = 2.0
    assert source(2).observation_id.tolist() == [1, 2]
    assert snapshots[0][0]["reflectorPositions"]["R"][0] == 1.0
    assert snapshots[1][0]["reflectorPositions"]["R"][0] == 2.0
    assert snapshots[0][0]["stationStates"]["S"][0][0] == 1.0
    assert snapshots[1][0]["stationStates"]["S"][0][0] == 2.0
    assert snapshots[1][1].progress_description == "linearization 2"
    assert snapshots[1][2]["quiet"] is True


def _batch(identity):
    from lunarops.classes.observation.equations import EquationBatch
    from lunarops.classes.time import Epoch, TimeScale
    epoch = Epoch(2458849.5, 0., TimeScale.UTC)
    return EquationBatch.single(oc=0., sigma=1., observation_id=identity, station="S", reflector="R", epoch=epoch, receive_epoch=epoch)
