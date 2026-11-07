"""Datasets, observation options, parametrizations and equation sources of the LLR workflow."""

from lunarops.classes.observation import NptDataset, ObservationResultDetail
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.station_range_bias import StationRangeBiasParametrization
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.context import RunContext
from lunarops.config.schema import ConfigSchema
from lunarops.fileio.normal_points import write_normal_points
from lunarops.llr_workflow import (
    EquationSource,
    MpiEquationBackend,
    build_equation_source,
    build_parametrization,
    load_datasets,
    make_processing_options,
    output_level,
)
from lunarops.programs.specs import observation_fields
from tests.support.observation import npt_record

EPOCH = Epoch(2458849.5, 0.0, TimeScale.UTC)


def _observation_config(**changes):
    return dict(ConfigSchema(observation_fields()).resolve({}), **changes)


def _batch(identity):
    return EquationBatch.single(
        oc=0.0, sigma=1.0, observation_id=identity, station="S", reflector="R", epoch=EPOCH, receive_epoch=EPOCH
    )


def test_datasets_are_read_from_the_working_directory_filtered_and_indexed_globally(tmp_path):
    def record(day):
        item = npt_record(index=day)
        item.transmit_epoch = Epoch.from_isot(f"2020-01-{day:02d}T00:00:00", scale=TimeScale.UTC)
        return item

    write_normal_points(NptDataset([record(1), record(2), record(20)]), tmp_path / "a.txt")
    write_normal_points(NptDataset([record(3)]), tmp_path / "b.txt")

    datasets = load_datasets(
        _observation_config(inputFilesNormalPoints=["a.txt", "b.txt"], startTime="2020-01-01", endTime="2020-01-10"),
        RunContext(working_dir=tmp_path),
    )

    assert list(datasets) == ["a", "b"]
    assert [record.index for record in datasets["a"].records] == [0, 1]
    assert [record.index for record in datasets["b"].records] == [2]


def test_options_and_parametrization_follow_the_configuration():
    config = {
        "overrideStation": "APOLLO",
        "overrideReflector": "APOLLO15",
        "minElevationDeg": 12.5,
        "showProgress": False,
        "outputLevel": "full",
        "parametrization": [{"type": "stationRangeBias", "per": "station"}],
    }
    options = make_processing_options(_observation_config(**config), include_design=True)
    blocks = build_parametrization(config, RunContext(), None)

    assert (options.station_identifier, options.reflector_identifier) == ("APOLLO", "APOLLO15")
    assert options.min_elevation_deg == 12.5 and options.include_reflector_position_partials
    assert output_level(config) is ObservationResultDetail.FULL
    assert output_level({"outputLevel": "standard"}, include_design=True) is ObservationResultDetail.FULL
    assert [type(block) for block in blocks.blocks] == [StationRangeBiasParametrization]


def test_serial_equation_source_concatenates_every_dataset_with_an_iteration_label():
    calls = []

    class Processor:
        def equations(self, dataset, *, options):
            calls.append(options)
            return dataset

    datasets = {"a": _batch(1), "b": _batch(2)}
    source = build_equation_source(
        _observation_config(showProgress=True, parametrization=[]), RunContext(), datasets, Processor()
    )

    assert source(4).observation_id.tolist() == [1, 2]
    assert all(not item.include_reflector_position_partials for item in calls)
    assert {item.progress_description for item in calls} == {"linearization (pass 4)"}


def test_mpi_equation_source_sends_the_current_catalog_state_each_iteration(monkeypatch):
    sent = []

    class State:
        position = 1.0

        def reflector_positions_pa_m(self):
            return {"R": [self.position, 0.0, 0.0]}

        def station_states_itrf_m(self):
            return {"S": ([self.position, 0.0, 0.0], [0.0, 0.0, 0.0])}

    def evaluate(runtime, spec, datasets, options, **kwargs):
        sent.append((kwargs["catalog_state"], options, kwargs))
        return datasets

    # The MPI transport is replaced at its public entry point; the source logic is real.
    monkeypatch.setattr("lunarops.parallel.mpi.mpi_observations", evaluate)
    state = State()
    source = EquationSource(
        MpiEquationBackend(object(), {"specId": "test"}, 8),
        {"a": _batch(1), "b": _batch(2)},
        make_processing_options(_observation_config(showProgress=False), include_design=True),
        state,
    )

    assert source(1).observation_id.tolist() == [1, 2]
    state.position = 2.0
    source(2)
    assert [item[0]["reflectorPositions"]["R"][0] for item in sent] == [1.0, 2.0]
    assert [item[0]["stationStates"]["S"][0][0] for item in sent] == [1.0, 2.0]
    assert sent[1][1].progress_description == "linearization (pass 2)" and sent[1][2]["quiet"] is True
