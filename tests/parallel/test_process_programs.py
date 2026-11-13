"""Programs on local worker processes give bitwise the results of a serial run."""

import multiprocessing
import pickle
import signal
from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest

from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings
from lunarops.classes.dynamics.epoch_data import SerialEpochSource
from lunarops.classes.observation import NptDataset, ReflectorRecord, StationRecord
from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog
from lunarops.fileio.normal_points import write_normal_points
from lunarops.fileio.structured_reports import read_processing_report
from lunarops.parallel.epoch_stream import RuntimeEpochSource
from lunarops.parallel.process import ProcessRuntime
from lunarops.programs.llr_observation_prediction import llr_observation_prediction
from lunarops.programs.llr_processing import llr_processing
from lunarops.programs.llr_residuals import llr_residuals
from lunarops.programs.registry import ensure_builtin_programs, resolve_program_config
from tests.parallel import probe_handlers
from tests.support.dynamics import EPOCH, History, build
from tests.support.observation import (
    TRANSMIT_UTC,
    StaticKernelContext,
    npt_record,
    static_frames,
    station_facing_the_moon,
)


@pytest.fixture(autouse=True)
def _guarded():
    def hung(signum, frame):
        raise TimeoutError("the test hung")

    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(120)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert multiprocessing.active_children() == []


@pytest.fixture
def epoch_runtime():
    with ProcessRuntime(2, initializer=probe_handlers.register_test_factories) as runtime:
        yield runtime


# A parallel backward arc on worker processes: tests/dynamics/test_parallel_arcs.py.
@pytest.mark.parametrize("backward", [0.0, 86400.0], ids=["forward", "two-sided"])
def test_epoch_data_from_worker_processes_integrate_bitwise_as_serial_data(epoch_runtime, backward):
    system, state = build(fused=True, rotation=False)
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=600.0, order=4, startup_step_s=75.0))
    history = History(state)

    def run(source):
        calls = []

        def record(epoch, node_state, batch, slot, history):
            derivative = system.derivatives(epoch, node_state, batch, index=slot, history=history)
            calls.append((float(EPOCH.seconds_until(epoch)), node_state.tolist(), derivative.tolist()))

        trajectory = integrator.integrate(
            system,
            EPOCH,
            state,
            86400.0,
            backward_duration_s=backward,
            external_history_provider=history,
            epoch_source=source,
            node_diagnostic=record,
        )
        return trajectory.states, sorted(calls)

    serial = run(SerialEpochSource(pickle.loads(pickle.dumps(system.epoch_inputs)), system.ephemeris))
    source = RuntimeEpochSource(
        epoch_runtime, system.epoch_inputs, {"type": "testAnalyticSolarSystem"}, ".", f"epochs-{backward}"
    )
    parallel = run(source)
    np.testing.assert_array_equal(parallel[0], serial[0])
    assert parallel[1] == serial[1]


MODELS = {
    "ephemerides": {"type": "chebyshev", "directory": "kernels", "lunarRelativisticScaleConvention": "alreadyScaled"},
    "earthRotation": {"type": "file", "file": "eop.txt"},
    "troposphere": {"type": "none"},
    "relativity": {"type": "none"},
    "stationDisplacement": [{"type": "iers2010SolidEarthTide"}],
    "reflectorDisplacement": {"type": "none"},
}
CATALOGS = {"inputFileStationCatalog": "stations.txt", "inputFileReflectorCatalog": "reflectors.txt"}


@pytest.fixture
def observation_dir(tmp_path):
    ensure_builtin_programs()
    write_station_catalog(
        {"APOLLO": StationRecord(name="Apache Point", itrf_xyz_m=tuple(station_facing_the_moon(static_frames())))},
        tmp_path / "stations.txt",
    )
    write_reflector_catalog(
        {"APOLLO15": ReflectorRecord(name="Apollo 15", moon_fixed_xyz_m=(1_737_400.0, 0.0, 0.0))},
        tmp_path / "reflectors.txt",
    )
    # Inputs recorded by the processing provenance; the static system reads none of them.
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "static.bsp").write_bytes(b"static")
    (tmp_path / "eop.txt").touch()
    records = [replace(npt_record(index), transmit_epoch=TRANSMIT_UTC.shifted(60.0 * index)) for index in range(7)]
    write_normal_points(NptDataset(records), tmp_path / "normal.txt")
    return tmp_path


def _run_both(observation_dir, program, name, config, outputs):
    """Run a program serially and on two worker processes; return the bytes of its outputs per run."""
    resolved = resolve_program_config(name, {**config, "showProgress": False, "mpi": {"chunksize": 2}})
    contents = {}
    for mode in ("serial", "process"):
        with ExitStack() as stack:
            runtime = None
            if mode == "process":
                runtime = stack.enter_context(ProcessRuntime(2, initializer=probe_handlers.install_static_observation))
            context = stack.enter_context(StaticKernelContext(working_dir=observation_dir, runtime=runtime))
            context.resolve_defaults()  # as the CLI does before running programs
            program(resolved, context)
        contents[mode] = {output: (observation_dir / output).read_bytes() for output in outputs}
        for output in outputs:
            (observation_dir / output).unlink()
    return contents


def _without(text: bytes, key: str) -> list[bytes]:
    return [line for line in text.splitlines() if key.encode() not in line]


def test_prediction_on_worker_processes_is_bitwise_serial(observation_dir):
    config = {
        **MODELS,
        **CATALOGS,
        "outputFilePrediction": "prediction.txt",
        "outputFileWindows": "windows.txt",
        "meteorologyPolicy": "nominal",
        "startTime": "2020-01-01T08:00:00",
        "endTime": "2020-01-01T08:10:00",
        "utcOffsetHours": 8.0,
        "stationName": "APOLLO",
        "reflectorName": "APOLLO15",
        "minElevationDeg": 0.0,
        "minReflectorElevationDeg": -90.0,
        "maxSunElevationDeg": 90.0,
    }
    contents = _run_both(
        observation_dir,
        llr_observation_prediction,
        "LlrObservationPrediction",
        config,
        ("prediction.txt", "windows.txt"),
    )
    # The files differ only in the execution mode of their metadata.
    for output, serial in contents["serial"].items():
        parallel = contents["process"][output]
        assert serial.count(b"executionMode") == parallel.count(b"executionMode") == 1, output
        assert _without(parallel, "executionMode") == _without(serial, "executionMode"), output


def test_residuals_on_worker_processes_are_bitwise_serial(observation_dir):
    config = {
        **MODELS,
        **CATALOGS,
        "inputFilesNormalPoints": ["normal.txt"],
        "rangeBias": {"type": "none"},
        "outputFileObservationResults": "results.txt",
        "outputLevel": "full",
        "includeReflectorDesign": True,
    }
    contents = _run_both(observation_dir, llr_residuals, "LlrResiduals", config, ("results.txt",))
    assert contents["process"] == contents["serial"]


def test_processing_on_worker_processes_is_bitwise_serial(observation_dir):
    config = {
        **MODELS,
        **CATALOGS,
        "inputFilesNormalPoints": ["normal.txt"],
        "rangeBias": {"type": "none"},
        "parametrization": [{"type": "stationRangeBias"}],
        "varianceComponents": [{"id": "A", "station": "APOLLO", "start": "2020-01-01", "endExclusive": None}],
        "processingSteps": [
            {"type": "screenObservations", "residual": {"maximumAbsoluteM": None}},
            {
                "type": "estimate",
                "name": "bias",
                "estimateVarianceFactors": False,
                "estimateRobustWeights": False,
                "maxIterationCount": 2,
            },
            {"type": "writeResiduals", "outputFile": "residuals.txt", "outputLevel": "full"},
            {
                "type": "writeResults",
                "outputFileSolution": "solution.txt",
                "outputFileReport": "report.txt",
                "outputFileState": "state.txt",
            },
        ],
    }
    outputs = ("residuals.txt", "solution.txt", "state.txt")
    contents = _run_both(observation_dir, llr_processing, "LlrProcessing", config, (*outputs, "report.txt"))
    serial, parallel = contents["serial"], contents["process"]
    for output in outputs:
        assert parallel[output] == serial[output], output
    reports = {}
    for mode in ("serial", "process"):
        path = observation_dir / f"{mode}-report.txt"
        path.write_bytes(contents[mode]["report.txt"])
        reports[mode] = read_processing_report(path)
    assert reports["serial"].pop("execution") == {"mode": "serial", "workerCount": 0}
    assert reports["process"].pop("execution") == {"mode": "process", "workerCount": 2}
    # Apart from the execution mode, the reports differ only in timings.
    differences = _differences(reports["process"], reports["serial"])
    assert [path for path, *_ in differences if ".performanceSeconds." not in path] == []


def _differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        return [d for key in keys for d in _differences(left.get(key), right.get(key), f"{path}.{key}")]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [d for i, (a, b) in enumerate(zip(left, right)) for d in _differences(a, b, f"{path}[{i}]")]
    return [] if left == right else [(path, left, right)]
