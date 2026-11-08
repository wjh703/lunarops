"""Observation specs broadcast to MPI workers."""

import pickle

from lunarops.classes.frames import EarthOrientationSample, TabulatedEarthOrientation
from lunarops.classes.observation.assembly import resolve_observation_assembly
from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord
from lunarops.config.context import RunContext
from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog
from lunarops.parallel.observation_spec import make_observation_spec

MODELS = {
    key: {"type": "test"}
    for key in ("ephemerides", "troposphere", "relativity", "stationDisplacement", "reflectorDisplacement")
}


def _context(tmp_path, **models):
    return RunContext(
        working_dir=tmp_path,
        model_configs={"rangeBias": None, **MODELS, "earthRotation": {"type": "test"}, **models},
    )


def test_each_spec_has_its_own_id_and_the_explicit_catalogs(tmp_path):
    context = _context(tmp_path)
    first = make_observation_spec({}, context, station_catalog={"a": object()}, reflector_catalog={"r": object()})
    second = make_observation_spec({}, context, station_catalog={"b": object()}, reflector_catalog={"s": object()})

    assert first["specId"] != second["specId"]
    assert set(first["stationCatalog"]) == {"a"} and set(second["reflectorCatalog"]) == {"s"}


def test_serial_and_mpi_assemblies_read_the_same_catalog_files(tmp_path):
    write_station_catalog({"TEST": StationRecord("Test station", [1.0, 2.0, 3.0])}, tmp_path / "stations.txt")
    write_reflector_catalog({"REF": ReflectorRecord("Test reflector", [4.0, 5.0, 6.0])}, tmp_path / "reflectors.txt")
    config = {"inputFileStationCatalog": "stations.txt", "inputFileReflectorCatalog": "reflectors.txt"}
    context = _context(tmp_path)

    serial, spec = resolve_observation_assembly(context, config), make_observation_spec(config, context)

    assert set(spec["stationCatalog"]) == set(serial.station_catalog) == {"TEST"}
    assert set(spec["reflectorCatalog"]) == set(serial.reflector_catalog) == {"REF"}


def test_a_worker_resolves_the_spec_models_without_shared_sections(tmp_path):
    context = _context(tmp_path)
    serial = resolve_observation_assembly(context, {}, station_catalog={}, reflector_catalog={})
    spec = make_observation_spec({}, context, station_catalog={}, reflector_catalog={})

    worker = resolve_observation_assembly(
        RunContext(working_dir=tmp_path), spec["programConfig"], station_catalog={}, reflector_catalog={}
    )

    assert "rangeBias" not in serial.model_configs
    assert worker.model_configs == serial.model_configs


def test_a_file_eop_table_travels_in_the_spec_and_replaces_the_worker_file_read(tmp_path):
    table = TabulatedEarthOrientation((EarthOrientationSample(60000.0, 0.1, 0.2, -0.3),), source_file_path="eop.txt")

    class Context(RunContext):
        def create_class(self, category, config, *, cache=True):
            return table if category == "earthRotation" else super().create_class(category, config, cache=cache)

    context = Context(
        working_dir=tmp_path,
        model_configs={"rangeBias": None, **MODELS, "earthRotation": {"type": "file", "file": "eop.txt"}},
    )
    spec = make_observation_spec({}, context, station_catalog={}, reflector_catalog={})
    received = pickle.loads(pickle.dumps(spec["sharedResources"]))
    restored = TabulatedEarthOrientation.from_config(
        {"file": "unavailable.txt"}, RunContext(working_dir=tmp_path, mpi_resources=received)
    )

    assert restored.samples == table.samples and restored.source_file_path == table.source_file_path
