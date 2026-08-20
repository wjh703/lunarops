from __future__ import annotations

from lunarops.config.context import RunContext
from lunarops.fileio.catalogs import read_reflector_catalog, read_station_catalog
from lunarops.programs.registry import ensure_builtin_programs, run_program


def test_reflector_catalog_create_from_inline_coordinates(tmp_path):
    ensure_builtin_programs()
    context = RunContext(working_dir=tmp_path)
    output = run_program(
        "ReflectorCatalogCreate",
        {
            "reflectorCoordinates": [{"key": "APOLLO11", "xyzM": [1.25, 2.5, 3.75]}],
            "outputFileReflectorCatalog": "reflectors.txt",
        },
        context,
    )

    record = read_reflector_catalog(output)["APOLLO11"]
    assert list(record.moon_fixed_xyz_m) == [1.25, 2.5, 3.75]
    assert record.name == "APOLLO11"
    assert record.aliases == ()


def test_catalog_create_programs_accept_coordinate_source_files(tmp_path):
    ensure_builtin_programs()
    source = tmp_path / "coordinates.yml"
    source.write_text(
        "stationCoordinates:\n"
        "  - {key: S, xyzM: [1.0, 2.0, 3.0], velocityMPerYear: [0.1, 0.2, 0.3]}\n",
        encoding="ascii",
    )
    context = RunContext(working_dir=tmp_path)
    output = run_program(
        "StationCatalogCreate",
        {
            "inputFileStationCoordinates": "coordinates.yml",
            "outputFileStationCatalog": "stations.txt",
        },
        context,
    )
    record = read_station_catalog(output)["S"]
    assert list(record.itrf_xyz_m) == [1.0, 2.0, 3.0]
    assert list(record.itrf_velocity_m_per_year) == [0.1, 0.2, 0.3]
