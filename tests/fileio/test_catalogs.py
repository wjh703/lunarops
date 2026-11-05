"""Station, reflector and mass catalogs, and the programs that create them."""

import numpy as np
import pytest

from lunarops.base.station_identity import station_id_from_ilrs_code, station_ilrs_code, validate_station_id
from lunarops.classes.displacement.terrestrial_geometry import itrf2geodetic
from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord, resolve_catalog_key
from lunarops.config.context import RunContext
from lunarops.fileio.catalogs import (
    read_reflector_catalog,
    read_station_catalog,
    write_reflector_catalog,
    write_station_catalog,
)
from lunarops.fileio.mass_catalog import BodyMass, MassCatalog, read_mass_catalog
from lunarops.programs.catalog_create import AU_M, DAY_S, earth_moon_gm_from_gmb_emrat
from lunarops.programs.registry import ensure_builtin_programs, run_program
from tests.tolerances import Tolerance, assert_close

# WGS84 round trips through geodetic coordinates.
GEODETIC_DEG = Tolerance("GEODETIC_DEG", atol=1e-10, unit="deg")
GEODETIC_HEIGHT_M = Tolerance("GEODETIC_HEIGHT_M", atol=1e-5, unit="m")


@pytest.fixture
def run(tmp_path):
    ensure_builtin_programs()
    return lambda name, config: run_program(name, config, RunContext(working_dir=tmp_path))


def test_catalog_keys_match_exactly():
    catalog = {"Foo-Bar": StationRecord(name="Foo Bar Station", itrf_xyz_m=(1.0, 2.0, 3.0))}
    assert resolve_catalog_key("Foo-Bar", catalog, "Station") == "Foo-Bar"
    for value in ("foo-bar", "foobar", "fb01", "Foo Bar Station", "missing"):
        with pytest.raises(KeyError):
            resolve_catalog_key(value, catalog, "Station")


def test_station_identity_uses_registered_ids_and_ilrs_codes_only():
    assert station_id_from_ilrs_code("70610") == "APOLLO"
    assert station_ilrs_code("APOLLO") == "70610"
    assert validate_station_id("custom-station") == "custom-station"
    for alias in ("Apache Point Observatory", "70610", "APOL", "apollo", "custom-station"):
        with pytest.raises(ValueError, match="Unknown registered station"):
            station_ilrs_code(alias)


def test_catalog_files_round_trip(tmp_path):
    stations = {"TEST": StationRecord("TEST", [1.0, 2.0, 3.0], itrf_velocity_m_per_year=[0.1, 0.2, 0.3],
                                      position_epoch_utc="2020-01-01T00:00:00")}
    reflectors = {"REF": ReflectorRecord("REF", [4.0, 5.0, 6.0])}
    write_station_catalog(stations, tmp_path / "stations.txt.gz")
    write_reflector_catalog(reflectors, tmp_path / "reflectors.txt.gz")
    station, reflector = read_station_catalog(tmp_path / "stations.txt.gz")["TEST"], \
        read_reflector_catalog(tmp_path / "reflectors.txt.gz")["REF"]

    assert (station.name, reflector.name) == ("TEST", "REF")
    np.testing.assert_array_equal(station.itrf_xyz_m, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(station.itrf_velocity_m_per_year, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(reflector.moon_fixed_xyz_m, [4.0, 5.0, 6.0])


def test_reflector_catalog_program_takes_inline_coordinates(run):
    record = read_reflector_catalog(run("ReflectorCatalogCreate", {
        "reflectorCoordinates": [{"key": "APOLLO11", "xyzM": [1.25, 2.5, 3.75]}],
        "outputFileReflectorCatalog": "reflectors.txt"}))["APOLLO11"]
    assert (list(record.moon_fixed_xyz_m), record.name) == ([1.25, 2.5, 3.75], "APOLLO11")


def test_station_catalog_program_reads_a_coordinate_file(run, tmp_path):
    (tmp_path / "coordinates.yml").write_text(
        "stationCoordinates:\n  - {key: S, xyzM: [1.0, 2.0, 3.0], velocityMPerYear: [0.1, 0.2, 0.3]}\n")
    record = read_station_catalog(run("StationCatalogCreate", {"inputFileStationCoordinates": "coordinates.yml",
                                                               "outputFileStationCatalog": "stations.txt"}))["S"]
    assert (list(record.itrf_xyz_m), list(record.itrf_velocity_m_per_year)) == ([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])


def test_station_catalog_program_converts_wgs84_geodetic_coordinates(run):
    record = read_station_catalog(run("StationCatalogCreate", {
        "stationCoordinates": [{"key": "PRIVATE", "longitudeDeg": 116.391, "latitudeDeg": 39.907, "heightM": 45.0}],
        "outputFileStationCatalog": "stations.txt"}))["PRIVATE"]
    site = itrf2geodetic(record.itrf_xyz_m)

    assert_close([site.longitude_deg, site.latitude_deg], [116.391, 39.907], GEODETIC_DEG)
    assert_close(site.ellipsoidal_height_m, 45.0, GEODETIC_HEIGHT_M)


@pytest.mark.parametrize(("coordinate", "message"), [
    ({"key": "T", "xyzM": [1.0, 2.0, 3.0], "longitudeDeg": 10.0, "latitudeDeg": 20.0, "heightM": 30.0},
     "either xyzM or all geodetic fields"),
    ({"key": "T", "longitudeDeg": 10.0, "latitudeDeg": 20.0}, "requires xyzM or all geodetic fields"),
])
def test_station_coordinates_must_be_cartesian_or_complete_geodetic(run, coordinate, message):
    with pytest.raises(ValueError, match=message):
        run("StationCatalogCreate", {"stationCoordinates": [coordinate], "outputFileStationCatalog": "stations.txt"})


def test_mass_catalog_program_converts_units_and_selects_groups(run):
    catalog = read_mass_catalog(run("MassCatalogCreate", {
        "catalogName": "test system", "source": "unit test", "expectedBodyCount": 2,
        "expectedGroupCounts": {"major": 1, "sb441": 1}, "outputFileMassCatalog": "masses.txt",
        "bodyMasses": [
            {"bodyId": "EARTH", "ephemerisTarget": 399, "gm": 3.986e14, "groups": ["major"]},
            {"bodyId": "NAIF:2000001", "name": "Ceres", "ephemerisTarget": 2000001, "gm": 1.0e-13,
             "gmUnit": "au3/day2", "groups": ["sb441", "asteroid"], "source": "table"},
        ]}))

    assert catalog.require("EARTH").gm_m3_s2 == 3.986e14
    assert catalog.require("NAIF:2000001").gm_m3_s2 == pytest.approx(1.0e-13 * AU_M**3 / DAY_S**2)
    assert [body.name for body in catalog.select(groups=["asteroid"])] == ["Ceres"]


def test_mass_catalog_program_splits_gmb_with_emrat(run):
    catalog = read_mass_catalog(run("MassCatalogCreate", {
        "catalogName": "DE440 masses", "source": "DE440 integration constants",
        "earthMoonCanonical": {"gmbAu3Day2": 8.9970113929473466e-10, "emrat": 81.300568221497215},
        "outputFileMassCatalog": "masses.txt",
        "bodyMasses": [{"bodyId": "EARTH", "ephemerisTarget": 399, "groups": ["major"]},
                       {"bodyId": "MOON", "ephemerisTarget": 301, "groups": ["major"]}]}))
    earth, moon = earth_moon_gm_from_gmb_emrat(8.9970113929473466e-10, 81.300568221497215)

    assert (catalog.require("EARTH").gm_m3_s2, catalog.require("MOON").gm_m3_s2) == (earth, moon)
    assert earth / moon == pytest.approx(81.300568221497215)
    assert earth + moon == pytest.approx(8.9970113929473466e-10 * AU_M**3 / DAY_S**2, rel=1e-15)


def test_mass_catalogs_reject_duplicate_ids_and_wrong_naif_targets():
    earth = BodyMass("EARTH", 1.0, ("major",), 399)
    with pytest.raises(ValueError, match="body IDs"):
        MassCatalog((earth, earth), name="duplicate")
    with pytest.raises(ValueError, match="canonical NAIF ID"):
        BodyMass("MOON", 1.0, ("major",), 399)
