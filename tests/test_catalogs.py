import numpy as np
import pytest

from lunarops.base.station_identity import (
    canonical_station_id,
    normalize_station_key,
    registered_station_id,
    station_aliases,
    station_display_name,
    station_ilrs_code,
    station_names,
)
from lunarops.classes.displacement.terrestrial_geometry import itrf2geodetic
from lunarops.classes.observation.catalogs import (
    ReflectorRecord,
    StationRecord,
    resolve_catalog_key,
)
from lunarops.fileio.catalogs import (
    load_reflector_catalog,
    load_station_catalog,
    read_reflector_catalog,
    read_station_catalog,
    write_reflector_catalog,
    write_station_catalog,
)
from lunarops.fileio.catalog_sources import station_catalog_from_coordinates


def test_resolve_catalog_key_exact_case_compact_and_alias():
    catalog = {
        "Foo-Bar": StationRecord(
            name="Foo Bar Station",
            itrf_xyz_m=(1.0, 2.0, 3.0),
            aliases=("FB-01",),
        )
    }

    assert resolve_catalog_key("Foo-Bar", catalog, "Station") == "Foo-Bar"
    assert resolve_catalog_key("foo-bar", catalog, "Station") == "Foo-Bar"
    assert resolve_catalog_key("foobar", catalog, "Station") == "Foo-Bar"
    assert resolve_catalog_key("fb01", catalog, "Station") == "Foo-Bar"
    with pytest.raises(KeyError):
        resolve_catalog_key("missing", catalog, "Station")


def test_builtin_catalog_loaders_are_rejected():
    with pytest.raises(ValueError, match="Builtin station catalogs"):
        load_station_catalog("builtin")
    with pytest.raises(ValueError, match="Builtin reflector catalogs"):
        load_reflector_catalog("builtin")


def test_station_identity_is_independent_of_coordinate_catalogs():
    assert canonical_station_id("Apache Point Observatory") == "APOLLO"
    assert canonical_station_id("70610") == "APOLLO"
    assert station_ilrs_code("APOL") == "70610"
    assert station_display_name("APOLLO") == "Apache Point Observatory"


def test_station_identity_separates_normalization_and_registered_resolution():
    assert normalize_station_key("Apache Point Observatory") == "APACHEPOINTOBSERVATORY"
    assert registered_station_id("Apache Point Observatory") == "APOLLO"
    assert canonical_station_id("custom-station") == "CUSTOMSTATION"
    with pytest.raises(ValueError, match="Unknown registered station"):
        registered_station_id("custom-station")


def test_station_names_includes_every_registered_spelling():
    names = station_names("APOL")

    assert names[0] == "APOLLO"
    assert "70610" in names
    assert "Apache Point Observatory" in names
    assert "APOL" in names
    assert station_aliases("APOLLO") == ("70610", "APOL", "APACHE", "APACHEPOINT", "7045")


def test_typed_catalog_files_round_trip(tmp_path):
    stations = {
        "TEST": StationRecord(
            "TEST",
            [1.0, 2.0, 3.0],
            itrf_velocity_m_per_year=[0.1, 0.2, 0.3],
            position_epoch_utc="2020-01-01T00:00:00",
        )
    }
    reflectors = {"REF": ReflectorRecord("REF", [4.0, 5.0, 6.0])}
    station_path = tmp_path / "stations.txt.gz"
    reflector_path = tmp_path / "reflectors.txt.gz"

    write_station_catalog(stations, station_path)
    write_reflector_catalog(reflectors, reflector_path)
    recovered_station = read_station_catalog(station_path)["TEST"]
    recovered_reflector = read_reflector_catalog(reflector_path)["REF"]

    assert recovered_station.name == "TEST"
    assert recovered_station.aliases == ()
    assert np.allclose(recovered_station.itrf_velocity_m_per_year, [0.1, 0.2, 0.3])
    assert recovered_reflector.name == "REF"
    assert recovered_reflector.aliases == ()


def test_station_catalog_accepts_wgs84_geodetic_coordinates():
    catalog = station_catalog_from_coordinates(
        [
            {
                "key": "TEST",
                "longitudeDeg": -105.0,
                "latitudeDeg": 40.0,
                "heightM": 1600.0,
            }
        ]
    )

    position = itrf2geodetic(catalog["TEST"].itrf_xyz_m)
    assert position.longitude_deg == pytest.approx(-105.0, abs=1.0e-10)
    assert position.latitude_deg == pytest.approx(40.0, abs=1.0e-10)
    assert position.ellipsoidal_height_m == pytest.approx(1600.0, abs=1.0e-5)


@pytest.mark.parametrize(
    "coordinate, message",
    [
        (
            {
                "key": "TEST",
                "xyzM": [1.0, 2.0, 3.0],
                "longitudeDeg": 10.0,
                "latitudeDeg": 20.0,
                "heightM": 30.0,
            },
            "either xyzM or all geodetic fields",
        ),
        (
            {"key": "TEST", "longitudeDeg": 10.0, "latitudeDeg": 20.0},
            "requires xyzM or all geodetic fields",
        ),
    ],
)
def test_station_catalog_rejects_ambiguous_or_incomplete_coordinate_forms(coordinate, message):
    with pytest.raises(ValueError, match=message):
        station_catalog_from_coordinates([coordinate])
