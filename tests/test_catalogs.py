import numpy as np
import pytest

from lunarops.base.station_identity import (
    validate_station_id,
    station_id_from_ilrs_code,
    station_ilrs_code,
)
from lunarops.classes.displacement.terrestrial_geometry import itrf2geodetic
from lunarops.classes.observation.catalogs import (
    ReflectorRecord,
    StationRecord,
    resolve_catalog_key,
)
from lunarops.fileio.catalog_sources import station_catalog_from_coordinates
from lunarops.fileio.catalogs import (
    read_reflector_catalog,
    read_station_catalog,
    write_reflector_catalog,
    write_station_catalog,
)


def test_resolve_catalog_key_exact_case_compact_and_alias():
    catalog = {
        "Foo-Bar": StationRecord(
            name="Foo Bar Station",
            itrf_xyz_m=(1.0, 2.0, 3.0),
        )
    }

    assert resolve_catalog_key("Foo-Bar", catalog, "Station") == "Foo-Bar"
    for value in ("foo-bar", "foobar", "fb01", "Foo Bar Station", "missing"):
        with pytest.raises(KeyError):
            resolve_catalog_key(value, catalog, "Station")


def test_station_identity_is_independent_of_coordinate_catalogs():
    assert station_id_from_ilrs_code("70610") == "APOLLO"
    assert station_ilrs_code("APOLLO") == "70610"
    assert validate_station_id("custom-station") == "custom-station"
    for alias in ("Apache Point Observatory", "70610", "APOL", "apollo", "custom-station"):
        with pytest.raises(ValueError, match="Unknown registered station"):
            station_ilrs_code(alias)


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
    assert np.allclose(recovered_station.itrf_velocity_m_per_year, [0.1, 0.2, 0.3])
    assert recovered_reflector.name == "REF"


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
