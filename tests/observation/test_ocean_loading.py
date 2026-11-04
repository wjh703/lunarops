"""BLQ ocean-loading catalogs and the HARDISP station-displacement model."""

import numpy as np
import pytest

from lunarops.classes.displacement import Iers2010OceanTidalLoading, OceanTidalLoadingCatalog, StationDisplacementInput
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.context import RunContext
from tests.support.samples import (
    FES2022B_APOLLO_AMPLITUDES_M,
    FES2022B_APOLLO_PHASES_DEG,
    ONSALA_AMPLITUDES_M,
    ONSALA_PHASES_DEG,
    OREKIT_FES2022B_APOLLO,
    blq_row,
    blq_text,
)
from tests.tolerances import HARDISP_M, Tolerance, assert_close

# Orekit and LunarOps use different HARDISP interpolation details; 5 micrometres agree.
OREKIT_M = Tolerance("OREKIT_M", atol=5e-6, unit="m")
# WGS84 conversion of the APOLLO BLQ header's position.
APOLLO_ITRF_M = np.array([-1_463_996.2265579700, -5_166_630.8462324440, 3_435_016.8950683116])


@pytest.fixture
def blq_file(tmp_path):
    path = tmp_path / "fes2022b.txt"
    path.write_text(blq_text(), encoding="utf-8")
    return path


def _input(text, station_id="APOLLO", position=(6_378_137.0, 0.0, 0.0)):
    return StationDisplacementInput(reference_position_itrf_m=np.asarray(position, dtype=float),
                                    epoch_utc=Epoch.from_isot(text, scale=TimeScale.UTC), station_id=station_id)


def test_catalog_reads_metadata_and_read_only_coefficients(blq_file):
    catalog = OceanTidalLoadingCatalog(blq_file)
    coefficients = catalog.coefficients_for("APOLLO")

    assert catalog.station_ids == ("APOLLO",)
    assert catalog.tidal_model == "FES2022b"
    assert (coefficients.station_id, coefficients.source_station_name) == ("APOLLO", "APOLLO")
    np.testing.assert_array_equal(coefficients.amplitudes_m, ONSALA_AMPLITUDES_M)
    np.testing.assert_array_equal(coefficients.phases_deg, ONSALA_PHASES_DEG)
    assert not coefficients.amplitudes_m.flags.writeable and not coefficients.phases_deg.flags.writeable


@pytest.mark.parametrize(("edit", "message"), [
    (lambda text: text.replace(blq_row(ONSALA_AMPLITUDES_M[0]), "0.1 0.2"), "11 values"),
    (lambda text: text.replace("0.00123", "not-a-number", 1), "invalid BLQ numeric value"),
    (lambda text: text.replace(f"{blq_row(ONSALA_PHASES_DEG[-1])}\n$$ END TABLE", "$$ END TABLE"), "expected 6"),
    (lambda text: text.replace("$$ END TABLE", "  APOLLO\n" + "\n".join(
        [*map(blq_row, ONSALA_AMPLITUDES_M), *map(blq_row, ONSALA_PHASES_DEG), "$$ END TABLE"])),
     "duplicate BLQ station"),
    (lambda text: text.replace("M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA", "S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SSA", 1),
     "column order"),
], ids=["short-row", "non-numeric", "missing-row", "duplicate-station", "column-order"])
def test_catalog_rejects_malformed_files(tmp_path, edit, message):
    path = tmp_path / "bad.blq"
    path.write_text(edit(blq_text()), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        OceanTidalLoadingCatalog(path)
    with pytest.raises(FileNotFoundError, match="BLQ file not found"):
        OceanTidalLoadingCatalog(tmp_path / "missing.blq")


@pytest.mark.parametrize(("epoch_text", "_usw", "_enu", "expected_itrf_m"), OREKIT_FES2022B_APOLLO)
def test_displacement_matches_an_independent_orekit_series(tmp_path, epoch_text, _usw, _enu, expected_itrf_m):
    path = tmp_path / "fes2022b.txt"
    path.write_text(blq_text(amplitudes=FES2022B_APOLLO_AMPLITUDES_M, phases=FES2022B_APOLLO_PHASES_DEG))
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(path))
    assert_close(model.evaluate(_input(epoch_text, position=APOLLO_ITRF_M)), expected_itrf_m, OREKIT_M)


@pytest.mark.parametrize(("text", "up_west_south_m"), [
    ("2016-12-31T23:59:59", (0.004732082132250071, -0.0012600324116647243, -0.0005435922648757696)),
    ("2017-01-01T00:00:00", (0.00473177433013916, -0.0012598361354321241, -0.0005435359198600054)),
    ("2024-01-01T00:00:00", (0.0042690373957157135, -0.0016440704930573702, -0.00022759537387173623)),
])
def test_displacement_around_a_leap_second_keeps_the_utc_calendar(blq_file, text, up_west_south_m):
    up, west, south = up_west_south_m
    # On the equator at zero longitude, East/North/Up are the ITRF y, z and x axes.
    expected_itrf = np.array([up, -west, -south])
    assert_close(Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(blq_file)).evaluate(_input(text)), expected_itrf,
                 HARDISP_M)


def test_unrepresentable_epochs_and_missing_stations_are_rejected(blq_file):
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(blq_file))
    with pytest.raises(ValueError, match="exact UTC leap-second label"):
        model.displacement_itrf_m(_input("2016-12-31T23:59:60"))
    with pytest.raises(ValueError, match="supports UTC epochs only"):
        model.displacement_itrf_m(_input("2027-07-01T00:00:00"))
    with pytest.raises(ValueError, match="station_id"):
        model.displacement_itrf_m(_input("2009-06-25T01:10:45", station_id=None))
    with pytest.raises(KeyError, match="WETTZELL"):
        model.displacement_itrf_m(_input("2009-06-25T01:10:45", station_id="WETTZELL"))


def test_factory_checks_the_declared_model_against_the_file_header(blq_file):
    context = RunContext(working_dir=blq_file.parent)
    config = {"type": "iers2010OceanTidalLoading", "coefficientFile": blq_file.name, "model": "FES2022b"}

    assert isinstance(context.create_class("stationDisplacement", config, cache=False), Iers2010OceanTidalLoading)
    with pytest.raises(ValueError, match="model mismatch"):
        context.create_class("stationDisplacement", {**config, "model": "FES2014b"}, cache=False)
