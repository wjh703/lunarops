"""HARDISP ocean-loading synthesis: published example, epoch policy and the ENU/ITRF axes."""

import json

import numpy as np
import pytest

from lunarops import _iers2010
from lunarops.classes.displacement import Iers2010OceanTidalLoading, OceanTidalLoadingCatalog, StationDisplacementInput
from lunarops.classes.displacement.terrestrial_geometry import geodetic2itrf
from lunarops.classes.time import Epoch, TimeScale
from tests.conftest import DATA_DIR
from tests.support.samples import ONSALA_AMPLITUDES_M, ONSALA_HOURLY_UP_WEST_SOUTH_M, ONSALA_PHASES_DEG, blq_text
from tests.tolerances import DISPLACEMENT_M, HARDISP_M, Tolerance, assert_close

# The published example prints six decimals (half a micrometre).
PRINTED_M = Tolerance("PRINTED_M", atol=6e-7, unit="m")
# Frozen Cython outputs captured before the public Up/West/South axis order change.
FROZEN_AXES = json.loads((DATA_DIR / "hardisp_axis_baseline.json").read_text())["cases"]


def _onsala(*calendar, count=1, step_s=1.0):
    return np.column_stack(_iers2010.hardisp(*calendar, count, step_s, ONSALA_AMPLITUDES_M, ONSALA_PHASES_DEG))


def test_onsala_published_hourly_series():
    assert_close(_onsala(2009, 6, 25, 1, 10, 45, count=24, step_s=3600.0), ONSALA_HOURLY_UP_WEST_SOUTH_M, PRINTED_M)


@pytest.mark.parametrize(
    ("calendar", "up_west_south_m"),
    [
        ((2016, 12, 31, 23, 59, 59), (0.004732082132250071, -0.0012600324116647243, -0.0005435922648757696)),
        ((2017, 1, 1, 0, 0, 0), (0.00473177433013916, -0.0012598361354321241, -0.0005435359198600054)),
        ((2009, 6, 25, 1, 10, 45), (0.003094081999734044, -0.0008953595533967018, -0.0015382986748591065)),
        ((2024, 1, 1, 0, 0, 0), (0.0042690373957157135, -0.0016440704930573702, -0.00022759537387173623)),
    ],
)
def test_single_epochs_around_a_leap_second(calendar, up_west_south_m):
    assert_close(_onsala(*calendar)[0], up_west_south_m, HARDISP_M)


def test_regular_series_matches_individual_epochs():
    series = _onsala(2009, 6, 25, 1, 10, 45, count=4, step_s=900.0)
    scalar = np.vstack([_onsala(2009, 6, 25, 1, minute, 45) for minute in (10, 25, 40, 55)])
    assert_close(series, scalar, HARDISP_M)


def test_exact_leap_second_label_and_offset_transitions_are_rejected():
    with pytest.raises(ValueError, match="exact UTC leap-second label"):
        _onsala(2016, 12, 31, 23, 59, 60)
    with pytest.raises(ValueError, match="must not cross a UTC offset transition"):
        _onsala(2016, 12, 31, 23, 59, 59, count=3)


@pytest.mark.parametrize("case", FROZEN_AXES, ids=lambda c: f"{c['station']}-{c['calendar'][:3]}")
def test_frozen_hourly_series_keep_up_west_south_axes(case):
    actual = np.asarray(_iers2010.hardisp(*case["calendar"], 24, 3600.0, case["amplitudes"], case["phases"]))
    assert_close(actual, case["upWestSouthM"], HARDISP_M)
    assert_close([-actual[1], -actual[2], actual[0]], case["eastNorthUpM"], HARDISP_M)


@pytest.mark.parametrize("latitude,longitude", [(0.0, 0.0), (32.7803, -105.8203), (59.395, 11.926)])
def test_station_displacement_rotates_east_north_up_into_itrf(tmp_path, latitude, longitude):
    source = tmp_path / "loading.blq"
    source.write_text(blq_text(station_name="ONSALA"))
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(source))
    lat, lon = np.deg2rad([latitude, longitude])
    position = geodetic2itrf(latitude_rad=lat, longitude_rad=lon, ellipsoidal_height_m=2000.0)
    epoch = Epoch.from_calendar(2009, 6, 25, 1, 10, 45, scale=TimeScale.UTC)

    result = model.evaluate(StationDisplacementInput(reference_position_itrf_m=position, epoch_utc=epoch,
                                                     station_id="ONSALA"))

    up, west, south = _onsala(2009, 6, 25, 1, 10, 45)[0]
    # Independent East/North/Up basis at the site.
    basis = np.array([
        [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
        [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
        [0.0, np.cos(lat), np.sin(lat)],
    ])
    assert_close(result, basis @ np.array([-west, -south, up]), DISPLACEMENT_M)
