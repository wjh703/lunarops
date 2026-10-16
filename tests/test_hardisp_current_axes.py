"""Frozen full-amplitude/phase regression for the current Up/West/South API."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_ocean_tidal_loading import _blq_text

from lunarops import _iers2010
from lunarops.classes.displacement import Iers2010OceanTidalLoading, OceanTidalLoadingCatalog, StationDisplacementInput
from lunarops.classes.displacement.terrestrial_geometry import geodetic2itrf
from lunarops.classes.time import Epoch, TimeScale

CASES = json.loads((Path(__file__).parent / "data/hardisp_axis_baseline.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['station']}-{c['calendar'][:3]}")
def test_full_hourly_series_matches_frozen_physical_axes(case):
    result = np.asarray(_iers2010.hardisp(*case["calendar"], 24, 3600.0, case["amplitudes"], case["phases"]))
    np.testing.assert_array_equal(result, case["upWestSouthM"])
    np.testing.assert_array_equal(np.asarray([-result[1], -result[2], result[0]]), case["eastNorthUpM"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['station']}-{c['calendar'][:3]}")
@pytest.mark.parametrize("latitude,longitude", [(0.0, 0.0), (32.7803, -105.8203), (59.395, 11.926)])
def test_station_enu_and_itrf_preserve_frozen_displacements(tmp_path, case, latitude, longitude):
    source = tmp_path / "loading.blq"
    source.write_text(
        _blq_text(
            station_name=case["station"], amplitudes=np.asarray(case["amplitudes"]), phases=np.asarray(case["phases"])
        )
    )
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(source))
    lat, lon = np.deg2rad([latitude, longitude])
    position = geodetic2itrf(latitude_rad=lat, longitude_rad=lon, ellipsoidal_height_m=2000.0)
    epoch = Epoch.from_calendar(*case["calendar"], scale=TimeScale.UTC)
    result = model.evaluate(
        StationDisplacementInput(reference_position_itrf_m=position, epoch_utc=epoch, station_id=case["station"])
    )
    expected_enu = np.asarray(case["eastNorthUpM"])[:, 0]
    np.testing.assert_array_equal(result.displacement_enu_m, expected_enu)
    np.testing.assert_array_equal(result.displacement_up_west_south_m, np.asarray(case["upWestSouthM"])[:, 0])
    # Independent explicit East/North/Up basis, in the same terrestrial frame.
    basis = np.array(
        [
            [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
            [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
            [0.0, np.cos(lat), np.sin(lat)],
        ]
    )
    np.testing.assert_allclose(result.displacement_itrf_m, basis @ expected_enu, rtol=0, atol=1e-17)
