"""The published IERS C04 series, when it is available next to the repository."""

import pickle
from dataclasses import replace

import pytest

from lunarops.classes.frames.earth_orientation import TabulatedEarthOrientation, read_iers_c04
from lunarops.classes.time import Epoch, TimeScale
from lunarops.fileio.earth_orientation import (
    load_earth_orientation_parameter,
    read_earth_orientation_parameter,
    write_earth_orientation_parameter,
)
from tests.conftest import REPO_ROOT

C04 = REPO_ROOT.parent / "data/auxiliary/eopc04.1962-now.txt"


@pytest.mark.requires_data
@pytest.mark.skipif(not C04.exists(), reason=f"{C04} is not available")
def test_real_c04_converts_to_native_files_and_filters_predictions(tmp_path):
    samples = read_iers_c04(C04)
    target = tmp_path / "eop.txt"
    write_earth_orientation_parameter(samples[:2], target)

    assert (samples[0].source, samples[0].quality) == ("C04-20", "final")
    assert read_earth_orientation_parameter(target) == samples[:2]
    assert [line for line in target.read_text().splitlines() if line.startswith("field ")] == [
        "field mjdUtc float d",
        "field xpArcsec float arcsec",
        "field ypArcsec float arcsec",
        "field ut1MinusUtcSec float s",
        "field lodSec float s",
        "field dxArcsec float arcsec",
        "field dyArcsec float arcsec",
        "field source text 1",
        "field quality text 1",
        "field sourceSha256 text 1",
    ]
    table = TabulatedEarthOrientation(samples[:2])
    assert pickle.loads(pickle.dumps(table)).samples == table.samples
    prediction = replace(samples[1], quality="prediction", source="finals2000A")
    write_earth_orientation_parameter([samples[0], prediction], target)
    filtered = load_earth_orientation_parameter(target, allow_predicted=False)
    assert len(filtered.samples) == 1
    with pytest.raises(ValueError, match="outside"):
        filtered.interpolate(Epoch(2400000.5, prediction.mjd_utc, TimeScale.UTC))
