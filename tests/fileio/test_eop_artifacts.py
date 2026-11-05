"""Native earthOrientationParameter files."""

import pytest

from lunarops.classes.frames import EarthOrientationSample
from lunarops.fileio.earth_orientation import read_earth_orientation_parameter, write_earth_orientation_parameter

C04 = (
    EarthOrientationSample(60000.0, 0.1, 0.2, 0.3, 0.004, -0.005, 0.0012),
    EarthOrientationSample(60001.0, 0.11, 0.21, 0.31, 0.006, -0.007, 0.0013),
)
RAPID = (
    EarthOrientationSample(60001.0, 0.12, 0.22, 0.32, 0.008, -0.009, 0.0014),
    EarthOrientationSample(60002.0, 0.13, 0.23, 0.33, 0.010, -0.011, 0.0015),
)


def test_round_trip_keeps_every_sample_field(tmp_path):
    path = write_earth_orientation_parameter(C04, tmp_path / "c04.txt")
    assert read_earth_orientation_parameter(path) == C04


def test_merged_file_prefers_c04_where_both_series_exist(tmp_path):
    by_mjd = {sample.mjd_utc: sample for sample in RAPID} | {sample.mjd_utc: sample for sample in C04}
    path = write_earth_orientation_parameter(tuple(by_mjd[mjd] for mjd in sorted(by_mjd)), tmp_path / "merged.txt")

    assert read_earth_orientation_parameter(path) == (C04[0], C04[1], RAPID[1])


def test_reader_rejects_a_file_without_a_records_header(tmp_path):
    path = tmp_path / "bad_eop.txt"
    path.write_text("not an eop row\nstill not eop\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_earth_orientation_parameter(path)
