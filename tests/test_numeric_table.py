from __future__ import annotations

import numpy as np
import pytest

from lunarops.fileio.numeric_table import read_numeric_table, read_numeric_table_type, write_numeric_table


def test_numeric_table_round_trip(tmp_path):
    values = np.arange(12, dtype=float).reshape(4, 3)
    path = write_numeric_table(
        tmp_path / "orbit.dat",
        "lunarOrbit",
        ("t", "x", "y"),
        values,
        metadata={"frame": "BCRS"},
    )
    artifact_type, columns, restored, metadata = read_numeric_table(path, "lunarOrbit")
    assert artifact_type == read_numeric_table_type(path) == "lunarOrbit"
    assert columns == ("t", "x", "y")
    assert metadata == {"frame": "BCRS"}
    np.testing.assert_array_equal(restored, values)


def test_numeric_table_rejects_wrong_artifact_type(tmp_path):
    path = write_numeric_table(tmp_path / "orbit.dat", "lunarOrbit", ("t",), [[0.0]])
    with pytest.raises(ValueError, match="Expected"):
        read_numeric_table(path, "lunarAccelerationDiagnostics")
