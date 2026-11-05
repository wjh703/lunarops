"""Equation batches, observation result files and their output levels."""

import pickle

import numpy as np
import pytest

from lunarops.classes.observation.equations import STANDARD_OUTPUT_FIELDS, EquationBatch, ObservationResultDetail
from lunarops.classes.time import Epoch, TimeScale
from lunarops.fileio.observation_results import read_observation_results, write_observation_results

UTC_EPOCH = Epoch(2458849.5, 0.0, TimeScale.UTC)


def _equation(**changes):
    values = {"oc": 0.12, "sigma": 0.02, "reflector_position_pa": np.asarray([0.1, 0.2, 0.3]), "observation_id": 3,
              "station": "STA", "reflector": "REF", "epoch": UTC_EPOCH, "wavelength_nm": 532.0,
              "receive_epoch": UTC_EPOCH}
    return EquationBatch.single(**{**values, **changes})


def test_output_level_parses_enum_names_and_defaults_to_standard():
    assert ObservationResultDetail.parse(ObservationResultDetail.FULL) is ObservationResultDetail.FULL
    assert ObservationResultDetail.parse("standard") is ObservationResultDetail.STANDARD
    assert ObservationResultDetail.parse(None) is ObservationResultDetail.STANDARD
    with pytest.raises(ValueError):
        ObservationResultDetail.parse("")


def test_standard_output_contains_only_the_per_record_oc_fields():
    assert STANDARD_OUTPUT_FIELDS == (
        "obs_time_utc", "obs_time_local", "normal_point_index", "station_id", "station_name", "reflector_id",
        "reflector_name", "observed_rtt_s", "computed_rtt_s", "oc_one_way_m", "observation_sigma_one_way_m",
        "elevation_up_deg", "light_time_converged", "status",
    )


def test_equation_batch_survives_pickling_for_mpi_transport():
    restored = pickle.loads(pickle.dumps(_equation()))

    assert restored.observation_id[0] == 3
    assert restored.epoch(0) == UTC_EPOCH
    assert restored.wavelength_nm[0] == 532.0
    np.testing.assert_array_equal(restored.reflector_position_pa[0], [0.1, 0.2, 0.3])


def test_equation_batch_selects_rows_by_mask_and_is_not_row_iterable():
    batch = EquationBatch.concatenate([_equation(observation_id=1), _equation(observation_id=2, oc=0.3)])

    assert batch[np.array([False, True])].oc.tolist() == [0.3]
    with pytest.raises(TypeError, match="not iterable"):
        list(batch)


def test_result_files_round_trip_sources_nulls_and_native_types(tmp_path):
    rows = {
        "source-a": [
            {"normal_point_index": 2, "oc_one_way_m": 0.1, "light_time_converged": True, "label": "station A"},
            {"normal_point_index": 1, "oc_one_way_m": 1.0e-15, "light_time_converged": False, "label": "station B"},
        ],
        "source-b": [{"normal_point_index": 3, "light_time_converged": True, "label": "station C"}],
    }
    for name in ("rows.txt", "rows.txt.gz"):
        restored = read_observation_results(write_observation_results(rows, tmp_path / name))

        assert [row["source"] for row in restored] == ["source-a", "source-a", "source-b"]
        assert [row["oc_one_way_m"] for row in restored] == [0.1, 1.0e-15, None]
        assert [row["light_time_converged"] for row in restored] == [True, False, True]
        assert restored[0]["label"] == "station A"
        assert type(restored[0]["normal_point_index"]) is int
