import pickle

import numpy as np
import pytest

from lunarops.classes.observation.equations import (
    STANDARD_OUTPUT_FIELDS,
    EquationBatch,
    ObservationResultDetail,
)
from lunarops.classes.time import Epoch, TimeScale

_UTC_EPOCH = Epoch(2458849.5, 0.0, TimeScale.UTC)


def _equation() -> EquationBatch:
    return EquationBatch.single(
        oc=0.12,
        sigma=0.02,
        reflector_position_pa=np.asarray([0.1, 0.2, 0.3]),
        observation_id=3,
        station="STA",
        reflector="REF",
        epoch=_UTC_EPOCH,
        wavelength_nm=532.0,
        receive_epoch=_UTC_EPOCH,
    )


def test_output_level_accepts_enum_and_string():
    assert ObservationResultDetail.parse(ObservationResultDetail.FULL) is ObservationResultDetail.FULL
    assert ObservationResultDetail.parse("standard") is ObservationResultDetail.STANDARD
    assert ObservationResultDetail.parse(None) is ObservationResultDetail.STANDARD
    with pytest.raises(ValueError):
        ObservationResultDetail.parse("")


def test_equation_is_pickle_safe_for_mpi_transport():
    restored = pickle.loads(pickle.dumps(_equation()))
    assert restored.observation_id[0] == 3
    assert restored.epoch(0) == _UTC_EPOCH
    assert restored.wavelength_nm[0] == 532.0
    assert np.allclose(restored.reflector_position_pa[0], [0.1, 0.2, 0.3])


def test_standard_output_schema_contains_only_per_record_oc_fields():
    assert STANDARD_OUTPUT_FIELDS == (
        "obs_time_utc",
        "obs_time_local",
        "normal_point_index",
        "station_id",
        "station_name",
        "reflector_id",
        "reflector_name",
        "observed_rtt_s",
        "computed_rtt_s",
        "oc_one_way_m",
        "observation_sigma_one_way_m",
        "elevation_up_deg",
        "light_time_converged",
        "status",
    )


def test_r5_batch_report_rows_write_native_bool_columns(tmp_path):
    from types import SimpleNamespace

    from lunarops.base.station_interval import StationInterval
    from lunarops.classes.parametrization.base import ParametrizationList
    from lunarops.classes.parametrization.station_range_bias import StationRangeBiasParametrization
    from lunarops.config.context import RunContext
    from lunarops.estimation.adjustment_reporting import observation_records
    from lunarops.estimation.variance_component_groups import VarianceComponentDefinition
    from lunarops.fileio.observation_results import read_observation_results
    from lunarops.programs.llr_processing import _write_residuals

    batch = EquationBatch.concatenate([_equation(), _equation()])
    batch.observation_id[:] = [3, 4]
    batch.converged[:] = [True, False]
    batch.clamped[:] = [False, True]
    parametrization = ParametrizationList([StationRangeBiasParametrization(per="station")])
    parametrization.setup(batch, None)
    values = {3: 0.12, 4: -0.02}
    weights = {3: 1.0, 4: 0.5}
    screening = {key: {"reportedSigmaM": 0.02, "minimumValidSigmaM": 0.001,
                       "status": "RETAINED", "reason": None} for key in values}
    rows = observation_records(
        batch, values, values, {3: 0.01, 4: 0.01}, {3: 1., 4: -1.}, {"A": 1.}, weights, weights,
        assignments={3: "A", 4: "A"}, parametrization=parametrization,
        components=[VarianceComponentDefinition("A", StationInterval("STA", StationInterval.parse("2020-01-01"), None))],
        accuracy_screening_records=screening,
    )
    assert [type(row["lightTimeConverged"]) for row in rows] == [bool, bool]
    assert [type(row["troposphereClamped"]) for row in rows] == [bool, bool]
    assert all(type(row["stationId"]) is str and type(row["sigma0M"]) is float for row in rows)
    datasets = {"input": SimpleNamespace(records=[SimpleNamespace(index=3), SimpleNamespace(index=4)])}
    _write_residuals(SimpleNamespace(output_level="full", output_file="residuals.txt"),
                     SimpleNamespace(observations=rows), datasets, RunContext(working_dir=tmp_path))
    restored = read_observation_results(tmp_path / "residuals.txt")
    assert [row["light_time_converged"] for row in restored] == [True, False]
    assert [row["troposphere_clamped"] for row in restored] == [False, True]
