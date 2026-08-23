import pickle

import numpy as np
import pytest

from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.observation.equations import (
    ObservationEquation,
    ObservationResultDetail,
    STANDARD_OUTPUT_FIELDS,
)

_UTC_EPOCH = Epoch(2458849.5, 0.0, TimeScale.UTC)


def _equation() -> ObservationEquation:
    return ObservationEquation(
        observed_minus_computed_one_way_m=0.12,
        sigma_one_way_m=0.02,
        design_partials={
            "station_range_bias": np.asarray([1.0]),
            "reflector_position_pa": np.asarray([0.1, 0.2, 0.3]),
        },
        observation_id=3,
        station_key="STA",
        reflector_key="REF",
        transmit_epoch_utc=_UTC_EPOCH,
        wavelength_nm=532.0,
    )


def test_output_level_accepts_enum_and_string():
    assert ObservationResultDetail.parse(ObservationResultDetail.FULL) is ObservationResultDetail.FULL
    assert ObservationResultDetail.parse("standard") is ObservationResultDetail.STANDARD
    assert ObservationResultDetail.parse(None) is ObservationResultDetail.STANDARD
    with pytest.raises(ValueError):
        ObservationResultDetail.parse("")


def test_equation_is_pickle_safe_for_mpi_transport():
    restored = pickle.loads(pickle.dumps(_equation()))
    assert restored.observation_id == 3
    assert restored.transmit_epoch_utc == _UTC_EPOCH
    assert restored.wavelength_nm == 532.0
    assert np.allclose(restored.design_partials["reflector_position_pa"], [0.1, 0.2, 0.3])


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
