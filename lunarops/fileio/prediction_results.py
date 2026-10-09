"""Prediction and visibility-window schemas for the scalar table archive."""

from pathlib import Path

from .result_table import read_table, write_table

PREDICTION_ARTIFACT_TYPE = "observationPrediction"
WINDOW_ARTIFACT_TYPE = "predictionWindow"
PREDICTION_FORMAT_VERSION = 5
WINDOW_FORMAT_VERSION = 4

_PREDICTION_FIELDS = (
    ("utc_t1", "text"),
    ("local_t1", "text"),
    ("station", "text"),
    ("reflector", "text"),
    ("station_itrf_x_m", "float"),
    ("station_itrf_y_m", "float"),
    ("station_itrf_z_m", "float"),
    ("reflector_itrf_x_m", "float"),
    ("reflector_itrf_y_m", "float"),
    ("reflector_itrf_z_m", "float"),
    ("range_up_geometric_m", "float"),
    ("azimuth_deg", "float"),
    ("elevation_deg", "float"),
    ("observable", "bool"),
)

_WINDOW_FIELDS = (
    ("station", "text"),
    ("reflector", "text"),
    ("start_utc", "text"),
    ("end_utc", "text"),
    ("start_local", "text"),
    ("end_local", "text"),
    ("sample_count", "int"),
    ("duration_s", "float"),
)


def write_prediction_results(rows, path) -> Path:
    return write_table(
        rows,
        path,
        artifact_type=PREDICTION_ARTIFACT_TYPE,
        version=PREDICTION_FORMAT_VERSION,
        fields=_PREDICTION_FIELDS,
        precision={
            name: ".3f" if name == "range_up_geometric_m" else ".6f"
            for name, kind in _PREDICTION_FIELDS
            if kind == "float"
        },
    )


def read_prediction_results(path) -> list[dict[str, object]]:
    return read_table(
        path,
        artifact_type=PREDICTION_ARTIFACT_TYPE,
        version=PREDICTION_FORMAT_VERSION,
        expected_fields=_PREDICTION_FIELDS,
    )


def write_prediction_windows(rows, path) -> Path:
    return write_table(
        rows,
        path,
        artifact_type=WINDOW_ARTIFACT_TYPE,
        version=WINDOW_FORMAT_VERSION,
        fields=_WINDOW_FIELDS,
        precision={"duration_s": ".6f"},
    )


def read_prediction_windows(path) -> list[dict[str, object]]:
    return read_table(
        path, artifact_type=WINDOW_ARTIFACT_TYPE, version=WINDOW_FORMAT_VERSION, expected_fields=_WINDOW_FIELDS
    )


__all__ = [
    "PREDICTION_ARTIFACT_TYPE",
    "WINDOW_ARTIFACT_TYPE",
    "read_prediction_results",
    "read_prediction_windows",
    "write_prediction_results",
    "write_prediction_windows",
]
