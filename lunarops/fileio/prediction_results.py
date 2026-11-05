"""Read and write observation-prediction and visibility-window files.

Both are records tables (see :mod:`lunarops.fileio.result_table`) whose header is the caller's metadata
mapping. Times are ISO 8601 text timestamps: UTC ones carry no offset suffix, local ones carry the
``±hh:mm`` offset of the configured local civil time (omitted when it is zero).

``observationPrediction`` has one row per predicted transmit epoch ``t1``:

- ``utc_t1``, ``local_t1`` (text): transmit epoch in UTC and in local civil time.
- ``station``, ``reflector`` (text): station and reflector catalog keys.
- ``station_itrf_x_m``, ``station_itrf_y_m``, ``station_itrf_z_m`` (float, m): ITRF station position at
  transmit.
- ``reflector_itrf_x_m``, ``reflector_itrf_y_m``, ``reflector_itrf_z_m`` (float, m): ITRF reflector position
  at the bounce epoch.
- ``range_up_geometric_m`` (float, m): geometric uplink range.
- ``azimuth_deg``, ``elevation_deg`` (float, deg): direction of the uplink vector in the station's local frame,
  without refraction.
- ``observable`` (bool): True when every visibility criterion holds.

Floats are written with 6 decimals, except ``range_up_geometric_m`` with 3.

``predictionWindow`` has one row per visibility window: ``station`` and ``reflector`` (text); ``start_utc``,
``end_utc``, ``start_local`` and ``end_local`` (text timestamps of the first and last observable samples);
``sample_count`` (int); and ``duration_s`` (float, s, 6 decimals).
"""

from pathlib import Path

from .result_table import read_records, write_records

PREDICTION_ARTIFACT_TYPE = "observationPrediction"
WINDOW_ARTIFACT_TYPE = "predictionWindow"

PREDICTION_FIELDS = (
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


def write_prediction_results(rows, path, *, metadata=None) -> Path:
    """Write an ``observationPrediction`` file.

    Parameters
    ----------
    rows : Sequence of Mapping
        Prediction rows holding every prediction field; other keys are ignored.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.
    metadata : dict or None, optional
        Header mapping; empty when None.

    Returns
    -------
    Path
        The written path.
    """
    return write_records(
        path, PREDICTION_ARTIFACT_TYPE, metadata or {}, PREDICTION_FIELDS, rows,
        precision={
            name: ".3f" if name == "range_up_geometric_m" else ".6f"
            for name, kind in PREDICTION_FIELDS
            if kind == "float"
        },
    )


def read_prediction_results(path) -> list[dict[str, object]]:
    """Read the rows of an ``observationPrediction`` file.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    list of dict
        Prediction rows in file order; the header is discarded.

    Raises
    ------
    ValueError
        If the schema or units differ, or the records table is invalid.
    """
    return read_records(
        path,
        artifact_type=PREDICTION_ARTIFACT_TYPE,
        expected_fields=PREDICTION_FIELDS,
    )[1]


def write_prediction_windows(rows, path, *, metadata=None) -> Path:
    """Write a ``predictionWindow`` file.

    Parameters
    ----------
    rows : Sequence of Mapping
        Window rows holding every window field.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.
    metadata : dict or None, optional
        Header mapping; empty when None.

    Returns
    -------
    Path
        The written path.
    """
    return write_records(
        path, WINDOW_ARTIFACT_TYPE, metadata or {}, _WINDOW_FIELDS, rows,
        precision={"duration_s": ".6f"},
    )


def read_prediction_windows(path) -> list[dict[str, object]]:
    """Read the rows of a ``predictionWindow`` file.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    list of dict
        Window rows in file order; the header is discarded.

    Raises
    ------
    ValueError
        If the schema or units differ, or the records table is invalid.
    """
    return read_records(
        path, artifact_type=WINDOW_ARTIFACT_TYPE, expected_fields=_WINDOW_FIELDS
    )[1]


__all__ = [
    "PREDICTION_ARTIFACT_TYPE",
    "WINDOW_ARTIFACT_TYPE",
    "read_prediction_results",
    "read_prediction_windows",
    "write_prediction_results",
    "write_prediction_windows",
]
