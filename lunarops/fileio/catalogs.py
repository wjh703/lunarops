"""Read and write native station and reflector catalogs with explicit reference frames.

Both catalogs are records tables (see :mod:`lunarops.fileio.result_table`) with one row per catalog entry,
sorted by key on write.

``stationCatalog`` has the header ``{"frame": "ITRF"}`` and the fields:

- ``key`` (text): station catalog key.
- ``x_m``, ``y_m``, ``z_m`` (float, m): ITRF Cartesian position at the reference epoch.
- ``vx_m_per_year``, ``vy_m_per_year``, ``vz_m_per_year`` (float): ITRF velocity in metres per Julian year.
  The unit token in the file is ``1`` because the name suffix maps to no unit.
- ``epoch`` (text): reference epoch of the position, an ISO 8601 UTC timestamp.

``reflectorCatalog`` has the header ``{"frame": "MOON_PA"}`` and the fields ``key`` (text) and ``x_m``,
``y_m``, ``z_m`` (float, m): position in the lunar principal-axis (PA) frame.
"""

from pathlib import Path

from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord

from .result_table import read_records, write_records

_POSITION_FIELDS = tuple((axis + "_m", "float") for axis in "xyz")
_STATION_FIELDS = (
    ("key", "text"),
    *_POSITION_FIELDS,
    *((axis + "_m_per_year", "float") for axis in ("vx", "vy", "vz")),
    ("epoch", "text"),
)
_REFLECTOR_FIELDS = (("key", "text"), *_POSITION_FIELDS)


def _write_catalog(catalog, path, *, station):
    rows = []
    fields = _STATION_FIELDS if station else _REFLECTOR_FIELDS
    for key, record in sorted(catalog.items()):
        if station:
            values = (key, *record.itrf_xyz_m, *record.itrf_velocity_m_per_year, record.position_epoch_utc)
        else:
            values = (key, *record.moon_fixed_xyz_m)
        rows.append(dict(zip((name for name, _ in fields), values)))
    return write_records(
        path,
        "stationCatalog" if station else "reflectorCatalog",
        {"frame": "ITRF" if station else "MOON_PA"},
        fields,
        rows,
    )


def _read_catalog(path, *, station):
    header, rows = read_records(
        path,
        "stationCatalog" if station else "reflectorCatalog",
        expected_fields=_STATION_FIELDS if station else _REFLECTOR_FIELDS,
    )
    if header["frame"] != ("ITRF" if station else "MOON_PA"):
        raise ValueError("Unexpected catalog reference frame.")
    catalog: dict[str, StationRecord | ReflectorRecord] = {}
    keys = set()
    for row in rows:
        key = row["key"]
        if not key or key.casefold() in keys:
            raise ValueError(f"Invalid or duplicate catalog key {key!r}.")
        keys.add(key.casefold())
        position = [row[axis + "_m"] for axis in "xyz"]
        if station:
            velocity = [row[axis + "_m_per_year"] for axis in ("vx", "vy", "vz")]
            catalog[key] = StationRecord(key, position, velocity, row["epoch"])
        else:
            catalog[key] = ReflectorRecord(key, position)
    return catalog


def write_station_catalog(catalog: dict[str, StationRecord], path: str | Path) -> Path:
    """Write a ``stationCatalog`` file.

    Parameters
    ----------
    catalog : dict of str to StationRecord
        Stations by catalog key. The key, not ``StationRecord.name``, is written.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    Path
        The written path.
    """
    return _write_catalog(catalog, path, station=True)


def read_station_catalog(path: str | Path) -> dict[str, StationRecord]:
    """Read a ``stationCatalog`` file.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    dict of str to StationRecord
        Stations by key, named by their key, with ITRF position (m), velocity (m per Julian year) and UTC
        reference epoch.

    Raises
    ------
    ValueError
        If the schema or units differ, the header frame is not ``ITRF``, or a key is empty or repeated
        (case-insensitive).
    """
    return _read_catalog(path, station=True)


def write_reflector_catalog(catalog: dict[str, ReflectorRecord], path: str | Path) -> Path:
    """Write a ``reflectorCatalog`` file.

    Parameters
    ----------
    catalog : dict of str to ReflectorRecord
        Reflectors by catalog key. The key, not ``ReflectorRecord.name``, is written.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    Path
        The written path.
    """
    return _write_catalog(catalog, path, station=False)


def read_reflector_catalog(path: str | Path) -> dict[str, ReflectorRecord]:
    """Read a ``reflectorCatalog`` file.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    dict of str to ReflectorRecord
        Reflectors by key, named by their key, with Moon PA position in metres.

    Raises
    ------
    ValueError
        If the schema or units differ, the header frame is not ``MOON_PA``, or a key is empty or repeated
        (case-insensitive).
    """
    return _read_catalog(path, station=False)
