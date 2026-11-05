"""Config schemas of station and reflector coordinate entries.

A station entry gives an ITRF Cartesian position in metres, or WGS84 geodetic longitude, latitude and height
that are converted to it, with a linear velocity and a UTC reference epoch. A reflector entry gives a
position in the Moon principal-axis (PA) frame of the selected ephemeris, in metres.
"""

import math

from lunarops.classes.displacement.terrestrial_geometry import geodetic2itrf

from .schema import ConfigSchema, number, sequence, string


def _validate_station_coordinate(config: dict, path: str) -> dict:
    """Replace WGS84 geodetic coordinates by ITRF ``xyzM``; reject entries with both or neither form."""
    geodetic_fields = ("longitudeDeg", "latitudeDeg", "heightM")
    has_xyz = config.get("xyzM") is not None
    supplied_geodetic = [name for name in geodetic_fields if config.get(name) is not None]
    if has_xyz and supplied_geodetic:
        raise ValueError(f"{path} must specify either xyzM or all geodetic fields, not both.")
    if not has_xyz:
        missing = [name for name in geodetic_fields if config.get(name) is None]
        if missing:
            raise ValueError(f"{path} requires xyzM or all geodetic fields; missing {missing}.")
        config["xyzM"] = geodetic2itrf(
            latitude_rad=math.radians(float(config["latitudeDeg"])),
            longitude_rad=math.radians(float(config["longitudeDeg"])),
            ellipsoidal_height_m=float(config["heightM"]),
        ).tolist()
        for field in geodetic_fields:
            config.pop(field, None)
    return config


STATION_COORDINATE_SCHEMA = ConfigSchema(
    fields=(
        string("key", required=True, non_empty=True, description="Station catalog key."),
        sequence(
            "xyzM",
            item_kind="number",
            min_items=3,
            max_items=3,
            allow_none=True,
            description=(
                "ITRF Cartesian position (X, Y, Z) at `positionEpochUtc`, in metres. Give either this or the three "
                "geodetic fields."
            ),
        ),
        number(
            "longitudeDeg",
            ge=-180.0,
            le=180.0,
            allow_none=True,
            description="WGS84 geodetic longitude, east positive, in degrees.",
        ),
        number(
            "latitudeDeg",
            ge=-90.0,
            le=90.0,
            allow_none=True,
            description="WGS84 geodetic latitude, north positive, in degrees.",
        ),
        number("heightM", allow_none=True, description="WGS84 ellipsoidal height, in metres."),
        sequence(
            "velocityMPerYear",
            default=[0.0, 0.0, 0.0],
            item_kind="number",
            min_items=3,
            max_items=3,
            description="ITRF linear velocity (X, Y, Z), in metres per year.",
        ),
        string(
            "positionEpochUtc",
            default="2010-01-01T00:00:00",
            non_empty=True,
            description="UTC reference epoch of the position, as an ISO timestamp.",
        ),
    ),
    description=(
        "One station position given as ITRF XYZ or WGS84 geodetic longitude, "
        "latitude, and ellipsoidal height, with an optional linear velocity model."
    ),
    validator=_validate_station_coordinate,
)


REFLECTOR_COORDINATE_SCHEMA = ConfigSchema(
    fields=(
        string("key", required=True, non_empty=True, description="Reflector catalog key."),
        sequence(
            "xyzM",
            required=True,
            item_kind="number",
            min_items=3,
            max_items=3,
            description=(
                "Reflector position (X, Y, Z) in the lunar principal-axis (PA) frame of the selected ephemeris, in "
                "metres."
            ),
        ),
    ),
    description="One reflector position in the selected ephemeris Moon PA frame.",
)


def validate_coordinate_keys(entries, location):
    """Check that coordinate entries have distinct keys, ignoring case and surrounding whitespace.

    Parameters
    ----------
    entries : list of dict
        Coordinate entries, each with a ``key``.
    location : str
        Config location used in error messages.

    Returns
    -------
    list of dict
        The entries, unchanged.

    Raises
    ------
    ValueError
        If two entries have the same key.
    """
    seen = set()
    for entry in entries:
        key = entry["key"].strip()
        if key.casefold() in seen:
            raise ValueError(f"{location} contains duplicate key {key!r}.")
        seen.add(key.casefold())
    return entries
