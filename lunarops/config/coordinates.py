import math

from lunarops.classes.displacement.terrestrial_geometry import geodetic2itrf

from .schema import ConfigSchema, number, sequence, string


def _validate_station_coordinate(config: dict, path: str) -> dict:
    """Normalize either XYZ or WGS84 geodetic coordinates to XYZ."""
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
        string(
            "key",
            required=True,
            non_empty=True,
        ),
        sequence(
            "xyzM",
            item_kind="number",
            min_items=3,
            max_items=3,
            allow_none=True,
        ),
        number(
            "longitudeDeg",
            ge=-180.0,
            le=180.0,
            allow_none=True,
        ),
        number(
            "latitudeDeg",
            ge=-90.0,
            le=90.0,
            allow_none=True,
        ),
        number(
            "heightM",
            allow_none=True,
        ),
        sequence(
            "velocityMPerYear",
            default=[0.0, 0.0, 0.0],
            item_kind="number",
            min_items=3,
            max_items=3,
        ),
        string(
            "positionEpochUtc",
            default="2010-01-01T00:00:00",
            non_empty=True,
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
        string(
            "key",
            required=True,
            non_empty=True,
        ),
        sequence(
            "xyzM",
            required=True,
            item_kind="number",
            min_items=3,
            max_items=3,
        ),
    ),
    description="One reflector position in the selected ephemeris Moon PA frame.",
)


def validate_coordinate_keys(entries, location):
    seen = set()
    for entry in entries:
        key = entry["key"].strip()
        if key.casefold() in seen:
            raise ValueError(f"{location} contains duplicate key {key!r}.")
        seen.add(key.casefold())
    return entries
