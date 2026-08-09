"""Builtin station and reflector catalog data.

The records are domain fixtures used when a run selects ``builtin`` catalogs;
native catalog persistence remains in :mod:`lunarops.fileio.catalogs`.
"""

from __future__ import annotations

from lunarops.base.station_identity import station_aliases, station_display_name
from lunarops.classes.observation.catalogs import ReflectorRecord as _ReflectorRecord
from lunarops.classes.observation.catalogs import StationRecord as _StationRecord

# Station coordinates and velocities from the INPOP21a station table.
# Coordinates are ITRF XYZ in meters. Velocities are meters/year.
# IMPORTANT: If your source table defines a different reference epoch, change
# position_epoch_utc below. The default used here is J2000.0.
STATION_POSITION_EPOCH_UTC = "2000-01-01T00:00:00"

STATIONS = {
    "APOLLO": _StationRecord(
        name=station_display_name("APOLLO"),
        aliases=station_aliases("APOLLO"),
        itrf_xyz_m=(-1463998.9079, -5166632.7663, 3435012.8921),
        itrf_velocity_m_per_year=(-0.0139, -0.0003, -0.0023),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "GRASSE": _StationRecord(
        name=station_display_name("GRASSE"),
        aliases=station_aliases("GRASSE"),
        itrf_xyz_m=(4581692.1686, 556196.0742, 4389355.1225),
        itrf_velocity_m_per_year=(-0.0151, 0.0193, 0.0114),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "HALEAKALA": _StationRecord(
        name=station_display_name("HALEAKALA"),
        aliases=station_aliases("HALEAKALA"),
        itrf_xyz_m=(-5466003.7272, -2404425.9189, 2242197.8916),
        itrf_velocity_m_per_year=(-0.0122, 0.0622, 0.0310),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "MATERA": _StationRecord(
        name=station_display_name("MATERA"),
        aliases=station_aliases("MATERA"),
        itrf_xyz_m=(4641978.8100, 1393067.5310, 4133249.4800),
        itrf_velocity_m_per_year=(-0.0180, 0.0192, 0.0140),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "MCDONALD": _StationRecord(
        name=station_display_name("MCDONALD"),
        aliases=station_aliases("MCDONALD"),
        itrf_xyz_m=(-1330781.6134, -5328756.4702, 3235697.8262),
        itrf_velocity_m_per_year=(-0.0244, -0.0319, 0.0091),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "MLRS1": _StationRecord(
        name=station_display_name("MLRS1"),
        aliases=station_aliases("MLRS1"),
        itrf_xyz_m=(-1330121.0057, -5328532.3595, 3236146.0225),
        itrf_velocity_m_per_year=(-0.0124, 0.0009, -0.0053),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "MLRS2": _StationRecord(
        name=station_display_name("MLRS2"),
        aliases=station_aliases("MLRS2"),
        itrf_xyz_m=(-1330021.4931, -5328403.3401, 3236481.6472),
        itrf_velocity_m_per_year=(-0.0121, 0.0015, -0.0036),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
    "WETTZELL": _StationRecord(
        name=station_display_name("WETTZELL"),
        aliases=station_aliases("WETTZELL"),
        itrf_xyz_m=(4075576.7721, 931785.5248, 4801583.5601),
        itrf_velocity_m_per_year=(-0.0139, 0.0170, 0.0124),
        position_epoch_utc=STATION_POSITION_EPOCH_UTC,
    ),
}

# Lunar reflector coordinates in the INPOP21a PA frame, meters.
REFLECTORS = {
    "APOLLO11": _ReflectorRecord(
        name="Apollo 11",
        aliases=("apollo11", "Apollo11", "A11", "AP11"),
        moon_fixed_xyz_m=(1591966.6407, 690699.4669, 21003.7578),
    ),
    "LUNOKHOD1": _ReflectorRecord(
        name="Lunokhod 1",
        aliases=("lunokhod1", "Luna17", "L1", "LUNA17", "luna17"),
        moon_fixed_xyz_m=(1114292.2303, -781298.4355, 1076058.6227),
    ),
    "APOLLO14": _ReflectorRecord(
        name="Apollo 14",
        aliases=("apollo14", "Apollo14", "A14", "AP14"),
        moon_fixed_xyz_m=(1652689.5625, -520997.5929, -109730.5181),
    ),
    "APOLLO15": _ReflectorRecord(
        name="Apollo 15",
        aliases=("apollo15", "Apollo15", "A15", "AP15"),
        moon_fixed_xyz_m=(1554678.3071, 98095.5262, 765005.2077),
    ),
    "LUNOKHOD2": _ReflectorRecord(
        name="Lunokhod 2",
        aliases=("lunokhod2", "Luna21", "L2", "LUNA21", "luna21"),
        moon_fixed_xyz_m=(1339363.3937, 801871.9437, 756358.6633),
    ),
}


__all__ = ["REFLECTORS", "STATIONS"]
