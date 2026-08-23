"""Compare ERFA and LunarOps topocentric TDB-TT corrections.

Both corrections are evaluated on a TDB grid.  The LunarOps value is the
current ``v_E dot r_GCRS / c^2`` term.  The ERFA value is isolated by
subtracting a geocentric ``dtdb`` call from a call with the station's UT1 and
ITRF site parameters.

Example::

    python scripts/compare_tdb_tt_topocentric.py ../data/kernels/inpop21a \
        --eop ../data/auxiliary/eopc04.1962-now.txt --station APOLLO \
        --start-tdb-jd 2451545.0 --end-tdb-jd 2451910.0 --step-days 1 \
        --output output/tdb_tt_topocentric_ephemeris_vs_erfa.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import erfa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lunarops.base.array_validation import vector3
from lunarops.base.constants import C2
from lunarops.classes.ephemerides import Ephemeris, load_calceph_ephemeris
from lunarops.classes.frames import EarthOrientationProvider, TerrestrialFrameTransform
from lunarops.fileio.earth_orientation import load_earth_orientation_parameter
from lunarops.classes.frames.high_frequency_eop import high_frequency_eop_correction
from lunarops.classes.observation.catalogs import StationRecord, resolve_catalog_key
from lunarops.classes.time import Epoch, TimeScale, TimeScaleConverter
from lunarops.fileio.catalogs import load_station_catalog


def tdb_epochs(
    start_tdb_jd: float,
    end_tdb_jd: float,
    step_days: float,
) -> Iterable[Epoch]:
    """Yield an inclusive TDB Julian-date grid."""

    start = float(start_tdb_jd)
    end = float(end_tdb_jd)
    step = float(step_days)
    if not all(math.isfinite(value) for value in (start, end, step)):
        raise ValueError("start_tdb_jd, end_tdb_jd, and step_days must be finite.")
    if end < start:
        raise ValueError("end_tdb_jd must not precede start_tdb_jd.")
    if step <= 0.0:
        raise ValueError("step_days must be positive.")

    count = math.floor((end - start) / step)
    for index in range(count + 1):
        jd = start + index * step
        yield Epoch(2451545.0, jd - 2451545.0, TimeScale.TDB)
    last = start + count * step
    if not math.isclose(last, end, rel_tol=0.0, abs_tol=1.0e-12):
        yield Epoch(2451545.0, end - 2451545.0, TimeScale.TDB)


@dataclass(frozen=True, slots=True)
class TopocentricComparison:
    """One pure topocentric TDB-TT comparison at a TDB epoch."""

    tdb_jd: float
    utc_isot: str
    lunarops_topocentric_tdb_minus_tt_s: float
    erfa_topocentric_tdb_minus_tt_s: float
    lunarops_minus_erfa_s: float


def erfa_topocentric_tdb_minus_tt_s(
    epoch_tdb: Epoch,
    *,
    ut1_fraction_of_day: float,
    elong_rad: float,
    u_km: float,
    v_km: float,
) -> float:
    """Return only ERFA's topocentric term, removing its geocentric model."""

    epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
    geocentric = erfa.dtdb(epoch_tdb.jd1, epoch_tdb.jd2, 0.0, 0.0, 0.0, 0.0)
    topocentric = erfa.dtdb(
        epoch_tdb.jd1,
        epoch_tdb.jd2,
        float(ut1_fraction_of_day),
        float(elong_rad),
        float(u_km),
        float(v_km),
    )
    return float(topocentric - geocentric)


def site_parameters_from_itrf(itrf_xyz_m: np.ndarray) -> tuple[float, float, float]:
    """Return the ERFA longitude, spin-axis distance, and north distance."""

    x_m, y_m, z_m = vector3(itrf_xyz_m, name="itrf_xyz_m")
    return (
        float(math.atan2(y_m, x_m)),
        float(math.hypot(x_m, y_m) / 1000.0),
        float(z_m / 1000.0),
    )


def ut1_fraction_of_day(epoch_utc: Epoch, earth_orientation: EarthOrientationProvider) -> float:
    """Return the UT1 fraction measured from 0h, including high-frequency EOP."""

    epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
    background_dut1_s = earth_orientation.ut1_minus_utc_s(epoch_utc)
    high_frequency = high_frequency_eop_correction(
        epoch_utc,
        background_ut1_minus_utc_s=background_dut1_s,
    )
    ut1_jd1, ut1_jd2 = erfa.utcut1(
        epoch_utc.jd1,
        epoch_utc.jd2,
        background_dut1_s + high_frequency.delta_ut1_s,
    )
    # Julian Dates start at noon; ERFA dtdb's UT fraction starts at 0h.
    return float((math.fmod(ut1_jd1, 1.0) + math.fmod(ut1_jd2, 1.0) + 0.5) % 1.0)


def station_state_at_tdb(
    epoch_tdb: Epoch,
    *,
    converter: TimeScaleConverter,
    station: StationRecord,
    terrestrial: TerrestrialFrameTransform,
) -> tuple[Epoch, np.ndarray, np.ndarray]:
    """Resolve the fixed station's UTC, ITRF, and GCRS coordinates at TDB."""

    epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
    def observer(epoch_utc: Epoch):
        return terrestrial.tdb_topocentric_arguments(station.itrf_xyz_at(epoch_utc), epoch_utc)

    epoch_utc = converter.convert(
        epoch_tdb,
        TimeScale.UTC,
        topocentric_observer=observer,
    )
    itrf = station.itrf_xyz_at(epoch_utc)
    return epoch_utc, itrf, terrestrial.itrf2gcrs(itrf, epoch_utc)


def compare_topocentric_tdb_tt(
    ephemeris: Ephemeris,
    earth_orientation: EarthOrientationProvider,
    station: StationRecord,
    *,
    start_tdb_jd: float,
    end_tdb_jd: float,
    step_days: float,
) -> list[TopocentricComparison]:
    """Compare only the ERFA and LunarOps station-dependent corrections."""

    converter = TimeScaleConverter()
    terrestrial = TerrestrialFrameTransform(earth_orientation)
    rows: list[TopocentricComparison] = []
    for epoch_tdb in tdb_epochs(start_tdb_jd, end_tdb_jd, step_days):
        epoch_utc, itrf_m, gcrs_m = station_state_at_tdb(
            epoch_tdb,
            converter=converter,
            station=station,
            terrestrial=terrestrial,
        )
        arguments = terrestrial.tdb_topocentric_arguments(itrf_m, epoch_utc)
        lunarops_topocentric_s = float(
            np.dot(ephemeris.body_state_bcrs("EARTH", epoch_tdb).velocity_mps, gcrs_m) / C2
        )
        erfa_topocentric_s = erfa_topocentric_tdb_minus_tt_s(
            epoch_tdb,
            ut1_fraction_of_day=arguments.ut1_fraction_of_day,
            elong_rad=arguments.longitude_rad,
            u_km=arguments.distance_from_spin_axis_km,
            v_km=arguments.north_of_equatorial_plane_km,
        )
        rows.append(
            TopocentricComparison(
                tdb_jd=epoch_tdb.jd,
                utc_isot=epoch_utc.isot(),
                lunarops_topocentric_tdb_minus_tt_s=lunarops_topocentric_s,
                erfa_topocentric_tdb_minus_tt_s=erfa_topocentric_s,
                lunarops_minus_erfa_s=lunarops_topocentric_s - erfa_topocentric_s,
            )
        )
    return rows


def write_csv(rows: Iterable[TopocentricComparison], stream: TextIO) -> None:
    writer = csv.DictWriter(stream, fieldnames=list(TopocentricComparison.__dataclass_fields__))
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))


def print_summary(rows: list[TopocentricComparison], stream: TextIO) -> None:
    if not rows:
        raise ValueError("At least one comparison row is required.")
    differences = [row.lunarops_minus_erfa_s for row in rows]
    print(f"samples: {len(rows)}", file=stream)
    print(
        "LunarOps topocentric TDB-TT minus ERFA [s]: "
        f"min={min(differences):+.16e}, max={max(differences):+.16e}, "
        f"max_abs={max(abs(value) for value in differences):.16e}",
        file=stream,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ephemeris", type=Path, help="SPICE kernel directory used for Earth's BCRS velocity")
    parser.add_argument("--eop", type=Path, required=True, help="IERS C04 EOP file used for UT1 and GCRS")
    parser.add_argument("--station", required=True, help="builtin station key or alias, for example APOLLO")
    parser.add_argument("--start-tdb-jd", type=float, required=True, help="first TDB Julian Date, inclusive")
    parser.add_argument("--end-tdb-jd", type=float, required=True, help="last TDB Julian Date, inclusive")
    parser.add_argument("--step-days", type=float, default=1.0, help="TDB grid spacing in days (default: 1)")
    parser.add_argument(
        "--lunar-relativistic-scale-convention",
        default="alreadyScaled",
        choices=("alreadyScaled", "tdbCompatibleLunarSurface"),
        help="CALCEPH coordinate-scale convention (default: alreadyScaled)",
    )
    parser.add_argument("--output", type=Path, help="write detailed CSV here; omit to write to standard output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stations = load_station_catalog("builtin")
    station_key = resolve_catalog_key(args.station, stations, "Station")
    earth_orientation = load_earth_orientation_parameter(args.eop)
    with load_calceph_ephemeris(
        args.ephemeris,
        lunar_relativistic_scale_convention=args.lunar_relativistic_scale_convention,
    ) as ephemeris:
        rows = compare_topocentric_tdb_tt(
            ephemeris,
            earth_orientation,
            stations[station_key],
            start_tdb_jd=args.start_tdb_jd,
            end_tdb_jd=args.end_tdb_jd,
            step_days=args.step_days,
        )

    if args.output is None:
        write_csv(rows, sys.stdout)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            write_csv(rows, stream)
    print_summary(rows, sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
