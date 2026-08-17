"""Compare an ephemeris TDB-TT table with ERFA's analytic model.

The comparison grid is explicitly TDB.  ERFA ``dtdb`` and the ephemeris
target-16 table both take TDB as their independent variable.  For the
diagnostic that starts from TT, this script iterates the implicit relation
``TDB = TT + (TDB-TT)(TDB)``.

Example::

    python scripts/compare_tdb_tt.py ../data/kernels/inpop21a_TDB_m100_p100_tt.dat \
        --start-tdb-jd 2451545.0 --end-tdb-jd 2451910.0 --step-days 1 \
        --output output/tdb_tt_ephemeris_vs_erfa.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TextIO

import erfa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lunarops.classes.ephemerides import load_calceph_ephemeris
from lunarops.classes.time import Epoch, TimeScale

_MAX_TDB_ITERATIONS = 6
_TDB_TT_TOLERANCE_S = 1.0e-12


@dataclass(frozen=True, slots=True)
class TdbTtComparison:
    """One ephemeris/ERFA comparison evaluated at a TDB epoch."""

    tdb_jd: float
    ephemeris_tdb_minus_tt_s: float
    erfa_tdb_minus_tt_s: float
    ephemeris_minus_erfa_s: float
    ephemeris_tt2tdb_minus_input_s: float
    erfa_tt2tdb_minus_input_s: float
    erfa_tt2tdb_minus_ephemeris_tt2tdb_s: float


class Target16Source(Protocol):
    """Minimal diagnostic interface for a CALCEPH target-16 table."""

    def target16_tdb_minus_tt_s(self, epoch_tdb: Epoch) -> float: ...


def erfa_tdb_minus_tt_s(epoch_tdb: Epoch) -> float:
    """Return geocentric TDB-TT from ERFA with TDB as the argument."""

    epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
    # Zero site coordinates suppress the Moyer/Murray topocentric contribution.
    return float(erfa.dtdb(epoch_tdb.jd1, epoch_tdb.jd2, 0.0, 0.0, 0.0, 0.0))


def tt2tdb_erfa(
    epoch_tt: Epoch,
    *,
    max_iterations: int = _MAX_TDB_ITERATIONS,
    tolerance_s: float = _TDB_TT_TOLERANCE_S,
) -> Epoch:
    """Invert ERFA's TDB-TT relation for a TT input by fixed-point iteration."""

    epoch_tt.require_scale(TimeScale.TT, name="epoch_tt")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one.")
    if not math.isfinite(tolerance_s) or tolerance_s <= 0.0:
        raise ValueError("tolerance_s must be positive and finite.")

    current = Epoch(epoch_tt.jd1, epoch_tt.jd2, TimeScale.TDB)
    for _ in range(max_iterations):
        delta_s = erfa_tdb_minus_tt_s(current)
        shifted = epoch_tt.shifted(delta_s)
        updated = Epoch(shifted.jd1, shifted.jd2, TimeScale.TDB)
        if abs(current.seconds_until(updated)) < tolerance_s:
            return updated
        current = updated
    return current


def tt2tdb_target16(
    epoch_tt: Epoch,
    source: Target16Source,
    *,
    max_iterations: int = _MAX_TDB_ITERATIONS,
    tolerance_s: float = _TDB_TT_TOLERANCE_S,
) -> Epoch:
    """Invert target 16 without using LunarOps production time conversion."""

    epoch_tt.require_scale(TimeScale.TT, name="epoch_tt")
    current = Epoch(epoch_tt.jd1, epoch_tt.jd2, TimeScale.TDB)
    for _ in range(max_iterations):
        shifted = epoch_tt.shifted(source.target16_tdb_minus_tt_s(current))
        updated = Epoch(shifted.jd1, shifted.jd2, TimeScale.TDB)
        if abs(current.seconds_until(updated)) < tolerance_s:
            return updated
        current = updated
    return current


def tdb_epochs(
    start_tdb_jd: float,
    end_tdb_jd: float,
    step_days: float,
) -> Iterable[Epoch]:
    """Yield an inclusive TDB Julian-date grid using J2000 two-part dates."""

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
        yield _tdb_epoch_from_jd(start + index * step)
    last = start + count * step
    if not math.isclose(last, end, rel_tol=0.0, abs_tol=1.0e-12):
        yield _tdb_epoch_from_jd(end)


def compare_tdb_tt(
    ephemeris: Target16Source,
    *,
    start_tdb_jd: float,
    end_tdb_jd: float,
    step_days: float,
) -> list[TdbTtComparison]:
    """Evaluate target 16 and ERFA at the same TDB epochs."""

    rows: list[TdbTtComparison] = []
    for epoch_tdb in tdb_epochs(start_tdb_jd, end_tdb_jd, step_days):
        ephemeris_offset_s = ephemeris.target16_tdb_minus_tt_s(epoch_tdb)
        erfa_offset_s = erfa_tdb_minus_tt_s(epoch_tdb)

        # This TT is defined by the ephemeris relation at the sampled TDB point.
        tt_shifted = epoch_tdb.shifted(-ephemeris_offset_s)
        epoch_tt = Epoch(tt_shifted.jd1, tt_shifted.jd2, TimeScale.TT)
        ephemeris_tdb = tt2tdb_target16(epoch_tt, ephemeris)
        erfa_tdb = tt2tdb_erfa(epoch_tt)

        rows.append(
            TdbTtComparison(
                tdb_jd=epoch_tdb.jd,
                ephemeris_tdb_minus_tt_s=ephemeris_offset_s,
                erfa_tdb_minus_tt_s=erfa_offset_s,
                ephemeris_minus_erfa_s=ephemeris_offset_s - erfa_offset_s,
                ephemeris_tt2tdb_minus_input_s=epoch_tdb.seconds_until(ephemeris_tdb),
                erfa_tt2tdb_minus_input_s=epoch_tdb.seconds_until(erfa_tdb),
                erfa_tt2tdb_minus_ephemeris_tt2tdb_s=ephemeris_tdb.seconds_until(erfa_tdb),
            )
        )
    return rows


def write_csv(rows: Iterable[TdbTtComparison], stream: TextIO) -> None:
    writer = csv.DictWriter(stream, fieldnames=list(TdbTtComparison.__dataclass_fields__))
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))


def print_summary(rows: list[TdbTtComparison], stream: TextIO) -> None:
    if not rows:
        raise ValueError("At least one comparison row is required.")
    differences = [row.ephemeris_minus_erfa_s for row in rows]
    inversion_differences = [row.erfa_tt2tdb_minus_ephemeris_tt2tdb_s for row in rows]
    print(f"samples: {len(rows)}", file=stream)
    print(
        "ephemeris TDB-TT minus ERFA [s]: "
        f"min={min(differences):+.16e}, max={max(differences):+.16e}, "
        f"max_abs={max(abs(value) for value in differences):.16e}",
        file=stream,
    )
    print(
        "ERFA TT->TDB minus ephemeris TT->TDB [s]: "
        f"max_abs={max(abs(value) for value in inversion_differences):.16e}",
        file=stream,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ephemeris", type=Path, help="CALCEPH ephemeris containing target 16 (TT-TDB).")
    parser.add_argument("--start-tdb-jd", type=float, required=True, help="first TDB Julian Date, inclusive")
    parser.add_argument("--end-tdb-jd", type=float, required=True, help="last TDB Julian Date, inclusive")
    parser.add_argument("--step-days", type=float, default=1.0, help="TDB grid spacing in days (default: 1)")
    parser.add_argument(
        "--lunar-relativistic-scale-convention",
        default="alreadyScaled",
        choices=("alreadyScaled", "tdbCompatibleLunarSurface"),
        help="CALCEPH coordinate-scale convention (default: alreadyScaled)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write detailed CSV here; omit to write CSV to standard output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with load_calceph_ephemeris(
        args.ephemeris,
        lunar_relativistic_scale_convention=args.lunar_relativistic_scale_convention,
    ) as ephemeris:
        rows = compare_tdb_tt(
            ephemeris,
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


def _tdb_epoch_from_jd(jd: float) -> Epoch:
    return Epoch(2451545.0, jd - 2451545.0, TimeScale.TDB)


if __name__ == "__main__":
    raise SystemExit(main())
