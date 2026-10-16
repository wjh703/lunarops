#!/usr/bin/env python3
"""Compare O-C residuals with and without the longitude-libration correction."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WITH_CORRECTION = PROJECT_ROOT / "output" / "oc_residuals_w_libration_correction.csv"
DEFAULT_WITHOUT_CORRECTION = PROJECT_ROOT / "output" / "oc_residuals_wo_libration_correction.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "libration_correction_comparison"

KEY_FIELDS = (
    "normal_point_index",
    "obs_time_utc",
    "station_id",
    "reflector_id",
)
REQUIRED_FIELDS = set(KEY_FIELDS) | {
    "station_name",
    "oc_one_way_m",
    "light_time_converged",
    "status",
}


@dataclass(frozen=True)
class Station:
    csv_name: str
    display_name: str
    output_name: str


@dataclass(frozen=True)
class Residual:
    timestamp: datetime
    station_name: str
    oc_one_way_m: float


@dataclass(frozen=True)
class Comparison:
    timestamp: datetime
    with_correction_m: float
    without_correction_m: float


STATIONS = (
    Station("APOL", "Apache", "apache_libration_correction_comparison.png"),
    Station("Grasse", "Grasse", "grasse_libration_correction_comparison.png"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one-way O-C residuals with and without the INPOP21a "
            "longitude-libration correction for Apache Point and Grasse."
        )
    )
    parser.add_argument(
        "--with-correction",
        type=Path,
        default=DEFAULT_WITH_CORRECTION,
        help=f"CSV with correction (default: {DEFAULT_WITH_CORRECTION})",
    )
    parser.add_argument(
        "--without-correction",
        type=Path,
        default=DEFAULT_WITHOUT_CORRECTION,
        help=f"CSV without correction (default: {DEFAULT_WITHOUT_CORRECTION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for PNG files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date(2014, 1, 1),
        metavar="YYYY-MM-DD",
        help="first observation date, inclusive (default: 2014-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date(2021, 12, 31),
        metavar="YYYY-MM-DD",
        help="last observation date, inclusive (default: 2021-12-31)",
    )
    return parser.parse_args()


def parse_timestamp(value: str, *, path: Path, row_number: int) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path}:{row_number}: invalid obs_time_utc value {value!r}") from exc
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def load_residuals(
    path: Path,
    *,
    start_date: date,
    end_date: date,
) -> tuple[dict[tuple[str, ...], Residual], int]:
    residuals: dict[tuple[str, ...], Residual] = {}
    skipped_unsuccessful = 0
    station_names = {station.csv_name for station in STATIONS}

    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise OSError(f"Cannot open result file {path}: {exc}") from exc

    with stream:
        reader = csv.DictReader(stream)
        missing_fields = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(f"{path} is missing required CSV fields: {', '.join(sorted(missing_fields))}")

        for row_number, row in enumerate(reader, start=2):
            timestamp = parse_timestamp(row["obs_time_utc"], path=path, row_number=row_number)
            if not start_date <= timestamp.date() <= end_date:
                continue
            if row["station_name"] not in station_names:
                continue
            if row["status"].strip().lower() != "ok" or row["light_time_converged"].strip().lower() not in {
                "true",
                "1",
                "yes",
            }:
                skipped_unsuccessful += 1
                continue

            try:
                residual_m = float(row["oc_one_way_m"])
            except ValueError as exc:
                raise ValueError(f"{path}:{row_number}: invalid oc_one_way_m value {row['oc_one_way_m']!r}") from exc
            if not math.isfinite(residual_m):
                raise ValueError(f"{path}:{row_number}: non-finite oc_one_way_m value {row['oc_one_way_m']!r}")

            key = tuple(row[field] for field in KEY_FIELDS)
            if key in residuals:
                raise ValueError(f"{path}:{row_number}: duplicate observation key {key}")
            residuals[key] = Residual(
                timestamp=timestamp,
                station_name=row["station_name"],
                oc_one_way_m=residual_m,
            )

    return residuals, skipped_unsuccessful


def pair_residuals(
    with_correction: dict[tuple[str, ...], Residual],
    without_correction: dict[tuple[str, ...], Residual],
    *,
    strict: bool = False,
) -> dict[str, list[Comparison]]:
    """Pair observations successful in both result files."""
    only_with = with_correction.keys() - without_correction.keys()
    only_without = without_correction.keys() - with_correction.keys()
    if only_with or only_without:
        message = (
            "The selected CSV rows do not match: "
            f"{len(only_with)} only in the corrected result and "
            f"{len(only_without)} only in the uncorrected result."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(
            f"{message} Comparing only observations present in both results.",
            RuntimeWarning,
            stacklevel=2,
        )

    paired: dict[str, list[Comparison]] = {station.csv_name: [] for station in STATIONS}
    for key in with_correction.keys() & without_correction.keys():
        corrected = with_correction[key]
        uncorrected = without_correction[key]
        if corrected.station_name != uncorrected.station_name:
            raise ValueError(
                f"Station mismatch for observation key {key}: "
                f"{corrected.station_name!r} != {uncorrected.station_name!r}"
            )
        paired[corrected.station_name].append(
            Comparison(
                timestamp=corrected.timestamp,
                with_correction_m=corrected.oc_one_way_m,
                without_correction_m=uncorrected.oc_one_way_m,
            )
        )

    for station in STATIONS:
        paired[station.csv_name].sort(key=lambda item: item.timestamp)
        if not paired[station.csv_name]:
            raise ValueError(f"No {station.csv_name!r} observations found in the selected date range.")
    return paired


def render_station_plot(
    station: Station,
    comparisons: list[Comparison],
    *,
    output_path: Path,
) -> None:
    plot_data: list[tuple[datetime, float, float]] = []
    for item in comparisons:
        corrected_cm = item.with_correction_m * 100.0
        uncorrected_cm = item.without_correction_m * 100.0
        if abs(corrected_cm) <= 12.0 and abs(uncorrected_cm) <= 12.0:
            plot_data.append((item.timestamp, corrected_cm, uncorrected_cm))

    if not plot_data:
        raise ValueError(f"No {station.csv_name!r} paired observations remain within +/-12 cm.")

    timestamps = [item[0] for item in plot_data]
    corrected_values_cm = [item[1] for item in plot_data]
    uncorrected_values_cm = [item[2] for item in plot_data]
    corrected_rms_cm = math.sqrt(sum(value * value for value in corrected_values_cm) / len(corrected_values_cm))
    uncorrected_rms_cm = math.sqrt(sum(value * value for value in uncorrected_values_cm) / len(uncorrected_values_cm))
    plot_dates = mdates.date2num(timestamps)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(
        plot_dates,
        uncorrected_values_cm,
        s=15,
        alpha=1,
        label="Without libration correction",
        color="k",
    )
    ax.scatter(plot_dates, corrected_values_cm, s=15, alpha=1, label="With libration correction", color="red")

    ax.set_title(station.display_name, pad=1, fontsize=18)
    ax.set_ylabel("One-way O-C residual (cm)", fontsize=18, labelpad=1)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_ylim(-12, 12)
    ax.set_yticks(range(-12, 13, 4))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)
    ax.grid(True)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.legend(loc="upper right", fontsize=18, frameon=True)
    # ax.text(
    #     0.02,
    #     0.96,
    #     f"With libration correction RMS = {corrected_rms_cm:.2f} cm\n"
    #     f"Without libration correction RMS = {uncorrected_rms_cm:.2f} cm",
    #     transform=ax.transAxes,
    #     va="top",
    #     fontsize=12,
    # )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(
        f"{station.display_name}: {len(plot_data):,} paired observations within "
        f"+/-12 cm, with RMS {corrected_rms_cm:.3f} cm, "
        f"without RMS {uncorrected_rms_cm:.3f} cm"
    )
    print(f"  wrote {output_path}")


def main() -> int:
    args = parse_args()
    if args.start_date > args.end_date:
        raise ValueError("--start-date must not be later than --end-date.")

    corrected, corrected_skipped = load_residuals(
        args.with_correction,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    uncorrected, uncorrected_skipped = load_residuals(
        args.without_correction,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    paired = pair_residuals(corrected, uncorrected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if corrected_skipped or uncorrected_skipped:
        print(
            "Skipped unsuccessful target rows: "
            f"{corrected_skipped} with correction, "
            f"{uncorrected_skipped} without correction"
        )

    for station in STATIONS:
        render_station_plot(
            station,
            paired[station.csv_name],
            output_path=args.output_dir / station.output_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
