#!/usr/bin/env python3
"""Plot full-station O-C residuals with and without station-bias correction."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WITH_BIAS = PROJECT_ROOT / "output" / "oc_residuals_w_libration_correction.csv"
DEFAULT_WITHOUT_BIAS = PROJECT_ROOT / "output" / "oc_residuals_w_libration_correction_wo_bias.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "station_bias_comparison"

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
class Residual:
    timestamp: datetime
    station_name: str
    oc_one_way_m: float


@dataclass(frozen=True)
class Comparison:
    timestamp: datetime
    with_bias_m: float
    without_bias_m: float


@dataclass(frozen=True)
class StationPlot:
    station_name: str
    output_name: str
    y_limit_m: float
    y_tick_step_m: float


STATION_PLOTS = (StationPlot("McDonald", "mcdonald_station_bias_comparison.png", 2.0, 0.5),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-bias",
        type=Path,
        default=DEFAULT_WITH_BIAS,
        help=f"CSV with station-bias correction (default: {DEFAULT_WITH_BIAS})",
    )
    parser.add_argument(
        "--without-bias",
        type=Path,
        default=DEFAULT_WITHOUT_BIAS,
        help=f"CSV without station-bias correction (default: {DEFAULT_WITHOUT_BIAS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for PNG files (default: {DEFAULT_OUTPUT_DIR})",
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


def load_residuals(path: Path) -> dict[tuple[str, ...], Residual]:
    residuals: dict[tuple[str, ...], Residual] = {}
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
            if row["status"].strip().lower() != "ok":
                continue
            if row["light_time_converged"].strip().lower() not in {"true", "1", "yes"}:
                continue

            timestamp = parse_timestamp(row["obs_time_utc"], path=path, row_number=row_number)
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

    return residuals


def pair_station(
    with_bias: dict[tuple[str, ...], Residual],
    without_bias: dict[tuple[str, ...], Residual],
    station_plot: StationPlot,
) -> list[Comparison]:
    comparisons: list[Comparison] = []
    for key in with_bias.keys() & without_bias.keys():
        corrected = with_bias[key]
        uncorrected = without_bias[key]
        if corrected.station_name != uncorrected.station_name:
            raise ValueError(
                f"Station mismatch for observation key {key}: "
                f"{corrected.station_name!r} != {uncorrected.station_name!r}"
            )
        if corrected.station_name != station_plot.station_name:
            continue
        comparisons.append(
            Comparison(
                timestamp=corrected.timestamp,
                with_bias_m=corrected.oc_one_way_m,
                without_bias_m=uncorrected.oc_one_way_m,
            )
        )

    comparisons.sort(key=lambda item: item.timestamp)
    if not comparisons:
        raise ValueError(f"No matched {station_plot.station_name} observations found.")
    return comparisons


def render_station(
    station_plot: StationPlot,
    comparisons: list[Comparison],
    output_path: Path,
) -> None:
    plot_data = [
        item
        for item in comparisons
        if abs(item.with_bias_m) <= station_plot.y_limit_m and abs(item.without_bias_m) <= station_plot.y_limit_m
    ]
    if not plot_data:
        raise ValueError(
            f"No matched {station_plot.station_name} observations remain within +/-{station_plot.y_limit_m:g} m."
        )

    timestamps = [item.timestamp for item in plot_data]
    with_bias_m = [item.with_bias_m for item in plot_data]
    without_bias_m = [item.without_bias_m for item in plot_data]
    plot_dates = mdates.date2num(timestamps)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(
        plot_dates,
        without_bias_m,
        s=15,
        alpha=1,
        label="Without station bias correction",
        color="k",
    )
    ax.scatter(
        plot_dates,
        with_bias_m,
        s=15,
        alpha=1,
        label="With station bias correction",
        color="red",
    )

    ax.set_title(station_plot.station_name, pad=1, fontsize=18)
    ax.set_ylabel("One-way O-C residual (m)", fontsize=18, labelpad=1)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_ylim(-station_plot.y_limit_m, station_plot.y_limit_m)
    tick_count = round(2 * station_plot.y_limit_m / station_plot.y_tick_step_m)
    ax.set_yticks([-station_plot.y_limit_m + index * station_plot.y_tick_step_m for index in range(tick_count + 1)])
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)
    ax.grid(True)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.legend(loc="upper right", fontsize=18, frameon=True)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(
        f"{station_plot.station_name}: {len(plot_data):,} observations within "
        f"+/-{station_plot.y_limit_m:g} m, "
        f"{plot_data[0].timestamp.date()} to {plot_data[-1].timestamp.date()}"
    )
    print(f"  wrote {output_path}")


def main() -> int:
    args = parse_args()
    with_bias = load_residuals(args.with_bias)
    without_bias = load_residuals(args.without_bias)

    only_with = with_bias.keys() - without_bias.keys()
    only_without = without_bias.keys() - with_bias.keys()
    if only_with or only_without:
        warnings.warn(
            "Successful CSV rows do not match exactly: "
            f"{len(only_with)} only with bias and "
            f"{len(only_without)} only without bias. "
            "Plotting their intersection.",
            RuntimeWarning,
            stacklevel=1,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for station_plot in STATION_PLOTS:
        comparisons = pair_station(with_bias, without_bias, station_plot)
        render_station(
            station_plot,
            comparisons,
            args.output_dir / station_plot.output_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
