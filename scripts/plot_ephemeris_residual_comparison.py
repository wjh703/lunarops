#!/usr/bin/env python3
"""Plot O-C residuals produced with several ephemeris kernels."""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from lunarops.fileio.observation_results import read_observation_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_PLOTS = {
    "INPOP21a": DEFAULT_OUTPUT_DIR / "oc_residuals_inpop21a.txt",
    "DE440": DEFAULT_OUTPUT_DIR / "oc_residuals_de440.txt",
    "EPM2021": DEFAULT_OUTPUT_DIR / "oc_residuals_epm2021.txt",
}
STATION_ORDER = (
    "MCDONALD",
    "MLRS1",
    "MLRS2",
    "HALEAKALA",
    "GRASSE",
    "MATERA",
    "APOLLO",
    "WETTZELL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label, default in DEFAULT_PLOTS.items():
        parser.add_argument(f"--{label.lower()}", type=Path, default=default, metavar="FILE")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ephemeris_residual_comparison" / "stations_by_name",
        help="directory for per-station PNG files",
    )
    parser.add_argument("--y-limit-cm", type=float, default=None, metavar="CM")
    return parser.parse_args()


def parse_epoch(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"invalid observation epoch {value!r}")
    epoch = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=UTC)).astimezone(UTC)


def load_series(path: Path) -> dict[str, tuple[list[datetime], np.ndarray]]:
    rows = read_observation_results(path)
    data: dict[str, tuple[list[datetime], list[float]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("light_time_converged") is not True:
            continue
        value = row.get("oc_one_way_m")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        station = row.get("station_name")
        if not isinstance(station, str) or not station:
            continue
        epochs, values = data.setdefault(station, ([], []))
        epochs.append(parse_epoch(row.get("obs_time_utc")))
        values.append(float(value) * 100.0)
    if not data:
        raise ValueError(f"{path} contains no successful finite observations")
    result: dict[str, tuple[list[datetime], np.ndarray]] = {}
    for station, (epochs, values) in data.items():
        order = np.argsort(np.asarray([epoch.timestamp() for epoch in epochs], dtype=float))
        result[station] = ([epochs[index] for index in order], np.asarray(values, dtype=float)[order])
    return result


def station_names(series_by_kernel: dict[str, dict[str, tuple[list[datetime], np.ndarray]]]) -> list[str]:
    stations = set().union(*(series.keys() for series in series_by_kernel.values()))
    ordered = [station for station in STATION_ORDER if station in stations]
    return ordered + sorted(stations - set(ordered))


def render_station(
    station: str,
    series_by_kernel: dict[str, dict[str, tuple[list[datetime], np.ndarray]]],
    output_path: Path,
    configured_y_limit: float | None,
) -> None:
    station_series = {
        label: series[station]
        for label, series in series_by_kernel.items()
        if station in series
    }
    all_values = np.concatenate([values for _, values in station_series.values()])
    y_limit = configured_y_limit
    if y_limit is None:
        y_limit = max(1.0, math.ceil(float(np.percentile(np.abs(all_values), 99.5)) / 10.0) * 10.0)
    if not math.isfinite(y_limit) or y_limit <= 0.0:
        raise ValueError("--y-limit-cm must be finite and positive")

    colors = {"INPOP21a": "#1f77b4", "DE440": "#d62728", "EPM2021": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for label, (epochs, values) in station_series.items():
        dates = mdates.date2num(epochs)
        ax.scatter(dates, values, s=2, alpha=0.35, color=colors[label], label=label, rasterized=True)
        print(
            f"{station} {label}: N={len(values):,}, mean={np.mean(values):.3f} cm, "
            f"RMS={np.sqrt(np.mean(values**2)):.3f} cm, median={np.median(values):.3f} cm"
        )
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title(f"{station} LLR O-C residuals")
    ax.set_ylabel("O-C (cm)")
    ax.set_xlabel("Observation epoch (UTC)")
    ax.set_ylim(-y_limit, y_limit)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> int:
    args = parse_args()
    paths = {"INPOP21a": args.inpop21a, "DE440": args.de440, "EPM2021": args.epm2021}
    series_by_kernel = {label: load_series(path) for label, path in paths.items()}
    for station in station_names(series_by_kernel):
        render_station(
            station,
            series_by_kernel,
            args.output_dir / f"oc_residuals_{station.lower()}.png",
            args.y_limit_cm,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
