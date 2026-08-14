#!/usr/bin/env python3
"""Plot final LLR post-fit residuals for IGG3 and direct rejection."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IGG3_REPORT = PROJECT_ROOT / "output" / "reflector_bias_processing_report_igg3.txt"
DEFAULT_DIRECT_REJECTION_REPORT = PROJECT_ROOT / "output" / "reflector_bias_processing_report_directRejection.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "robust_postfit_residuals"

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


@dataclass(frozen=True)
class ReportSpec:
    label: str
    path: Path
    output_name: str


@dataclass(frozen=True)
class PostfitResidual:
    timestamp: datetime
    station_id: str
    residual_cm: float


@dataclass(frozen=True)
class LoadedReport:
    label: str
    observations: tuple[PostfitResidual, ...]
    total_observation_count: int
    rejected_observation_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--igg3-report",
        type=Path,
        default=DEFAULT_IGG3_REPORT,
        help=f"IGG3 adjustment report (default: {DEFAULT_IGG3_REPORT})",
    )
    parser.add_argument(
        "--direct-rejection-report",
        type=Path,
        default=DEFAULT_DIRECT_REJECTION_REPORT,
        help=(f"direct-rejection adjustment report (default: {DEFAULT_DIRECT_REJECTION_REPORT})"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for PNG files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="also plot zero robust-weight observations",
    )
    parser.add_argument(
        "--y-limit-cm",
        type=float,
        default=None,
        metavar="CM",
        help=("symmetric y-axis limit in cm; by default a common limit is computed from both plotted data sets"),
    )
    return parser.parse_args()


def parse_timestamp(value: object, *, path: Path, row_number: int) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{path}: observation {row_number} has a non-string epoch.")
    try:
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path}: observation {row_number} has invalid epoch {value!r}.") from exc
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def finite_float(
    value: object,
    *,
    path: Path,
    row_number: int,
    field: str,
) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: observation {row_number} has invalid {field} value {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: observation {row_number} has non-finite {field} value {value!r}.")
    return result


def robust_factor(record: Mapping[str, object], *, path: Path, row_number: int) -> float:
    for field in ("applied_robust_factor", "applied_igg3_factor"):
        if field in record:
            factor = finite_float(record[field], path=path, row_number=row_number, field=field)
            if not 0.0 <= factor <= 1.0:
                raise ValueError(f"{path}: observation {row_number} has {field} outside [0, 1].")
            return factor
    raise ValueError(f"{path}: observation {row_number} is missing an applied robust factor.")


def scalar_text(value: str) -> str:
    """Decode the simple scalar forms emitted in an adjustment report."""
    result = value.strip()
    if len(result) >= 2 and result[0] == result[-1] and result[0] in "\"'":
        return result[1:-1]
    return result


def iter_observation_records(path: Path):
    """Stream the final top-level observations list from a report.

    Reports contain large nonlinear-iteration histories. Loading all YAML into a
    dictionary is unnecessarily memory intensive when only four observation
    fields are needed for this plot.
    """
    fields = {
        "epoch",
        "station_id",
        "current_state_residual_m",
        "applied_robust_factor",
        "applied_igg3_factor",
    }
    in_observations = False
    record: dict[str, object] | None = None
    record_start = 0

    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Cannot open adjustment report {path}: {exc}") from exc

    with stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.rstrip("\n")
            if not in_observations:
                if stripped == "observations:":
                    in_observations = True
                continue

            if stripped.startswith("- observation_id:"):
                if record is not None:
                    yield record_start, record
                record = {}
                record_start = line_number
                continue

            if record is None:
                continue
            if stripped and not stripped.startswith(" "):
                break
            if not stripped.startswith("  "):
                continue
            key, separator, value = stripped[2:].partition(":")
            if separator and key in fields:
                record[key] = scalar_text(value)

    if not in_observations:
        raise ValueError(f"{path} does not contain a top-level observations section.")
    if record is not None:
        yield record_start, record


def load_postfit_residuals(spec: ReportSpec, *, include_rejected: bool) -> LoadedReport:
    observations: list[PostfitResidual] = []
    rejected_count = 0
    total_observation_count = 0
    for row_number, raw_record in iter_observation_records(spec.path):
        total_observation_count += 1
        station_id = raw_record.get("station_id")
        if not isinstance(station_id, str) or not station_id.strip():
            raise ValueError(f"{spec.path}: observation {row_number} has invalid station_id.")
        factor = robust_factor(raw_record, path=spec.path, row_number=row_number)
        if factor == 0.0:
            rejected_count += 1
            if not include_rejected:
                continue
        residual_m = finite_float(
            raw_record.get("current_state_residual_m"),
            path=spec.path,
            row_number=row_number,
            field="current_state_residual_m",
        )
        observations.append(
            PostfitResidual(
                timestamp=parse_timestamp(raw_record.get("epoch"), path=spec.path, row_number=row_number),
                station_id=station_id.strip(),
                residual_cm=residual_m * 100.0,
            )
        )

    if not observations:
        raise ValueError(f"{spec.path} has no observations selected for plotting.")
    return LoadedReport(
        label=spec.label,
        observations=tuple(observations),
        total_observation_count=total_observation_count,
        rejected_observation_count=rejected_count,
    )


def nice_positive(value: float) -> float:
    """Round a positive axis value up to a 1/2/5 x 10**n step."""
    if value <= 0.0 or not math.isfinite(value):
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(value))
    scaled = value / magnitude
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        if scaled <= multiplier:
            return multiplier * magnitude
    raise AssertionError("unreachable")


def resolve_y_limit_cm(reports: Sequence[LoadedReport], configured_limit_cm: float | None) -> float:
    if configured_limit_cm is not None:
        if not math.isfinite(configured_limit_cm) or configured_limit_cm <= 0.0:
            raise ValueError("--y-limit-cm must be finite and positive.")
        return configured_limit_cm
    maximum = max(abs(observation.residual_cm) for report in reports for observation in report.observations)
    return nice_positive(maximum * 1.05)


def station_colors(reports: Sequence[LoadedReport]) -> dict[str, object]:
    stations = {observation.station_id for report in reports for observation in report.observations}
    ordered = [station for station in STATION_ORDER if station in stations]
    ordered.extend(sorted(stations - set(ordered)))
    colormap = plt.get_cmap("tab10")
    palette = [colormap(index) for index in range(colormap.N)]
    return {station: palette[index % len(palette)] for index, station in enumerate(ordered)}


def render_report(
    report: LoadedReport,
    *,
    output_path: Path,
    colors: Mapping[str, object],
    y_limit_cm: float,
    include_rejected: bool,
) -> None:
    by_station: dict[str, list[PostfitResidual]] = defaultdict(list)
    for observation in report.observations:
        by_station[observation.station_id].append(observation)

    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    for station_id in colors:
        observations = by_station.get(station_id)
        if not observations:
            continue
        observations.sort(key=lambda item: item.timestamp)
        plot_dates = mdates.date2num([item.timestamp for item in observations])
        ax.scatter(
            plot_dates,
            [item.residual_cm for item in observations],
            s=10,
            alpha=1,
            color=colors[station_id],
            edgecolors="none",
            label=station_id,
            rasterized=True,
        )

    selected_label = "all retained observations" if include_rejected else "nonzero robust-weight observations"
    ax.set_title(f"{report.label} LLR post-fit residuals", fontsize=14, pad=1)
    # ax.set_xlabel("Epoch (UTC)", fontsize=15)
    ax.set_ylabel("LLR post-fit residual (cm)", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(-y_limit_cm, y_limit_cm)
    ax.yaxis.set_major_locator(MultipleLocator(nice_positive(y_limit_cm / 4.0)))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axhline(0.0, color="0.25", linewidth=0.8, zorder=0)
    ax.margins(x=0.01)
    ax.grid(True, color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, fontsize=10, frameon=True)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    outside = sum(abs(item.residual_cm) > y_limit_cm for item in report.observations)
    print(
        f"{report.label}: {len(report.observations):,}/{report.total_observation_count:,} "
        f"{selected_label}; {outside:,} outside +/-{y_limit_cm:g} cm"
    )
    print(f"  wrote {output_path}")


def main() -> int:
    args = parse_args()
    specs = (
        ReportSpec("IGG3", args.igg3_report, "llr_postfit_residuals_igg3.png"),
        ReportSpec(
            "Direct rejection",
            args.direct_rejection_report,
            "llr_postfit_residuals_direct_rejection.png",
        ),
    )
    reports = tuple(load_postfit_residuals(spec, include_rejected=args.include_rejected) for spec in specs)
    y_limit_cm = resolve_y_limit_cm(reports, args.y_limit_cm)
    colors = station_colors(reports)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec, report in zip(specs, reports, strict=True):
        render_report(
            report,
            output_path=args.output_dir / spec.output_name,
            colors=colors,
            y_limit_cm=y_limit_cm,
            include_rejected=args.include_rejected,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
