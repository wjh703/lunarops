"""Native LunarOps Earth-orientation parameter artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lunarops.classes.frames.earth_orientation import (
    DuplicateMjdPolicy,
    EarthOrientationSample,
    TabulatedEarthOrientation,
)

from .archive import (
    atomic_text_writer,
    data_lines,
    format_float,
    open_text_reader,
    parse_float,
    parse_header,
)

ARTIFACT_TYPE = "earthOrientationParameter"
FORMAT_VERSION = 1
FIELDS = (
    "mjdUtc",
    "xpArcsec",
    "ypArcsec",
    "ut1MinusUtcSec",
    "lodSec",
    "dxArcsec",
    "dyArcsec",
)


def _ordered_samples(samples: Sequence[EarthOrientationSample]) -> tuple[EarthOrientationSample, ...]:
    if not samples:
        raise ValueError("Earth-orientation output requires at least one sample.")
    ordered = tuple(sorted(samples, key=lambda sample: sample.mjd_utc))
    mjd = [sample.mjd_utc for sample in ordered]
    if any(right <= left for left, right in zip(mjd, mjd[1:])):
        raise ValueError("Earth-orientation output requires strictly increasing MJD values.")
    return ordered


def write_earth_orientation_parameter(
    samples: Sequence[EarthOrientationSample],
    path: str | Path,
) -> Path:
    ordered = _ordered_samples(samples)
    target = Path(path).expanduser()
    with atomic_text_writer(target, ARTIFACT_TYPE, version=FORMAT_VERSION) as stream:
        stream.write(f"recordCount {len(ordered)}\n")
        stream.write("fields " + " ".join(FIELDS) + "\n")
        stream.write("data\n")
        for sample in ordered:
            stream.write(
                " ".join(
                    format_float(value)
                    for value in (
                        sample.mjd_utc,
                        sample.xp_arcsec,
                        sample.yp_arcsec,
                        sample.ut1_minus_utc_s,
                        sample.lod_s,
                        sample.dx_arcsec,
                        sample.dy_arcsec,
                    )
                )
                + "\n"
            )
    return target


def read_earth_orientation_parameter(path: str | Path) -> tuple[EarthOrientationSample, ...]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, ARTIFACT_TYPE, expected_version=FORMAT_VERSION)
        lines = iter(data_lines(stream))
        try:
            count_parts = next(lines).split()
            field_line = next(lines)
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated {ARTIFACT_TYPE} header in {source}.") from exc
        if (
            len(count_parts) != 2
            or count_parts[0] != "recordCount"
            or field_line != "fields " + " ".join(FIELDS)
            or marker != "data"
        ):
            raise ValueError(f"Malformed {ARTIFACT_TYPE} header in {source}.")
        count = int(count_parts[1])
        if count <= 0:
            raise ValueError(f"{ARTIFACT_TYPE} recordCount must be positive.")
        samples = []
        for line_number, line in enumerate(lines, start=1):
            values = line.split()
            if len(values) != len(FIELDS):
                raise ValueError(
                    f"{ARTIFACT_TYPE} row {line_number} has {len(values)} fields; expected {len(FIELDS)}."
                )
            numbers = [
                parse_float(value, field=field)
                for field, value in zip(FIELDS, values)
            ]
            samples.append(
                EarthOrientationSample(
                    mjd_utc=numbers[0],
                    xp_arcsec=numbers[1],
                    yp_arcsec=numbers[2],
                    ut1_minus_utc_s=numbers[3],
                    lod_s=numbers[4],
                    dx_arcsec=numbers[5],
                    dy_arcsec=numbers[6],
                )
            )
    if len(samples) != count:
        raise ValueError(f"{ARTIFACT_TYPE} declares {count} rows, found {len(samples)}.")
    _ordered_samples(samples)
    return tuple(samples)


def load_earth_orientation_parameter(
    path: str | Path,
    *,
    duplicate_mjd_policy: DuplicateMjdPolicy = "error",
) -> TabulatedEarthOrientation:
    source = Path(path).expanduser()
    return TabulatedEarthOrientation(
        read_earth_orientation_parameter(source),
        source_file_path=source,
        duplicate_mjd_policy=duplicate_mjd_policy,
    )


__all__ = [
    "ARTIFACT_TYPE",
    "FIELDS",
    "FORMAT_VERSION",
    "load_earth_orientation_parameter",
    "read_earth_orientation_parameter",
    "write_earth_orientation_parameter",
]
