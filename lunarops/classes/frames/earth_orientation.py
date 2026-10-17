"""Explicit Earth-orientation data sources used by ERFA frame transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeGuard, cast, runtime_checkable

import erfa
import numpy as np

from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.classes.time import Epoch, TimeScale


@dataclass(frozen=True, slots=True)
class PolarMotion:
    xp_arcsec: float
    yp_arcsec: float


@dataclass(frozen=True, slots=True)
class CelestialPoleOffsets:
    dx_arcsec: float
    dy_arcsec: float


@dataclass(frozen=True, slots=True)
class EarthOrientationSample:
    mjd_utc: float
    xp_arcsec: float
    yp_arcsec: float
    ut1_minus_utc_s: float
    dx_arcsec: float = 0.0
    dy_arcsec: float = 0.0
    lod_s: float = 0.0
    source: str = "unspecified"
    quality: str = "unspecified"
    source_sha256: str = "unspecified"

    def __post_init__(self) -> None:
        for name in (
            "mjd_utc",
            "xp_arcsec",
            "yp_arcsec",
            "ut1_minus_utc_s",
            "dx_arcsec",
            "dy_arcsec",
            "lod_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)


@runtime_checkable
class EarthOrientationProvider(Protocol):
    """Typed access to the Earth-orientation quantities used by LunarOps."""

    @property
    def source_file_path(self) -> Path | None: ...

    def polar_motion(self, epoch_utc: Epoch) -> PolarMotion: ...

    def celestial_pole_offsets(self, epoch_utc: Epoch) -> CelestialPoleOffsets:
        return CelestialPoleOffsets(0.0, 0.0)

    def ut1_minus_utc_s(self, epoch_utc: Epoch) -> float: ...


class TabulatedEarthOrientation(EarthOrientationProvider):
    """Linearly interpolated Earth-orientation table."""

    _mjd: np.ndarray
    _xp_arcsec: np.ndarray
    _yp_arcsec: np.ndarray
    _ut1_minus_tai_sec: np.ndarray
    _dx_arcsec: np.ndarray
    _dy_arcsec: np.ndarray
    _lod_s: np.ndarray


    __slots__ = (
        "_dx_arcsec",
        "_dy_arcsec",
        "_lod_s",
        "_mjd",
        "_provenance",
        "_source_file_path",
        "_ut1_minus_tai_sec",
        "_xp_arcsec",
        "_yp_arcsec",
    )

    def __init__(
        self,
        samples: Sequence[EarthOrientationSample],
        *,
        source_file_path: str | Path | None = None,
    ) -> None:
        if not samples:
            raise ValueError("EarthOrientationProvider requires at least one EOP sample.")
        ordered = sorted(samples, key=lambda item: item.mjd_utc)
        if len({sample.mjd_utc for sample in ordered}) != len(ordered):
            raise ValueError("EOP table contains duplicate MJD values.")
        self._set_columns(
            *(
                np.array([getattr(sample, name) for sample in ordered], dtype=float)
                for name in ("mjd_utc", "xp_arcsec", "yp_arcsec", "ut1_minus_utc_s", "dx_arcsec", "dy_arcsec", "lod_s")
            ),
            source_file_path=source_file_path,
            provenance=tuple((sample.source, sample.quality, sample.source_sha256) for sample in ordered),
        )

    def _set_columns(
        self,
        mjd_utc: np.ndarray,
        xp_arcsec: np.ndarray,
        yp_arcsec: np.ndarray,
        ut1_minus_utc_s: np.ndarray,
        dx_arcsec: np.ndarray,
        dy_arcsec: np.ndarray,
        lod_s: np.ndarray,
        *,
        source_file_path: str | Path | None,
        provenance: tuple[tuple[str, str, str], ...],
    ) -> None:
        columns = (
            mjd_utc,
            xp_arcsec,
            yp_arcsec,
            ut1_minus_utc_s - self._tai_minus_utc_at_mjd(mjd_utc),
            dx_arcsec,
            dy_arcsec,
            lod_s,
        )
        names = ("_mjd", "_xp_arcsec", "_yp_arcsec", "_ut1_minus_tai_sec", "_dx_arcsec", "_dy_arcsec", "_lod_s")
        for name, values in zip(names, columns, strict=True):
            column = np.array(values, dtype=float, copy=True, order="C")
            column.setflags(write=False)
            setattr(self, name, column)
        self._source_file_path = Path(source_file_path).expanduser() if source_file_path else None
        self._provenance = provenance

    def to_mpi_payload(self) -> dict[str, object]:
        """Return the compact, picklable columns broadcast to worker ranks."""
        return {
            "kind": "earthOrientationArrays",
            "provenance": self._provenance,
            "sourceFile": None if self.source_file_path is None else str(self.source_file_path),
            "mjdUtc": self._mjd,
            "xpArcsec": self._xp_arcsec,
            "ypArcsec": self._yp_arcsec,
            "ut1MinusUtcSec": self._ut1_minus_tai_sec + self._tai_minus_utc_at_mjd(self._mjd),
            "dxArcsec": self._dx_arcsec,
            "dyArcsec": self._dy_arcsec,
            "lodSec": self._lod_s,
        }

    @classmethod
    def from_mpi_payload(
        cls,
        payload: Mapping[str, object],
    ) -> TabulatedEarthOrientation:
        """Rebuild the table from :meth:`to_mpi_payload` without reparsing samples."""
        result = cls.__new__(cls)
        result._set_columns(
            *(
                np.asarray(payload[key], dtype=float)
                for key in ("mjdUtc", "xpArcsec", "ypArcsec", "ut1MinusUtcSec", "dxArcsec", "dyArcsec", "lodSec")
            ),
            source_file_path=cast(str | None, payload["sourceFile"]),
            provenance=tuple((row[0], row[1], row[2]) for row in cast(Sequence[Sequence[str]], payload["provenance"])),
        )
        return result

    @property
    def source_file_path(self) -> Path | None:
        return self._source_file_path

    @property
    def mjd_utc_range(self) -> tuple[float, float]:
        return float(self._mjd[0]), float(self._mjd[-1])

    @property
    def samples(self) -> tuple[EarthOrientationSample, ...]:
        return tuple(
            replace(
                EarthOrientationSample(float(mjd), float(xp), float(yp), float(dut1), float(dx), float(dy), float(lod)),
                source=provenance[0],
                quality=provenance[1],
                source_sha256=provenance[2],
            )
            for mjd, xp, yp, dut1, dx, dy, lod, provenance in zip(
                self._mjd,
                self._xp_arcsec,
                self._yp_arcsec,
                self._ut1_minus_tai_sec + self._tai_minus_utc_at_mjd(self._mjd),
                self._dx_arcsec,
                self._dy_arcsec,
                self._lod_s,
                self._provenance,
            )
        )

    @staticmethod
    def _tai_minus_utc_at_mjd(mjd_utc) -> np.ndarray:
        values = np.asarray(mjd_utc, dtype=float)
        year, month, day, fraction = erfa.jd2cal(2_400_000.5, values)
        return np.asarray(erfa.dat(year, month, day, fraction), dtype=float)

    @staticmethod
    def _tai_minus_utc_at_epoch(epoch_utc: Epoch) -> float:
        year, month, day, fields = erfa.d2dtf("UTC", 9, epoch_utc.jd1, epoch_utc.jd2)
        seconds = fields["h"] * 3600.0 + fields["m"] * 60.0 + fields["s"] + fields["f"] * 1.0e-9
        fraction = min(seconds / SECONDS_PER_DAY, np.nextafter(1.0, 0.0))
        return float(erfa.dat(year, month, day, fraction))

    @staticmethod
    def _epoch_to_mjd_utc(epoch_utc: Epoch) -> float:
        if not isinstance(epoch_utc, Epoch):
            raise TypeError("Earth-orientation queries require an Epoch.")
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        return float(epoch_utc.mjd)

    def _interpolate_column(
        self,
        column_values: np.ndarray,
        epoch_utc: Epoch,
        *,
        column_name: str,
    ) -> float:
        mjd = self._epoch_to_mjd_utc(epoch_utc)
        start, end = self.mjd_utc_range
        if mjd < start or mjd > end:
            raise ValueError(
                f"EOP {column_name} interpolation requested MJD {mjd:.6f}, outside "
                f"loaded range [{start:.6f}, {end:.6f}]."
            )
        return float(np.interp(mjd, self._mjd, column_values))

    def polar_motion(self, epoch_utc: Epoch) -> PolarMotion:
        return PolarMotion(
            xp_arcsec=self._interpolate_column(self._xp_arcsec, epoch_utc, column_name="xp"),
            yp_arcsec=self._interpolate_column(self._yp_arcsec, epoch_utc, column_name="yp"),
        )

    def ut1_minus_utc_s(self, epoch_utc: Epoch) -> float:
        ut1_minus_tai = self._interpolate_column(
            self._ut1_minus_tai_sec,
            epoch_utc,
            column_name="UT1-TAI",
        )
        return ut1_minus_tai + self._tai_minus_utc_at_epoch(epoch_utc)

    def celestial_pole_offsets(self, epoch_utc: Epoch) -> CelestialPoleOffsets:
        return CelestialPoleOffsets(
            dx_arcsec=self._interpolate_column(self._dx_arcsec, epoch_utc, column_name="dX"),
            dy_arcsec=self._interpolate_column(self._dy_arcsec, epoch_utc, column_name="dY"),
        )


def _is_int_token(value: str) -> bool:
    return value.lstrip("+-").isdigit()


def _is_mjd(value: float | None) -> TypeGuard[float]:
    return value is not None and 15_000.0 < value < 90_000.0


def _sample_if_plausible(
    mjd: float,
    xp: float,
    yp: float,
    dut1: float,
    dx: float,
    dy: float,
    lod: float,
) -> EarthOrientationSample | None:
    # Polar motion is in arcseconds and UT1-UTC is in seconds.  These generous
    # bounds reject obvious mis-parses such as choosing x-error as y-pole while
    # still covering historical and prediction rows.
    if abs(xp) > 5.0 or abs(yp) > 5.0 or abs(dut1) > 5.0 or abs(dx) > 5.0 or abs(dy) > 5.0 or abs(lod) > 5.0:
        return None
    return EarthOrientationSample(mjd, xp, yp, dut1, dx, dy, lod)


def _parse_c04_line(line: str) -> EarthOrientationSample | None:
    """Explicit C04 14 (16 columns) and C04 20 (21 columns) layouts."""
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "%")):
        return None
    parts = stripped.split()
    if len(parts) == 16:
        date_count, mjd_index, xp_index, lod_index, dx_index = 3, 3, 4, 7, 8
        source = "C04-14"
    elif len(parts) == 21:
        date_count, mjd_index, xp_index, lod_index, dx_index = 4, 4, 5, 12, 8
        source = "C04-20"
    else:
        raise ValueError(f"C04 requires 16 (14 C04) or 21 (20 C04) columns, found {len(parts)}")
    if not all(_is_int_token(parts[i]) for i in range(date_count)):
        raise ValueError("C04 calendar fields must be integers")
    year, month, day = map(int, parts[:3])
    hour = int(parts[3]) if date_count == 4 else 0
    if not 0 <= hour <= 23:
        raise ValueError("C04 hour is outside [0, 23]")
    epoch = Epoch.from_calendar(year, month, day, hour, scale=TimeScale.UTC)
    values = [float(part) for part in parts[date_count:]]
    if not np.all(np.isfinite(values)):
        raise ValueError("C04 numeric fields must be finite")
    mjd = float(parts[mjd_index])
    if abs(epoch.mjd - mjd) > 0.0051:
        raise ValueError("C04 calendar date and MJD disagree")
    return EarthOrientationSample(
        mjd,
        float(parts[xp_index]),
        float(parts[xp_index + 1]),
        float(parts[xp_index + 2]),
        float(parts[dx_index]),
        float(parts[dx_index + 1]),
        float(parts[lod_index]),
        source,
        "final",
    )


def _parse_finals_fixed_width(line: str) -> EarthOrientationSample | None:
    """Read the Bulletin-A columns from one IERS finals2000A row.

    The file has two EOP sets: Bulletin A (rapid/prediction) and Bulletin B.
    The A set is deliberately selected here; the merge program supplies the
    C04 final values wherever they exist. Rows without all required Bulletin-A
    values return ``None``; IERS publishes them only at the prediction tail.
    """
    if len(line) < 134:
        raise ValueError("finals2000A rows must contain the Bulletin-A columns")
    try:
        mjd = float(line[7:15])
    except ValueError:
        raise ValueError("finals2000A MJD must be numeric") from None
    if not _is_mjd(mjd):
        raise ValueError(f"finals2000A MJD {mjd} is outside the supported range")

    def field(start: int, end: int) -> float | None:
        text = line[start:end].strip()
        if not text:
            return None
        try:
            return float(text.replace("D", "E").replace("d", "e"))
        except ValueError:
            raise ValueError(f"malformed finals2000A value {text!r}") from None

    xp = field(18, 27)
    yp = field(37, 46)
    dut1 = field(58, 68)
    lod_ms = field(79, 86)
    dx_mas = field(97, 106)
    dy_mas = field(116, 125)
    if xp is None or yp is None or dut1 is None or lod_ms is None or dx_mas is None or dy_mas is None:
        return None
    flags = (line[16:17], line[57:58], line[95:96])
    if any(flag not in {"I", "P"} for flag in flags):
        raise ValueError("finals2000A requires I/P quality flags")
    sample = _sample_if_plausible(
        mjd,
        xp,
        yp,
        dut1,
        0.001 * dx_mas,
        0.001 * dy_mas,
        0.001 * lod_ms,
    )

    if sample is None:
        raise ValueError("Invalid finals2000A numeric values")
    return replace(sample, source="finals2000A", quality="prediction" if "P" in flags else "rapid")


def _read_iers(path: Path, parser, *, encoding: str) -> tuple[EarthOrientationSample, ...]:
    from lunarops.fileio.archive import sha256_file

    samples = []
    incomplete_tail = False
    digest = sha256_file(path)
    for line_no, line in enumerate(path.read_text(encoding=encoding).splitlines(), 1):
        if not line.strip() or line.lstrip().startswith(("#", "%")):
            continue
        try:
            sample = parser(line)
            if sample is None:
                incomplete_tail = True
                continue
            if incomplete_tail:
                raise ValueError("complete EOP row follows an incomplete row")
            samples.append(replace(sample, source_sha256=digest))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: line {line_no}: {exc}") from exc
    if not samples:
        raise ValueError(f"Could not read IERS samples from {path}")
    return tuple(samples)


def read_iers_c04(eop_file: str | Path) -> tuple[EarthOrientationSample, ...]:
    return _read_iers(Path(eop_file).expanduser(), _parse_c04_line, encoding="ascii")


def read_iers_rapid(eop_file: str | Path) -> tuple[EarthOrientationSample, ...]:
    return _read_iers(Path(eop_file).expanduser(), _parse_finals_fixed_width, encoding="ascii")


__all__ = [
    "CelestialPoleOffsets",
    "EarthOrientationProvider",
    "EarthOrientationSample",
    "PolarMotion",
    "TabulatedEarthOrientation",
    "read_iers_c04",
    "read_iers_rapid",
]
