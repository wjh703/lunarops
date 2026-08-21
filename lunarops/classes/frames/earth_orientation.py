"""Explicit Earth-orientation data sources used by ERFA frame transforms."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeGuard, cast

import erfa
import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.classes.time import Epoch, TimeScale


@dataclass(frozen=True, slots=True)
class PolarMotion:
    xp_arcsec: float
    yp_arcsec: float

    def __post_init__(self) -> None:
        for name in ("xp_arcsec", "yp_arcsec"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CelestialPoleOffsets:
    dx_arcsec: float
    dy_arcsec: float

    def __post_init__(self) -> None:
        for name in ("dx_arcsec", "dy_arcsec"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class EarthOrientationSample:
    mjd_utc: float
    xp_arcsec: float
    yp_arcsec: float
    ut1_minus_utc_s: float
    dx_arcsec: float = 0.0
    dy_arcsec: float = 0.0
    lod_s: float = 0.0

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


DuplicateMjdPolicy = Literal["error", "first", "last", "mean"]


def _parse_duplicate_mjd_policy(value: str | None) -> DuplicateMjdPolicy:
    policy = str(value or "error").strip().lower()
    if policy not in {"error", "first", "last", "mean"}:
        raise ValueError(f"duplicateMjdPolicy must be one of 'error', 'first', 'last', or 'mean', got {value!r}.")
    # The membership check above validates the runtime string; the cast tells
    # static type checkers that it is now one of the allowed literal values.
    return cast(DuplicateMjdPolicy, policy)


def _deduplicate_samples(
    sorted_samples: Sequence[EarthOrientationSample],
    *,
    policy: DuplicateMjdPolicy,
) -> tuple[EarthOrientationSample, ...]:
    if policy == "error":
        mjd_values = np.array([sample.mjd_utc for sample in sorted_samples], dtype=float)
        if np.unique(mjd_values).size != mjd_values.size:
            duplicate_values = sorted(
                {float(value) for value in mjd_values if np.count_nonzero(mjd_values == value) > 1}
            )
            preview = ", ".join(f"{value:.1f}" for value in duplicate_values[:10])
            suffix = "" if len(duplicate_values) <= 10 else f", ... ({len(duplicate_values)} duplicate MJDs)"
            raise ValueError(
                "EOP table contains duplicate MJD values. "
                "Set earthRotation.duplicateMjdPolicy explicitly to 'first', 'last', or 'mean' "
                f"if this is an intentional concatenated C04 file. Duplicate MJD(s): {preview}{suffix}"
            )
        return tuple(sorted_samples)

    grouped: dict[float, list[EarthOrientationSample]] = {}
    for sample in sorted_samples:
        grouped.setdefault(float(sample.mjd_utc), []).append(sample)

    result: list[EarthOrientationSample] = []
    for mjd in sorted(grouped):
        samples = grouped[mjd]
        if len(samples) == 1 or policy == "first":
            result.append(samples[0])
        elif policy == "last":
            result.append(samples[-1])
        elif policy == "mean":
            result.append(
                EarthOrientationSample(
                    mjd_utc=mjd,
                    xp_arcsec=float(np.mean([sample.xp_arcsec for sample in samples])),
                    yp_arcsec=float(np.mean([sample.yp_arcsec for sample in samples])),
                    ut1_minus_utc_s=float(np.mean([sample.ut1_minus_utc_s for sample in samples])),
                    dx_arcsec=float(np.mean([sample.dx_arcsec for sample in samples])),
                    dy_arcsec=float(np.mean([sample.dy_arcsec for sample in samples])),
                    lod_s=float(np.mean([sample.lod_s for sample in samples])),
                )
            )
        else:  # pragma: no cover - guarded by _parse_duplicate_mjd_policy
            raise AssertionError(policy)
    return tuple(result)


class EarthOrientationProvider(ABC):
    """Typed access to the Earth-orientation quantities used by LunarOps."""

    @property
    @abstractmethod
    def source_file_path(self) -> Path | None: ...

    @abstractmethod
    def polar_motion(self, epoch_utc: Epoch) -> PolarMotion: ...

    def celestial_pole_offsets(self, epoch_utc: Epoch) -> CelestialPoleOffsets:
        if not isinstance(epoch_utc, Epoch):
            raise TypeError("Earth-orientation queries require an Epoch.")
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        return CelestialPoleOffsets(0.0, 0.0)

    @abstractmethod
    def ut1_minus_utc_s(self, epoch_utc: Epoch) -> float: ...

    def close(self) -> None:
        """Release resources; the default implementation owns none."""
        return


class TabulatedEarthOrientation(EarthOrientationProvider):
    """Linearly interpolated Earth-orientation table."""

    _duplicate_mjd_policy: DuplicateMjdPolicy

    __slots__ = (
        "_duplicate_mjd_policy",
        "_dx_arcsec",
        "_dy_arcsec",
        "_lod_s",
        "_mjd",
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
        duplicate_mjd_policy: DuplicateMjdPolicy = "error",
    ) -> None:
        if not samples:
            raise ValueError("EarthOrientationProvider requires at least one EOP sample.")
        policy = _parse_duplicate_mjd_policy(duplicate_mjd_policy)
        ordered = _deduplicate_samples(
            sorted(samples, key=lambda item: item.mjd_utc),
            policy=policy,
        )
        mjd = np.array([sample.mjd_utc for sample in ordered], dtype=float)
        self._mjd = mjd
        self._xp_arcsec = np.array([sample.xp_arcsec for sample in ordered], dtype=float)
        self._yp_arcsec = np.array([sample.yp_arcsec for sample in ordered], dtype=float)
        dut1_sec = np.array([sample.ut1_minus_utc_s for sample in ordered], dtype=float)
        self._ut1_minus_tai_sec = dut1_sec - self._tai_minus_utc_at_mjd(mjd)
        self._dx_arcsec = np.array([sample.dx_arcsec for sample in ordered], dtype=float)
        self._dy_arcsec = np.array([sample.dy_arcsec for sample in ordered], dtype=float)
        self._lod_s = np.array([sample.lod_s for sample in ordered], dtype=float)
        for name in (
            "_mjd",
            "_xp_arcsec",
            "_yp_arcsec",
            "_ut1_minus_tai_sec",
            "_dx_arcsec",
            "_dy_arcsec",
            "_lod_s",
        ):
            values = getattr(self, name)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"EOP column {name} contains non-finite values.")
            values.setflags(write=False)
        self._source_file_path = Path(source_file_path).expanduser() if source_file_path else None
        self._duplicate_mjd_policy = policy

    @classmethod
    def from_columns(
        cls,
        mjd_utc: ArrayLike,
        xp_arcsec: ArrayLike,
        yp_arcsec: ArrayLike,
        ut1_minus_utc_s: ArrayLike,
        dx_arcsec: ArrayLike | None = None,
        dy_arcsec: ArrayLike | None = None,
        lod_s: ArrayLike | None = None,
        *,
        source_file_path: str | Path | None = None,
        duplicate_mjd_policy: DuplicateMjdPolicy = "error",
    ) -> TabulatedEarthOrientation:
        """Construct directly from already parsed EOP columns.

        This path is used by MPI workers after rank 0 broadcasts the parsed
        columns.  It deliberately avoids rebuilding thousands of
        :class:`EarthOrientationSample` objects or reparsing the text file.
        The broadcast payload must already be sorted and deduplicated.
        """
        policy = _parse_duplicate_mjd_policy(duplicate_mjd_policy)
        columns = [
            np.asarray(mjd_utc, dtype=float),
            np.asarray(xp_arcsec, dtype=float),
            np.asarray(yp_arcsec, dtype=float),
            np.asarray(ut1_minus_utc_s, dtype=float),
        ]
        if any(values.ndim != 1 for values in columns):
            raise ValueError("Broadcast EOP columns must be one-dimensional arrays.")
        sizes = {int(values.size) for values in columns}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError("Broadcast EOP columns must have the same non-zero length.")
        size = next(iter(sizes))
        optional_columns = [
            np.zeros(size, dtype=float) if dx_arcsec is None else np.asarray(dx_arcsec, dtype=float),
            np.zeros(size, dtype=float) if dy_arcsec is None else np.asarray(dy_arcsec, dtype=float),
            np.zeros(size, dtype=float) if lod_s is None else np.asarray(lod_s, dtype=float),
        ]
        if any(values.ndim != 1 or values.size != size for values in optional_columns):
            raise ValueError("Broadcast optional EOP columns must match the EOP column length.")
        columns.extend(optional_columns)
        if any(not np.all(np.isfinite(values)) for values in columns):
            raise ValueError("Broadcast EOP columns contain non-finite values.")
        if np.any(np.diff(columns[0]) <= 0.0):
            raise ValueError("Broadcast EOP MJD values must be strictly increasing after rank-0 duplicate handling.")

        self = cls.__new__(cls)
        for name, values in zip(("_mjd", "_xp_arcsec", "_yp_arcsec"), columns[:3]):
            copied = np.array(values, dtype=float, copy=True, order="C")
            copied.setflags(write=False)
            setattr(self, name, copied)
        ut1_minus_tai = columns[3] - self._tai_minus_utc_at_mjd(columns[0])
        stored = (ut1_minus_tai, columns[4], columns[5], columns[6])
        for name, values in zip(("_ut1_minus_tai_sec", "_dx_arcsec", "_dy_arcsec", "_lod_s"), stored):
            copied = np.array(values, dtype=float, copy=True, order="C")
            copied.setflags(write=False)
            setattr(self, name, copied)
        self._source_file_path = Path(source_file_path).expanduser() if source_file_path else None
        self._duplicate_mjd_policy = policy
        return self

    def to_mpi_payload(self) -> dict[str, object]:
        """Return the compact, picklable columns broadcast to worker ranks."""
        return {
            "kind": "earthOrientationArrays",
            "sourceFile": None if self.source_file_path is None else str(self.source_file_path),
            "duplicateMjdPolicy": self.duplicate_mjd_policy,
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
        if not isinstance(payload, Mapping) or payload.get("kind") != "earthOrientationArrays":
            raise ValueError("Invalid MPI Earth-orientation payload.")
        required = ("mjdUtc", "xpArcsec", "ypArcsec", "ut1MinusUtcSec")
        if any(key not in payload for key in required):
            raise ValueError("MPI Earth-orientation payload is missing required columns.")
        source_file = payload.get("sourceFile")
        if source_file is not None and not isinstance(source_file, (str, Path)):
            raise TypeError("MPI Earth-orientation sourceFile must be a path string.")
        return cls.from_columns(
            cast(ArrayLike, payload["mjdUtc"]),
            cast(ArrayLike, payload["xpArcsec"]),
            cast(ArrayLike, payload["ypArcsec"]),
            cast(ArrayLike, payload["ut1MinusUtcSec"]),
            cast(ArrayLike | None, payload.get("dxArcsec")),
            cast(ArrayLike | None, payload.get("dyArcsec")),
            cast(ArrayLike | None, payload.get("lodSec")),
            source_file_path=source_file,
            duplicate_mjd_policy=_parse_duplicate_mjd_policy(
                cast(str | None, payload.get("duplicateMjdPolicy", "error"))
            ),
        )

    @property
    def source_file_path(self) -> Path | None:
        return self._source_file_path

    @property
    def duplicate_mjd_policy(self) -> DuplicateMjdPolicy:
        return self._duplicate_mjd_policy

    @property
    def mjd_utc_range(self) -> tuple[float, float]:
        return float(self._mjd[0]), float(self._mjd[-1])

    @property
    def samples(self) -> tuple[EarthOrientationSample, ...]:
        return tuple(
            EarthOrientationSample(
                float(mjd), float(xp), float(yp), float(dut1), float(dx), float(dy), float(lod)
            )
            for mjd, xp, yp, dut1, dx, dy, lod in zip(
                self._mjd,
                self._xp_arcsec,
                self._yp_arcsec,
                self._ut1_minus_tai_sec + self._tai_minus_utc_at_mjd(self._mjd),
                self._dx_arcsec,
                self._dy_arcsec,
                self._lod_s,
            )
        )

    @staticmethod
    def _tai_minus_utc_at_mjd(mjd_utc) -> np.ndarray:
        values = np.asarray(mjd_utc, dtype=float)
        year, month, day, fraction = erfa.jd2cal(2_400_000.5, values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", erfa.ErfaWarning)
            return np.asarray(erfa.dat(year, month, day, fraction), dtype=float)

    @staticmethod
    def _tai_minus_utc_at_epoch(epoch_utc: Epoch) -> float:
        year, month, day, fields = erfa.d2dtf("UTC", 9, epoch_utc.jd1, epoch_utc.jd2)
        seconds = fields["h"] * 3600.0 + fields["m"] * 60.0 + fields["s"] + fields["f"] * 1.0e-9
        fraction = min(seconds / SECONDS_PER_DAY, np.nextafter(1.0, 0.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", erfa.ErfaWarning)
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


def _float_or_none(value: str) -> float | None:
    try:
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _is_int_token(value: str) -> bool:
    return value.lstrip("+-").isdigit()


def _is_mjd(value: float | None) -> TypeGuard[float]:
    return value is not None and 15_000.0 < value < 90_000.0


def _sample_if_plausible(
    mjd: float,
    xp: float,
    yp: float,
    dut1: float,
    dx: float = 0.0,
    dy: float = 0.0,
    lod: float = 0.0,
) -> EarthOrientationSample | None:
    # Polar motion is in arcseconds and UT1-UTC is in seconds.  These generous
    # bounds reject obvious mis-parses such as choosing x-error as y-pole while
    # still covering historical and prediction rows.
    if (
        abs(xp) > 5.0
        or abs(yp) > 5.0
        or abs(dut1) > 5.0
        or abs(dx) > 5.0
        or abs(dy) > 5.0
        or abs(lod) > 5.0
    ):
        return None
    return EarthOrientationSample(mjd, xp, yp, dut1, dx, dy, lod)


def _parse_c04_line(line: str) -> EarthOrientationSample | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "%")):
        return None
    parts = stripped.split()
    if len(parts) < 4:
        return None

    # Common eopc04 layout: year month day MJD xp yp UT1-UTC ...
    if len(parts) >= 7 and all(_is_int_token(parts[i]) for i in range(3)):
        mjd = _float_or_none(parts[3])
        if _is_mjd(mjd):
            xp = _float_or_none(parts[4])
            yp = _float_or_none(parts[5])
            dut1 = _float_or_none(parts[6])
            dx = _float_or_none(parts[8]) if len(parts) > 9 else 0.0
            dy = _float_or_none(parts[9]) if len(parts) > 9 else 0.0
            lod = _float_or_none(parts[12]) if len(parts) > 12 else 0.0
            if xp is not None and yp is not None and dut1 is not None:
                sample = _sample_if_plausible(mjd, xp, yp, dut1, dx or 0.0, dy or 0.0, lod or 0.0)
                if sample is not None:
                    return sample
            sample = _parse_split_finals_row(parts, 3, mjd)
            if sample is not None:
                return sample

    # C04 variants with an hour column: year month day hour MJD xp yp UT1-UTC ...
    if len(parts) >= 8 and all(_is_int_token(parts[i]) for i in range(4)):
        mjd = _float_or_none(parts[4])
        if _is_mjd(mjd):
            xp = _float_or_none(parts[5])
            yp = _float_or_none(parts[6])
            dut1 = _float_or_none(parts[7])
            dx = _float_or_none(parts[8]) if len(parts) > 9 else 0.0
            dy = _float_or_none(parts[9]) if len(parts) > 9 else 0.0
            lod = _float_or_none(parts[12]) if len(parts) > 12 else 0.0
            if xp is not None and yp is not None and dut1 is not None:
                sample = _sample_if_plausible(mjd, xp, yp, dut1, dx or 0.0, dy or 0.0, lod or 0.0)
                if sample is not None:
                    return sample
            sample = _parse_split_finals_row(parts, 4, mjd)
            if sample is not None:
                return sample

    # Compact numeric layout: MJD xp yp UT1-UTC ...
    mjd = _float_or_none(parts[0])
    if _is_mjd(mjd) and len(parts) >= 4:
        xp = _float_or_none(parts[1])
        yp = _float_or_none(parts[2])
        dut1 = _float_or_none(parts[3])
        if xp is not None and yp is not None and dut1 is not None:
            sample = _sample_if_plausible(mjd, xp, yp, dut1)
            if sample is not None:
                return sample

    # Last-resort MJD discovery for whitespace-separated derived tables.  This
    # handles files that prepend a label or version column before the MJD.
    numeric = [(index, value) for index, part in enumerate(parts) if (value := _float_or_none(part)) is not None]
    for index, value in numeric:
        if _is_mjd(value):
            if index + 3 < len(parts):
                xp = _float_or_none(parts[index + 1])
                yp = _float_or_none(parts[index + 2])
                dut1 = _float_or_none(parts[index + 3])
                if xp is not None and yp is not None and dut1 is not None:
                    sample = _sample_if_plausible(value, xp, yp, dut1)
                    if sample is not None:
                        return sample

    return None


def _parse_split_finals_row(
    parts: list[str],
    mjd_index: int,
    mjd: float,
) -> EarthOrientationSample | None:
    """Fallback for whitespace-normalized finals2000A rows."""
    if len(parts) <= mjd_index + 10 or _float_or_none(parts[mjd_index + 1]) is not None:
        return None
    xp = _float_or_none(parts[mjd_index + 2])
    yp = _float_or_none(parts[mjd_index + 4])
    dut1 = _float_or_none(parts[mjd_index + 7])
    lod_ms = _float_or_none(parts[mjd_index + 9])
    dx_mas = _float_or_none(parts[mjd_index + 12]) if len(parts) > mjd_index + 12 else 0.0
    dy_mas = _float_or_none(parts[mjd_index + 14]) if len(parts) > mjd_index + 14 else 0.0
    if xp is None or yp is None or dut1 is None:
        return None
    return _sample_if_plausible(
        mjd,
        xp,
        yp,
        dut1,
        0.001 * (dx_mas or 0.0),
        0.001 * (dy_mas or 0.0),
        0.001 * (lod_ms or 0.0),
    )


def _parse_finals_fixed_width(line: str) -> EarthOrientationSample | None:
    """Read the Bulletin-A columns from one IERS finals2000A row.

    The file has two EOP sets: Bulletin A (rapid/prediction) and Bulletin B.
    The A set is deliberately selected here; the merge program supplies the
    C04 final values wherever they exist.
    """
    if len(line) < 134 or not line.strip():
        return None
    try:
        mjd = float(line[7:15])
    except ValueError:
        return None
    if not _is_mjd(mjd):
        return None

    def field(start: int, end: int) -> float | None:
        text = line[start:end].strip()
        return _float_or_none(text) if text else None

    xp = field(18, 27)
    yp = field(37, 46)
    dut1 = field(58, 68)
    lod_ms = field(79, 86)
    dx_mas = field(97, 106)
    dy_mas = field(116, 125)
    if xp is None or yp is None or dut1 is None:
        return None
    return _sample_if_plausible(
        mjd,
        xp,
        yp,
        dut1,
        0.001 * (dx_mas or 0.0),
        0.001 * (dy_mas or 0.0),
        0.001 * (lod_ms or 0.0),
    )


def read_iers_c04(eop_file: str | Path) -> tuple[EarthOrientationSample, ...]:
    path = Path(eop_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"IERS C04 file not found: {path}")
    samples = [
        sample
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if (sample := _parse_c04_line(line)) is not None
    ]
    if not samples:
        raise ValueError(f"Could not read IERS C04 samples from {path}.")
    return tuple(samples)


def read_iers_rapid(eop_file: str | Path) -> tuple[EarthOrientationSample, ...]:
    path = Path(eop_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"IERS finals2000A file not found: {path}")
    samples = [
        sample
        for line in path.read_text(encoding="ascii", errors="ignore").splitlines()
        if (sample := _parse_finals_fixed_width(line)) is not None
    ]
    if not samples:
        raise ValueError(f"Could not read Bulletin-A samples from {path}.")
    return tuple(samples)


def read_iers_eop(eop_file: str | Path) -> tuple[EarthOrientationSample, ...]:
    path = Path(eop_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"IERS C04/EOP file not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    samples = [sample for line in lines if (sample := _parse_finals_fixed_width(line)) is not None]
    if not samples:
        samples = [sample for line in lines if (sample := _parse_c04_line(line)) is not None]
    if not samples:
        preview_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "%")):
                preview_lines.append(stripped[:180])
            if len(preview_lines) >= 5:
                break
        preview = "\n".join(f"  {line}" for line in preview_lines) or "  <no non-comment text rows>"
        raise ValueError(
            f"Could not read EOP samples from {path}. Expected IERS C04 rows "
            "(year month day MJD xp yp UT1-UTC), compact rows "
            "(MJD xp yp UT1-UTC), or finals.all/finals2000A rows with I/P flags. "
            f"First non-comment rows seen:\n{preview}"
        )
    return tuple(samples)


def load_iers_eop(
    eop_file: str | Path,
    *,
    duplicate_mjd_policy: DuplicateMjdPolicy = "error",
) -> TabulatedEarthOrientation:
    path = Path(eop_file).expanduser()
    return TabulatedEarthOrientation(
        read_iers_eop(path),
        source_file_path=path,
        duplicate_mjd_policy=duplicate_mjd_policy,
    )


__all__ = [
    "CelestialPoleOffsets",
    "DuplicateMjdPolicy",
    "EarthOrientationProvider",
    "EarthOrientationSample",
    "PolarMotion",
    "TabulatedEarthOrientation",
    "load_iers_eop",
    "read_iers_c04",
    "read_iers_rapid",
    "read_iers_eop",
]
