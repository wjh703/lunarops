"""Explicit Earth-orientation data sources used by ERFA frame transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Protocol, TypeGuard

import erfa
import numpy as np

from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.schema import ConfigSchema, boolean, path


@dataclass(frozen=True, slots=True)
class EarthOrientationValues:
    """Earth-orientation parameters at one UTC epoch.

    Attributes
    ----------
    xp_arcsec, yp_arcsec : float
        Polar-motion coordinates x_p, y_p of the CIP in the ITRS, in arcseconds.
    ut1_minus_utc_s : float
        UT1 - UTC, in seconds.
    dx_arcsec, dy_arcsec : float
        Celestial pole offsets dX, dY added to the IAU 2006/2000A CIP coordinates, in arcseconds.
    """

    xp_arcsec: float
    yp_arcsec: float
    ut1_minus_utc_s: float
    dx_arcsec: float = 0.0
    dy_arcsec: float = 0.0


@dataclass(frozen=True, slots=True)
class EarthOrientationSample:
    """One tabulated Earth-orientation record.

    Attributes
    ----------
    mjd_utc : float
        Epoch of the record as a UTC Modified Julian Date.
    xp_arcsec, yp_arcsec : float
        Polar-motion coordinates x_p, y_p, in arcseconds.
    ut1_minus_utc_s : float
        UT1 - UTC, in seconds.
    dx_arcsec, dy_arcsec : float
        Celestial pole offsets dX, dY, in arcseconds.
    lod_s : float
        Excess length of day, in seconds.
    source : str
        Product the record was read from (for example ``"C04-20"`` or ``"finals2000A"``).
    quality : str
        Record quality: ``"final"``, ``"rapid"``, ``"prediction"`` or ``"unspecified"``.
    source_sha256 : str
        SHA-256 digest of the source file.
    """

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


class EarthOrientationProvider(Protocol):
    """Interface of a source of Earth-orientation parameters at UTC epochs."""

    @property
    def source_file_path(self) -> Path | None:
        """Path of the Earth-orientation file, or ``None`` if the data have no file source."""
        ...

    def interpolate(self, epoch_utc: Epoch) -> EarthOrientationValues:
        """Return the Earth-orientation parameters at a UTC epoch."""
        ...


class TabulatedEarthOrientation(EarthOrientationProvider):
    """Earth-orientation parameters linearly interpolated from a tabulated EOP file.

    UT1 - UTC is interpolated as UT1 - TAI, so that leap seconds do not produce jumps, and TAI - UTC at the
    requested epoch is added back. The other parameters are interpolated directly. No high-frequency
    (tidal or libration) terms are included.

    Parameters
    ----------
    samples : sequence of EarthOrientationSample
        Records with distinct UTC MJDs, in any order.
    source_file_path : str or Path or None, optional
        File the records were read from.

    Raises
    ------
    ValueError
        If ``samples`` is empty or contains duplicate MJDs.
    """

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(
        fields=(
            path(
                "file",
                required=True,
                non_empty=True,
                description="Native EOP file, usually the merged C04/Bulletin-A product.",
            ),
            boolean(
                "allowPredicted",
                default=True,
                description=(
                    "Use rows whose quality is `prediction`. If false, prediction rows are dropped and a file with "
                    "`unspecified` quality is rejected."
                ),
            ),
        ),
        type_name="file",
    )

    @classmethod
    def from_config(cls, cfg, ctx):
        """Return the table shared through ``ctx.mpi_resources`` if present, otherwise load ``file``."""
        from lunarops.fileio.earth_orientation import load_earth_orientation_parameter

        payload = ctx.mpi_resources.get("earthRotation")
        if payload is not None:
            return payload
        return load_earth_orientation_parameter(ctx.resolve_path(cfg["file"]), allow_predicted=cfg["allowPredicted"])

    _mjd: np.ndarray
    _values: np.ndarray
    _lod_s: np.ndarray

    __slots__ = ("_lod_s", "_mjd", "_provenance", "_source_file_path", "_values")

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
        self._mjd = np.array([sample.mjd_utc for sample in ordered], dtype=float)
        self._values = np.array(
            [
                [getattr(sample, name) for sample in ordered]
                for name in ("xp_arcsec", "yp_arcsec", "ut1_minus_utc_s", "dx_arcsec", "dy_arcsec")
            ],
            dtype=float,
        )
        self._values[2] -= self._tai_minus_utc_at_mjd(self._mjd)
        self._lod_s = np.array([sample.lod_s for sample in ordered], dtype=float)
        self._source_file_path = Path(source_file_path).expanduser() if source_file_path else None
        self._provenance = tuple((sample.source, sample.quality, sample.source_sha256) for sample in ordered)

    @property
    def source_file_path(self) -> Path | None:
        """File the records were read from, or ``None``."""
        return self._source_file_path

    @property
    def mjd_utc_range(self) -> tuple[float, float]:
        """First and last tabulated epochs as UTC Modified Julian Dates."""
        return float(self._mjd[0]), float(self._mjd[-1])

    @property
    def samples(self) -> tuple[EarthOrientationSample, ...]:
        """Tabulated records in ascending MJD order."""
        return tuple(
            replace(
                EarthOrientationSample(float(mjd), float(xp), float(yp), float(dut1), float(dx), float(dy), float(lod)),
                source=provenance[0],
                quality=provenance[1],
                source_sha256=provenance[2],
            )
            for mjd, xp, yp, dut1, dx, dy, lod, provenance in zip(
                self._mjd,
                self._values[0],
                self._values[1],
                self._values[2] + self._tai_minus_utc_at_mjd(self._mjd),
                self._values[3],
                self._values[4],
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

    def interpolate(self, epoch_utc: Epoch) -> EarthOrientationValues:
        """Return the linearly interpolated Earth-orientation parameters at a UTC epoch.

        Parameters
        ----------
        epoch_utc : Epoch
            Epoch on the UTC scale, within ``mjd_utc_range``.

        Returns
        -------
        EarthOrientationValues
            Polar motion, UT1 - UTC and celestial pole offsets.

        Raises
        ------
        ValueError
            If the epoch is not on UTC or lies outside the tabulated range.
        """
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        mjd = float(epoch_utc.mjd)
        start, end = self.mjd_utc_range
        if mjd < start or mjd > end:
            raise ValueError(
                f"EOP interpolation requested MJD {mjd:.6f}, outside loaded range [{start:.6f}, {end:.6f}]."
            )
        values = [float(np.interp(mjd, self._mjd, column)) for column in self._values]
        values[2] += self._tai_minus_utc_at_epoch(epoch_utc)
        return EarthOrientationValues(*values)


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
    """Parse one IERS EOP C04 row in the 14 C04 (16 columns) or 20 C04 (21 columns) layout."""
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
    [float(part) for part in parts[date_count:]]
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
    The Bulletin-A set is read; the merge program supplies the C04 final
    values wherever they exist. Rows without all required Bulletin-A values
    return ``None``; IERS publishes such rows only at the prediction tail.
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
    columns = np.asarray(
        [(s.mjd_utc, s.xp_arcsec, s.yp_arcsec, s.ut1_minus_utc_s, s.dx_arcsec, s.dy_arcsec, s.lod_s) for s in samples]
    )
    if not np.all(np.isfinite(columns)):
        raise ValueError(f"{path}: EOP numeric fields must be finite")
    return tuple(samples)


def read_iers_c04(eop_file: str | Path) -> tuple[EarthOrientationSample, ...]:
    """Read the records of an IERS EOP 14 C04 or 20 C04 file.

    Parameters
    ----------
    eop_file : str or Path
        ASCII C04 file; lines starting with ``#`` or ``%`` are skipped.

    Returns
    -------
    tuple of EarthOrientationSample
        Records with quality ``"final"``, polar motion and pole offsets in arcseconds, UT1 - UTC and LOD in
        seconds.

    Raises
    ------
    ValueError
        If a row is malformed, its calendar date and MJD disagree, or the file contains no records.
    """
    return _read_iers(Path(eop_file).expanduser(), _parse_c04_line, encoding="ascii")


def read_iers_rapid(eop_file: str | Path) -> tuple[EarthOrientationSample, ...]:
    """Read the Bulletin-A records of an IERS ``finals2000A`` file.

    Rows without complete Bulletin-A values are accepted only at the end of the file and are skipped.
    Pole offsets dX, dY are converted from milliarcseconds and LOD from milliseconds.

    Parameters
    ----------
    eop_file : str or Path
        Fixed-width ASCII ``finals2000A`` file.

    Returns
    -------
    tuple of EarthOrientationSample
        Records with quality ``"rapid"`` or ``"prediction"``, polar motion and pole offsets in arcseconds,
        UT1 - UTC and LOD in seconds.

    Raises
    ------
    ValueError
        If a row is malformed, has values out of range, or the file contains no records.
    """
    return _read_iers(Path(eop_file).expanduser(), _parse_finals_fixed_width, encoding="ascii")


__all__ = [
    "EarthOrientationProvider",
    "EarthOrientationSample",
    "EarthOrientationValues",
    "TabulatedEarthOrientation",
    "read_iers_c04",
    "read_iers_rapid",
]
