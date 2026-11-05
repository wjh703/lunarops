"""Read and write native ``earthOrientationParameter`` files of tabulated Earth-orientation parameters.

The file is a records table (see :mod:`lunarops.fileio.result_table`) with artifact type
``earthOrientationParameter``, an empty header and one row per sample, sorted by ``mjdUtc`` on write. Fields,
with their kind and unit token:

- ``mjdUtc`` (float, d): Modified Julian Date of the sample, UTC.
- ``xpArcsec``, ``ypArcsec`` (float, arcsec): polar motion x and y.
- ``ut1MinusUtcSec`` (float, s): UT1 - UTC.
- ``lodSec`` (float, s): excess length of day.
- ``dxArcsec``, ``dyArcsec`` (float, arcsec): celestial pole offsets dX and dY.
- ``source`` (text, 1): source product of the sample, for example ``finals2000A``.
- ``quality`` (text, 1): sample quality label, for example ``rapid`` or ``prediction``.
- ``sourceSha256`` (text, 1): SHA-256 digest of the source file.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

from lunarops.classes.frames.earth_orientation import EarthOrientationSample, TabulatedEarthOrientation

from .result_table import read_records, write_records

ARTIFACT_TYPE = "earthOrientationParameter"
FIELDS = (
    "mjdUtc",
    "xpArcsec",
    "ypArcsec",
    "ut1MinusUtcSec",
    "lodSec",
    "dxArcsec",
    "dyArcsec",
)

PROVENANCE_FIELDS = ("source", "quality", "sourceSha256")
_FIELD_UNITS = dict(zip(FIELDS, ("d", "arcsec", "arcsec", "s", "s", "arcsec", "arcsec"), strict=True))


def _ordered_samples(samples: Sequence[EarthOrientationSample]) -> tuple[EarthOrientationSample, ...]:
    """Return samples sorted by MJD after checking there is at least one and no MJD repeats."""
    if not samples:
        raise ValueError("Earth-orientation output requires at least one sample.")
    ordered = tuple(sorted(samples, key=lambda sample: sample.mjd_utc))
    mjd = [sample.mjd_utc for sample in ordered]
    if any(right <= left for left, right in pairwise(mjd)):
        raise ValueError("Earth-orientation output requires strictly increasing MJD values.")
    return ordered


_ATTRIBUTES = ("mjd_utc", "xp_arcsec", "yp_arcsec", "ut1_minus_utc_s", "lod_s", "dx_arcsec", "dy_arcsec",
               "source", "quality", "source_sha256")
_SCHEMA = tuple((name, "float" if name in FIELDS else "text") for name in FIELDS + PROVENANCE_FIELDS)


def write_earth_orientation_parameter(samples: Sequence[EarthOrientationSample], path: str | Path) -> Path:
    """Write an ``earthOrientationParameter`` file with samples sorted by UTC MJD.

    Parameters
    ----------
    samples : Sequence of EarthOrientationSample
        Earth-orientation samples with their provenance.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    Path
        The written path.
    """
    rows = [dict(zip(FIELDS + PROVENANCE_FIELDS, (getattr(sample, name) for name in _ATTRIBUTES)))
            for sample in sorted(samples, key=lambda sample: sample.mjd_utc)]
    return write_records(path, ARTIFACT_TYPE, {}, _SCHEMA, rows, field_units=_FIELD_UNITS)


def read_earth_orientation_parameter(path: str | Path) -> tuple[EarthOrientationSample, ...]:
    """Read an ``earthOrientationParameter`` file.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    tuple of EarthOrientationSample
        Samples in file order.

    Raises
    ------
    ValueError
        If the schema or units differ, the file has no samples, or two samples share an MJD.
    """
    _, rows = read_records(path, ARTIFACT_TYPE, expected_fields=_SCHEMA, field_units=_FIELD_UNITS)
    samples = tuple(EarthOrientationSample(**dict(zip(_ATTRIBUTES, row.values()))) for row in rows)
    _ordered_samples(samples)
    return samples


def load_earth_orientation_parameter(path: str | Path, *, allow_predicted: bool = True) -> TabulatedEarthOrientation:
    """Load an ``earthOrientationParameter`` file as an interpolating Earth-orientation provider.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``; ``~`` is expanded.
    allow_predicted : bool, optional
        When False, samples with quality ``prediction`` are dropped, and a file containing any sample of
        quality ``unspecified`` is rejected.

    Returns
    -------
    TabulatedEarthOrientation
        Provider over the retained samples, recording ``path`` as its source file.

    Raises
    ------
    ValueError
        If the file is invalid, or ``allow_predicted`` is False and a sample's quality is ``unspecified``.
    """
    source = Path(path).expanduser()
    samples = read_earth_orientation_parameter(source)
    if not allow_predicted:
        if any(sample.quality == "unspecified" for sample in samples):
            raise ValueError("EOP quality is unspecified; cannot enforce a prediction-free policy")
        samples = tuple(sample for sample in samples if sample.quality != "prediction")
    return TabulatedEarthOrientation(samples, source_file_path=source)


__all__ = [
    "ARTIFACT_TYPE",
    "FIELDS",
    "load_earth_orientation_parameter",
    "read_earth_orientation_parameter",
    "write_earth_orientation_parameter",
]
