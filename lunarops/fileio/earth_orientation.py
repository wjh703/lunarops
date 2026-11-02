"""Native LunarOps Earth-orientation parameter artifacts."""

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
    rows = [dict(zip(FIELDS + PROVENANCE_FIELDS, (getattr(sample, name) for name in _ATTRIBUTES)))
            for sample in sorted(samples, key=lambda sample: sample.mjd_utc)]
    return write_records(path, ARTIFACT_TYPE, {}, _SCHEMA, rows, field_units=_FIELD_UNITS)


def read_earth_orientation_parameter(path: str | Path) -> tuple[EarthOrientationSample, ...]:
    _, rows = read_records(path, ARTIFACT_TYPE, expected_fields=_SCHEMA, field_units=_FIELD_UNITS)
    samples = tuple(EarthOrientationSample(**dict(zip(_ATTRIBUTES, row.values()))) for row in rows)
    _ordered_samples(samples)
    return samples


def load_earth_orientation_parameter(path: str | Path, *, allow_predicted: bool = True) -> TabulatedEarthOrientation:
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
