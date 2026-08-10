"""IERS 2010 ocean tidal loading from external Onsala BLQ coefficients."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.base.array_validation import readonly_vector3
from lunarops.classes.time import Epoch, TimeScale
from lunarops.base.station_identity import canonical_station_id, normalize_station_key

from .base import StationDisplacementInput
from .terrestrial_geometry import enu2itrf, itrf2geodetic

BLQ_TIDE_NAMES = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA")
BLQ_NATIVE_COMPONENT_NAMES = ("up", "west", "south")
HARDISP_MIN_UTC = _iers2010.HARDISP_MIN_UTC
# This application policy is kept in the ERFA facade, not the Cython core.
HARDISP_VALID_UNTIL_UTC_EXCLUSIVE = _iers2010.HARDISP_VALID_UNTIL_UTC_EXCLUSIVE
_MODEL_LINE = re.compile(r"^\s*\$+\s*([A-Za-z][A-Za-z0-9_.-]*)\s*:\s*M2\b", re.IGNORECASE)
_CMC_LINE = re.compile(r"\bCMC\s*:\s*(YES|NO)\b", re.IGNORECASE)
_COLUMN_ORDER_LINE = re.compile(r"\bCOLUMN\s+ORDER\s*:\s*(.*)$", re.IGNORECASE)


def _readonly_matrix(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    expected_shape = (len(BLQ_NATIVE_COMPONENT_NAMES), len(BLQ_TIDE_NAMES))
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class OceanTidalLoadingCoefficients:
    """One station's 11-constituent BLQ amplitudes and Greenwich phase lags."""

    station_id: str
    source_station_name: str
    amplitudes_m: np.ndarray
    phases_deg: np.ndarray

    def __post_init__(self) -> None:
        source_station_name = str(self.source_station_name).strip()
        station_id = canonical_station_id(self.station_id)
        if not source_station_name:
            raise ValueError("source_station_name must not be empty.")
        if not station_id:
            raise ValueError("station_id must not be empty.")
        object.__setattr__(self, "station_id", station_id)
        object.__setattr__(self, "source_station_name", source_station_name)
        object.__setattr__(
            self,
            "amplitudes_m",
            _readonly_matrix(self.amplitudes_m, name="amplitudes_m"),
        )
        object.__setattr__(
            self,
            "phases_deg",
            _readonly_matrix(self.phases_deg, name="phases_deg"),
        )


@dataclass(frozen=True, slots=True)
class OceanTidalLoadingCatalogInfo:
    coefficient_file: Path
    station_count: int
    tidal_model: str | None
    center_of_mass_correction: bool | None


@dataclass(frozen=True, slots=True, eq=False)
class OceanTidalLoadingResult:
    """Ocean-tide displacement with both native and project-local components."""

    displacement_itrf_m: np.ndarray
    displacement_enu_m: np.ndarray
    displacement_up_south_west_m: np.ndarray
    coefficients: OceanTidalLoadingCoefficients

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "displacement_itrf_m",
            readonly_vector3(self.displacement_itrf_m, name="displacement_itrf_m"),
        )
        object.__setattr__(
            self,
            "displacement_enu_m",
            readonly_vector3(self.displacement_enu_m, name="displacement_enu_m"),
        )
        object.__setattr__(
            self,
            "displacement_up_south_west_m",
            readonly_vector3(
                self.displacement_up_south_west_m,
                name="displacement_up_south_west_m",
            ),
        )


class OceanTidalLoadingCatalog:
    """Strict parser and immutable lookup table for an Onsala BLQ file.

    The accepted table has one station identifier followed by six rows of
    eleven finite values: amplitudes in Up/West/South order, then Greenwich
    phase lags in the same order.  Metadata and station diagnostic lines from
    the Onsala provider are comments beginning with ``$``.  If a column-order
    declaration is present, it must exactly match the 11 HARDISP constituents.
    """

    __slots__ = ("_coefficients", "_info", "coefficient_file")

    def __init__(self, coefficient_file: str | Path) -> None:
        path = Path(coefficient_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Ocean tidal-loading BLQ file not found: {path}")
        self.coefficient_file = path
        self._coefficients, self._info = self._read(path)

    @staticmethod
    def _open(path: Path) -> TextIO:
        if path.suffix.lower() == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return path.open("rt", encoding="utf-8")

    @staticmethod
    def _numeric_row(text: str, *, path: Path, line_number: int) -> np.ndarray | None:
        fields = text.split()
        if not fields:
            return None
        try:
            float(fields[0])
        except ValueError:
            return None
        if len(fields) != len(BLQ_TIDE_NAMES):
            raise ValueError(
                f"{path}:{line_number}: BLQ numeric row must contain {len(BLQ_TIDE_NAMES)} values, got {len(fields)}."
            )
        try:
            values = np.asarray([float(field) for field in fields], dtype=float)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid BLQ numeric value.") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}:{line_number}: BLQ numeric row contains a non-finite value.")
        return values

    @classmethod
    def _read(
        cls,
        path: Path,
    ) -> tuple[dict[str, OceanTidalLoadingCoefficients], OceanTidalLoadingCatalogInfo]:
        coefficients: dict[str, OceanTidalLoadingCoefficients] = {}
        current_name: str | None = None
        current_line_number: int | None = None
        current_rows: list[np.ndarray] = []
        tidal_model: str | None = None
        center_of_mass_correction: bool | None = None
        table_ended = False

        def finish_current() -> None:
            nonlocal current_name, current_line_number, current_rows
            if current_name is None:
                return
            if len(current_rows) != 2 * len(BLQ_NATIVE_COMPONENT_NAMES):
                raise ValueError(
                    f"{path}:{current_line_number}: station {current_name!r} has "
                    f"{len(current_rows)} BLQ numeric rows; expected 6."
                )
            station_id = canonical_station_id(current_name)
            if station_id in coefficients:
                raise ValueError(f"{path}:{current_line_number}: duplicate BLQ station {station_id!r}.")
            values = np.asarray(current_rows, dtype=float)
            coefficients[station_id] = OceanTidalLoadingCoefficients(
                station_id=station_id,
                source_station_name=current_name,
                amplitudes_m=values[: len(BLQ_NATIVE_COMPONENT_NAMES)],
                phases_deg=values[len(BLQ_NATIVE_COMPONENT_NAMES) :],
            )
            current_name = None
            current_line_number = None
            current_rows = []

        with cls._open(path) as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                text = raw_line.strip()
                if not text:
                    continue
                if text.startswith(("$", "#", "%", "!", "//")):
                    column_order_match = _COLUMN_ORDER_LINE.search(text)
                    if column_order_match is not None:
                        parsed_order = tuple(value.upper() for value in column_order_match.group(1).split())
                        if parsed_order != BLQ_TIDE_NAMES:
                            raise ValueError(
                                f"{path}:{line_number}: BLQ column order {parsed_order!r} "
                                f"does not match {BLQ_TIDE_NAMES!r}."
                            )
                    model_match = _MODEL_LINE.match(text)
                    if model_match is not None:
                        parsed_model = model_match.group(1)
                        if tidal_model is None:
                            tidal_model = parsed_model
                        elif tidal_model.casefold() != parsed_model.casefold():
                            raise ValueError(
                                f"{path}:{line_number}: conflicting tidal models {tidal_model!r} and {parsed_model!r}."
                            )
                    cmc_match = _CMC_LINE.search(text)
                    if cmc_match is not None:
                        parsed_cmc = cmc_match.group(1).upper() == "YES"
                        if center_of_mass_correction is None:
                            center_of_mass_correction = parsed_cmc
                        elif center_of_mass_correction != parsed_cmc:
                            raise ValueError(f"{path}:{line_number}: conflicting CMC declarations.")
                    if "END TABLE" in text.upper():
                        finish_current()
                        table_ended = True
                    continue
                if table_ended:
                    raise ValueError(f"{path}:{line_number}: content found after END TABLE.")

                numeric_values = cls._numeric_row(text, path=path, line_number=line_number)
                if numeric_values is not None:
                    if current_name is None:
                        raise ValueError(f"{path}:{line_number}: BLQ numeric row has no preceding station name.")
                    if len(current_rows) >= 2 * len(BLQ_NATIVE_COMPONENT_NAMES):
                        raise ValueError(f"{path}:{line_number}: station {current_name!r} has more than six BLQ rows.")
                    current_rows.append(numeric_values)
                    continue

                if current_name is not None:
                    finish_current()
                if not normalize_station_key(text):
                    raise ValueError(f"{path}:{line_number}: invalid BLQ station name {text!r}.")
                current_name = text
                current_line_number = line_number

        finish_current()
        if not coefficients:
            raise ValueError(f"No station BLQ coefficients found in {path}.")
        return coefficients, OceanTidalLoadingCatalogInfo(
            coefficient_file=path,
            station_count=len(coefficients),
            tidal_model=tidal_model,
            center_of_mass_correction=center_of_mass_correction,
        )

    @property
    def info(self) -> OceanTidalLoadingCatalogInfo:
        return self._info

    @property
    def station_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._coefficients))

    def coefficients_for(self, station_id: object) -> OceanTidalLoadingCoefficients:
        normalized = canonical_station_id(station_id)
        try:
            return self._coefficients[normalized]
        except KeyError as exc:
            available = ", ".join(self.station_ids)
            raise KeyError(
                f"BLQ file {self.coefficient_file} has no coefficients for station "
                f"{normalized!r}; available stations: {available}."
            ) from exc


class Iers2010OceanTidalLoading:
    """Evaluate one arbitrary UTC epoch with the Cython IERS ``HARDISP`` model.

    ``HARDISP`` also supports a regularly sampled series, but LLR
    transmit times are irregular and receive times are resolved iteratively.
    The production displacement interface therefore deliberately uses one
    ``n=1`` call per event instead of forcing those epochs onto a grid.
    """

    def __init__(self, catalog: OceanTidalLoadingCatalog) -> None:
        if not isinstance(catalog, OceanTidalLoadingCatalog):
            raise TypeError("catalog must be an OceanTidalLoadingCatalog.")
        self.catalog = catalog

    @staticmethod
    def _utc_calendar_second(epoch_utc: Epoch) -> tuple[int, int, int, int, int, int]:
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        # HARDISP has an integer UTC-second interface. ERFA formatting rounds
        # across minute/day boundaries and retains a leap-second label as 60 so
        # the caller can reject the scalar interface's ambiguous representation.
        date_text, time_text = epoch_utc.isot(precision=0).split("T", maxsplit=1)
        year, month, day = (int(value) for value in date_text.split("-"))
        hour, minute, second = (int(value) for value in time_text.split(":"))
        return year, month, day, hour, minute, second

    @staticmethod
    def _hardisp_up_south_west_m(
        coefficients: OceanTidalLoadingCoefficients,
        epoch_utc: Epoch,
    ) -> np.ndarray:
        year, month, day, hour, minute, second = Iers2010OceanTidalLoading._utc_calendar_second(epoch_utc)
        if second == 60:
            raise ValueError(
                "Iers2010OceanTidalLoading cannot evaluate an exact UTC leap-second "
                "label: HARDISP's scalar calendar interface cannot distinguish "
                "23:59:60 from the following midnight."
            )
        calendar = (year, month, day, hour, minute, second)
        if calendar < HARDISP_MIN_UTC or calendar >= HARDISP_VALID_UNTIL_UTC_EXCLUSIVE:
            raise ValueError(
                "Iers2010OceanTidalLoading supports UTC epochs only from "
                "1960-01-01T00:00:00 through 2027-06-30T23:59:59; "
                f"got {year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}."
            )
        # The N-series interface is valid only on a fixed UTC-sampled
        # grid. Production light-time events are arbitrary, so use one sample.
        up, south, west = _iers2010.hardisp(
            year,
            month,
            day,
            hour,
            minute,
            second,
            1,
            1.0,
            coefficients.amplitudes_m,
            coefficients.phases_deg,
        )
        return readonly_vector3((up[0], south[0], west[0]), name="HARDISP result")

    def evaluate(self, data: StationDisplacementInput) -> OceanTidalLoadingResult:
        if data.station_id is None:
            raise ValueError("Iers2010OceanTidalLoading requires StationDisplacementInput.station_id.")
        coefficients = self.catalog.coefficients_for(data.station_id)
        up_south_west_m = self._hardisp_up_south_west_m(coefficients, data.epoch_utc)
        # HARDISP returns positive Up, South, West.  LunarOps uses East, North, Up.
        enu_m = np.array(
            [-up_south_west_m[2], -up_south_west_m[1], up_south_west_m[0]],
            dtype=float,
        )
        site = itrf2geodetic(data.reference_position_itrf_m)
        itrf_m = enu2itrf(
            enu_m,
            latitude_rad=site.latitude_rad,
            longitude_rad=site.longitude_rad,
        )
        return OceanTidalLoadingResult(
            displacement_itrf_m=itrf_m,
            displacement_enu_m=enu_m,
            displacement_up_south_west_m=up_south_west_m,
            coefficients=coefficients,
        )

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        return np.array(self.evaluate(data).displacement_itrf_m, copy=True)


__all__ = [
    "BLQ_NATIVE_COMPONENT_NAMES",
    "BLQ_TIDE_NAMES",
    "Iers2010OceanTidalLoading",
    "OceanTidalLoadingCatalog",
    "OceanTidalLoadingCatalogInfo",
    "OceanTidalLoadingCoefficients",
    "OceanTidalLoadingResult",
]
