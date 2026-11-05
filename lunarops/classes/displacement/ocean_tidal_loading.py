"""IERS 2010 ocean tidal loading displacement from BLQ coefficients.

``BLQ_TIDE_NAMES`` lists the 11 BLQ constituents in file column order and ``BLQ_NATIVE_COMPONENT_NAMES``
the BLQ axis order (Up, West, South). BLQ files are typically produced by the Onsala Space Observatory
ocean loading service.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TextIO

import numpy as np

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.base.station_identity import validate_station_id
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.schema import ConfigSchema, path, string

from .base import StationDisplacementInput
from .terrestrial_geometry import GeodeticPosition, enu2itrf, itrf2geodetic

BLQ_TIDE_NAMES = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA")
BLQ_NATIVE_COMPONENT_NAMES = ("up", "west", "south")
HARDISP_MIN_UTC = _iers2010.HARDISP_MIN_UTC
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
    """Ocean tidal loading BLQ coefficients of one station for the 11 HARDISP constituents.

    Rows are in BLQ axis order (Up, West, South) and columns in ``BLQ_TIDE_NAMES`` order
    (M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, Ssa). The arrays are stored read-only.

    Attributes
    ----------
    station_id : str
        Validated LunarOps station identifier.
    source_station_name : str
        Station name as written in the BLQ file.
    amplitudes_m : ndarray, shape (3, 11)
        Displacement amplitudes, in metres.
    phases_deg : ndarray, shape (3, 11)
        Greenwich phase lags, in degrees.

    Raises
    ------
    ValueError
        If a name is empty or an array has the wrong shape or non-finite values.
    """

    station_id: str
    source_station_name: str
    amplitudes_m: np.ndarray
    phases_deg: np.ndarray

    def __post_init__(self) -> None:
        """Validate the names and store the coefficient arrays as read-only copies."""
        source_station_name = str(self.source_station_name).strip()
        station_id = self.station_id
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


class OceanTidalLoadingCatalog:
    """Per-station ocean tidal loading coefficients read from a BLQ file.

    Each station entry is a station name line followed by six rows of eleven finite values: amplitudes
    in metres in Up, West, South order, then Greenwich phase lags in degrees in the same order. Lines
    starting with ``$``, ``#``, ``%``, ``!`` or ``//`` are comments; among them the ocean-tide model name,
    the ``CMC`` declaration, a ``COLUMN ORDER`` declaration and ``END TABLE`` are interpreted. A column
    order, if declared, must match ``BLQ_TIDE_NAMES``. Plain and gzip-compressed files are accepted.

    Parameters
    ----------
    coefficient_file : str or Path
        BLQ file, optionally ``.gz``.

    Attributes
    ----------
    coefficient_file : Path
        Resolved BLQ file.
    tidal_model : str or None
        Ocean-tide model name declared in the file header, if any.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If an entry is malformed, a station is duplicated, declarations conflict, content follows
        ``END TABLE``, or the file holds no station.
    """

    __slots__ = ("_coefficients", "coefficient_file", "tidal_model")

    def __init__(self, coefficient_file: str | Path) -> None:
        path = Path(coefficient_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Ocean tidal-loading BLQ file not found: {path}")
        self.coefficient_file = path
        self._coefficients, self.tidal_model = self._read(path)

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
    ) -> tuple[dict[str, OceanTidalLoadingCoefficients], str | None]:
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
            station_id = validate_station_id(current_name)
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
                if not text.strip():
                    raise ValueError(f"{path}:{line_number}: invalid BLQ station name {text!r}.")
                current_name = text
                current_line_number = line_number

        finish_current()
        if not coefficients:
            raise ValueError(f"No station BLQ coefficients found in {path}.")
        return coefficients, tidal_model

    @property
    def station_ids(self) -> tuple[str, ...]:
        """Identifiers of the stations in the file, sorted."""
        return tuple(sorted(self._coefficients))

    def coefficients_for(self, station_id: str) -> OceanTidalLoadingCoefficients:
        """Return the coefficients of a station.

        Parameters
        ----------
        station_id : str
            LunarOps station identifier.

        Returns
        -------
        OceanTidalLoadingCoefficients
            The station's BLQ coefficients.

        Raises
        ------
        KeyError
            If the file has no entry for the station.
        """
        normalized = station_id
        try:
            return self._coefficients[normalized]
        except KeyError as exc:
            available = ", ".join(self.station_ids)
            raise KeyError(
                f"BLQ file {self.coefficient_file} has no coefficients for station "
                f"{normalized!r}; available stations: {available}."
            ) from exc


class Iers2010OceanTidalLoading:
    """IERS 2010 ocean tidal loading displacement of a station from BLQ coefficients.

    The IERS ``HARDISP`` model expands the 11 BLQ constituents to 342 tidal constituents by admittance
    interpolation and returns the displacement in (Up, West, South) order, in metres. The displacement
    is converted to (East, North, Up) = (-West, -South, Up) and rotated to ITRF axes at the station's
    WGS84 geodetic latitude and longitude. Each event is evaluated as a single ``HARDISP`` sample at its
    UTC epoch rounded to the whole second, because LLR transmit and receive epochs are irregular.
    Supported UTC epochs are 1960-01-01T00:00:00 through 2027-06-30T23:59:59, excluding leap seconds.

    Parameters
    ----------
    catalog : OceanTidalLoadingCatalog
        BLQ coefficients of the stations.

    References
    ----------
    .. [1] Petit, G., & Luzum, B. (Eds.) (2010). IERS Conventions (2010), IERS Technical Note 36,
       Sec. 7.1.2.
    """

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(
        fields=(
            path(
                "coefficientFile",
                required=True,
                non_empty=True,
                description="Ocean-loading coefficients in BLQ format.",
            ),
            string(
                "model",
                non_empty=True,
                description=(
                    "Expected ocean-tide model name, such as `FES2022b`. The run fails if the BLQ header declares a "
                    "different model."
                ),
            ),
        ),
        type_name="iers2010OceanTidalLoading",
    )

    @classmethod
    def from_config(cls, cfg, ctx):
        """Create the model from the cached BLQ catalog of ``coefficientFile``.

        Raises
        ------
        ValueError
            If ``model`` is set and the BLQ header declares no model or a different one.
        """
        from lunarops.classes.displacement import (
            OceanTidalLoadingCatalog,
        )
        from lunarops.classes.displacement.coefficients import coefficient_file as _coefficient_file

        catalog = _coefficient_file(OceanTidalLoadingCatalog, ctx.resolve_path(cfg["coefficientFile"]))
        expected_model = cfg.get("model")
        actual_model = catalog.tidal_model
        if expected_model is not None and (
            actual_model is None or str(expected_model).casefold() != actual_model.casefold()
        ):
            raise ValueError(
                f"stationDisplacement/iers2010OceanTidalLoading model mismatch: config requests {expected_model!r}, BLQ file declares {actual_model!r}."
            )
        return cls(catalog=catalog)

    def __init__(self, catalog: OceanTidalLoadingCatalog) -> None:
        self.catalog = catalog
        self._hardisp_cache: dict[tuple, np.ndarray] = {}
        self._site_cache: dict[tuple, GeodeticPosition] = {}

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
    def _hardisp_up_west_south_m(
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
        up, west, south = _iers2010.hardisp(
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
        return np.asarray((up[0], west[0], south[0]), float)

    def evaluate(self, data: StationDisplacementInput) -> np.ndarray:
        """Return the ocean tidal loading displacement of a station.

        Parameters
        ----------
        data : StationDisplacementInput
            Station ITRF position, UTC epoch and station identifier.

        Returns
        -------
        ndarray, shape (3,)
            Displacement (X, Y, Z) in the ITRF, in metres.

        Raises
        ------
        ValueError
            If ``data.station_id`` is ``None``, or the epoch is a leap second or outside the supported range.
        KeyError
            If the catalog has no coefficients for the station.
        """
        if data.station_id is None:
            raise ValueError("Iers2010OceanTidalLoading requires StationDisplacementInput.station_id.")
        coefficients = self.catalog.coefficients_for(data.station_id)
        key = (data.station_id, self._utc_calendar_second(data.epoch_utc))
        if key not in self._hardisp_cache:
            if len(self._hardisp_cache) >= 64:
                self._hardisp_cache.clear()
            self._hardisp_cache[key] = self._hardisp_up_west_south_m(coefficients, data.epoch_utc)
        up_west_south_m = self._hardisp_cache[key]
        # HARDISP returns positive Up, West, South.  LunarOps uses East, North, Up.
        enu_m = np.array(
            [-up_west_south_m[1], -up_west_south_m[2], up_west_south_m[0]],
            dtype=float,
        )
        position_key = tuple(data.reference_position_itrf_m)
        if position_key not in self._site_cache:
            self._site_cache[position_key] = itrf2geodetic(data.reference_position_itrf_m)
        site = self._site_cache[position_key]
        itrf_m = enu2itrf(
            enu_m,
            latitude_rad=site.latitude_rad,
            longitude_rad=site.longitude_rad,
        )
        return itrf_m

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        """Return the ocean tidal loading displacement (X, Y, Z) in the ITRF, shape (3,), in metres."""
        return self.evaluate(data)


__all__ = [
    "BLQ_NATIVE_COMPONENT_NAMES",
    "BLQ_TIDE_NAMES",
    "Iers2010OceanTidalLoading",
    "OceanTidalLoadingCatalog",
    "OceanTidalLoadingCoefficients",
]
