"""Linearized LLR observation equations, evaluation options and result-table fields.

``EquationBatch`` holds one row per normal point: the one-way O-C residual in metres, its a-priori sigma and
the partial derivatives of the computed one-way range used by the parametrizations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from functools import cached_property

import erfa
import numpy as np

from lunarops.classes.time import Epoch, TimeScale

from .resolver import ObservationCatalogSelection


def diagnostic_values[T](fields: Sequence[tuple[str, str, Callable[[T], object]]], values: T) -> dict[str, object]:
    """Evaluate a table of ``(name, unit, getter)`` field declarations on one value object.

    Parameters
    ----------
    fields : Sequence[tuple[str, str, Callable]]
        Field declarations; the unit entry is not used here.
    values : T
        Object passed to every getter.

    Returns
    -------
    dict[str, object]
        Getter result by field name, in declaration order.
    """
    return {name: getter(values) for name, _, getter in fields}


class ObservationResultDetail(str, Enum):
    """Detail level of per-observation result rows.

    ``standard`` writes ``STANDARD_OUTPUT_FIELDS`` (plus reflector partials when computed); ``full`` adds the
    diagnostic fields of the observation model.
    """

    STANDARD = "standard"
    FULL = "full"

    @classmethod
    def parse(cls, value: str | None) -> ObservationResultDetail:
        """Return the detail level named by ``value`` (case-insensitive); ``None`` selects ``standard``.

        Raises
        ------
        ValueError
            If ``value`` names no detail level.
        """
        candidate = cls.STANDARD.value if value is None else value.strip().lower()
        try:
            return cls(candidate)
        except ValueError as exc:
            raise ValueError(f"Unknown observation result detail {value!r}.") from exc


STANDARD_OUTPUT_FIELDS = (
    "obs_time_utc",
    "obs_time_local",
    "normal_point_index",
    "station_id",
    "station_name",
    "reflector_id",
    "reflector_name",
    "observed_rtt_s",
    "computed_rtt_s",
    "oc_one_way_m",
    "observation_sigma_one_way_m",
    "elevation_up_deg",
    "light_time_converged",
    "status",
)

REFLECTOR_DESIGN_OUTPUT_FIELDS = (
    "design_reflector_pa_x",
    "design_reflector_pa_y",
    "design_reflector_pa_z",
)


@dataclass(frozen=True, slots=True)
class ObservationOptions:
    """Options for evaluating normal points with the LLR observation model.

    Attributes
    ----------
    station_identifier : str or None
        Station catalog key used for every record instead of the record's station name.
    reflector_identifier : str or None
        Reflector catalog key used for every record instead of the record's reflector name.
    min_elevation_deg : float
        Minimum vacuum elevation of the uplink and the downlink at the station, in degrees. Observations
        below it are excluded from equations and result rows.
    low_elevation_policy : str
        ``rejectClamped`` also excludes observations whose troposphere elevation was clamped to a model floor;
        ``allowFlagged`` keeps them with the ``clamped`` flag set.
    utc_offset_hours : float
        Offset of local time from UTC, in hours, used for the ``obs_time_local`` column.
    include_reflector_position_partials : bool
        Compute partials of the one-way range with respect to the reflector Moon PA coordinates.
    include_station_position_partials : bool
        Compute partials of the one-way range with respect to the station ITRF reference coordinates.
    include_lunar_orientation_partials : bool
        Compute partials of the one-way range with respect to small rotations of the lunar PA frame.
    show_progress : bool
        Show a progress bar while evaluating a dataset.
    progress_description : str or None
        Label of the progress bar.

    Raises
    ------
    ValueError
        If ``low_elevation_policy`` is unknown or ``min_elevation_deg`` is not finite.
    """

    station_identifier: str | None = None
    reflector_identifier: str | None = None
    min_elevation_deg: float = 0.0
    low_elevation_policy: str = "rejectClamped"
    utc_offset_hours: float = 0.0
    include_reflector_position_partials: bool = False
    include_station_position_partials: bool = False
    include_lunar_orientation_partials: bool = False
    show_progress: bool = False
    progress_description: str | None = None

    def __post_init__(self) -> None:
        """Validate the elevation settings."""
        if self.low_elevation_policy not in {"rejectClamped", "allowFlagged"}:
            raise ValueError("low_elevation_policy must be rejectClamped or allowFlagged")
        min_elevation = float(self.min_elevation_deg)
        if not np.isfinite(min_elevation):
            raise ValueError("min_elevation_deg must be finite.")
        object.__setattr__(self, "min_elevation_deg", min_elevation)

    @property
    def catalog_selection(self) -> ObservationCatalogSelection:
        """Catalog key overrides as an ``ObservationCatalogSelection``."""
        return ObservationCatalogSelection(self.station_identifier, self.reflector_identifier)

    def with_progress(
        self,
        description: str | None,
        *,
        enabled: bool | None = None,
    ) -> ObservationOptions:
        """Return a copy with a progress-bar label and, optionally, a changed progress switch.

        Parameters
        ----------
        description : str or None
            Progress-bar label.
        enabled : bool, optional
            New value of ``show_progress``; ``None`` keeps the current value.

        Returns
        -------
        ObservationOptions
            Updated copy.
        """
        return replace(
            self,
            progress_description=description,
            show_progress=self.show_progress if enabled is None else bool(enabled),
        )


@dataclass(frozen=True, eq=False)
class EquationBatch:
    """Columnar batch of linearized LLR observation equations, one row per normal point.

    The batch is selected with a boolean mask or an index array (``batch[mask]``) and combined with
    ``concatenate``; rows are not iterable. Partial-derivative columns are with respect to the computed one-way
    range, so a parameter correction ``dx`` predicts ``oc ~ A dx``. A partial block has shape ``(n, 0)`` when it
    was not requested.

    Attributes
    ----------
    oc : ndarray, shape (n,)
        One-way observed-minus-computed residual ``0.5 * c * (observed - computed)`` of the round-trip time,
        in metres. The computed round-trip time includes the range-bias correction.
    sigma : ndarray, shape (n,)
        A-priori one-way observation sigma ``0.5 * c * uncertainty_two_way_s``, in metres.
    observation_id : ndarray of object, shape (n,)
        Observation identity (the normal-point index).
    station_idx : ndarray of int, shape (n,)
        Row index into ``stations``.
    reflector_idx : ndarray of int, shape (n,)
        Row index into ``reflectors``.
    epoch_jd1, epoch_jd2 : ndarray, shape (n,)
        Two-part Julian date of the transmit epoch, UTC.
    receive_jd1, receive_jd2 : ndarray, shape (n,)
        Two-part Julian date of the receive epoch, UTC.
    converged : ndarray of bool, shape (n,)
        Whether the light-time iteration converged.
    wavelength_nm : ndarray of object, shape (n,)
        Laser wavelength in nanometres, or ``None`` when unknown.
    clamped : ndarray of bool, shape (n,)
        Whether the troposphere elevation of the uplink or downlink was clamped to a model floor.
    reflector_position_pa : ndarray, shape (n, 3) or (n, 0)
        Partials of the one-way range with respect to the reflector Moon PA coordinates (x, y, z), in m/m.
    station_position_itrf : ndarray, shape (n, 3) or (n, 0)
        Partials of the one-way range with respect to the station ITRF reference coordinates (X, Y, Z), in m/m.
    lunar_orientation_pa : ndarray, shape (n, 3) or (n, 0)
        Partials of the one-way range with respect to small rotations of the lunar PA frame about its
        x, y and z axes, in m/rad.
    lunar_ecliptic_libration : ndarray, shape (n, 3) or (n, 0)
        Partials of the one-way range with respect to the physical-libration corrections rho, I*sigma and tau
        referred to the J2000 mean ecliptic, in m/rad.
    stations : ndarray of str
        Station catalog keys referenced by ``station_idx``.
    reflectors : ndarray of str
        Reflector catalog keys referenced by ``reflector_idx``.
    """

    oc: np.ndarray
    sigma: np.ndarray
    observation_id: np.ndarray
    station_idx: np.ndarray
    reflector_idx: np.ndarray
    epoch_jd1: np.ndarray
    epoch_jd2: np.ndarray
    receive_jd1: np.ndarray
    receive_jd2: np.ndarray
    converged: np.ndarray
    wavelength_nm: np.ndarray
    clamped: np.ndarray
    reflector_position_pa: np.ndarray
    station_position_itrf: np.ndarray
    lunar_orientation_pa: np.ndarray
    lunar_ecliptic_libration: np.ndarray
    stations: np.ndarray
    reflectors: np.ndarray

    _COLUMNS = (
        "oc", "sigma", "observation_id", "epoch_jd1", "epoch_jd2", "receive_jd1", "receive_jd2",
        "converged", "wavelength_nm", "clamped", "reflector_position_pa", "station_position_itrf",
        "lunar_orientation_pa", "lunar_ecliptic_libration",
    )

    @cached_property
    def fundamental_arguments(self) -> tuple[np.ndarray, ...]:
        """Delaunay arguments (l, l', F, D, Omega) of IERS Conventions (2003), in radians.

        Each array has shape ``(n,)``. The time argument is Julian centuries from J2000.0 of the transmit-epoch
        Julian date (UTC).
        """
        t = ((self.epoch_jd1 + self.epoch_jd2) - 2_451_545.0) / 36_525.0
        return tuple(function(t) for function in (erfa.fal03, erfa.falp03, erfa.faf03, erfa.fad03, erfa.faom03))

    @cached_property
    def dates(self) -> np.ndarray:
        """UTC calendar date of each transmit epoch, as an object array of ``datetime.date``."""
        return np.asarray([date.fromisoformat(self.epoch(i).date_iso()) for i in range(len(self))])

    @cached_property
    def year_columns(self) -> dict[str, np.ndarray]:
        """Cache of Julian years since a station reference epoch, keyed by that epoch string."""
        return {}

    @cached_property
    def libration_bases(self) -> dict[tuple, np.ndarray]:
        """Cache of empirical-libration design bases, keyed by the term tuple."""
        return {}

    # Rows are selected by mask or index array; per-row iteration is not a batch operation.
    __iter__ = None

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.oc)

    def __getitem__(self, rows):
        """Return the rows selected by a boolean mask or index array as a new batch."""
        return EquationBatch(
            **{name: getattr(self, name)[rows] for name in self._COLUMNS},
            station_idx=self.station_idx[rows], reflector_idx=self.reflector_idx[rows],
            stations=self.stations, reflectors=self.reflectors,
        )

    @property
    def station(self) -> np.ndarray:
        """Station catalog key of each row, shape ``(n,)``."""
        return self.stations[self.station_idx]

    @property
    def reflector(self) -> np.ndarray:
        """Reflector catalog key of each row, shape ``(n,)``."""
        return self.reflectors[self.reflector_idx]

    def epoch(self, index: int) -> Epoch:
        """Return the transmit epoch (UTC) of row ``index``."""
        return Epoch(self.epoch_jd1[index], self.epoch_jd2[index], TimeScale.UTC)

    @classmethod
    def single(
        cls, *, oc, sigma, observation_id, station, reflector, epoch, receive_epoch,
        reflector_position_pa=(), station_position_itrf=(), lunar_orientation_pa=(), lunar_ecliptic_libration=(),
        converged=True, wavelength_nm=None, clamped=False,
    ) -> EquationBatch:
        """Return a one-row batch.

        Parameters
        ----------
        oc : float
            One-way O-C residual, in metres.
        sigma : float
            A-priori one-way sigma, in metres.
        observation_id : object
            Observation identity.
        station : str
            Station catalog key.
        reflector : str
            Reflector catalog key.
        epoch : Epoch
            Transmit epoch, UTC.
        receive_epoch : Epoch
            Receive epoch, UTC.
        reflector_position_pa, station_position_itrf : array_like, shape (3,) or (0,)
            One-way range partials with respect to reflector Moon PA and station ITRF coordinates, in m/m.
        lunar_orientation_pa, lunar_ecliptic_libration : array_like, shape (3,) or (0,)
            One-way range partials with respect to PA-frame rotations and ecliptic libration corrections, in m/rad.
        converged : bool
            Whether the light-time iteration converged.
        wavelength_nm : float or None
            Laser wavelength, in nanometres.
        clamped : bool
            Whether a troposphere elevation was clamped.

        Returns
        -------
        EquationBatch
            Batch with one row.
        """
        identities = np.empty(1, dtype=object)
        identities[0] = observation_id
        return cls(
            oc=np.asarray([oc], dtype=float), sigma=np.asarray([sigma], dtype=float), observation_id=identities,
            station_idx=np.zeros(1, dtype=int), reflector_idx=np.zeros(1, dtype=int),
            epoch_jd1=np.asarray([epoch.jd1]), epoch_jd2=np.asarray([epoch.jd2]),
            receive_jd1=np.asarray([receive_epoch.jd1]), receive_jd2=np.asarray([receive_epoch.jd2]),
            converged=np.asarray([converged], dtype=bool), wavelength_nm=np.asarray([wavelength_nm], dtype=object),
            clamped=np.asarray([clamped], dtype=bool),
            reflector_position_pa=np.asarray(reflector_position_pa, dtype=float).reshape(1, -1),
            station_position_itrf=np.asarray(station_position_itrf, dtype=float).reshape(1, -1),
            lunar_orientation_pa=np.asarray(lunar_orientation_pa, dtype=float).reshape(1, -1),
            lunar_ecliptic_libration=np.asarray(lunar_ecliptic_libration, dtype=float).reshape(1, -1),
            stations=np.asarray([station]), reflectors=np.asarray([reflector]),
        )

    @classmethod
    def empty(cls) -> EquationBatch:
        """Return a batch with no rows."""
        return cls(
            oc=np.empty(0), sigma=np.empty(0), observation_id=np.empty(0, dtype=object),
            station_idx=np.empty(0, dtype=int), reflector_idx=np.empty(0, dtype=int),
            epoch_jd1=np.empty(0), epoch_jd2=np.empty(0), receive_jd1=np.empty(0), receive_jd2=np.empty(0),
            converged=np.empty(0, dtype=bool), wavelength_nm=np.empty(0, dtype=object), clamped=np.empty(0, dtype=bool),
            reflector_position_pa=np.empty((0, 0)), station_position_itrf=np.empty((0, 0)),
            lunar_orientation_pa=np.empty((0, 0)), lunar_ecliptic_libration=np.empty((0, 0)),
            stations=np.empty(0, dtype=str), reflectors=np.empty(0, dtype=str),
        )

    @classmethod
    def concatenate(cls, chunks) -> EquationBatch:
        """Concatenate batches row-wise.

        Empty batches are skipped. The station and reflector key arrays of the result are the sorted unique keys
        of all chunks, and the row indices are remapped to them.

        Parameters
        ----------
        chunks : Iterable[EquationBatch]
            Batches with matching partial-column widths.

        Returns
        -------
        EquationBatch
            Combined batch, or an empty batch when no chunk has rows.
        """
        chunks = [chunk for chunk in chunks if len(chunk)]
        if not chunks:
            return cls.empty()
        stations = np.unique(np.concatenate([chunk.stations for chunk in chunks]))
        reflectors = np.unique(np.concatenate([chunk.reflectors for chunk in chunks]))
        return cls(
            **{name: np.concatenate([getattr(chunk, name) for chunk in chunks]) for name in cls._COLUMNS},
            station_idx=np.concatenate([np.searchsorted(stations, chunk.station) for chunk in chunks]),
            reflector_idx=np.concatenate([np.searchsorted(reflectors, chunk.reflector) for chunk in chunks]),
            stations=stations, reflectors=reflectors,
        )

__all__ = [
    "REFLECTOR_DESIGN_OUTPUT_FIELDS",
    "STANDARD_OUTPUT_FIELDS",
    "EquationBatch",
    "ObservationResultDetail",
]
