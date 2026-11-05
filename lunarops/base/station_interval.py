"""Station selection on a half-open UTC date interval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class StationInterval:
    """Half-open UTC date interval ``[start, end_exclusive)`` of one station.

    ``start`` and ``end_exclusive`` may be given as ISO dates (``YYYY-MM-DD``) or ``date`` objects and are
    stored as ``date``; ``name`` is stripped of surrounding whitespace.

    Attributes
    ----------
    station : str
        Station ID.
    start : date
        First UTC date of the interval (inclusive).
    end_exclusive : date or None
        First UTC date after the interval; None for an open-ended interval.
    name : str or None
        Optional name of the interval, used as its key.

    Raises
    ------
    ValueError
        If ``end_exclusive`` is not after ``start``.
    """

    station: str
    start: date
    end_exclusive: date | None
    name: str | None = None

    @staticmethod
    def parse(value: object) -> date:
        """Parse an ISO date (``YYYY-MM-DD``), ignoring surrounding whitespace."""
        return date.fromisoformat(str(value).strip())

    def __post_init__(self) -> None:
        """Parse the dates and check that the interval is not empty."""
        start = self.parse(self.start)
        end = None if self.end_exclusive is None else self.parse(self.end_exclusive)
        if end is not None and end <= start:
            raise ValueError("interval end_exclusive must be after start.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end_exclusive", end)
        object.__setattr__(self, "name", None if self.name is None else self.name.strip())

    def contains(self, station, day):
        """Return whether observations of a station on a UTC date fall in this interval.

        Parameters
        ----------
        station : str or ndarray
            Station ID, or an array of station IDs.
        day : date or ndarray
            UTC date, or an array of UTC dates of the same shape as ``station``.

        Returns
        -------
        bool or ndarray of bool
            True where the station matches and ``start <= day < end_exclusive``.
        """
        return (self.station == station) & (self.start <= day) & (True if self.end_exclusive is None else day < self.end_exclusive)

    @property
    def key(self) -> str:
        """Interval key: ``name``, or ``STATION_START_END`` with ISO dates and ``present`` for an open end."""
        end = self.end_exclusive.isoformat() if self.end_exclusive else "present"
        return self.name or f"{self.station}_{self.start.isoformat()}_{end}"
