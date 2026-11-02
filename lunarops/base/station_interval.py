"""Station selection on a half-open UTC date interval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class StationInterval:
    station: str
    start: date
    end_exclusive: date | None
    name: str | None = None

    @staticmethod
    def parse(value: object) -> date:
        return date.fromisoformat(str(value).strip())

    def __post_init__(self) -> None:
        start = self.parse(self.start)
        end = None if self.end_exclusive is None else self.parse(self.end_exclusive)
        if end is not None and end <= start:
            raise ValueError("interval end_exclusive must be after start.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end_exclusive", end)
        object.__setattr__(self, "name", None if self.name is None else self.name.strip())

    def contains(self, station, day):
        return (self.station == station) & (self.start <= day) & (True if self.end_exclusive is None else day < self.end_exclusive)

    @property
    def key(self) -> str:
        end = self.end_exclusive.isoformat() if self.end_exclusive else "present"
        return self.name or f"{self.station}_{self.start.isoformat()}_{end}"
