"""Parameter names use ``object:type:temporal:interval``.

For example, ``apollo15:position.x::`` is a reflector PA x coordinate and
``GRASSE:rangeBias::`` is a station range bias. Normal systems align columns
by these names."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class ParameterName:
    object_name: str = ""
    parameter_type: str = ""
    temporal: str = ""
    interval: str = ""

    def __str__(self) -> str:
        return f"{self.object_name}:{self.parameter_type}:{self.temporal}:{self.interval}"

    @classmethod
    def parse(cls, text: str) -> ParameterName:
        parts = str(text).split(":")
        if len(parts) > 4:
            raise ValueError(f"Structured parameter name has too many fields: {text!r}")
        return cls(*(parts + ["", "", "", ""])[:4])


def names_to_strings(names: Sequence[ParameterName]) -> list[str]:
    return [str(n) for n in names]


def parameter_unit(name: ParameterName) -> str:
    """Return the canonical unit implied by a structured parameter type."""
    if not isinstance(name, ParameterName):
        raise TypeError("parameter_unit expects a ParameterName.")
    kind = name.parameter_type.casefold()
    if kind.startswith("position.") or "rangebias" in kind or kind.endswith("offset"):
        return "m"
    if "time" in kind or kind.endswith("clock"):
        return "s"
    return "1"


__all__ = [
    "ParameterName",
    "names_to_strings",
    "parameter_unit",
]
