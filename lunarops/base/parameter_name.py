"""Structured parameter names inspired by GROOPS.

LunarOps uses a GROOPS-inspired structured parameter name,
``object:type:temporal:interval``. Structured names are what make normal
equations *combinable across programs*: two normal-equation files can be
merged by aligning parameter names instead of hoping the column order agrees.

Examples
--------
``apollo15:position.x::``                     reflector PA x-coordinate
``GRASSE:rangeBias::``                        per-station range bias
``earth:polarMotion.xp:trend:``               (future) EOP parameter
``moon:orbitState.x0::``                      (future) integrated orbit ICs
``moon:loveNumber.h2::``                      (future) lunar tide parameter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True, order=True, slots=True)
class ParameterName:
    object_name: str = ""
    parameter_type: str = ""
    temporal: str = ""
    interval: str = ""

    def __post_init__(self) -> None:
        fields = ("object_name", "parameter_type", "temporal", "interval")
        for field_name in fields:
            value = getattr(self, field_name)
            text = str(value or "").strip()
            if ":" in text:
                raise ValueError(f"ParameterName.{field_name} must not contain ':' characters.")
            object.__setattr__(self, field_name, text)
        if not self.parameter_type:
            raise ValueError("ParameterName.parameter_type must not be empty.")

    def __str__(self) -> str:
        return f"{self.object_name}:{self.parameter_type}:{self.temporal}:{self.interval}"

    @classmethod
    def parse(cls, text: str) -> "ParameterName":
        parts = str(text).split(":")
        if len(parts) > 4:
            raise ValueError(f"Structured parameter name has too many fields: {text!r}")
        return cls(*(parts + ["", "", "", ""])[:4])


def names_to_strings(names: Sequence[ParameterName]) -> List[str]:
    return [str(n) for n in names]


def strings_to_names(strings: Sequence[str]) -> List[ParameterName]:
    return [ParameterName.parse(s) for s in strings]


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
    "strings_to_names",
]
