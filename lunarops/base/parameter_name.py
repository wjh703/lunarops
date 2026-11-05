"""Structured names of estimated parameters.

A parameter name has four colon-separated fields, ``object:type:temporal:interval``. For example,
``apollo15:position.x::`` is the x coordinate of a reflector in the Moon principal-axis (PA) frame and
``GRASSE:rangeBias::`` is a station range bias. Normal-equation systems align their columns by these names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class ParameterName:
    """Structured parameter name ``object:type:temporal:interval``.

    Names are ordered and compared field by field.

    Attributes
    ----------
    object_name : str
        Object the parameter belongs to, such as a station or reflector ID.
    parameter_type : str
        Parameter type, such as ``position.x`` or ``rangeBias``.
    temporal : str
        Temporal representation of the parameter, such as ``interval`` for a piecewise-constant parameter;
        empty for a constant.
    interval : str
        Validity interval of the parameter, such as ``2020-01-01/present`` (UTC dates); empty for a constant.
    """

    object_name: str = ""
    parameter_type: str = ""
    temporal: str = ""
    interval: str = ""

    def __str__(self) -> str:
        """Return the name as ``object:type:temporal:interval``."""
        return f"{self.object_name}:{self.parameter_type}:{self.temporal}:{self.interval}"

    @classmethod
    def parse(cls, text: str) -> ParameterName:
        """Parse a name of up to four colon-separated fields; missing trailing fields are empty.

        Parameters
        ----------
        text : str
            Name text, such as ``GRASSE:rangeBias::``.

        Returns
        -------
        ParameterName
            Parsed name.

        Raises
        ------
        ValueError
            If the text has more than four fields.
        """
        parts = str(text).split(":")
        if len(parts) > 4:
            raise ValueError(f"Structured parameter name has too many fields: {text!r}")
        return cls(*(parts + ["", "", "", ""])[:4])


def names_to_strings(names: Sequence[ParameterName]) -> list[str]:
    """Return the string form of each parameter name, in order.

    Parameters
    ----------
    names : sequence of ParameterName
        Parameter names.

    Returns
    -------
    list of str
        Names as ``object:type:temporal:interval``.
    """
    return [str(n) for n in names]


def parameter_unit(name: ParameterName) -> str:
    """Return the unit implied by the type of a parameter.

    Types starting with ``position.``, containing ``rangeBias`` or ending in ``offset`` are in metres; types
    containing ``time`` or ending in ``clock`` are in seconds; all others are dimensionless. The comparison
    ignores case.

    Parameters
    ----------
    name : ParameterName
        Parameter name.

    Returns
    -------
    str
        ``"m"``, ``"s"`` or ``"1"``.

    Raises
    ------
    TypeError
        If ``name`` is not a ``ParameterName``.
    """
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
