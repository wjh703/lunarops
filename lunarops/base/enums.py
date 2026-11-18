"""Parsing of string-valued enumerations from configuration text."""

from __future__ import annotations

from enum import StrEnum


def parse_enum[E: StrEnum](cls: type[E], value: object, *, default: E | None = None, what: str) -> E:
    """Return the member of a string enumeration named by ``value``.

    Names are matched case-insensitively after stripping surrounding whitespace.

    Parameters
    ----------
    cls : type of StrEnum
        Enumeration to parse into.
    value : object
        A member of ``cls``, its value as a string, or ``None`` when ``default`` is given.
    default : StrEnum, optional
        Member returned for ``None``; without it ``None`` is rejected.
    what : str
        Description of the value used in error messages, for example ``"time scale"``.

    Returns
    -------
    StrEnum
        The parsed member of ``cls``.

    Raises
    ------
    TypeError
        If ``value`` is neither a member, a string nor (with ``default``) ``None``.
    ValueError
        If the string names no member.
    """
    if isinstance(value, cls):
        return value
    if value is None and default is not None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string or {cls.__name__}, got {type(value).__name__}.")
    text = value.strip().casefold()
    for member in cls:
        if member.value.casefold() == text:
            return member
    allowed = ", ".join(member.value for member in cls)
    raise ValueError(f"Unsupported {what} {value!r}; expected one of: {allowed}.")


__all__ = ["parse_enum"]
