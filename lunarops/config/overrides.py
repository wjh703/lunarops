"""Parsing for command-line variable overrides."""

from __future__ import annotations

import re
from typing import Any

from .expressions import is_variable_name


def _parse_value(value: str) -> Any:
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none"}:
        return None

    if text.startswith(("'", '"', "[", "{")):
        import yaml

        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML value in --set: {value!r}") from exc
        if isinstance(parsed, (str, int, float, bool, list, dict)) or parsed is None:
            return parsed
        raise TypeError(f"Unsupported --set value {value!r}.")

    if re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)", text):
        return int(text)
    if re.fullmatch(
        r"[+-]?(?:(?:[0-9]+\.[0-9]*)|(?:\.[0-9]+)|(?:[0-9]+[eE][+-]?[0-9]+))",
        text,
    ):
        return float(text)
    return text


def parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse repeated CLI ``--set name=value`` arguments."""
    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if not isinstance(pair, str):
            raise TypeError(f"--set entries must be strings, got {pair!r}")
        if "=" not in pair:
            raise ValueError(f"--set expects name=value, got {pair!r}")
        name, value = pair.split("=", 1)
        name = name.strip()
        if not is_variable_name(name):
            raise ValueError(f"--set expects an identifier variable name, got {pair!r}")
        if name in overrides:
            raise ValueError(f"--set variable {name!r} was provided more than once.")
        overrides[name] = _parse_value(value)
    return overrides


__all__ = ["parse_set_overrides"]
