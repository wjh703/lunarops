from __future__ import annotations

from typing import Any

import yaml

from .expressions import is_variable_name


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


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
