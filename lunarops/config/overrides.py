from __future__ import annotations

from typing import Any

import yaml

from .expressions import is_variable_name


def parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse repeated CLI ``--set name=value`` arguments."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects name=value, got {pair!r}")
        name, value = pair.split("=", 1)
        name = name.strip()
        if not is_variable_name(name):
            raise ValueError(f"--set expects an identifier variable name, got {pair!r}")
        if name in overrides:
            raise ValueError(f"--set variable {name!r} was provided more than once.")
        overrides[name] = yaml.safe_load(value)
    return overrides


__all__ = ["parse_set_overrides"]
