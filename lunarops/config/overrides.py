"""Parsing of ``--set NAME=VALUE`` command-line overrides of config variables."""

from __future__ import annotations

from typing import Any

import yaml

from .expressions import is_variable_name


def parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse repeated ``--set name=value`` command-line arguments.

    Each value is parsed as YAML, so ``3`` becomes an int and ``[a, b]`` a list.

    Parameters
    ----------
    pairs : list of str
        Arguments of the form ``name=value``.

    Returns
    -------
    dict
        Override values keyed by variable name.

    Raises
    ------
    ValueError
        If an argument has no ``=``, the name is not an identifier, or a name is given more than once.
    """
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
