"""Deterministic fingerprints for scientific configuration and referenced data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from lunarops.base.serialization import plain_data as _plain_data

from .archive import sha256_file


def _referenced_files(value, context, files: set[Path]) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _referenced_files(item, context, files)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _referenced_files(item, context, files)
    elif isinstance(value, str):
        candidate = context.resolve_path(value)
        if candidate.is_file():
            files.add(candidate.resolve())


def scientific_fingerprint(
    config: Mapping[str, object],
    context,
    *,
    excluded_keys: Iterable[str] = (),
) -> str:
    """Hash selected program/global configuration and referenced file contents."""
    excluded = set(excluded_keys)
    selected = {key: value for key, value in config.items() if key not in excluded}
    files: set[Path] = set()
    _referenced_files(selected, context, files)
    _referenced_files(context.global_class_configs, context, files)
    payload = {
        "program": _plain_data(selected),
        "globals": _plain_data(context.global_class_configs),
        "files": {str(path): sha256_file(path) for path in sorted(files)},
    }
    encoded = yaml.safe_dump(
        payload,
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["scientific_fingerprint"]
