"""Typed text reports and restart state using a YAML scalar grammar."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from lunarops.base.serialization import plain_data as _plain_data

from .archive import atomic_text_writer, open_text_reader, parse_header


def write_structured_text(
    path: str | Path,
    artifact_type: str,
    payload: Mapping[str, object],
    *,
    version: int,
    validator=None,
) -> Path:
    payload = _plain_data(payload) if validator is None else validator(payload)
    target = Path(path).expanduser()
    with atomic_text_writer(target, artifact_type, version=version) as stream:
        yaml.dump(
            payload,
            stream,
            Dumper=yaml.CSafeDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    return target


def read_structured_text(
    path: str | Path,
    artifact_type: str,
    *,
    version: int,
) -> dict[str, object]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, artifact_type, expected_version=version)
        payload = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"LunarOps {artifact_type} payload must be a mapping: {source}")
    return payload


__all__ = ["read_structured_text", "write_structured_text"]
