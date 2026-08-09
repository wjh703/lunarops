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
) -> Path:
    target = Path(path).expanduser()
    with atomic_text_writer(target, artifact_type) as stream:
        yaml.safe_dump(
            _plain_data(payload),
            stream,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    return target


def read_structured_text(path: str | Path, artifact_type: str) -> dict[str, object]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, artifact_type)
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"LunarOps {artifact_type} payload must be a mapping: {source}")
    return payload


__all__ = ["read_structured_text", "write_structured_text"]
