"""Read and write structured text artifacts: a ``lunarops <artifactType>`` header followed by a YAML mapping.

Structured text artifacts hold reports, metadata, catalogs and restart state. The payload is converted to
plain data and written in YAML block style with key order preserved; both directions use the YAML safe
dumper and loader, so only plain scalars, lists and mappings are represented.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from lunarops.base.serialization import plain_data

from .archive import atomic_text_writer, open_text_reader, parse_header


def load_yaml_file(path: Path) -> Any:
    """Parse a whole UTF-8 YAML file with the safe loader (``yaml.safe_load``).

    Parameters
    ----------
    path : Path
        YAML file.

    Returns
    -------
    Any
        Parsed document; ``None`` for an empty file.

    Raises
    ------
    ValueError
        If the file is not valid YAML; the message names the file.
    """
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc


def write_structured_text(
    path: str | Path,
    artifact_type: str,
    payload: Mapping[str, object],
) -> Path:
    """Write a structured text artifact atomically.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.
    artifact_type : str
        Artifact type written in the ``lunarops <artifactType>`` line.
    payload : Mapping of str to object
        Content, converted to plain data before it is written as YAML.

    Returns
    -------
    Path
        The written path with ``~`` expanded.

    Raises
    ------
    ValueError
        If the suffix or the artifact type is invalid.
    """
    payload = plain_data(payload)
    target = Path(path).expanduser()
    with atomic_text_writer(target, artifact_type) as stream:
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
) -> dict[str, object]:
    """Read a structured text artifact.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.
    artifact_type : str
        Artifact type the header line must declare.

    Returns
    -------
    dict
        YAML payload.

    Raises
    ------
    ValueError
        If the suffix or header is invalid, the artifact type differs, or the payload is not a mapping.
    FileNotFoundError
        If the file does not exist.
    """
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, artifact_type)
        payload = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"LunarOps {artifact_type} payload must be a mapping: {source}")  # noqa: TRY004 -- invalid file content
    return payload


__all__ = ["load_yaml_file", "read_structured_text", "write_structured_text"]
