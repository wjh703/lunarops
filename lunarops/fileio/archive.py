"""Typed LunarOps ASCII and binary archive primitives inspired by GROOPS.

Native LunarOps artifacts identify their scientific type in the first line.  The
filename selects only the physical encoding: ``.txt[.gz]`` is text and
``.dat[.gz]`` is binary.  JSON, CSV, and NPZ are intentionally not supported
by this layer.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, ContextManager, Iterator, TextIO, cast
from urllib.parse import quote, unquote_to_bytes

TEXT_SUFFIXES = (".txt", ".txt.gz")
BINARY_SUFFIXES = (".dat", ".dat.gz")
_ARTIFACT_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def is_text_path(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in TEXT_SUFFIXES)


def is_binary_path(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in BINARY_SUFFIXES)


def require_text_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if not is_text_path(target):
        raise ValueError(f"LunarOps text archives require .txt or .txt.gz: {target}")
    return target


def require_binary_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if not is_binary_path(target):
        raise ValueError(f"LunarOps binary archives require .dat or .dat.gz: {target}")
    return target


@contextmanager
def _open_text(path: Path, mode: str) -> Iterator[TextIO]:
    if path.name.lower().endswith(".gz"):
        if "w" in mode:
            with path.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                        yield stream
            return
        with gzip.open(path, mode, encoding="utf-8", newline="\n") as stream:
            yield cast(TextIO, stream)
        return
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        yield cast(TextIO, stream)


@contextmanager
def _open_binary(path: Path, mode: str) -> Iterator[BinaryIO]:
    if path.name.lower().endswith(".gz"):
        if "w" in mode:
            with path.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                    yield cast(BinaryIO, stream)
            return
        with gzip.open(path, mode) as stream:
            yield cast(BinaryIO, stream)
        return
    with path.open(mode) as stream:
        yield cast(BinaryIO, stream)


def encode_token(value: object) -> str:
    """Encode one scalar token without whitespace or comment ambiguity."""
    text = str(value)
    if text == "~":
        return "%7E"
    return quote(text, safe="-._:+/=@") or "~"


def decode_token(value: str) -> str:
    if value == "~":
        return ""
    if _PERCENT_ESCAPE.search(value):
        raise ValueError(f"Invalid percent escape in LunarOps token {value!r}.")
    try:
        return unquote_to_bytes(value).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 in LunarOps token {value!r}.") from exc


def format_float(value: object) -> str:
    number = float(cast(Any, value))
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"LunarOps archives reject non-finite float {value!r}.")
    return format(number, ".17e")


def parse_float(value: str, *, field: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {field}: {value!r}") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"Non-finite float for {field}: {value!r}")
    return number


def write_header(stream: TextIO, artifact_type: str, *, version: int) -> None:
    if _ARTIFACT_TYPE.fullmatch(str(artifact_type)) is None:
        raise ValueError(f"Invalid LunarOps artifact type {artifact_type!r}.")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError(f"Invalid LunarOps format version {version!r}.")
    stream.write(f"lunarops {artifact_type} version={version}\n")


def parse_header(
    stream: TextIO,
    expected_type: str | None = None,
    *,
    expected_version: int | None = None,
) -> str:
    for raw in stream:
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) != 3 or parts[0] != "lunarops" or not parts[2].startswith("version="):
            raise ValueError(f"Invalid LunarOps archive header: {text!r}")
        try:
            version = int(parts[2].split("=", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid LunarOps archive version in {text!r}") from exc
        if version <= 0:
            raise ValueError(f"Invalid LunarOps format version {version}.")
        artifact_type = parts[1]
        if _ARTIFACT_TYPE.fullmatch(artifact_type) is None:
            raise ValueError(f"Invalid LunarOps artifact type {artifact_type!r}.")
        if expected_type is not None and artifact_type != expected_type:
            raise ValueError(f"Expected LunarOps {expected_type!r} archive, found {artifact_type!r}.")
        if expected_version is not None and version != expected_version:
            raise ValueError(
                f"Unsupported LunarOps {artifact_type} format version {version}; expected {expected_version}."
            )
        return artifact_type
    raise ValueError("LunarOps archive is empty.")


def data_lines(stream: TextIO) -> Iterator[str]:
    """Yield nonblank, non-comment lines after an archive header."""
    for raw in stream:
        line = raw.split("#", 1)[0].strip()
        if line:
            yield line


@contextmanager
def atomic_text_writer(path: str | Path, artifact_type: str, *, version: int = 1) -> Iterator[TextIO]:
    target = require_text_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp.gz" if target.name.lower().endswith(".gz") else ".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with _open_text(temporary, "wt") as stream:
            write_header(stream, artifact_type, version=version)
            yield stream
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def atomic_binary_writer(path: str | Path) -> Iterator[BinaryIO]:
    target = require_binary_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp.gz" if target.name.lower().endswith(".gz") else ".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with _open_binary(temporary, "wb") as stream:
            yield stream
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def atomic_directory_writer(path: str | Path) -> Iterator[Path]:
    """Build and atomically publish one extensionless directory artifact."""
    target = require_file_group_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    backup: Path | None = None
    try:
        yield temporary
        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"Artifact target exists and is not a directory: {target}")
            backup = target.parent / f".{target.name}.old.{os.getpid()}"
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def open_text_reader(path: str | Path) -> ContextManager[TextIO]:
    target = require_text_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"LunarOps text archive not found: {target}")
    return _open_text(target, "rt")


def open_binary_reader(path: str | Path) -> ContextManager[BinaryIO]:
    target = require_binary_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"LunarOps binary archive not found: {target}")
    return _open_binary(target, "rb")


def require_file_group_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if target.suffix:
        raise ValueError(f"LunarOps file groups require an extensionless directory: {target}")
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_artifact_type(path: str | Path) -> str:
    """Return the declared type of a text artifact without consuming payload."""
    with open_text_reader(path) as stream:
        return parse_header(stream)


__all__ = [
    "BINARY_SUFFIXES",
    "TEXT_SUFFIXES",
    "atomic_binary_writer",
    "atomic_directory_writer",
    "atomic_text_writer",
    "data_lines",
    "decode_token",
    "encode_token",
    "format_float",
    "is_binary_path",
    "is_text_path",
    "open_binary_reader",
    "open_text_reader",
    "parse_float",
    "parse_header",
    "read_artifact_type",
    "require_binary_path",
    "require_file_group_path",
    "require_text_path",
    "sha256_file",
    "write_header",
]
