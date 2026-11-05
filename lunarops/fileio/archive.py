"""Text and binary archive primitives for native LunarOps artifacts.

Every native text artifact starts with the header line ``lunarops <artifactType>``, where the artifact type
matches ``[A-Za-z][A-Za-z0-9]*`` and names the scientific content. The file name selects only the physical
encoding: ``.txt`` or ``.txt.gz`` is UTF-8 text and ``.dat`` or ``.dat.gz`` is binary; a ``.gz`` suffix selects
gzip compression, written with a zero modification time so that identical content gives identical bytes.
File groups (covariance and normal-equation directories) use extensionless directory paths. JSON, CSV and NPZ
files are not supported by this layer.

Writers are atomic: content is written to a temporary sibling that replaces the target only after the write
succeeds.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast
from urllib.parse import quote, unquote_to_bytes

TEXT_SUFFIXES = (".txt", ".txt.gz")
BINARY_SUFFIXES = (".dat", ".dat.gz")
_ARTIFACT_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def is_text_path(path: str | Path) -> bool:
    """Return whether a file name has a native text suffix.

    Parameters
    ----------
    path : str or Path
        File path; only its name is inspected, case-insensitively.

    Returns
    -------
    bool
        True if the name ends with ``.txt`` or ``.txt.gz``.
    """
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in TEXT_SUFFIXES)


def is_binary_path(path: str | Path) -> bool:
    """Return whether a file name has a native binary suffix.

    Parameters
    ----------
    path : str or Path
        File path; only its name is inspected, case-insensitively.

    Returns
    -------
    bool
        True if the name ends with ``.dat`` or ``.dat.gz``.
    """
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in BINARY_SUFFIXES)


def require_text_path(path: str | Path) -> Path:
    """Return a user-expanded text archive path after checking its suffix.

    Parameters
    ----------
    path : str or Path
        Path of a native text archive.

    Returns
    -------
    Path
        The path with ``~`` expanded.

    Raises
    ------
    ValueError
        If the name does not end with ``.txt`` or ``.txt.gz``.
    """
    target = Path(path).expanduser()
    if not is_text_path(target):
        raise ValueError(f"LunarOps text archives require .txt or .txt.gz: {target}")
    return target


def require_binary_path(path: str | Path) -> Path:
    """Return a user-expanded binary archive path after checking its suffix.

    Parameters
    ----------
    path : str or Path
        Path of a native binary archive.

    Returns
    -------
    Path
        The path with ``~`` expanded.

    Raises
    ------
    ValueError
        If the name does not end with ``.dat`` or ``.dat.gz``.
    """
    target = Path(path).expanduser()
    if not is_binary_path(target):
        raise ValueError(f"LunarOps binary archives require .dat or .dat.gz: {target}")
    return target


@contextmanager
def _open_text(path: Path, mode: str, *, encoding: str = "utf-8", newline: str | None = "\n") -> Iterator[TextIO]:
    """Open a text stream, using gzip with a zero header modification time for ``.gz`` names."""
    if path.name.lower().endswith(".gz"):
        if "w" in mode:
            with (
                path.open("wb") as raw,
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
                io.TextIOWrapper(compressed, encoding=encoding, newline=newline) as stream,
            ):
                yield stream
            return
        with gzip.open(path, mode, encoding=encoding, newline=newline) as stream:
            yield cast(TextIO, stream)
        return
    with path.open(mode, encoding=encoding, newline=newline) as stream:
        yield cast(TextIO, stream)


@contextmanager
def _open_binary(path: Path, mode: str) -> Iterator[BinaryIO]:
    """Open a binary stream, using gzip with a zero header modification time for ``.gz`` names."""
    if path.name.lower().endswith(".gz"):
        if "w" in mode:
            with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                yield cast(BinaryIO, stream)
            return
        with gzip.open(path, mode) as stream:
            yield cast(BinaryIO, stream)
        return
    with path.open(mode) as stream:
        yield cast(BinaryIO, stream)


def encode_token(value: object) -> str:
    """Encode one scalar as an archive token without whitespace or comment ambiguity.

    The value is converted with ``str`` and percent-encoded as UTF-8, leaving ASCII letters, digits and
    ``-._~:+/=@`` unescaped. The empty string is written as ``~`` and a token consisting only of ``~`` as
    ``%7E``.

    Parameters
    ----------
    value : object
        Scalar to encode.

    Returns
    -------
    str
        Token without whitespace or ``#``.
    """
    text = str(value)
    if text == "~":
        return "%7E"
    return quote(text, safe="-._:+/=@") or "~"


def decode_token(value: str) -> str:
    """Decode a token written by ``encode_token``.

    Parameters
    ----------
    value : str
        Encoded token; ``~`` decodes to the empty string.

    Returns
    -------
    str
        Decoded text.

    Raises
    ------
    ValueError
        If the token contains a malformed percent escape or the decoded bytes are not valid UTF-8.
    """
    if value == "~":
        return ""
    if _PERCENT_ESCAPE.search(value):
        raise ValueError(f"Invalid percent escape in LunarOps token {value!r}.")
    try:
        return unquote_to_bytes(value).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 in LunarOps token {value!r}.") from exc


def format_float(value: object, precision: str = ".17e") -> str:
    """Format a finite float for an archive.

    Parameters
    ----------
    value : object
        Value convertible with ``float``.
    precision : str, optional
        Format specification passed to ``format``; the default ``.17e`` round-trips every float64.

    Returns
    -------
    str
        Formatted number.

    Raises
    ------
    ValueError
        If the value is NaN or infinite.
    """
    number = float(cast(Any, value))
    if not math.isfinite(number):
        raise ValueError(f"LunarOps archives reject non-finite float {value!r}.")
    return format(number, precision)


def parse_float(value: str, *, field: str) -> float:
    """Parse a finite float token.

    Parameters
    ----------
    value : str
        Token to parse.
    field : str
        Field name used in error messages.

    Returns
    -------
    float
        Parsed value.

    Raises
    ------
    ValueError
        If the token is not a number or is NaN or infinite.
    """
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite float for {field}: {value!r}")
    return number


def write_header(stream: TextIO, artifact_type: str) -> None:
    """Write the ``lunarops <artifactType>`` header line.

    Parameters
    ----------
    stream : TextIO
        Output text stream.
    artifact_type : str
        Artifact type matching ``[A-Za-z][A-Za-z0-9]*``.

    Raises
    ------
    ValueError
        If the artifact type is not a valid identifier.
    """
    if _ARTIFACT_TYPE.fullmatch(str(artifact_type)) is None:
        raise ValueError(f"Invalid LunarOps artifact type {artifact_type!r}.")
    stream.write(f"lunarops {artifact_type}\n")


def parse_header(stream: TextIO, expected_type: str | None = None) -> str:
    """Read and validate the ``lunarops <artifactType>`` header line.

    Blank lines and lines starting with ``#`` before the header are skipped. The stream is left positioned
    after the header line.

    Parameters
    ----------
    stream : TextIO
        Input text stream positioned at the start of the archive.
    expected_type : str or None, optional
        Artifact type the header must declare; any valid type is accepted when None.

    Returns
    -------
    str
        Declared artifact type.

    Raises
    ------
    ValueError
        If the header is malformed, the artifact type is invalid or differs from ``expected_type``, or the
        stream holds no header.
    """
    for raw in stream:
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) != 2 or parts[0] != "lunarops":
            raise ValueError(f"Invalid LunarOps archive header: {text!r}")
        artifact_type = parts[1]
        if _ARTIFACT_TYPE.fullmatch(artifact_type) is None:
            raise ValueError(f"Invalid LunarOps artifact type {artifact_type!r}.")
        if expected_type is not None and artifact_type != expected_type:
            raise ValueError(f"Expected LunarOps {expected_type!r} archive, found {artifact_type!r}.")
        return artifact_type
    raise ValueError("LunarOps archive is empty.")


def data_lines(stream: Iterable[str]) -> Iterator[str]:
    """Yield nonblank, non-comment lines after an archive header.

    Text from the first ``#`` to the end of a line is a comment and is removed; surrounding whitespace is
    stripped and lines that are then empty are skipped.

    Parameters
    ----------
    stream : Iterable of str
        Remaining lines of an archive.

    Yields
    ------
    str
        Stripped data line.
    """
    for raw in stream:
        line = raw.split("#", 1)[0].strip()
        if line:
            yield line


@contextmanager
def atomic_text_writer(path: str | Path, artifact_type: str) -> Iterator[TextIO]:
    """Open a native text archive for atomic writing, with its header line already written.

    Parent directories are created. Content goes to a temporary file in the target directory, which replaces
    the target when the ``with`` block exits normally and is removed otherwise.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.
    artifact_type : str
        Artifact type written in the ``lunarops <artifactType>`` header.

    Yields
    ------
    TextIO
        UTF-8 text stream with LF line endings, gzip-compressed for ``.gz`` names.

    Raises
    ------
    ValueError
        If the suffix or the artifact type is invalid.
    """
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
            write_header(stream, artifact_type)
            yield stream
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def atomic_binary_writer(path: str | Path) -> Iterator[BinaryIO]:
    """Open a native binary archive for atomic writing.

    Parent directories are created. Content goes to a temporary file in the target directory, which replaces
    the target when the ``with`` block exits normally and is removed otherwise.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.dat`` or ``.dat.gz``.

    Yields
    ------
    BinaryIO
        Binary stream, gzip-compressed for ``.gz`` names.

    Raises
    ------
    ValueError
        If the suffix is invalid.
    """
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
    """Write a file-group directory atomically.

    The caller fills a temporary directory created next to the target. When the ``with`` block exits
    normally, an existing target directory is moved to a backup ``.<name>.old.<pid>``, the temporary directory
    is renamed to the target and the backup is removed. If this fails, the backup is restored when the target
    is missing. The temporary directory is always removed.

    Parameters
    ----------
    path : str or Path
        Extensionless target directory path.

    Yields
    ------
    Path
        Temporary directory to write the group's files into.

    Raises
    ------
    ValueError
        If the path has a suffix.
    FileExistsError
        If the target exists and is not a directory.
    """
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


def open_text_reader(path: str | Path) -> AbstractContextManager[TextIO]:
    """Open a native text archive for reading.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    AbstractContextManager of TextIO
        Context manager yielding a UTF-8 text stream, decompressed for ``.gz`` names.

    Raises
    ------
    ValueError
        If the suffix is invalid.
    FileNotFoundError
        If the file does not exist.
    """
    target = require_text_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"LunarOps text archive not found: {target}")
    return _open_text(target, "rt")


def open_ascii_source(path: str | Path) -> AbstractContextManager[TextIO]:
    """Open an external ASCII source file, optionally gzipped, with universal newlines.

    The suffix is not checked; names ending in ``.gz`` are decompressed. Decoding is strict ASCII.

    Parameters
    ----------
    path : str or Path
        Source file path.

    Returns
    -------
    AbstractContextManager of TextIO
        Context manager yielding the text stream.
    """
    return _open_text(Path(path), "rt", encoding="ascii", newline=None)


def open_binary_reader(path: str | Path) -> AbstractContextManager[BinaryIO]:
    """Open a native binary archive for reading.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.dat`` or ``.dat.gz``.

    Returns
    -------
    AbstractContextManager of BinaryIO
        Context manager yielding a binary stream, decompressed for ``.gz`` names.

    Raises
    ------
    ValueError
        If the suffix is invalid.
    FileNotFoundError
        If the file does not exist.
    """
    target = require_binary_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"LunarOps binary archive not found: {target}")
    return _open_binary(target, "rb")


def require_file_group_path(path: str | Path) -> Path:
    """Return a user-expanded file-group directory path after checking it has no suffix.

    Parameters
    ----------
    path : str or Path
        Directory path of a file group.

    Returns
    -------
    Path
        The path with ``~`` expanded.

    Raises
    ------
    ValueError
        If the final path component has a suffix.
    """
    target = Path(path).expanduser()
    if target.suffix:
        raise ValueError(f"LunarOps file groups require an extensionless directory: {target}")
    return target


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's bytes.

    Digests are cached by resolved path, size and modification time in nanoseconds.

    Parameters
    ----------
    path : str or Path
        File to hash.

    Returns
    -------
    str
        Lower-case hexadecimal digest.
    """
    path = Path(path).resolve()
    stat = path.stat()
    return _sha256_file(path, stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=4096)
def _sha256_file(path: Path, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_artifact_type(path: str | Path) -> str:
    """Return the artifact type declared in a native text archive's header.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    str
        Artifact type from the ``lunarops <artifactType>`` line.

    Raises
    ------
    ValueError
        If the suffix or the header is invalid.
    FileNotFoundError
        If the file does not exist.
    """
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
    "open_ascii_source",
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
