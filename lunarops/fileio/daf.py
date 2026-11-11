"""Read-only access to NAIF DAF files: SPK (``.bsp``) and binary PCK (``.bpc``) kernels.

A DAF (Double precision Array File) is a sequence of 1024-byte records. The first record holds the file
identification word (``DAF/SPK`` or ``DAF/PCK``), the summary layout (ND doubles and NI integers per
summary), the address of the first summary record, the binary format (``LTL-IEEE`` or ``BIG-IEEE``) and an
FTP validation string that detects files damaged by text-mode transfer. Summary records form a doubly
linked list; each summary describes one array (segment) by ND doubles and NI integers, the last two
integers being the 1-based word addresses of its first and last double.

SPK summaries have ND = 2 (start and end epoch, TDB seconds past J2000) and NI = 6 (target, centre, frame,
data type, first and last address); binary PCK summaries have ND = 2 and NI = 5 (body frame class ID,
reference frame, data type, first and last address). Data words are read through a read-only memory map
of the whole file, so only the pages of the arrays actually evaluated are loaded.

Reference: NAIF, *DAF Required Reading*, *SPK Required Reading* and *PCK Required Reading* (SPICE
Toolkit N0067).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

RECORD_BYTES = 1024
_FTP_VALIDATION = b"FTPSTR:\r:\n:\r\n:\r\x00:\x81:\x10\xce:ENDFTP"
_FTP_OFFSET = 699
_FORMATS = {b"LTL-IEEE": "<", b"BIG-IEEE": ">"}
_LAYOUTS: dict[str, tuple[int, int]] = {"spk": (2, 6), "pck": (2, 5)}

KernelKind = Literal["spk", "pck"]


@dataclass(frozen=True, slots=True)
class DafSegment:
    """Descriptor of one DAF array of an SPK or binary PCK kernel.

    Attributes
    ----------
    kind : {"spk", "pck"}
        Kernel kind of the file holding the segment.
    target : int
        NAIF ID of the target body (SPK) or body-fixed frame class ID (PCK).
    center : int or None
        NAIF ID of the centre body (SPK); ``None`` for PCK segments.
    frame : int
        NAIF ID of the reference frame (1 is J2000/ICRF).
    data_type : int
        SPK or PCK data type.
    start_s, end_s : float
        Coverage interval in TDB seconds past J2000, both ends included.
    first_word, last_word : int
        1-based word addresses of the first and last double of the array.
    """

    kind: KernelKind
    target: int
    center: int | None
    frame: int
    data_type: int
    start_s: float
    end_s: float
    first_word: int
    last_word: int


class DafFile:
    """Open DAF file: its segment descriptors and a read-only memory map of its data words.

    Parameters
    ----------
    path : str or Path
        SPK (``DAF/SPK``) or binary PCK (``DAF/PCK``) file.

    Attributes
    ----------
    path : Path
        Path of the file.
    kind : {"spk", "pck"}
        Kernel kind from the file identification word.
    byte_order : {"<", ">"}
        Byte order of the binary format.
    segments : tuple of DafSegment
        Segment descriptors in file order (the order of increasing SPICE priority).

    Raises
    ------
    ValueError
        If the file is not an SPK or binary PCK DAF, uses an unsupported binary format, fails the FTP
        validation check, or has an inconsistent summary layout or segment address.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        size = self.path.stat().st_size
        if size < RECORD_BYTES or size % 8:
            raise ValueError(f"Not a DAF file (size {size} bytes): {self.path}")
        with self.path.open("rb") as stream:
            header = stream.read(RECORD_BYTES)
            locidw = header[:8]
            kind = _kind(locidw, self.path)
            byte_order = _FORMATS.get(header[88:96])
            if byte_order is None:
                raise ValueError(f"Unsupported DAF binary format {header[88:96]!r}: {self.path}")
            if header[_FTP_OFFSET : _FTP_OFFSET + len(_FTP_VALIDATION)] != _FTP_VALIDATION:
                raise ValueError(f"DAF FTP validation string is damaged (text-mode transfer?): {self.path}")
            # ND and NI follow the identification word; FWARD follows the 60-character internal file name.
            nd, ni = (int(value) for value in np.frombuffer(header, dtype=f"{byte_order}i4", count=2, offset=8))
            forward = int(np.frombuffer(header, dtype=f"{byte_order}i4", count=1, offset=76)[0])
            if (nd, ni) != _LAYOUTS[kind]:
                raise ValueError(f"Unexpected {kind.upper()} summary layout ND={nd}, NI={ni}: {self.path}")
            self.kind: KernelKind = kind
            self.byte_order = byte_order
            self.segments = tuple(_read_summaries(stream, kind, byte_order, nd, ni, forward, size // 8, self.path))
        self._words: np.ndarray | None = None

    @property
    def words(self) -> np.ndarray:
        """Read-only memory map of the whole file as doubles in the file's byte order; word ``k`` is ``[k - 1]``."""
        if self._words is None:
            self._words = np.memmap(self.path, dtype=f"{self.byte_order}f8", mode="r")
        return self._words

    def segment_words(self, segment: DafSegment) -> np.ndarray:
        """Return the doubles of one segment, in native byte order.

        Parameters
        ----------
        segment : DafSegment
            Segment of this file.

        Returns
        -------
        ndarray, shape (last_word - first_word + 1,)
            Read-only view into the memory map when the file is in native byte order, otherwise a
            byte-swapped copy.
        """
        words = self.words[segment.first_word - 1 : segment.last_word]
        if words.dtype.isnative:
            return np.asarray(words)
        return words.astype(words.dtype.newbyteorder("="))


@dataclass(frozen=True, slots=True)
class ChebyshevLayout:
    """Directory of an SPK/PCK type 2 or 3 segment (fixed-length Chebyshev records).

    Each record is ``[MID, RADIUS, coefficients...]``; record ``i`` covers
    ``[INIT + i*INTLEN, INIT + (i + 1)*INTLEN]`` and the time argument is ``(t - MID) / RADIUS``.

    Attributes
    ----------
    init_s : float
        Start of the first record, TDB seconds past J2000.
    interval_s : float
        Length of every record interval, in seconds.
    record_size : int
        Number of doubles per record, including MID and RADIUS.
    record_count : int
        Number of records.
    coefficient_count : int
        Number of Chebyshev coefficients per component.
    components : int
        Number of fitted components (3 for type 2, 6 for type 3).
    records : ndarray, shape (record_count, record_size)
        Record matrix, native byte order.
    """

    init_s: float
    interval_s: float
    record_size: int
    record_count: int
    coefficient_count: int
    components: int
    records: np.ndarray


def chebyshev_layout(daf: DafFile, segment: DafSegment) -> ChebyshevLayout:
    """Return the record directory of an SPK/PCK type 2 or 3 segment.

    Parameters
    ----------
    daf : DafFile
        File holding the segment.
    segment : DafSegment
        Segment of data type 2 or 3.

    Returns
    -------
    ChebyshevLayout
        Record directory and record matrix.

    Raises
    ------
    ValueError
        If the data type is not 2 or 3 (type 3 only for SPK), or the directory is inconsistent with the
        segment size, the record size or the coverage interval.
    """
    components = {2: 3, 3: 6}.get(segment.data_type)
    if components is None or (segment.kind == "pck" and segment.data_type != 2):
        raise ValueError(f"{segment.kind.upper()} type {segment.data_type} is not a Chebyshev type 2/3 segment.")
    words = daf.segment_words(segment)
    init_s, interval_s, record_size_f, record_count_f = (float(value) for value in words[-4:])
    record_size, record_count = int(record_size_f), int(record_count_f)
    where = f"{daf.path} (target {segment.target}, type {segment.data_type})"
    if record_size != record_size_f or record_count != record_count_f or record_count < 1 or interval_s <= 0.0:
        raise ValueError(f"Invalid Chebyshev segment directory {words[-4:].tolist()} in {where}.")
    coefficient_count, remainder = divmod(record_size - 2, components)
    if remainder or coefficient_count < 1:
        raise ValueError(f"Record size {record_size} does not hold {components} components in {where}.")
    if record_size * record_count + 4 != words.size:
        raise ValueError(
            f"Segment of {words.size} words does not hold {record_count} records of {record_size} in {where}."
        )
    records = words[: record_size * record_count].reshape(record_count, record_size)
    first_mid, first_radius = float(records[0, 0]), float(records[0, 1])
    if (
        abs(first_mid - first_radius - init_s) > 1e-6 * interval_s
        or abs(2.0 * first_radius - interval_s) > 1e-9 * interval_s
    ):
        raise ValueError(f"First record MID/RADIUS disagree with INIT/INTLEN in {where}.")
    if init_s > segment.start_s or init_s + record_count * interval_s < segment.end_s:
        raise ValueError(f"Records do not cover the segment interval in {where}.")
    return ChebyshevLayout(init_s, interval_s, record_size, record_count, coefficient_count, components, records)


def _kind(locidw: bytes, path: Path) -> KernelKind:
    if locidw == b"DAF/SPK ":
        return "spk"
    if locidw == b"DAF/PCK ":
        return "pck"
    raise ValueError(f"Not an SPK or binary PCK DAF (identification word {locidw!r}): {path}")


def _read_summaries(stream, kind: KernelKind, byte_order: str, nd: int, ni: int, record: int, words: int, path: Path):
    summary_doubles = nd + (ni + 1) // 2
    visited: set[int] = set()
    while record:
        if record in visited or record < 2:
            raise ValueError(f"Corrupt DAF summary-record chain at record {record}: {path}")
        visited.add(record)
        stream.seek((record - 1) * RECORD_BYTES)
        raw = stream.read(RECORD_BYTES)
        if len(raw) != RECORD_BYTES:
            raise ValueError(f"Truncated DAF summary record {record}: {path}")
        control = np.frombuffer(raw, dtype=f"{byte_order}f8", count=3)
        next_record, count = int(control[0]), int(control[2])
        for index in range(count):
            offset = 24 + 8 * summary_doubles * index
            doubles = np.frombuffer(raw, dtype=f"{byte_order}f8", count=nd, offset=offset)
            ints = [
                int(value) for value in np.frombuffer(raw, dtype=f"{byte_order}i4", count=ni, offset=offset + 8 * nd)
            ]
            if kind == "spk":
                target, center, frame, data_type, first, last = ints
            else:
                target, frame, data_type, first, last = ints
                center = None
            if not 1 <= first <= last <= words:
                raise ValueError(f"Segment address range {first}..{last} outside the file: {path}")
            yield DafSegment(kind, target, center, frame, data_type, float(doubles[0]), float(doubles[1]), first, last)
        record = next_record


__all__ = ["ChebyshevLayout", "DafFile", "DafSegment", "chebyshev_layout"]
