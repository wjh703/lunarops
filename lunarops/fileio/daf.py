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
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.polynomial import chebyshev

RECORD_BYTES = 1024
_FTP_VALIDATION = b"FTPSTR:\r:\n:\r\n:\r\x00:\x81:\x10\xce:ENDFTP"
_FTP_OFFSET = 699
_FORMATS = {b"LTL-IEEE": "<", b"BIG-IEEE": ">"}
_LAYOUTS: dict[str, tuple[int, int]] = {"spk": (2, 6), "pck": (2, 5)}

_J2000_JD = 2_451_545

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
    """Fixed-length Chebyshev records of an SPK type 2, 3 or 20 or a PCK type 2 or 20 segment.

    Records are normalised to ``[MID, RADIUS, P_x, P_y, P_z, V_x, V_y, V_z]``: MID and RADIUS in TDB
    seconds past J2000 and seconds, ``P_*`` the ``position_count`` coefficients of each position (or
    angle) component in km (rad), and ``V_*`` the ``velocity_count`` coefficients of each velocity (rate)
    component in km/s (rad/s). With ``velocity_count == 0`` (type 2) the velocity is the time derivative
    of the position expansion. Record ``i`` covers ``[INIT + i*INTLEN, INIT + (i + 1)*INTLEN]`` and the time
    argument is ``(t - MID) / RADIUS``.

    Type 2 and 3 records are read in place from the memory map. Type 20 records (velocity coefficients and
    the midpoint position, in DSCALE and DSCALE/TSCALE units, epochs as Julian days) are converted on
    loading: the position coefficients are the integral of the velocity expansion, with the constant set so
    that the position at the midpoint equals the stored midpoint value.

    Attributes
    ----------
    data_type : int
        SPK or PCK data type of the segment.
    init_s : float
        Start of the first record, TDB seconds past J2000.
    interval_s : float
        Length of every record interval, in seconds.
    record_count : int
        Number of records.
    position_count : int
        Chebyshev coefficients per position component.
    velocity_count : int
        Chebyshev coefficients per velocity component; 0 when the velocity is the derivative.
    records : ndarray, shape (record_count, 2 + 3 * (position_count + velocity_count))
        Normalised record matrix, native byte order.
    """

    data_type: int
    init_s: float
    interval_s: float
    record_count: int
    position_count: int
    velocity_count: int
    records: np.ndarray


def chebyshev_layout(daf: DafFile, segment: DafSegment) -> ChebyshevLayout:
    """Return the normalised Chebyshev records of a segment.

    Parameters
    ----------
    daf : DafFile
        File holding the segment.
    segment : DafSegment
        SPK segment of type 2, 3 or 20, or PCK segment of type 2 or 20.

    Returns
    -------
    ChebyshevLayout
        Record directory and normalised record matrix.

    Raises
    ------
    ValueError
        If the data type is not supported, or the directory is inconsistent with the segment size, the
        record size or the coverage interval, or type 20 record epochs are not exact in TDB seconds.
    """
    supported = (2, 3, 20) if segment.kind == "spk" else (2, 20)
    if segment.data_type not in supported:
        raise ValueError(f"{segment.kind.upper()} type {segment.data_type} is not a supported Chebyshev segment.")
    words = daf.segment_words(segment)
    where = f"{daf.path} (target {segment.target}, type {segment.data_type})"
    if segment.data_type == 20:
        layout = _type20_layout(words, where)
    else:
        layout = _type23_layout(words, segment.data_type, where)
    if layout.init_s > segment.start_s or layout.init_s + layout.record_count * layout.interval_s < segment.end_s:
        raise ValueError(f"Records do not cover the segment interval in {where}.")
    return layout


def _directory(words: np.ndarray, count: int, where: str) -> tuple[float, ...]:
    if words.size < count:
        raise ValueError(f"Segment too short for its directory in {where}.")
    return tuple(float(value) for value in words[-count:])


def _sizes(record_size_f: float, record_count_f: float, where: str) -> tuple[int, int]:
    record_size, record_count = int(record_size_f), int(record_count_f)
    if record_size != record_size_f or record_count != record_count_f or record_count < 1 or record_size < 1:
        raise ValueError(f"Invalid record size {record_size_f} or count {record_count_f} in {where}.")
    return record_size, record_count


def _type23_layout(words: np.ndarray, data_type: int, where: str) -> ChebyshevLayout:
    init_s, interval_s, record_size_f, record_count_f = _directory(words, 4, where)
    record_size, record_count = _sizes(record_size_f, record_count_f, where)
    components = 3 if data_type == 2 else 6
    count, remainder = divmod(record_size - 2, components)
    if remainder or count < 1 or interval_s <= 0.0:
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
    return ChebyshevLayout(data_type, init_s, interval_s, record_count, count, 0 if data_type == 2 else count, records)


def _type20_layout(words: np.ndarray, where: str) -> ChebyshevLayout:
    scale, time_scale, init_jd, init_fraction, interval_days, record_size_f, record_count_f = _directory(
        words, 7, where
    )
    record_size, record_count = _sizes(record_size_f, record_count_f, where)
    count, remainder = divmod(record_size, 3)
    count -= 1  # each component holds its velocity coefficients and then the midpoint position
    if remainder or count < 1 or interval_days <= 0.0 or scale <= 0.0 or time_scale <= 0.0:
        raise ValueError(f"Invalid type 20 directory {words[-7:].tolist()} in {where}.")
    if record_size * record_count + 7 != words.size:
        raise ValueError(
            f"Segment of {words.size} words does not hold {record_count} records of {record_size} in {where}."
        )
    # Record epochs must be exact in seconds, or every MID would carry a rounding of up to 0.5 ulp(|MID|).
    init = (Fraction(init_jd) - _J2000_JD + Fraction(init_fraction)) * 86400
    radius = Fraction(interval_days) * 43200
    if init.denominator != 1 or radius.denominator != 1:
        raise ValueError(f"Type 20 record epochs are not whole TDB seconds in {where}.")
    init_s, radius_s = float(init), float(radius)
    raw = words[: record_size * record_count].reshape(record_count, 3, count + 1)
    velocity = raw[:, :, :count] * (scale / time_scale)  # km/s or rad/s
    midpoint = raw[:, :, count] * scale  # km or rad
    # Antiderivative in tau (zero at tau = 0), times dt/dtau = RADIUS, plus the midpoint value.
    position = chebyshev.chebint(velocity, lbnd=0.0, axis=-1) * radius_s
    position[:, :, 0] += midpoint
    records = np.empty((record_count, 2 + 3 * (2 * count + 1)))
    records[:, 0] = init_s + radius_s * (2.0 * np.arange(record_count) + 1.0)
    records[:, 1] = radius_s
    records[:, 2 : 2 + 3 * (count + 1)] = position.reshape(record_count, -1)
    records[:, 2 + 3 * (count + 1) :] = velocity.reshape(record_count, -1)
    return ChebyshevLayout(20, init_s, 2.0 * radius_s, record_count, count + 1, count, records)


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
