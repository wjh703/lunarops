"""Minimal writer of synthetic SPK / binary PCK type 2 kernels for tests.

The file has one file record, one summary record (at most 25 segments), one name record and the segment
arrays from record 4 on. Each segment is ``[records..., INIT, INTLEN, RSIZE, N]`` with records
``[MID, RADIUS, X coefficients, Y coefficients, Z coefficients]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FTP = b"FTPSTR:\r:\n:\r\n:\r\x00:\x81:\x10\xce:ENDFTP"


@dataclass
class SyntheticSegment:
    """One type 2 segment: ``coefficients`` has shape (n_records, 3, K), in km (SPK) or rad (PCK)."""

    target: int
    center: int
    coefficients: np.ndarray
    init_s: float
    interval_s: float
    start_s: float | None = None
    end_s: float | None = None
    frame: int = 1
    data_type: int = 2

    def words(self) -> np.ndarray:
        records, components, count = self.coefficients.shape
        assert components == 3
        mids = self.init_s + (np.arange(records) + 0.5) * self.interval_s
        body = np.column_stack(
            [mids, np.full(records, 0.5 * self.interval_s), self.coefficients.reshape(records, 3 * count)]
        )
        return np.concatenate([body.ravel(), [self.init_s, self.interval_s, 2 + 3 * count, records]])


def write_daf(path: Path, segments: list[SyntheticSegment], *, kind: str = "spk", byte_order: str = "<") -> Path:
    """Write a synthetic SPK (``kind="spk"``) or binary PCK (``kind="pck"``) file and return its path."""
    nd, ni = (2, 6) if kind == "spk" else (2, 5)
    summary_doubles = nd + (ni + 1) // 2
    assert len(segments) * summary_doubles + 3 <= 128
    f8, i4 = f"{byte_order}f8", f"{byte_order}i4"
    address = 3 * 128 + 1
    data, summaries = [], []
    for segment in segments:
        words = segment.words()
        first, last = address, address + words.size - 1
        start = segment.init_s if segment.start_s is None else segment.start_s
        end = (
            segment.init_s + segment.coefficients.shape[0] * segment.interval_s
            if segment.end_s is None
            else segment.end_s
        )
        ints = (
            [segment.target, segment.center, segment.frame, segment.data_type, first, last]
            if kind == "spk"
            else [segment.target, segment.frame, segment.data_type, first, last]
        )
        packed = np.asarray([start, end], dtype=f8).tobytes() + np.asarray(ints, dtype=i4).tobytes()
        summaries.append(packed.ljust(8 * summary_doubles, b"\0"))
        data.append(words)
        address = last + 1
    header = bytearray(1024)
    header[0:8] = b"DAF/SPK " if kind == "spk" else b"DAF/PCK "
    header[8:16] = np.asarray([nd, ni], dtype=i4).tobytes()
    header[16:76] = b"synthetic test kernel".ljust(60)
    header[76:88] = np.asarray([2, 2, address], dtype=i4).tobytes()
    header[88:96] = b"LTL-IEEE" if byte_order == "<" else b"BIG-IEEE"
    header[699 : 699 + len(_FTP)] = _FTP
    summary = bytearray(1024)
    summary[0:24] = np.asarray([0.0, 0.0, float(len(segments))], dtype=f8).tobytes()
    blob = b"".join(summaries)
    summary[24 : 24 + len(blob)] = blob
    names = bytearray(b" " * 1024)
    payload = np.concatenate(data).astype(f8).tobytes() if data else b""
    payload = payload.ljust(-(-len(payload) // 1024) * 1024, b"\0")
    path.write_bytes(bytes(header) + bytes(summary) + bytes(names) + payload)
    return path


__all__ = ["SyntheticSegment", "write_daf"]
