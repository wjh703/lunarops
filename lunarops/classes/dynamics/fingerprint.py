"""Fingerprint of everything that determines the right-hand side of a lunar propagation.

:func:`model_fingerprint` hashes the model parameters of a
:class:`~lunarops.classes.dynamics.lunar_dynamics.LunarDynamics`, the integrator settings and the identity of
the ephemeris files into one SHA-256. Two builds of a propagation problem with the same fingerprint integrate
identically; the backward-arc process (:mod:`~lunarops.classes.dynamics.arc_process`) uses it to check that it
built the problem of the process that seeds it.

The hash is taken over a canonical byte encoding of plain values, not over pickles: every value is written
with a type tag, floats as their IEEE-754 little-endian bytes, arrays as dtype, shape and contiguous
little-endian bytes, mappings in sorted key order, records by their declared parameters. Caches and compiled
state (the native kernel, prepared step data, ephemeris segment caches) are never read, so a system that has
already evaluated nodes has the fingerprint of a freshly built one.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import struct
import types
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import numpy as np

from lunarops.classes.ephemerides.kernels import kernel_paths

from .force_models import LunarForceGroup
from .gravity import GravityCoefficients
from .solar_environment import KuiperBeltRing

if TYPE_CHECKING:
    from .integrators import IntegratorSettings
    from .lunar_dynamics import LunarDynamics

FINGERPRINT_VERSION = 1
"""Version of the encoding; changing the encoding changes every fingerprint."""

_FILE_SAMPLE_BYTES = 65536
"""Bytes hashed at the start and at the end of each ephemeris file."""


def model_fingerprint(system: LunarDynamics, settings: IntegratorSettings) -> str:
    """Return the fingerprint of a propagation problem.

    It covers, for the right-hand side: the bodies and their GMs, the solar GM drift, the Kuiper-belt ring
    (its parameters and prepropagated trajectories), the Earth and Moon axes providers, every active force
    model with its parameters (tide Love numbers and delays, gravity-field coefficients, PPN and radiation
    flags, Earth J2 polynomial, degree-2 deformation), the integrated rotation (inertia, core, torques,
    geodetic precession, deformation with k2 and delay); the integrator settings; and the ephemeris: its
    type, scale convention, lunar-orientation correction and, for each kernel file, its name, size and a
    SHA-256 of its first and last 64 KiB.

    Parameters
    ----------
    system : LunarDynamics
        Dynamics, freshly built or already evaluated.
    settings : IntegratorSettings
        Integrator settings.

    Returns
    -------
    str
        Hexadecimal SHA-256.

    Raises
    ------
    TypeError
        If a model holds a value the encoding does not know, so that no parameter is silently ignored.

    Notes
    -----
    Functions (orientation providers, for example) are identified by their qualified name; what they read
    from the ephemeris is covered by the ephemeris identity.
    """
    encoder = _Encoder(system.ephemeris)
    encoder.encode(
        (
            "lunarops.model_fingerprint",
            FINGERPRINT_VERSION,
            system.epoch_inputs,
            system.force_group,
            system.rotation,
            settings,
            ephemeris_identity(system.ephemeris),
        )
    )
    return encoder.hexdigest()


def ephemeris_identity(ephemeris) -> tuple:
    """Return the identity of an ephemeris as plain values.

    Parameters
    ----------
    ephemeris : Ephemeris
        Ephemeris.

    Returns
    -------
    tuple
        Type name, lunar coordinate scale, lunar-orientation correction and, when ``source_path`` names
        existing files (a kernel directory, see
        :func:`~lunarops.classes.ephemerides.kernels.kernel_paths`, or one file), each file's name, size
        and SHA-256 of its first and last 64 KiB. A source that is not a file contributes its path only.
    """
    source = ephemeris.source_path
    files: tuple = ()
    if source is not None:
        path = Path(source)
        listed = kernel_paths(path) if path.is_dir() else (path,) if path.is_file() else ()
        files = tuple(_file_identity(item) for item in listed) if listed else (str(source),)
    orientation = getattr(ephemeris, "lunar_orientation", None)
    return (
        _type_name(type(ephemeris)),
        getattr(ephemeris, "relativistic_scale", None),
        getattr(orientation, "correction", None),
        files,
    )


def _file_identity(path: Path) -> tuple[str, int, str]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(_FILE_SAMPLE_BYTES))
        if size > _FILE_SAMPLE_BYTES:
            stream.seek(max(size - _FILE_SAMPLE_BYTES, _FILE_SAMPLE_BYTES))
            digest.update(stream.read())
    return path.name, size, digest.hexdigest()


def _type_name(kind: type) -> str:
    return f"{kind.__module__}.{kind.__qualname__}"


def _ring_parameters(ring: KuiperBeltRing) -> tuple:
    # The prepropagated trajectories depend on the covered interval: their step breakpoints identify them.
    solutions = tuple(
        None if solution is None else solution.ts for solution in (ring._solution, ring._backward_solution)
    )
    return (_record_fields(ring), ring._initial_states, solutions)


def _record_fields(value) -> tuple:
    return tuple((field.name, getattr(value, field.name)) for field in dataclasses.fields(value) if field.compare)


_PARAMETERS = {
    # The active models, not the configured ones: inactive models do not enter the right-hand side.
    LunarForceGroup: lambda group: group.active_models,
    KuiperBeltRing: _ring_parameters,
    # Not the name and file provenance: the same coefficients loaded from another path are the same field.
    GravityCoefficients: lambda coefficients: (coefficients.coeffs, coefficients.gm_m3_s2, coefficients.radius_m),
}
"""Parameters of the types whose declared fields or public attributes do not describe them."""


class _Encoder:
    """Canonical byte encoding of nested plain values, fed into SHA-256."""

    def __init__(self, ephemeris) -> None:
        self._digest = hashlib.sha256()
        self._ephemeris = ephemeris
        self._stack: set[int] = set()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def _write(self, tag: bytes, payload: bytes = b"") -> None:
        self._digest.update(tag + struct.pack("<Q", len(payload)) + payload)

    def _text(self, tag: bytes, text: str) -> None:
        self._write(tag, text.encode())

    def encode(self, value: Any) -> None:
        if value is None:
            self._write(b"N")
        elif isinstance(value, bool | np.bool_):
            self._write(b"B", b"\x01" if value else b"\x00")
        elif isinstance(value, enum.Enum):
            self._text(b"E", f"{_type_name(type(value))}.{value.name}")
        elif isinstance(value, int | np.integer):
            self._text(b"I", str(int(value)))
        elif isinstance(value, float | np.floating):
            self._write(b"F", struct.pack("<d", float(value)))
        elif isinstance(value, complex | np.complexfloating):
            self._write(b"C", struct.pack("<dd", value.real, value.imag))
        elif isinstance(value, str):
            self._text(b"S", value)
        elif isinstance(value, bytes):
            self._write(b"Y", value)
        elif isinstance(value, PurePath):
            self._text(b"P", str(value))
        elif isinstance(value, np.ndarray):
            self._array(value)
        elif value is self._ephemeris:
            # Identified once, by ephemeris_identity.
            self._write(b"H")
        elif isinstance(value, types.FunctionType | types.BuiltinFunctionType | type):
            self._text(b"G", f"{getattr(value, '__module__', '')}.{value.__qualname__}")
        elif isinstance(value, types.MethodType):
            self._text(b"M", f"{_type_name(type(value.__self__))}.{value.__func__.__qualname__}")
        else:
            self._container(value)

    def _array(self, value: np.ndarray) -> None:
        if value.dtype.hasobject:
            self._sequence(b"O", value.ravel().tolist(), suffix=str(value.shape))
            return
        data = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
        self._text(b"D", f"{data.dtype.str}{data.shape}")
        self._write(b"A", data.tobytes())

    def _sequence(self, tag: bytes, items, suffix: str = "") -> None:
        items = list(items)
        self._text(tag, f"{len(items)}{suffix}")
        for item in items:
            self.encode(item)

    def _container(self, value: Any) -> None:
        key = id(value)
        if key in self._stack:
            raise TypeError(f"Cannot fingerprint a model that refers to itself: {_type_name(type(value))}")
        self._stack.add(key)
        try:
            self._compound(value)
        finally:
            self._stack.discard(key)

    def _compound(self, value: Any) -> None:
        if isinstance(value, tuple | list):
            self._sequence(b"T", value)
        elif isinstance(value, Mapping):
            items = sorted(value.items(), key=lambda item: repr(item[0]))
            self._sequence(b"K", (pair for item in items for pair in item))
        elif isinstance(value, functools.partial):
            self._text(b"Q", "partial")
            self.encode((value.func, value.args, value.keywords))
        else:
            kind = type(value)
            parameters = _PARAMETERS.get(kind)
            if parameters is not None:
                described = parameters(value)
            elif dataclasses.is_dataclass(value):
                described = _record_fields(value)
            elif hasattr(value, "__dict__"):
                # Public attributes only: private ones are caches or derived from the public ones.
                public = [(name, item) for name, item in vars(value).items() if not name.startswith("_")]
                described = tuple(sorted(public, key=lambda pair: pair[0]))
            else:
                raise TypeError(f"Cannot fingerprint a value of type {_type_name(kind)}")
            self._text(b"R", _type_name(kind))
            self.encode(described)


__all__ = ["FINGERPRINT_VERSION", "ephemeris_identity", "model_fingerprint"]
