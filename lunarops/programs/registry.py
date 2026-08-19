"""Declarative program and artifact registry.

The registry is deliberately strict.  A program owns a small, typed contract
and its callable is only entered after the contract has been checked.  This
keeps YAML scenario files inspectable in the same way as GROOPS program
chains, while leaving the scientific implementation in ordinary Python
functions.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from threading import RLock
from typing import Callable, Dict, Mapping, Sequence

from lunarops.config.context import RunContext
from lunarops.config.schema import ConfigSchema, FieldSpec, SchemaValidator

ProgramFunc = Callable[[dict, RunContext], object]


@dataclass(frozen=True, slots=True)
class ArtifactSlot:
    key: str
    artifact_type: str
    many: bool = False
    required: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("Artifact slot keys must be non-empty strings.")
        if not isinstance(self.artifact_type, str) or not self.artifact_type.strip():
            raise ValueError(f"Artifact slot {self.key!r} needs an artifact type.")
        if not isinstance(self.many, bool) or not isinstance(self.required, bool):
            raise TypeError(f"Artifact slot {self.key!r} many/required flags must be booleans.")
        if not isinstance(self.description, str):
            raise TypeError(f"Artifact slot {self.key!r} description must be a string.")
        object.__setattr__(self, "key", self.key.strip())
        object.__setattr__(self, "artifact_type", self.artifact_type.strip())


@dataclass(frozen=True, slots=True)
class ProgramSpec:
    name: str
    summary: str
    inputs: tuple[ArtifactSlot, ...] = ()
    outputs: tuple[ArtifactSlot, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    validator: SchemaValidator | None = None
    _schema: ConfigSchema = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "fields", tuple(self.fields))
        if any(not isinstance(slot, ArtifactSlot) for slot in self.slots):
            raise TypeError(f"Program {self.name!r} artifact slots must be ArtifactSlot instances.")
        if any(not isinstance(field, FieldSpec) for field in self.fields):
            raise TypeError(f"Program {self.name!r} fields must be FieldSpec instances.")
        all_slots = (*self.inputs, *self.outputs)
        keys = [slot.key for slot in all_slots]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Program {self.name} declares duplicate artifact keys.")
        field_names = [field.name for field in self.fields]
        if len(set(field_names)) != len(field_names):
            raise ValueError(f"Program {self.name} declares duplicate schema fields: {field_names}")
        if set(field_names) & set(keys):
            raise ValueError(f"Program {self.name} schema fields overlap artifact keys.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Program names must not be empty.")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError(f"Program {self.name} needs a summary.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "summary", self.summary.strip())
        if self.validator is not None and not callable(self.validator):
            raise TypeError(f"Program {self.name} validator must be callable or None.")

        config_fields: list[FieldSpec] = []
        for slot in self.slots:
            if slot.many:
                config_fields.append(
                    FieldSpec(
                        name=slot.key,
                        kind="sequence",
                        required=slot.required,
                        item_kind="path",
                        min_items=1,
                        non_empty=True,
                        allow_none=not slot.required,
                        description=slot.description,
                    )
                )
            else:
                config_fields.append(
                    FieldSpec(
                        name=slot.key,
                        kind="path",
                        required=slot.required,
                        non_empty=True,
                        allow_none=not slot.required,
                        description=slot.description,
                    )
                )
        config_fields.extend(self.fields)
        object.__setattr__(
            self,
            "_schema",
            ConfigSchema(tuple(config_fields), description=self.summary, validator=self.validator),
        )

    @property
    def slots(self) -> tuple[ArtifactSlot, ...]:
        return (*self.inputs, *self.outputs)

    @property
    def allowed_keys(self) -> frozenset[str]:
        return self.schema.allowed_keys

    @property
    def schema(self) -> ConfigSchema:
        return self._schema

    def describe(self) -> dict[str, object]:
        def slot_data(slot: ArtifactSlot) -> dict[str, object]:
            return {
                "key": slot.key,
                "artifactType": slot.artifact_type,
                "many": slot.many,
                "required": slot.required,
                "description": slot.description,
            }

        return {
            "name": self.name,
            "summary": self.summary,
            "inputs": [slot_data(slot) for slot in self.inputs],
            "outputs": [slot_data(slot) for slot in self.outputs],
            "configuration": self.schema.describe(),
        }

    def json_schema(self) -> dict[str, object]:
        schema = self.schema.json_schema()
        schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "title": self.name})
        return schema


@dataclass(frozen=True, slots=True)
class RegisteredProgram:
    spec: ProgramSpec
    function: ProgramFunc


_PROGRAMS: Dict[str, RegisteredProgram] = {}
_PROGRAM_MODULES = (
    "lunarops.programs.llr_processing",
    "lunarops.programs.llr_residuals",
    "lunarops.programs.normal_points_convert",
    "lunarops.programs.reflector_catalog_create",
)
_PROGRAM_REGISTRY_LOCK = RLock()
_BUILTINS_REGISTERED = False


@contextmanager
def program_registration_transaction():
    """Roll back a program import batch when one declaration fails."""
    with _PROGRAM_REGISTRY_LOCK:
        snapshot = _PROGRAMS.copy()
        try:
            yield
        except Exception:
            _PROGRAMS.clear()
            _PROGRAMS.update(snapshot)
            raise


def ensure_builtin_programs() -> None:
    """Import the built-in program modules exactly once.

    Program modules register through decorators.  Keeping the import boundary
    here makes CLI commands, library callers, and MPI master setup share the
    same lifecycle and makes repeated discovery harmless.
    """
    global _BUILTINS_REGISTERED
    with _PROGRAM_REGISTRY_LOCK:
        if _BUILTINS_REGISTERED:
            return
        missing = object()
        previous_modules = {name: sys.modules.get(name, missing) for name in _PROGRAM_MODULES}
        try:
            with program_registration_transaction():
                for module_name in _PROGRAM_MODULES:
                    importlib.import_module(module_name)
        except Exception:
            # A failed import can leave earlier modules cached even though the
            # registry transaction removed their declarations.  Remove only
            # modules that this discovery attempt introduced so a later retry
            # executes their decorators again.
            for module_name, previous in previous_modules.items():
                if previous is missing:
                    sys.modules.pop(module_name, None)
            raise
        _BUILTINS_REGISTERED = True


def program(
    spec_or_name: ProgramSpec | str,
    *,
    summary: str | None = None,
    inputs: Sequence[ArtifactSlot] = (),
    outputs: Sequence[ArtifactSlot] = (),
    fields: Sequence[FieldSpec] = (),
    validator: SchemaValidator | None = None,
):
    """Register a callable with a :class:`ProgramSpec`.

    ``@program(ProgramSpec(...))`` is the preferred spelling.  The keyword
    form is retained solely to keep declarations compact in program modules;
    it still creates a complete strict spec.
    """
    if isinstance(spec_or_name, ProgramSpec):
        spec = spec_or_name
    else:
        spec = ProgramSpec(
            name=spec_or_name,
            summary=summary or "",
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            fields=tuple(fields),
            validator=validator,
        )

    def _wrap(func: ProgramFunc) -> ProgramFunc:
        key = spec.name.casefold()
        with _PROGRAM_REGISTRY_LOCK:
            if key in _PROGRAMS:
                raise RuntimeError(f"Program {spec.name!r} is already registered.")
            _PROGRAMS[key] = RegisteredProgram(spec, func)
        setattr(func, "program_name", spec.name)
        setattr(func, "program_spec", spec)
        return func

    return _wrap


def get_program(name: str) -> RegisteredProgram:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Program names must be non-empty strings.")
    key = name.strip().casefold()
    with _PROGRAM_REGISTRY_LOCK:
        entry = _PROGRAMS.get(key)
        if entry is not None:
            return entry
        available = sorted(
            (registered.spec.name for registered in _PROGRAMS.values()),
            key=str.casefold,
        )
    raise KeyError(f"Unknown program {name!r}. Available: {available}")


def program_specs() -> tuple[ProgramSpec, ...]:
    with _PROGRAM_REGISTRY_LOCK:
        specs = tuple(entry.spec for entry in _PROGRAMS.values())
    return tuple(sorted(specs, key=lambda item: item.name.casefold()))


def resolve_program_config(name: str, config: Mapping[str, object]) -> dict[str, object]:
    entry = get_program(name)
    spec = entry.spec
    if not isinstance(config, Mapping):
        raise TypeError(f"Program {spec.name} configuration must be a mapping.")
    resolved = spec.schema.resolve(config, path=spec.name)
    return spec.schema.resolve_classes(resolved, path=spec.name)


def validate_program_config(name: str, config: Mapping[str, object]) -> dict[str, object]:
    """Resolve defaults and class choices for one program configuration."""
    return resolve_program_config(name, config)


_TEXT_ARTIFACT_HEADERS = {
    "NormalPointFile": "normalPoint",
    "ObservationResultFile": "observationResult",
    "ProcessingStateFile": "processingState",
    "ImportReportFile": "normalPointImportReport",
    "ReflectorCatalogFile": "reflectorCatalog",
}


def _slot_values(slot: ArtifactSlot, value: object) -> list[object]:
    if not slot.many:
        return [value]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Artifact slot {slot.key} must contain a sequence of paths.")
    return list(value)


def _validate_program_artifacts_resolved(
    name: str,
    config: Mapping[str, object],
    context: RunContext,
    *,
    require_inputs: bool = True,
    available_artifacts: Mapping[Path, str] | None = None,
) -> None:
    """Validate one already-resolved program config against the artifact graph."""
    from lunarops.fileio.archive import is_text_path, read_artifact_type

    spec = get_program(name).spec
    available = {
        context.resolve_path(path).resolve(): artifact_type
        for path, artifact_type in (available_artifacts or {}).items()
    }
    input_keys = {slot.key for slot in spec.inputs}
    resolved_slots: dict[Path, str] = {}
    for slot in spec.slots:
        value = config.get(slot.key)
        if value is None:
            continue
        for raw_path in _slot_values(slot, value):
            path = context.resolve_path(raw_path)
            resolved = path.resolve()
            if resolved in resolved_slots:
                raise ValueError(f"{spec.name} reuses path {path} in both {resolved_slots[resolved]} and {slot.key}.")
            resolved_slots[resolved] = slot.key
            is_input = slot.key in input_keys
            generated_type = available.get(resolved)
            if is_input and generated_type is not None:
                if generated_type != slot.artifact_type:
                    raise ValueError(
                        f"{spec.name}.{slot.key} expects {slot.artifact_type}, but an earlier "
                        f"program produces {generated_type}: {path}"
                    )
                continue
            if is_input and require_inputs and not path.exists():
                raise FileNotFoundError(f"{spec.name}.{slot.key} does not exist: {path}")
            if slot.artifact_type in {
                "ExternalNormalPointFile",
                "ExternalReflectorCoordinatesFile",
            }:
                continue
            if slot.artifact_type in _TEXT_ARTIFACT_HEADERS:
                if not is_text_path(path):
                    raise ValueError(f"{spec.name}.{slot.key} must use .txt or .txt.gz: {path}")
                expected = _TEXT_ARTIFACT_HEADERS[slot.artifact_type]
                if is_input and require_inputs:
                    actual = read_artifact_type(path)
                    if expected is not None and actual != expected:
                        raise ValueError(f"{spec.name}.{slot.key} expects {expected!r}, found {actual!r}: {path}")
                continue
            raise RuntimeError(f"Program {spec.name} declares unknown artifact type {slot.artifact_type!r}.")
    return None


def validate_program_artifacts(
    name: str,
    config: Mapping[str, object],
    context: RunContext,
    *,
    require_inputs: bool = True,
    available_artifacts: Mapping[Path, str] | None = None,
) -> dict[str, object]:
    """Resolve and validate paths/types without running a program."""
    spec = get_program(name).spec
    resolved = resolve_program_config(spec.name, config)
    _validate_program_artifacts_resolved(
        spec.name,
        resolved,
        context,
        require_inputs=require_inputs,
        available_artifacts=available_artifacts,
    )
    return resolved


def run_program(name: str, config: Mapping[str, object], context: RunContext):
    entry = get_program(name)
    resolved = resolve_program_config(entry.spec.name, config)
    _validate_program_artifacts_resolved(
        entry.spec.name,
        resolved,
        context,
        require_inputs=True,
        available_artifacts=None,
    )
    return entry.function(resolved, context)


def available_programs() -> list[str]:
    return [spec.name for spec in program_specs()]


__all__ = [
    "ArtifactSlot",
    "ProgramSpec",
    "RegisteredProgram",
    "available_programs",
    "ensure_builtin_programs",
    "get_program",
    "program",
    "program_registration_transaction",
    "program_specs",
    "resolve_program_config",
    "run_program",
    "validate_program_config",
    "validate_program_artifacts",
]
