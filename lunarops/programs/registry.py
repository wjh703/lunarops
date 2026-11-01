"""Each program declares its configuration and input/output artifact types."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import cast

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


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    summary: str
    inputs: tuple[ArtifactSlot, ...] = ()
    outputs: tuple[ArtifactSlot, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    validator: SchemaValidator | None = None
    _schema: ConfigSchema = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
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
    def schema(self) -> ConfigSchema:
        return self._schema


_PROGRAMS: dict[str, tuple[ProgramSpec, ProgramFunc]] = {}
_PROGRAM_MODULES = (
    "lunarops.programs.lunar_orbit",
    "lunarops.programs.mass_catalog_create",
    "lunarops.programs.earth_orientation",
    "lunarops.programs.llr_observation_prediction",
    "lunarops.programs.llr_observation_prediction_merge",
    "lunarops.programs.llr_processing",
    "lunarops.programs.llr_residuals",
    "lunarops.programs.normal_points_convert",
    "lunarops.programs.catalog_create",
)
_BUILTINS_REGISTERED = False


def ensure_builtin_programs() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for module_name in _PROGRAM_MODULES:
        importlib.import_module(module_name)
    _BUILTINS_REGISTERED = True


def program(spec: ProgramSpec):

    def _wrap(func: ProgramFunc) -> ProgramFunc:
        _PROGRAMS[spec.name] = (spec, func)
        return func

    return _wrap


def get_program(name: str) -> tuple[ProgramSpec, ProgramFunc]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Program names must be non-empty strings.")
    key = name
    entry = _PROGRAMS.get(key)
    if entry is not None:
        return entry
    available = sorted(
        (registered[0].name for registered in _PROGRAMS.values()),
        key=str.casefold,
    )
    raise KeyError(f"Unknown program {name!r}. Available: {available}")


def program_specs() -> tuple[ProgramSpec, ...]:
    specs = tuple(entry[0] for entry in _PROGRAMS.values())
    return tuple(sorted(specs, key=lambda item: item.name.casefold()))


def resolve_program_config(name: str, config: Mapping[str, object]) -> dict[str, object]:
    spec = get_program(name)[0]
    return spec.schema.resolve(config, path=spec.name)


def validate_program_artifacts(
    name: str,
    config: Mapping[str, object],
    context: RunContext,
    *,
    require_inputs: bool = True,
    available_artifacts: Mapping[Path, str] | None = None,
) -> dict[str, object]:
    from lunarops.fileio.archive import is_binary_path, is_text_path, read_artifact_type
    from lunarops.fileio.numeric_table import read_numeric_table_type

    spec = get_program(name)[0]
    config = resolve_program_config(name, config)
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
        for raw_path in cast(Sequence[object], value) if slot.many else [value]:
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
                "ExternalStationCoordinatesFile",
                "ExternalReflectorCoordinatesFile",
                "ExternalEarthOrientationParameterFile",
            }:
                continue
            binary = slot.artifact_type in {"lunarAccelerationDiagnostics", "lunarOrbit", "lunarRotation"}
            if not (is_binary_path(path) if binary else is_text_path(path)):
                extension = ".dat or .dat.gz" if binary else ".txt or .txt.gz"
                raise ValueError(f"{spec.name}.{slot.key} must use {extension}: {path}")
            if is_input and require_inputs:
                actual = read_numeric_table_type(path) if binary else read_artifact_type(path)
                if actual != slot.artifact_type:
                    raise ValueError(f"{spec.name}.{slot.key} expects {slot.artifact_type!r}, found {actual!r}: {path}")

    return dict(config)


def run_program(name: str, config: Mapping[str, object], context: RunContext):
    entry = get_program(name)
    resolved = resolve_program_config(entry[0].name, config)
    return entry[1](resolved, context)


def available_programs() -> list[str]:
    return [spec.name for spec in program_specs()]


__all__ = [
    "ArtifactSlot",
    "ProgramSpec",
    "available_programs",
    "ensure_builtin_programs",
    "get_program",
    "program",
    "program_specs",
    "resolve_program_config",
    "run_program",
    "validate_program_artifacts",
]
