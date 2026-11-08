"""Registry of LunarOps programs and their artifact contracts.

Each program declares a ``ProgramSpec``: its name, a one-line summary, its config schema and the artifact
slots it reads and writes. An artifact slot is a config key holding one or more file paths, together with
the artifact type the file must have. Programs register with the ``@program`` decorator.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lunarops.config.context import RunContext
from lunarops.config.schema import ConfigSchema, FieldSpec, SchemaValidator

#: Program function called as ``func(config, context)`` with the resolved program config.
ProgramFunc = Callable[[dict, RunContext], object]
#: Returns (path, artifact type) of the output files a resolved config names outside the output slots.
NestedOutputs = Callable[[Mapping[str, object]], Iterable[tuple[object, str]]]


@dataclass(frozen=True, slots=True)
class ArtifactSlot:
    """Input or output file slot of a program.

    Attributes
    ----------
    key : str
        Config key holding the path (or the list of paths).
    artifact_type : str
        Artifact type recorded in the file header, checked for inputs.
    many : bool
        Whether the key holds a non-empty list of paths instead of one path.
    required : bool
        Whether the key must be given.
    description : str
        Description of the config key.
    binary : bool
        Whether the file is a binary numeric table (``.dat`` or ``.dat.gz``) rather than a text artifact
        (``.txt`` or ``.txt.gz``).
    external : bool
        Whether the file is in a format not written by LunarOps; its suffix and type are not checked.
    """

    key: str
    artifact_type: str
    many: bool = False
    required: bool = True
    description: str = ""
    binary: bool = False
    external: bool = False


@dataclass(frozen=True)
class ProgramSpec:
    """Declarative contract of a program.

    Attributes
    ----------
    name : str
        Program name used in the ``program`` entry of a run config.
    summary : str
        One-line summary of the program.
    schema : ConfigSchema
        Config schema, including one field per artifact slot.
    inputs : tuple of ArtifactSlot
        Files read by the program.
    outputs : tuple of ArtifactSlot
        Files written by the program.
    nested_outputs : callable or None
        Lists the output files named inside nested config entries, which have no artifact slot.
    """

    name: str
    summary: str
    schema: ConfigSchema
    inputs: tuple[ArtifactSlot, ...]
    outputs: tuple[ArtifactSlot, ...]
    nested_outputs: NestedOutputs | None = None

    @property
    def slots(self) -> tuple[ArtifactSlot, ...]:
        """Input slots followed by output slots."""
        return (*self.inputs, *self.outputs)


def program_spec(
    name: str,
    summary: str,
    inputs: tuple[ArtifactSlot, ...] = (),
    outputs: tuple[ArtifactSlot, ...] = (),
    fields: tuple[FieldSpec, ...] = (),
    validator: SchemaValidator | None = None,
    nested_outputs: NestedOutputs | None = None,
) -> ProgramSpec:
    """Build a program spec whose schema has one path field per artifact slot followed by ``fields``.

    Parameters
    ----------
    name : str
        Program name.
    summary : str
        One-line summary, also used as the schema description.
    inputs : tuple of ArtifactSlot, optional
        Files read by the program.
    outputs : tuple of ArtifactSlot, optional
        Files written by the program.
    fields : tuple of FieldSpec, optional
        Other config fields.
    validator : SchemaValidator, optional
        Cross-field check of the program config.
    nested_outputs : callable, optional
        Lists the output files named inside nested config entries.

    Returns
    -------
    ProgramSpec
        The program spec.
    """
    config_fields = tuple(
        FieldSpec(
            name=slot.key,
            kind="sequence" if slot.many else "path",
            required=slot.required,
            item_kind="path" if slot.many else None,
            min_items=1 if slot.many else None,
            non_empty=True,
            allow_none=not slot.required,
            description=slot.description,
        )
        for slot in (*inputs, *outputs)
    )
    return ProgramSpec(
        name,
        summary,
        ConfigSchema((*config_fields, *fields), description=summary, validator=validator),
        inputs,
        outputs,
        nested_outputs,
    )


_PROGRAMS: dict[str, tuple[ProgramSpec, ProgramFunc]] = {}
_PROGRAM_MODULES = (
    "lunarops.programs.lunar_orbit",
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
    """Import the built-in program modules once, which registers their programs."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for module_name in _PROGRAM_MODULES:
        importlib.import_module(module_name)
    _BUILTINS_REGISTERED = True


def program(spec: ProgramSpec):
    """Return a decorator that registers a program function under ``spec.name``.

    Parameters
    ----------
    spec : ProgramSpec
        Contract of the program; a later registration under the same name replaces an earlier one.

    Returns
    -------
    callable
        Decorator that registers the function and returns it unchanged.
    """

    def _wrap(func: ProgramFunc) -> ProgramFunc:
        _PROGRAMS[spec.name] = (spec, func)
        return func

    return _wrap


def get_program(name: str) -> tuple[ProgramSpec, ProgramFunc]:
    """Return the spec and function of a registered program.

    Parameters
    ----------
    name : str
        Program name, exactly as registered.

    Returns
    -------
    tuple of (ProgramSpec, callable)
        Spec and program function.

    Raises
    ------
    KeyError
        If no program has this name.
    """
    entry = _PROGRAMS.get(name)
    if entry is not None:
        return entry
    available = sorted(
        (registered[0].name for registered in _PROGRAMS.values()),
        key=str.casefold,
    )
    raise KeyError(f"Unknown program {name!r}. Available: {available}")


def program_specs() -> tuple[ProgramSpec, ...]:
    """Return the specs of the registered programs, sorted by name ignoring case."""
    specs = tuple(entry[0] for entry in _PROGRAMS.values())
    return tuple(sorted(specs, key=lambda item: item.name.casefold()))


def resolve_program_config(name: str, config: Mapping[str, object]) -> dict[str, object]:
    """Validate a program config against the program's schema and fill in defaults.

    Parameters
    ----------
    name : str
        Program name.
    config : Mapping
        Program config of one call.

    Returns
    -------
    dict
        Resolved program config.
    """
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
    """Resolve a program config and check its artifact slots.

    No path may appear in two slots. An input path produced by an earlier call (``available_artifacts``)
    must have the slot's artifact type. Other files must use the slot's suffix (``.txt``/``.txt.gz`` for text
    artifacts, ``.dat``/``.dat.gz`` for binary tables), and, with ``require_inputs``, input files must exist
    and carry the slot's artifact type. The suffix and type of files in external slots are not checked.

    Parameters
    ----------
    name : str
        Program name.
    config : Mapping
        Program config of one call.
    context : RunContext
        Run context used to resolve relative paths.
    require_inputs : bool, optional
        Require input files to exist and check their artifact type. Default True.
    available_artifacts : Mapping of Path to str, optional
        Artifact types of the files written by earlier program calls, keyed by path.

    Returns
    -------
    dict
        Resolved program config.

    Raises
    ------
    ValueError
        If a path is reused, has the wrong suffix or the wrong artifact type.
    FileNotFoundError
        If a required input file does not exist.
    """
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
            if slot.external:
                continue
            binary = slot.binary
            if not (is_binary_path(path) if binary else is_text_path(path)):
                extension = ".dat or .dat.gz" if binary else ".txt or .txt.gz"
                raise ValueError(f"{spec.name}.{slot.key} must use {extension}: {path}")
            if is_input and require_inputs:
                actual = read_numeric_table_type(path) if binary else read_artifact_type(path)
                if actual != slot.artifact_type:
                    raise ValueError(f"{spec.name}.{slot.key} expects {slot.artifact_type!r}, found {actual!r}: {path}")

    return dict(config)


def program_outputs(name: str, config: Mapping[str, object]) -> list[tuple[object, str]]:
    """Return the files a program call writes.

    Parameters
    ----------
    name : str
        Program name.
    config : Mapping
        Resolved program config.

    Returns
    -------
    list of (path, str)
        Path as given in the config and artifact type of every output file.
    """
    spec = get_program(name)[0]
    outputs: list[tuple[object, str]] = []
    for slot in spec.outputs:
        value = config.get(slot.key)
        if value is not None:
            paths = cast(Sequence[object], value) if slot.many else [value]
            outputs.extend((path, slot.artifact_type) for path in paths)
    if spec.nested_outputs is not None:
        outputs.extend(spec.nested_outputs(config))
    return outputs


def validate_program_calls(
    calls: Iterable[tuple[str, Mapping[str, object]]],
    context: RunContext,
    *,
    require_inputs: bool = True,
) -> list[tuple[str, dict[str, object]]]:
    """Validate the program calls of a run in order, without running them.

    Every call is checked against its schema and artifact contract. An input may be an output of an earlier
    call, and no two calls may write the same file.

    Parameters
    ----------
    calls : iterable of (str, Mapping)
        Program name and program config of each call, in run order.
    context : RunContext
        Run context used to resolve relative paths.
    require_inputs : bool, optional
        Require input files that no earlier call writes to exist. Default True.

    Returns
    -------
    list of (str, dict)
        Program name and resolved program config of each call.

    Raises
    ------
    ValueError
        If a call violates its contract, or two calls write the same file.
    FileNotFoundError
        If a required input file does not exist.
    """
    produced: dict[Path, str] = {}
    resolved_calls: list[tuple[str, dict[str, object]]] = []
    for name, config in calls:
        resolved = validate_program_artifacts(
            name, config, context, require_inputs=require_inputs, available_artifacts=produced
        )
        resolved_calls.append((name, resolved))
        for path, artifact_type in program_outputs(name, resolved):
            output = context.resolve_path(path).resolve()
            if output in produced:
                raise ValueError(f"Scenario publishes more than one artifact to {output}.")
            produced[output] = artifact_type
    return resolved_calls


def run_program(name: str, config: Mapping[str, object], context: RunContext):
    """Resolve a program config and run the program.

    Parameters
    ----------
    name : str
        Program name.
    config : Mapping
        Program config of one call.
    context : RunContext
        Run context of the run.

    Returns
    -------
    object
        Return value of the program function.
    """
    entry = get_program(name)
    resolved = resolve_program_config(entry[0].name, config)
    return entry[1](resolved, context)


def available_programs() -> list[str]:
    """Return the names of the registered programs, sorted ignoring case."""
    return [spec.name for spec in program_specs()]


__all__ = [
    "ArtifactSlot",
    "ProgramSpec",
    "available_programs",
    "ensure_builtin_programs",
    "get_program",
    "program",
    "program_outputs",
    "program_spec",
    "program_specs",
    "resolve_program_config",
    "run_program",
    "validate_program_artifacts",
    "validate_program_calls",
]
