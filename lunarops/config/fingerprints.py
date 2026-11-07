"""Manifests of scientific input files and fingerprints of processing configurations.

A manifest records the SHA-256 digest of every input file declared by a program's artifact slots or by a
``path`` field of a model schema, a digest of the LunarOps source files and the versions of the numerical
packages. A fingerprint is the SHA-256 digest of the scientific part of a program config, the run's model
configs and the manifest; it identifies the configuration that produced a result.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import yaml

from lunarops.base.serialization import plain_data
from lunarops.fileio.archive import sha256_file


def calceph_kernel_paths(directory: Path) -> tuple[Path, ...]:
    """Return the kernels of a CALCEPH directory in load order.

    Parameters
    ----------
    directory : Path
        Directory holding the kernels.

    Returns
    -------
    tuple of Path
        The ``*.bsp`` (position) kernels sorted by path, then the ``*.bpc`` (orientation) kernels sorted by path.
    """
    return (*sorted(directory.glob("*.bsp")), *sorted(directory.glob("*.bpc")))


def kernel_manifest(directory: Path) -> list[dict[str, object]]:
    """Return the manifest of the kernels of a CALCEPH directory.

    Parameters
    ----------
    directory : Path
        Directory holding the kernels.

    Returns
    -------
    list of dict
        One entry per kernel in load order, with ``relativePath``, ``sha256`` and ``loadOrder``.
    """
    return [
        {"relativePath": str(path.relative_to(directory)), "sha256": sha256_file(path), "loadOrder": index}
        for index, path in enumerate(calceph_kernel_paths(directory))
    ]


def _schema_paths(value, schema, context, resources: dict[str, object], location: str) -> None:
    from lunarops.config.registry import _declaration

    if not isinstance(value, Mapping):
        return
    for field in schema.fields:
        item = value.get(field.name)
        if item is None:
            continue
        name = f"{location}.{field.name}"
        if field.kind == "path":
            path = context.resolve_path(item)
            if path.is_dir():
                if field.name != "directory" or value.get("type") != "calceph":
                    raise ValueError(f"Unsupported scientific input directory at {name}: {path}")
                manifest = kernel_manifest(path)
                if not manifest:
                    raise ValueError(f"No CALCEPH kernels at {name}: {path}")
                resources[name] = [{"path": str(path), "kernels": manifest}]
            elif path.is_file():
                resources[name] = [{"path": str(path), "sha256": sha256_file(path)}]
            else:
                raise FileNotFoundError(f"Scientific input {name} does not exist: {path}")
        elif field.class_category:
            declarations = item if field.kind == "class_list" else [item]
            for index, declaration in enumerate(declarations):
                registered = _declaration(field.class_category, declaration["type"])
                _schema_paths(declaration, registered.schema, context, resources, f"{name}[{index}]")
        elif field.nested:
            _schema_paths(item, field.nested, context, resources, name)
        elif field.item_nested:
            for index, entry in enumerate(item):
                _schema_paths(entry, field.item_nested, context, resources, f"{name}[{index}]")


def scientific_input_manifest(config: Mapping[str, object], context) -> dict[str, object]:
    """Return the manifest of the scientific inputs of a program config.

    Only declared inputs are hashed: the input artifact slots of the registered programs and the ``path``
    fields of the run's model configs and of the program schemas. A directory is accepted only as the
    ``directory`` of a ``calceph`` ephemeris, whose kernels are listed in load order. Normal-point inputs that
    are directories are expanded to the supported files they contain. Output slots are ignored.

    Parameters
    ----------
    config : Mapping
        Resolved program config.
    context : RunContext
        Run context providing the model configs and resolving relative paths.

    Returns
    -------
    dict
        ``resources`` (file paths and SHA-256 digests keyed by config location), ``codeSha256`` (digest of
        the LunarOps source and extension files) and ``versions`` (versions of ``lunarops``, ``calcephpy``,
        ``pyerfa`` and ``numpy``).

    Raises
    ------
    FileNotFoundError
        If a declared input does not exist.
    ValueError
        If a directory is given where a file is expected, or a CALCEPH directory holds no kernels.
    """
    from lunarops.config.registry import _declaration
    from lunarops.programs.registry import ensure_builtin_programs, program_specs

    ensure_builtin_programs()
    output_keys = {slot.key for spec in program_specs() for slot in spec.outputs}
    config = {key: value for key, value in config.items() if key not in output_keys}
    resources: dict[str, object] = {}
    for category, declarations in context.class_configs.items():
        if declarations is None:
            continue
        entries = declarations if isinstance(declarations, (list, tuple)) else [declarations]
        for index, declaration in enumerate(entries):
            registered = _declaration(category, declaration["type"])
            _schema_paths(declaration, registered.schema, context, resources, f"models.{category}[{index}]")
    input_slots = {slot.key: slot for spec in program_specs() for slot in spec.inputs}
    for key, slot in sorted(input_slots.items()):
        if key not in config or config[key] is None:
            continue
        paths = cast(Iterable[str | Path], config[key]) if slot.many else [cast(str | Path, config[key])]
        if key == "inputFilesNormalPoints" and any(context.resolve_path(path).is_dir() for path in paths):
            from lunarops.fileio.formats.normal_point_sources import resolve_normal_point_inputs

            paths = resolve_normal_point_inputs([context.resolve_path(path) for path in paths])
        resources[f"program.{key}"] = [
            {"path": str(context.resolve_path(path)), "sha256": sha256_file(context.resolve_path(path))}
            for path in paths
        ]
    for spec in program_specs():
        model_options = {key: value for key, value in config.items() if key not in input_slots}
        _schema_paths(model_options, spec.schema, context, resources, "program")
    package_root = Path(__file__).resolve().parents[1]
    source_digest = hashlib.sha256()
    for path in sorted((*package_root.rglob("*.py"), *package_root.glob("*.so"), *package_root.glob("*.pyx"))):
        source_digest.update(str(path.relative_to(package_root)).encode())
        source_digest.update(bytes.fromhex(sha256_file(path)))
    backend_versions = {}
    for package in ("lunarops", "calcephpy", "pyerfa", "numpy"):
        try:
            backend_versions[package] = version(package)
        except PackageNotFoundError:
            backend_versions[package] = "not-installed"
    return {"resources": resources, "codeSha256": source_digest.hexdigest(), "versions": backend_versions}


#: Program config keys that do not affect the observation model and are left out of the compatibility fingerprint.
COMPATIBILITY_EXCLUDED_KEYS = frozenset(
    {
        "inputFilesNormalPoints",
        "inputFileProcessingState",
        "startTime",
        "endTime",
        "utcOffsetHours",
        "overrideStation",
        "overrideReflector",
        "minElevationDeg",
        "showProgress",
        "mpi",
        "outputLevel",
        "includeReflectorDesign",
    }
)


def processing_scientific_payload(config: Mapping[str, object], plan) -> dict:
    """Return the part of a processing config that determines its estimation result.

    ``varianceComponents`` and ``processingSteps`` are replaced by a ``processing`` entry holding the
    variance components of the plan and its observation-screening, parametrization-selection and estimation
    steps; other step kinds are left out.

    Parameters
    ----------
    config : Mapping
        Resolved processing program config.
    plan : LlrAdjustmentPlan
        Processing plan built from the config.

    Returns
    -------
    dict
        Config entries entering the processing fingerprint.
    """
    from lunarops.estimation.adjustment.plan import EstimateStep, ScreenObservationsStep, SelectParametrizationsStep

    selected = {key: value for key, value in config.items() if key not in {"varianceComponents", "processingSteps"}}
    selected["processing"] = {
        "varianceComponents": plan.variance_components,
        "steps": tuple(
            step
            for step in plan.processing_steps
            if isinstance(step, (ScreenObservationsStep, SelectParametrizationsStep, EstimateStep))
        ),
    }
    return selected


def processing_fingerprint(config: Mapping[str, object], context, *, plan, manifest=None) -> str:
    """Return the fingerprint of a processing configuration.

    The digest covers the scientific payload of the config (see ``processing_scientific_payload``) without
    ``inputFileProcessingState``, the progress, MPI, output-level and output-file entries, together with the
    run's model configs and the input manifest.

    Parameters
    ----------
    config : Mapping
        Resolved processing program config.
    context : RunContext
        Run context providing the model configs.
    plan : LlrAdjustmentPlan
        Processing plan built from the config.
    manifest : dict, optional
        Precomputed input manifest; computed with ``scientific_input_manifest`` when None.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return _fingerprint(
        processing_scientific_payload(config, plan),
        context,
        excluded_keys={"inputFileProcessingState"},
        manifest=manifest,
    )


def model_compatibility_fingerprint(config: Mapping[str, object], context) -> str:
    """Return the fingerprint of the observation model of a program config.

    Data selection, run-control and output entries (``COMPATIBILITY_EXCLUDED_KEYS`` and every
    ``outputFile*`` key) are left out, so results computed from different data with the same model share
    the fingerprint.

    Parameters
    ----------
    config : Mapping
        Resolved program config.
    context : RunContext
        Run context providing the model configs.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    output_keys = {key for key in config if key.startswith("outputFile")}
    return _fingerprint(config, context, excluded_keys=COMPATIBILITY_EXCLUDED_KEYS | output_keys)


def _fingerprint(config: Mapping[str, object], context, *, excluded_keys: Iterable[str] = (), manifest=None) -> str:
    from lunarops.programs.registry import ensure_builtin_programs, program_specs

    ensure_builtin_programs()
    excluded = set(excluded_keys) | {"showProgress", "mpi", "outputLevel"}
    excluded.update(slot.key for spec in program_specs() for slot in spec.outputs)
    selected = {key: value for key, value in config.items() if key not in excluded}
    payload = {
        "program": plain_data(selected),
        "models": plain_data(context.class_configs),
    }
    return _digest(payload, scientific_input_manifest(selected, context) if manifest is None else manifest)


def _digest(payload: dict, manifest) -> str:
    encoded = yaml.safe_dump({**payload, "manifest": manifest}, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "COMPATIBILITY_EXCLUDED_KEYS",
    "calceph_kernel_paths",
    "kernel_manifest",
    "model_compatibility_fingerprint",
    "processing_fingerprint",
    "processing_scientific_payload",
    "scientific_input_manifest",
]
