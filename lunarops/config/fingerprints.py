"""Schema-declared scientific input manifests and reproducible fingerprints."""

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
    """The CALCEPH load order: sorted position kernels, then orientation kernels."""
    return (*sorted(directory.glob("*.bsp")), *sorted(directory.glob("*.bpc")))


def kernel_manifest(directory: Path) -> list[dict[str, object]]:
    return [
        {"relativePath": str(path.relative_to(directory)), "sha256": sha256_file(path), "loadOrder": index}
        for index, path in enumerate(calceph_kernel_paths(directory))
    ]


def _schema_paths(value, schema, context, resources: dict[str, object], location: str) -> None:
    from lunarops.config.registry import get_class_declaration

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
                registered = get_class_declaration(field.class_category, declaration["type"])
                _schema_paths(declaration, registered.schema, context, resources, f"{name}[{index}]")
        elif field.nested:
            _schema_paths(item, field.nested, context, resources, name)
        elif field.item_nested:
            for index, entry in enumerate(item):
                _schema_paths(entry, field.item_nested, context, resources, f"{name}[{index}]")


def scientific_input_manifest(config: Mapping[str, object], context) -> dict[str, object]:
    """Hash input ArtifactSlots and model schema paths, never arbitrary strings."""
    from lunarops.classes.builtin_registry import ensure_registered
    from lunarops.config.registry import get_class_declaration
    from lunarops.programs.registry import ensure_builtin_programs, program_specs

    ensure_registered()
    ensure_builtin_programs()
    output_keys = {slot.key for spec in program_specs() for slot in spec.outputs}
    config = {key: value for key, value in config.items() if key not in output_keys}
    resources: dict[str, object] = {}
    for category, declarations in context.class_configs.items():
        entries = declarations if isinstance(declarations, (list, tuple)) else [declarations]
        for index, declaration in enumerate(entries):
            registered = get_class_declaration(category, declaration["type"])
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
    # Program override schemas carry model/file declarations as well.
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


def scientific_fingerprint(config: Mapping[str, object], context, *, excluded_keys: Iterable[str] = (), manifest=None) -> str:
    from lunarops.programs.registry import ensure_builtin_programs, program_specs

    ensure_builtin_programs()
    excluded = set(excluded_keys) | {"showProgress", "mpi", "outputLevel"}
    excluded.update(slot.key for spec in program_specs() for slot in spec.outputs)
    selected = {key: value for key, value in config.items() if key not in excluded}
    payload = {
        "formatVersion": 2,
        "program": plain_data(selected),
        "models": plain_data(context.class_configs),
        "manifest": scientific_input_manifest(selected, context) if manifest is None else manifest,
    }
    encoded = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["calceph_kernel_paths", "kernel_manifest", "scientific_fingerprint", "scientific_input_manifest"]
