"""AST guard against historical interface acceptance in production code.

Scientific source-name arrays and physical coordinate/time conversions remain
valid. This guard supplements behavioral tests; it does not infer data validity.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

FORBIDDEN_NAMES = {"supported_versions", "snake_to_camel", "legacyName", "global_scope"}
VERSION_HELPERS = {"atomic_text_writer", "read_structured_text", "write_structured_text"}


def check_source(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    errors = []
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    aliases = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    def report(node, reason):
        errors.append(f"{filename}:{node.lineno}: {reason}")

    def function_name(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return None

    for node in ast.walk(tree):
        name = None
        if isinstance(node, (ast.Name, ast.arg)):
            name = node.id if isinstance(node, ast.Name) else node.arg
        elif isinstance(node, ast.keyword):
            name = node.arg
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        if name in FORBIDDEN_NAMES or (name and name.startswith("_migrate_")):
            report(node, f"historical interface hook {name!r} is forbidden")
        if isinstance(node, ast.Constant) and node.value == "legacyName":
            report(node, "published schemas must not expose historical names")
        if not isinstance(node, ast.Call):
            continue
        callee = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        if callee is not None:
            callee = aliases.get(callee, callee)
        keywords = {keyword.arg for keyword in node.keywords}
        if callee in VERSION_HELPERS and "version" not in keywords:
            report(node, f"{callee} must specify the artifact's current version")
        if callee == "parse_header" and "expected_version" not in keywords:
            if not (filename.endswith("fileio/archive.py") and function_name(node) == "read_artifact_type"):
                report(node, "artifact readers must check their exact current version")
        if "source_names" in keywords and callee in {"validate", "convert"}:
            report(node, "report schema must not select historical field names")
        if callee == "record":
            for argument in node.args:
                for value in ast.walk(argument):
                    if (
                        isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and re.search(r"\w+=\w+", value.value)
                    ):
                        report(value, "report declarations must contain single current names")
        if isinstance(node.func, ast.Name) and callee == "ReportField" and "source" in keywords:
            report(node, "ReportField must have only one public name")
    return errors


def check_tests(source: str, filename: str) -> list[str]:
    """Reject explicit report-v1 success chains; tiny rejection fixtures are valid."""
    tree = ast.parse(source, filename=filename)
    errors = []
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        if not function.name.startswith("test_"):
            continue
        historical_paths = set()
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id == "write_structured_text" and len(call.args) >= 2:
                historical_type = isinstance(call.args[1], ast.Constant) and call.args[1].value in {
                    "processingReport",
                    "lunarOrbitMetadata",
                }
                version_one = any(
                    kw.arg == "version" and isinstance(kw.value, ast.Constant) and kw.value.value == 1
                    for kw in call.keywords
                )
                if historical_type and version_one:
                    historical_paths.add(ast.dump(call.args[0]))
        parents = {child: parent for parent in ast.walk(function) for child in ast.iter_child_nodes(parent)}
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            if (
                not isinstance(call.func, ast.Name)
                or call.func.id not in {"read_processing_report", "read_lunar_orbit_metadata"}
                or not call.args
            ):
                continue
            if ast.dump(call.args[0]) not in historical_paths:
                continue
            current: ast.AST = call
            inside_rejection = False
            while current in parents:
                current = parents[current]
                if isinstance(current, ast.With):
                    inside_rejection = any(
                        isinstance(item.context_expr, ast.Call)
                        and isinstance(item.context_expr.func, ast.Attribute)
                        and item.context_expr.func.attr == "raises"
                        for item in current.items
                    )
                    if inside_rejection:
                        break
            if not inside_rejection:
                errors.append(f"{filename}:{call.lineno}: historical report reads must be rejection tests")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = []
    for path in sorted((root / "lunarops").rglob("*.py")):
        errors.extend(check_source(path.read_text(), path.relative_to(root).as_posix()))
    for path in sorted((root / "tests").glob("test_*.py")):
        errors.extend(check_tests(path.read_text(), path.relative_to(root).as_posix()))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Current-contract AST guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
