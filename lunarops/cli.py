"""Run and inspect LunarOps processing scenarios.

Commands: "run" checks a YAML config file and executes its program sequence; "validate" checks a config file
without running it; "list-programs" and "describe-program NAME" show the registered programs and the
config fields of one program; "list-classes [CATEGORY]" lists the registered model classes; "schema" writes
the JSON Schema of config files for YAML editors.
"--set NAME=VALUE" (repeatable) overrides an entry of the config's "variables:" section.
"run --mpi" executes in master-worker MPI mode under mpirun, mpiexec or srun.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_MPI_NATIVE_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _configure_mpi_native_threads() -> None:
    for name in _MPI_NATIVE_THREAD_VARIABLES:
        os.environ[name] = "1"


def _import_programs() -> None:
    from lunarops.programs.registry import ensure_builtin_programs

    ensure_builtin_programs()


def _load_plan(args):
    from lunarops.config.loader import build_run_plan, load_config_file, parse_set_overrides

    return build_run_plan(load_config_file(args.config), parse_set_overrides(args.set or []))


def _run_context(plan, args, runtime=None):
    from lunarops.config.context import RunContext

    return RunContext(model_configs=plan.models, working_dir=args.working_dir, runtime=runtime)


def cmd_run(args) -> int:
    """Check every program call of a config file, then execute them in order.

    With ``args.mpi`` the process joins an MPI run: worker ranks serve model-evaluation tasks until the
    master shuts them down, and only rank 0 loads the config and runs the programs.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (``config``, ``set``, ``working_dir``, ``mpi``).

    Returns
    -------
    int
        Exit status, 0 on success.
    """
    from lunarops import console

    runtime = None
    context = None
    completed = 0
    current: str | None = None
    t0 = time.time()

    if args.mpi:
        _configure_mpi_native_threads()
        from lunarops.parallel.mpi import MpiRuntime

        runtime = MpiRuntime()
        if not runtime.is_master:
            runtime.worker_loop()
            return 0

    try:
        _import_programs()
        from lunarops.programs.registry import run_program, validate_program_calls

        plan = _load_plan(args)
        context = _run_context(plan, args, runtime)
        context.resolve_defaults()
        validate_program_calls(plan.calls, context)

        total = len(plan.calls)
        mode = "serial" if runtime is None else f"MPI, {runtime.size} ranks ({runtime.size - 1} workers)"
        console.run_header(args.config, call_count=total, mode=mode, working_dir=args.working_dir)
        for index, (name, program_config) in enumerate(plan.calls, start=1):
            current = f"[{index}/{total}] {name}"
            console.program_header(index, total, name)
            started = time.time()
            try:
                run_program(name, program_config, context)
            except BaseException:
                console.program_footer(time.time() - started, ok=False)
                raise
            console.program_footer(time.time() - started)
            completed += 1
        current = None
    except BaseException:
        if current is not None:
            console.run_footer(completed, time.time() - t0, failed=current)
        raise
    finally:
        try:
            if context is not None:
                context.close()
        finally:
            if runtime is not None:
                runtime.shutdown()

    console.run_footer(completed, time.time() - t0)
    return 0


def cmd_list_programs(_args) -> int:
    """Print the names of the registered programs, one per line.

    Parameters
    ----------
    _args : argparse.Namespace
        Parsed command-line arguments (unused).

    Returns
    -------
    int
        Exit status, 0.
    """
    _import_programs()
    from lunarops.programs.registry import available_programs

    for name in available_programs():
        print(name)
    return 0


def cmd_describe_program(args) -> int:
    """Print the configuration fields of one program as YAML.

    Each field is listed with its name, type, description and, where it has one, its default.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (``name``).

    Returns
    -------
    int
        Exit status, 0.
    """
    import yaml

    _import_programs()
    from lunarops.config.schema import MISSING
    from lunarops.programs.registry import get_program

    fields = []
    for field in get_program(args.name)[0].schema.fields:
        item = {"name": field.name, "type": field.kind, "description": field.description}
        if field.default is not MISSING:
            item["default"] = field.default
        fields.append(item)
    print(yaml.safe_dump(fields, sort_keys=False), end="")
    return 0


def cmd_validate(args) -> int:
    """Validate a config file without running its programs.

    Each program call is checked against its schema and artifact contract in order, so an input may be an
    output of an earlier call. Input files must exist unless ``args.resolve`` is set; in that case the
    resolved models and program configs are printed as YAML.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (``config``, ``set``, ``working_dir``, ``resolve``).

    Returns
    -------
    int
        Exit status, 0 when the config is valid.
    """
    _import_programs()
    from lunarops.programs.registry import validate_program_calls

    plan = _load_plan(args)
    with _run_context(plan, args) as context:
        context.resolve_defaults()
        calls = validate_program_calls(plan.calls, context, require_inputs=not args.resolve)
    if args.resolve:
        import yaml

        programs = [{"program": name, **config} for name, config in calls]
        print(yaml.safe_dump({"models": context.model_configs, "programs": programs}, sort_keys=False), end="")
    else:
        print(f"valid: {len(plan.calls)} program call(s)")
    return 0


def cmd_schema(args) -> int:
    """Write the JSON Schema of run configs, used by YAML editors for completion and checks.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (``output``, None for standard output).

    Returns
    -------
    int
        Exit status, 0.
    """
    _import_programs()
    from lunarops.config.json_schema import render_run_config_json_schema
    from lunarops.programs.registry import program_specs

    text = render_run_config_json_schema(program_specs())
    if args.output is None:
        print(text, end="")
    else:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


def cmd_list_classes(args) -> int:
    """Print the registered model classes, for every category or for one.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (``category``, None for every category).

    Returns
    -------
    int
        Exit status, 0.
    """
    from lunarops.config.class_configs import available

    _import_programs()
    listing = available(args.category) if args.category else available()
    if args.category is None:
        for category, types in sorted(listing.items()):
            print(f"{category}: {', '.join(types)}")
    else:
        for type_name in listing[args.category]:
            print(type_name)
    return 0


def main(argv=None) -> int:
    """Parse the command line and dispatch to the selected command.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments without the program name; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        Exit status of the command.
    """
    parser = argparse.ArgumentParser(prog="lunarops", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Execute the program sequence of a config file.")
    p_run.add_argument("config")
    p_run.add_argument(
        "--set",
        action="append",
        metavar="NAME=VALUE",
        help="Override a config variable (repeatable).",
    )
    p_run.add_argument(
        "--working-dir",
        default=".",
        help="Base directory for relative output paths.",
    )
    p_run.add_argument(
        "--mpi",
        action="store_true",
        help="Master-worker MPI execution; launch under mpirun/mpiexec/srun. "
        "Rank 0 runs the program sequence, ranks 1..N serve model-evaluation "
        "tasks (per-program option: mpi: {chunksize: 8}).",
    )
    p_run.set_defaults(func=cmd_run)

    p_lp = sub.add_parser("list-programs", help="List registered programs.")
    p_lp.set_defaults(func=cmd_list_programs)

    p_dp = sub.add_parser("describe-program", help="Show one declarative program contract.")
    p_dp.add_argument("name")
    p_dp.set_defaults(func=cmd_describe_program)

    p_validate = sub.add_parser("validate", help="Validate a YAML scenario and its typed inputs.")
    p_validate.add_argument("config")
    p_validate.add_argument("--set", action="append", metavar="NAME=VALUE")
    p_validate.add_argument("--working-dir", default=".")
    p_validate.add_argument(
        "--resolve", action="store_true", help="Print resolved native YAML without reading input files."
    )
    p_validate.set_defaults(func=cmd_validate)

    p_lc = sub.add_parser("list-classes", help="List registered model classes.")
    p_lc.add_argument("category", nargs="?")
    p_lc.set_defaults(func=cmd_list_classes)

    p_schema = sub.add_parser("schema", help="Write the JSON Schema of config files for YAML editors.")
    p_schema.add_argument("-o", "--output", help="Output file; standard output when omitted.")
    p_schema.set_defaults(func=cmd_schema)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
