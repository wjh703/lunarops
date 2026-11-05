"""Run and inspect LunarOps processing scenarios.

Commands: "run" executes the program sequence of a YAML config file; "validate" checks a config file
without running it; "list-programs" and "describe-program NAME" show the registered programs and the
config fields of one program; "list-classes [CATEGORY]" lists the registered model classes.
"--set NAME=VALUE" (repeatable) overrides an entry of the config's "variables:" section.
"run --mpi" executes in master-worker MPI mode under mpirun, mpiexec or srun.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

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


def cmd_run(args) -> int:
    """Execute every program call of a config file in order.

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
    runtime = None
    context = None
    n = 0
    t0 = time.time()

    if args.mpi:
        _configure_mpi_native_threads()
        from lunarops.parallel.mpi import MpiRuntime

        runtime = MpiRuntime()
        if not runtime.is_master:
            runtime.worker_loop()
            return 0

        print(
            f"=== MPI mode: {runtime.size} rank(s), {runtime.size - 1} worker(s) ===",
            flush=True,
        )

    try:
        _import_programs()
        from lunarops.config.context import RunContext
        from lunarops.config.loader import (
            build_run_plan,
            load_config_file,
            parse_set_overrides,
        )
        from lunarops.programs.registry import run_program

        config = load_config_file(args.config)
        overrides = parse_set_overrides(args.set or [])
        plan = build_run_plan(config, overrides)
        context = RunContext(
            shared_class_configs=plan.shared,
            observation_model_configs=plan.observation_model,
            working_dir=args.working_dir,
            runtime=runtime,
        )
        context.resolve_defaults()

        for name, program_config in plan.calls:
            n += 1
            print(
                f"=== [{n}] {name} " + "=" * max(8, 60 - len(name)),
                flush=True,
            )
            run_program(name, program_config, context)
    finally:
        try:
            if context is not None:
                context.close()
        finally:
            if runtime is not None:
                runtime.shutdown()

    print(
        f"=== done: {n} program call(s) in {time.time() - t0:.1f} s ===",
        flush=True,
    )
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
    resolved shared classes, observation model and program configs are printed as YAML.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (``config``, ``set``, ``working_dir``, ``resolve``).

    Returns
    -------
    int
        Exit status, 0 when the config is valid.

    Raises
    ------
    ValueError
        If more than one program call writes an artifact to the same path.
    """
    _import_programs()
    from lunarops.config.context import RunContext
    from lunarops.config.loader import (
        build_run_plan,
        load_config_file,
        parse_set_overrides,
    )
    from lunarops.programs.registry import (
        get_program,
        validate_program_artifacts,
    )

    config = load_config_file(args.config)
    overrides = parse_set_overrides(args.set or [])
    plan = build_run_plan(config, overrides)
    produced: dict[Path, str] = {}
    with RunContext(
        shared_class_configs=plan.shared,
        observation_model_configs=plan.observation_model,
        working_dir=args.working_dir,
    ) as context:
        context.resolve_defaults()
        programs = []
        for name, program_config in plan.calls:
            resolved_config = validate_program_artifacts(
                name,
                program_config,
                context,
                available_artifacts=produced,
                require_inputs=not args.resolve,
            )
            programs.append({"program": name, **resolved_config})
            for slot in get_program(name)[0].outputs:
                value = resolved_config.get(slot.key)
                if value is not None:
                    values = list(cast(Sequence[object], value)) if slot.many else [value]
                    for path in values:
                        resolved = context.resolve_path(path).resolve()
                        if resolved in produced:
                            raise ValueError(f"Scenario publishes more than one artifact to {resolved}.")
                        produced[resolved] = slot.artifact_type
    if args.resolve:
        import yaml

        print(
            yaml.safe_dump(
                {
                    "shared": context.shared_class_configs,
                    "observationModel": context.observation_model_configs,
                    "programs": programs,
                },
                sort_keys=False,
            ),
            end="",
        )
    else:
        print(f"valid: {len(plan.calls)} program call(s)")
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
