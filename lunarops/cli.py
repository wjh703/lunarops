"""Single command-line entry point.

Usage::

    python -m lunarops run config.yml [--set name=value ...] [--working-dir DIR]
    python -m lunarops list-programs
    python -m lunarops describe-program LlrNormalEquations
    python -m lunarops validate config.yml
    python -m lunarops list-classes [category]

The config drives everything (GROOPS style); ``--set`` overrides entries of
the ``variables:`` section for scripted batch runs (e.g. SLURM arrays).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


_MPI_NATIVE_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _configure_mpi_native_threads() -> None:
    for name in _MPI_NATIVE_THREAD_VARIABLES:
        os.environ[name] = "1"


def _import_programs() -> None:
    # Importing registers the @program entries.  This is deliberately called
    # only on rank 0 after MPI rank splitting; worker ranks never need the
    # program registry.
    import lunarops.programs.catalog_programs  # noqa: F401
    import lunarops.programs.inspection_programs  # noqa: F401
    import lunarops.programs.llr_adjustment  # noqa: F401
    import lunarops.programs.llr_normal_equations  # noqa: F401
    import lunarops.programs.llr_observation_equations  # noqa: F401
    import lunarops.programs.llr_residuals  # noqa: F401
    import lunarops.programs.normal_equation_programs  # noqa: F401
    import lunarops.programs.normal_point_programs  # noqa: F401


def cmd_run(args) -> int:
    runtime = None
    context = None
    n = 0
    t0 = time.time()

    if args.mpi:
        _configure_mpi_native_threads()
        # This is intentionally the first LunarOps subsystem imported by the run
        # command.  Worker ranks branch into the lightweight receive loop before
        # config/program/model modules are imported, avoiding a metadata storm
        # when many ranks start from a shared filesystem.
        from lunarops.parallel.mpi import MpiRuntime

        runtime = MpiRuntime()
        if not runtime.is_master:
            runtime.worker_loop()
            return 0

        print(
            f"=== MPI mode: {runtime.size} rank(s), {runtime.size - 1} worker(s) ===",
            flush=True,
        )
        if runtime.size == 1:
            print(
                "=== MPI size is 1; programs fall back to serial computation. ===",
                flush=True,
            )

    try:
        # Rank 0 alone imports the program registry and config machinery.  Keep
        # these imports inside the lifecycle guard so workers are still stopped
        # if registration or config loading fails.
        _import_programs()
        from lunarops.config.context import RunContext
        from lunarops.config.loader import (
            iter_program_calls,
            load_config_file,
            parse_set_overrides,
        )
        from lunarops.programs.registry import run_program

        config = load_config_file(args.config)
        overrides = parse_set_overrides(args.set or [])

        for name, program_config, global_configs in iter_program_calls(config, overrides):
            if context is None or context.global_class_configs != global_configs:
                if context is not None:
                    context.close()
                context = RunContext(
                    global_class_configs=global_configs,
                    working_dir=args.working_dir,
                    runtime=runtime,
                )
            n += 1
            print(
                f"=== [{n}] {name} " + "=" * max(8, 60 - len(name)),
                flush=True,
            )
            run_program(name, program_config, context)
    finally:
        if context is not None:
            context.close()
        if runtime is not None:
            runtime.shutdown()

    print(
        f"=== done: {n} program call(s) in {time.time() - t0:.1f} s ===",
        flush=True,
    )
    return 0


def cmd_list_programs(_args) -> int:
    _import_programs()
    from lunarops.programs.registry import available_programs

    for name in available_programs():
        print(name)
    return 0


def cmd_describe_program(args) -> int:
    import yaml

    _import_programs()
    from lunarops.programs.registry import get_program

    print(yaml.safe_dump(get_program(args.name).spec.describe(), sort_keys=False), end="")
    return 0


def cmd_validate(args) -> int:
    _import_programs()
    from lunarops.config.context import RunContext
    from lunarops.config.loader import (
        iter_program_calls,
        load_config_file,
        parse_set_overrides,
    )
    from lunarops.programs.registry import (
        get_program,
        validate_program_artifacts,
        validate_program_config,
    )

    config = load_config_file(args.config)
    overrides = parse_set_overrides(args.set or [])
    count = 0
    produced: dict[Path, str] = {}
    for name, program_config, globals_config in iter_program_calls(config, overrides):
        context = RunContext(
            global_class_configs=globals_config,
            working_dir=args.working_dir,
        )
        try:
            validate_program_config(name, program_config)
            validate_program_artifacts(
                name,
                program_config,
                context,
                available_artifacts=produced,
            )
            for slot in get_program(name).spec.outputs:
                value = program_config.get(slot.key)
                if value is not None:
                    values = value if slot.many else [value]
                    for path in values:
                        resolved = context.resolve_path(path).resolve()
                        if resolved in produced:
                            raise ValueError(f"Scenario publishes more than one artifact to {resolved}.")
                        produced[resolved] = slot.artifact_type
        finally:
            context.close()
        count += 1
    print(f"valid: {count} program call(s)")
    return 0


def cmd_list_classes(args) -> int:
    from lunarops.classes.observation_factory import ensure_registered
    from lunarops.config.registry import available

    ensure_registered()
    listing = available(args.category) if args.category else available()
    if isinstance(listing, dict):
        for category, types in sorted(listing.items()):
            print(f"{category}: {', '.join(types)}")
    else:
        for type_name in listing:
            print(type_name)
    return 0


def main(argv=None) -> int:
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
    p_validate.set_defaults(func=cmd_validate)

    p_lc = sub.add_parser("list-classes", help="List registered model classes.")
    p_lc.add_argument("category", nargs="?")
    p_lc.set_defaults(func=cmd_list_classes)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
