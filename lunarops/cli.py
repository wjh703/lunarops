"""Single command-line entry point.

Usage::

    python -m lunarops run config.yml [--set name=value ...] [--working-dir DIR]
    python -m lunarops list-programs
    python -m lunarops describe-program LlrNormalEquations
    python -m lunarops describe-config
    python -m lunarops validate config.yml
    python -m lunarops list-classes [category]

The config drives everything (GROOPS style); ``--set`` overrides entries of
the ``variables:`` section for scripted batch runs (e.g. SLURM arrays).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys
import time
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
    # The registry owns the import transaction and idempotence.  This is
    # deliberately called only on rank 0 after MPI rank splitting; worker ranks
    # never need the program registry.
    from lunarops.programs.registry import ensure_builtin_programs

    ensure_builtin_programs()


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
        from lunarops.classes.observation_factory import ensure_registered
        from lunarops.config.context import RunContext
        from lunarops.config.loader import (
            build_run_plan,
            load_config_file,
            parse_set_overrides,
        )
        from lunarops.programs.registry import run_program

        config = load_config_file(args.config)
        overrides = parse_set_overrides(args.set or [])
        ensure_registered()
        plan = build_run_plan(config, overrides)
        context = RunContext(
            global_class_configs=plan.globals,
            working_dir=args.working_dir,
            runtime=runtime,
        )
        context.validate_globals()

        for name, program_config in plan.calls:
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
    from lunarops.classes.observation_factory import ensure_registered
    from lunarops.programs.registry import get_program

    ensure_registered()
    print(yaml.safe_dump(get_program(args.name).spec.describe(), sort_keys=False), end="")
    return 0


def cmd_describe_config(_args) -> int:
    import json

    from lunarops.config.catalog import configuration_catalog

    print(json.dumps(configuration_catalog(), indent=2, sort_keys=False))
    return 0


def cmd_validate(args) -> int:
    _import_programs()
    from lunarops.classes.observation_factory import ensure_registered
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
    ensure_registered()
    plan = build_run_plan(config, overrides)
    produced: dict[Path, str] = {}
    with RunContext(
        global_class_configs=plan.globals,
        working_dir=args.working_dir,
    ) as context:
        context.validate_globals()
        for name, program_config in plan.calls:
            resolved_config = validate_program_artifacts(
                name,
                program_config,
                context,
                available_artifacts=produced,
            )
            for slot in get_program(name).spec.outputs:
                value = resolved_config.get(slot.key)
                if value is not None:
                    values = (
                        list(cast(Sequence[object], value))
                        if slot.many
                        else [value]
                    )
                    for path in values:
                        resolved = context.resolve_path(path).resolve()
                        if resolved in produced:
                            raise ValueError(f"Scenario publishes more than one artifact to {resolved}.")
                        produced[resolved] = slot.artifact_type
    print(f"valid: {len(plan.calls)} program call(s)")
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

    p_dc = sub.add_parser("describe-config", help="Show the complete GUI-oriented YAML contract.")
    p_dc.set_defaults(func=cmd_describe_config)

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
