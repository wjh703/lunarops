"""MPI start-up in the CLI: thread limits, worker branching and shutdown on errors."""

from types import SimpleNamespace

import pytest

import lunarops.config.loader as config_loader
import lunarops.programs.registry as program_registry
from lunarops import cli
from lunarops.parallel import mpi

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
ARGS = SimpleNamespace(mpi=True, config="unused.yml", set=[], working_dir=".")


def _runtime(*, master, events):
    class FakeRuntime:
        is_master = master
        size = 2

        def __init__(self):
            events.append(("threads", {name: cli.os.environ[name] for name in THREAD_VARIABLES}))

        def worker_loop(self):
            events.append("worker_loop")

        def shutdown(self):
            events.append("shutdown")

    return FakeRuntime


def test_workers_limit_native_threads_and_enter_their_loop_before_loading_programs(monkeypatch):
    events = []
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")
    monkeypatch.setattr(mpi, "MpiRuntime", _runtime(master=False, events=events))
    monkeypatch.setattr(program_registry, "ensure_builtin_programs",
                        lambda: (_ for _ in ()).throw(AssertionError("a worker imported the program registry")))

    assert cli.cmd_run(ARGS) == 0
    assert events == [("threads", dict.fromkeys(THREAD_VARIABLES, "1")), "worker_loop"]


def test_the_master_shuts_workers_down_when_the_run_fails(monkeypatch):
    events = []
    monkeypatch.setattr(mpi, "MpiRuntime", _runtime(master=True, events=events))
    monkeypatch.setattr(config_loader, "load_config_file",
                        lambda path: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        cli.cmd_run(ARGS)
    assert events[-1] == "shutdown"
