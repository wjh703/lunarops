"""Parallel start-up in the CLI: thread limits, worker branching, and shutdown or abort on errors."""

import multiprocessing
import os
from types import SimpleNamespace

import pytest

import lunarops.config.loader as config_loader
import lunarops.programs.registry as program_registry
from lunarops import cli
from lunarops.parallel import mpi, process

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
MPI_ARGS = SimpleNamespace(mpi=True, workers=None, config="unused.yml", set=[], working_dir=".")
WORKER_ARGS = SimpleNamespace(mpi=False, workers=3, config="unused.yml", set=[], working_dir=".")


def _runtime(*, master, events):
    class FakeRuntime:
        is_master = master
        name = "fake"
        workers = 1

        def __init__(self, *args):
            events.append(("threads", {name: os.environ[name] for name in THREAD_VARIABLES}, args))

        def describe(self):
            return "fake"

        def worker_loop(self):
            events.append("worker_loop")

        def shutdown(self, *, interrupted=False):
            events.append("abort" if interrupted else "shutdown")

    return FakeRuntime


def _fail_loading(monkeypatch, error):
    monkeypatch.setattr(config_loader, "load_config_file", lambda path: (_ for _ in ()).throw(error))


def test_workers_limit_native_threads_and_enter_their_loop_before_loading_programs(monkeypatch):
    events = []
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")
    monkeypatch.setattr(mpi, "MpiRuntime", _runtime(master=False, events=events))
    monkeypatch.setattr(
        program_registry,
        "ensure_builtin_programs",
        lambda: (_ for _ in ()).throw(AssertionError("a worker imported the program registry")),
    )

    assert cli.cmd_run(MPI_ARGS) == 0
    assert events == [("threads", dict.fromkeys(THREAD_VARIABLES, "1"), ()), "worker_loop"]


def test_the_master_shuts_workers_down_when_the_run_fails(monkeypatch):
    events = []
    monkeypatch.setattr(mpi, "MpiRuntime", _runtime(master=True, events=events))
    _fail_loading(monkeypatch, RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        cli.cmd_run(MPI_ARGS)
    assert events[-1] == "shutdown"


def test_the_master_aborts_the_job_when_the_run_is_interrupted(monkeypatch):
    events = []
    monkeypatch.setattr(mpi, "MpiRuntime", _runtime(master=True, events=events))
    _fail_loading(monkeypatch, KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_run(MPI_ARGS)
    assert events[-1] == "abort"


@pytest.mark.parametrize(("error", "end"), [(RuntimeError("boom"), "shutdown"), (KeyboardInterrupt(), "abort")])
def test_worker_processes_start_with_one_native_thread_and_stop_on_every_exit(monkeypatch, error, end):
    events = []
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")
    monkeypatch.setattr(process, "ProcessRuntime", _runtime(master=True, events=events))
    _fail_loading(monkeypatch, error)

    with pytest.raises(type(error)):
        cli.cmd_run(WORKER_ARGS)
    assert events == [("threads", dict.fromkeys(THREAD_VARIABLES, "1"), (3,)), end]


def test_a_serial_run_keeps_the_native_thread_pools_of_this_process(monkeypatch):
    # Only the processes a serial run starts (a backward-arc process) are limited to one thread, when they start.
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")
    _fail_loading(monkeypatch, RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        cli.cmd_run(SimpleNamespace(mpi=False, workers=None, config="unused.yml", set=[], working_dir="."))
    assert {name: os.environ[name] for name in THREAD_VARIABLES} == dict.fromkeys(THREAD_VARIABLES, "8")


def test_a_failed_run_leaves_no_worker_process(tmp_path, monkeypatch):
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    with pytest.raises(FileNotFoundError):
        cli.main(["run", str(tmp_path / "missing.yml"), "--workers", "2"])
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("options", [["--mpi", "--workers", "2"], ["--workers", "0"], ["--workers", "two"]])
def test_invalid_parallel_options_are_rejected(options, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["run", "unused.yml", *options])
    assert stop.value.code == 2
    error = capsys.readouterr().err
    assert "not allowed with argument" in error or "worker count" in error
