from types import SimpleNamespace

import pytest

import lunarops.cli as cli
import lunarops.config.loader as config_loader
import lunarops.parallel.mpi as mpi_module


def test_mpi_runtime_shutdown_on_config_parse_error(monkeypatch):
    instances = []

    class FakeRuntime:
        is_master = True
        size = 2

        def __init__(self):
            self.shutdown_called = False
            instances.append(self)

        def worker_loop(self):  # pragma: no cover - master path in this test
            raise AssertionError("worker_loop should not run on master")

        def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setattr(cli, "_import_programs", lambda: None)
    monkeypatch.setattr(mpi_module, "MpiRuntime", FakeRuntime)
    monkeypatch.setattr(
        config_loader,
        "load_config_file",
        lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    args = SimpleNamespace(
        mpi=True,
        config="bad.yml",
        set=[],
        working_dir=".",
    )
    with pytest.raises(RuntimeError, match="boom"):
        cli.cmd_run(args)

    assert instances and instances[0].shutdown_called


def test_mpi_worker_branches_before_program_registry(monkeypatch):
    events = []

    class FakeRuntime:
        is_master = False
        size = 4

        def worker_loop(self):
            events.append("worker_loop")

        def shutdown(self):  # pragma: no cover - worker returns before shutdown
            raise AssertionError("worker must not call master shutdown")

    monkeypatch.setattr(mpi_module, "MpiRuntime", FakeRuntime)
    monkeypatch.setattr(
        cli,
        "_import_programs",
        lambda: (_ for _ in ()).throw(AssertionError("worker imported program registry")),
    )

    args = SimpleNamespace(
        mpi=True,
        config="unused.yml",
        set=[],
        working_dir=".",
    )
    assert cli.cmd_run(args) == 0
    assert events == ["worker_loop"]


def test_mpi_forces_single_native_thread_before_runtime(monkeypatch):
    variables = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    for name in variables:
        monkeypatch.setenv(name, "8")

    class FakeRuntime:
        is_master = False
        size = 2

        def __init__(self):
            assert all(cli.os.environ[name] == "1" for name in variables)

        def worker_loop(self):
            return None

        def shutdown(self):  # pragma: no cover - worker returns before shutdown
            raise AssertionError("worker must not call master shutdown")

    monkeypatch.setattr(mpi_module, "MpiRuntime", FakeRuntime)
    args = SimpleNamespace(mpi=True, config="unused.yml", set=[], working_dir=".")

    assert cli.cmd_run(args) == 0
