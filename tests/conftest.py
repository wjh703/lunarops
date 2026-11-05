from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = Path(__file__).resolve().parent / "data"


def mpi_launcher() -> str | None:
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher is None or importlib.util.find_spec("mpi4py") is None:
        return None
    return launcher


def pytest_collection_modifyitems(config, items):
    launcher = mpi_launcher()
    nested = "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ
    reason = "MPI launcher and mpi4py are required" if launcher is None else "do not start a nested MPI job"
    if launcher is None or nested:
        for item in items:
            if "mpi" in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
