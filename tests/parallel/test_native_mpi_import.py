"""Compiled extensions import and evaluate inside MPI worker processes."""

import os
import subprocess
import sys

import pytest

from tests.conftest import mpi_launcher

WORKER = """
from mpi4py import MPI
from lunarops import _iers2010

values = MPI.COMM_WORLD.allgather(_iers2010.fcul_a(30.67166667, 2075.0, 300.15, 15.0))
assert len(values) == 2
assert all(abs(value - 3.800243667312344087) < 1e-11 for value in values)
"""


@pytest.mark.mpi
def test_native_extension_imports_in_mpi_workers():
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    subprocess.run([str(mpi_launcher()), "-n", "2", sys.executable, "-c", WORKER], check=True, capture_output=True,
                   env=env, text=True, timeout=60)
