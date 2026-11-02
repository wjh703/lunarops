"""Core registration stays usable when the optional dynamics dependency is absent."""

import subprocess
import sys


def test_core_imports_without_scipy_and_integrator_fails_when_used():
    script = """
import sys
class BlockSciPy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("SciPy blocked for core-profile test", name=fullname)
sys.meta_path.insert(0, BlockSciPy())
from lunarops.cli import _import_programs
from lunarops.fileio.normal_points import read_normal_points
_import_programs()
from lunarops.classes.dynamics.integrators import dop853_startup
import numpy as np
try:
    dop853_startup(lambda t, y: y, np.zeros(6), 1.0, step_s=1.0, sample_step_s=1.0)
except ImportError as exc:
    assert exc.name == "scipy"
else:
    raise AssertionError("missing SciPy must fail when integration uses it")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
