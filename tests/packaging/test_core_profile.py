"""Program registration works without the optional SciPy dependency; integration then fails clearly."""

import subprocess
import sys

SCRIPT = """
import sys

class BlockSciPy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("SciPy blocked for the core-profile test", name=fullname)

sys.meta_path.insert(0, BlockSciPy())
import numpy as np
from lunarops.programs.registry import ensure_builtin_programs, program_specs
from lunarops.fileio.normal_points import read_normal_points
ensure_builtin_programs()
assert program_specs()
from lunarops.classes.dynamics.integrators import dop853_startup
try:
    dop853_startup(lambda t, y: y, np.zeros(6), 1.0, step_s=1.0, sample_step_s=1.0)
except ImportError as exc:
    assert exc.name == "scipy"
else:
    raise AssertionError("missing SciPy must fail when integration uses it")
"""


def test_core_imports_without_scipy_and_integration_fails_when_used():
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
