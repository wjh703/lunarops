"""Core registration stays usable when the optional dynamics dependency is absent."""

import subprocess
import sys


def test_core_imports_without_scipy_and_orbit_fails_at_execution():
    script = """
import sys
class BlockSciPy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("SciPy blocked for core-profile test", name=fullname)
sys.meta_path.insert(0, BlockSciPy())
from lunarops.cli import _import_programs
from lunarops.classes.builtin_registry import ensure_registered
from lunarops.fileio.normal_points import read_normal_points
_import_programs()
ensure_registered()
from lunarops.programs.lunar_orbit import build_lunar_propagation_problem
try:
    build_lunar_propagation_problem({}, None)
except ImportError as exc:
    assert "lunarops[dynamics]" in str(exc)
else:
    raise AssertionError("missing SciPy must fail at the orbit entry")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
