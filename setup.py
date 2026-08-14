from __future__ import annotations

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).resolve().parent


class _BuildPy(_build_py):
    """Keep removed modules out of wheels built from an incremental cache."""

    _REMOVED_MODULES = (
        ("lunarops", "base", "epoch.py"),
        ("lunarops", "classes", "time_scale_converter.py"),
        ("lunarops", "fileio", "builtin_catalogs.py"),
        ("lunarops", "fileio", "adjustment_artifacts.py"),
        ("lunarops", "fileio", "linearized_observations.py"),
        ("lunarops", "fileio", "adjustment.py"),
        ("lunarops", "fileio", "crd.py"),
        ("lunarops", "fileio", "fingerprints.py"),
        ("lunarops", "fileio", "mini.py"),
        ("lunarops", "fileio", "normal_equation_file.py"),
        ("lunarops", "fileio", "normal_point_file.py"),
        ("lunarops", "fileio", "normal_point_inputs.py"),
        ("lunarops", "fileio", "observation_equation_file.py"),
        ("lunarops", "fileio", "parameters.py"),
        ("lunarops", "fileio", "structured_text.py"),
        ("lunarops", "estimation", "adjustment_options.py"),
        ("lunarops", "estimation", "adjustment_results.py"),
        ("lunarops", "estimation", "convergence.py"),
        ("lunarops", "estimation", "frozen_observation_equations.py"),
        ("lunarops", "estimation", "helmert_vce.py"),
        ("lunarops", "estimation", "linearized_least_squares.py"),
        ("lunarops", "estimation", "observation_equations.py"),
        ("lunarops", "estimation", "variance_components.py"),
        ("lunarops", "programs", "catalog_programs.py"),
        ("lunarops", "programs", "inspection_programs.py"),
        ("lunarops", "programs", "llr_adjustment.py"),
        ("lunarops", "programs", "llr_normal_equations.py"),
        ("lunarops", "programs", "llr_observation_equations.py"),
        ("lunarops", "programs", "normal_equation_programs.py"),
        ("lunarops", "programs", "normal_point_programs.py"),
    )

    def run(self) -> None:
        build_roots = {Path(self.build_lib), *(ROOT / "build").glob("lib*")}
        for build_root in build_roots:
            for relative_path in self._REMOVED_MODULES:
                (build_root / Path(*relative_path)).unlink(missing_ok=True)
        super().run()


setup(
    long_description=(ROOT / "docs" / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=("lunarops", "lunarops.*")),
    ext_modules=cythonize(
        [
            Extension(
                "lunarops._iers2010_core",
                ["lunarops/_iers2010_core.pyx"],
                include_dirs=[np.get_include()],
            ),
            Extension(
                "lunarops._normal_equations_core",
                ["lunarops/_normal_equations_core.pyx"],
                include_dirs=[np.get_include()],
            ),
        ],
        compiler_directives={
            "boundscheck": False,
            "initializedcheck": False,
            "language_level": 3,
            "wraparound": False,
        },
        build_dir="build/cython",
    ),
    package_data={
        "lunarops": [
            "_iers2010.pyi",
            "_iers2010_core.pyx",
            "_iers2010_tables.pxi",
            "_normal_equations_core.pyx",
            "_normal_equations_core.pyi",
        ]
    },
    include_package_data=False,
    cmdclass={"build_py": _BuildPy},
    zip_safe=False,
)
