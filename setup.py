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
        ("lunarops", "fileio", "builtin_catalogs.py"),
        ("lunarops", "fileio", "normal_equations.py"),
        ("lunarops", "fileio", "normal_points.py"),
    )

    def run(self) -> None:
        for relative_path in self._REMOVED_MODULES:
            Path(self.build_lib, *relative_path).unlink(missing_ok=True)
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
            "_external/iers2010/LICENSE",
            "_external/iers2010/README.md",
        ]
    },
    include_package_data=False,
    cmdclass={"build_py": _BuildPy},
    zip_safe=False,
)
