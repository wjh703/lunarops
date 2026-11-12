from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).resolve().parent
OPENMP_ENABLED = os.environ.get("LUNAROPS_ENABLE_OPENMP", "0").lower() in {"1", "true", "yes"}
OPENMP_FLAGS = ["-fopenmp"] if OPENMP_ENABLED and sys.platform.startswith("linux") else []
# Keep multiply/add rounding consistent with the frozen scientific references.
FP_FLAGS = ["/fp:strict"] if sys.platform == "win32" else ["-ffp-contract=off"]


setup(
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=("lunarops", "lunarops.*")),
    ext_modules=cythonize(
        [
            Extension(
                "lunarops._iers2010_core",
                ["lunarops/_iers2010_core.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=FP_FLAGS,
            ),
            Extension(
                "lunarops._dynamics_core",
                ["lunarops/_dynamics_core.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=FP_FLAGS + OPENMP_FLAGS,
                extra_link_args=OPENMP_FLAGS,
            ),
            Extension(
                "lunarops._ephemeris_core",
                ["lunarops/_ephemeris_core.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=FP_FLAGS,
            ),
        ],
        compiler_directives={
            "language_level": 3,
        },
        build_dir="build/cython",
    ),
    package_data={
        "lunarops": [
            "_iers2010.pyi",
            "_iers2010_core.pyx",
            "_iers2010_tables.pxi",
            "_dynamics_core.pyx",
            "_lunar_rhs.pxi",
            "_double_double.pxi",
            "_delayed_history.pxi",
            "_dynamics_core.pyi",
            "_ephemeris_core.pyx",
            "_ephemeris_core.pxd",
            "_ephemeris_core.pyi",
        ]
    },
    include_package_data=False,
    zip_safe=False,
)
