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


setup(
    long_description=(
        ROOT / "docs" / "README.md" if (ROOT / "docs" / "README.md").is_file() else ROOT / "README.md"
    ).read_text(encoding="utf-8"),
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
            Extension(
                "lunarops._dynamics_core",
                ["lunarops/_dynamics_core.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=OPENMP_FLAGS,
                extra_link_args=OPENMP_FLAGS,
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
            "_normal_equations_core.pyx",
            "_normal_equations_core.pyi",
            "_dynamics_core.pyx",
            "_dynamics_core.pyi",
        ]
    },
    include_package_data=False,
    zip_safe=False,
)
