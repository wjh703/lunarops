from __future__ import annotations

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).resolve().parent


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
            Extension(
                "lunarops._dynamics_core",
                ["lunarops/_dynamics_core.pyx"],
                include_dirs=[np.get_include()],
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
