from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).resolve().parent
# -ffp-contract=off keeps multiply/add rounding consistent with the frozen scientific references.
# -fno-math-errno lets sqrt compile to the bare (vectorisable) instruction; no result changes.
FP_FLAGS = ["/fp:strict"] if sys.platform == "win32" else ["-ffp-contract=off", "-fno-math-errno"]
NATIVE_MODULES = ("iers2010", "dynamics", "ephemeris")


setup(
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=("lunarops", "lunarops.*")),
    ext_modules=cythonize(
        [
            Extension(
                f"lunarops._native.{name}",
                [f"lunarops/_native/{name}.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=FP_FLAGS,
            )
            for name in NATIVE_MODULES
        ],
        compiler_directives={
            "language_level": 3,
        },
        build_dir="build/cython",
    ),
    # The Cython sources ship with the package (see scripts/verify_distribution.py).
    package_data={"lunarops._native": ["*.pyx", "*.pxi", "*.pxd", "*.pyi"]},
    include_package_data=False,
    zip_safe=False,
)
