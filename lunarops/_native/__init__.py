"""Compiled Cython extensions of LunarOps.

``dynamics`` holds the lunar dynamics kernels (fused right-hand side, point-mass sums, gravity fields,
double-double Adams accumulation, delayed-history evaluation), ``ephemeris`` the Chebyshev evaluation of
SPK and binary PCK segments, and ``iers2010`` the IERS Conventions (2010) routines behind
:mod:`lunarops._iers2010`. The modules are private; the public interfaces are in :mod:`lunarops.classes`.
"""
