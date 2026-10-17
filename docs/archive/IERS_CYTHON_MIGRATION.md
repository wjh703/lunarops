# IERS backend: ERFA and Cython migration record

## Status and decision

The migration is complete. LunarOps no longer builds, installs, or retains a
Fortran/f2py backend. A Python facade named `lunarops._iers2010` preserves the
existing private callable surface, and the compiled numerical module is
`lunarops._iers2010_core`.

Cython was selected instead of Python plus Numba because these routines contain
small fixed-size loops, large static coefficient tables, and scalar-heavy calls.
It also gives deterministic ahead-of-time builds. The tested environment's
Numba release did not support the project's NumPy version, while Cython is an
explicit isolated-build dependency.

This is an implementation migration, not a model change. Units, signs, frames,
permanent-tide conventions, and selected IERS 2010 corrections remain unchanged.

## Architecture

The facade in `lunarops/_iers2010.py` owns validation and all time handling.
ERFA provides:

- UTC calendar validation and two-part Julian dates;
- UTC to TAI to TT conversion;
- leap-second data;
- the IAU 2003 fundamental arguments `fal03`, `falp03`, `faf03`, `fad03`, and
  `faom03`.

The core in `lunarops/_iers2010_core.pyx` provides only algorithms for which
ERFA has no complete replacement:

- FCULa mapping and FCUL zenith delay;
- ORTHO_EOP orthotide terms;
- PMSDNUT2 and UTLIBR harmonic sums;
- DEHANT solid-Earth tide geometry and frequency corrections;
- HARDISP Doodson arguments, admittance expansion, spline interpolation, and
  regular-series recurrence.

The generated `lunarops/_iers2010_tables.pxi` contains the coefficient tables.
The Cython core has no calendar routines or leap-second table, and makes no ERFA
or Python calls inside its numerical loops.

## Removed and replaced routines

| Former responsibility | Final owner |
|---|---|
| Calendar-to-JD and UTC validation | ERFA `dtf2d` |
| UTC/TAI/TT and leap seconds | ERFA `utctai` and `taitt` |
| IERS FUNDARG implementation | Five ERFA IAU 2003 argument functions |
| HARDISP ETUTC, year/day conversion, and common-block date state | Python/ERFA epoch preparation |
| FCUL models | Typed Cython equations |
| Orthotide EOP and libration sums | Typed Cython loops |
| DEHANT and its vector helpers | Typed Cython loops |
| HARDISP admittance, spline, and recurrence | Typed Cython loops |

All derived core routines use `lunarops_` names. The derived `.pyx` and `.pxi`
sources are included in both source and binary distributions.

## HARDISP epoch contract

The supported UTC interval is `1960-01-01T00:00:00` through
`2027-06-30T23:59:59`. The lower bound is the start of ERFA's formal UTC domain;
the old pre-1960 HARDISP approximations are intentionally unsupported. The upper
bound is an application policy in the Python facade and is not embedded in the
numerical core.

Exact `23:59:60` labels are rejected because the inherited integer calendar API
cannot represent that label independently in the recurrence. A regular series
is also rejected if it crosses a UTC offset transition or the supported upper
bound. Production LLR processing evaluates irregular epochs with `n=1` and does
not encounter this regular-grid restriction.

## Fortran differential validation

Before removal, outputs from the pinned IERS Conventions v1.3.0 backend were
frozen in `tests/data/iers_fortran_baseline.npz` (SHA-256
`26ac513ae09027ef391ce2d58928e017ee688e1f07fa30677c2da92b496feb0e`).
The grid contains 128 randomized FCUL cases, 263 Earth-orientation epochs, 84
randomized DEHANT cases, and 54 randomized HARDISP cases across nine calendars.

| Output | Maximum absolute Cython/ERFA difference |
|---|---:|
| FCULa, FCUL zenith delay, ORTHO_EOP | exact in the frozen grid |
| PMSDNUT2 | `1.179e-10` output units |
| UTLIBR | `2.607e-10` output units |
| Fundamental arguments | `9.825e-12 rad` |
| DEHANT displacement | `3.421e-15 m` |
| HARDISP scalar displacement | `2.739e-7 m` |
| HARDISP eight-sample series | `3.037e-7 m` |

The PMSDNUT2 and UTLIBR differences come only from using ERFA's IAU 2003
fundamental-argument constants rather than the last printed digits in the old
routine. The largest HARDISP difference occurs at
`2016-12-31T23:59:59 UTC`: the old ETUTC calculation advances the decimal year
using one-based day-of-year and applies the 2017 leap-second offset before the
actual transition. ERFA correctly retains `TAI-UTC = 36 s` until midnight.
Away from that corrected boundary, the HARDISP maximum was `5.402e-8 m`.

The published Onsala 24-hour HARDISP case passes its `6e-7 m` source precision.
The committed differential test uses tolerances of `5e-10` for the ERFA-driven
EOP outputs, `2e-11 rad` for fundamental arguments, `1e-13 m` for DEHANT, and
`5e-7 m` for HARDISP. The full test suite also covers frames, signs, units,
end-to-end observation effects, read-only NumPy inputs, MPI imports, and package
contents.

On the migration host (CPython 3.14, GCC 9.4, best of five runs), the final
HARDISP Cython kernel took `76.7 us` for one Onsala sample versus `78.5 us` for
the former f2py call. The validated ERFA facade took `125.6 us`, including UTC
to TT conversion, leap policy, and BLQ checks. The DEHANT kernel took `4.2 us`
and its full facade took `42.2 us`; the difference is likewise validation plus
ERFA epoch conversion. These figures distinguish kernel performance from the
deliberately stronger Python boundary rather than hiding both in one number.

## Maintenance rules

- Keep calendar and time-scale logic in the facade and ERFA.
- Do not enable compiler fast-math for the Cython core.
- Transcribe coefficient changes mechanically and extend the frozen-grid test.
- Review the HARDISP upper-bound policy when new UTC information is adopted.
- Keep the derived Cython source in every distribution.
- Reject any Fortran or f2py source that reappears in a built archive.
