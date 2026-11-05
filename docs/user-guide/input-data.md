# Input data

LunarOps separates external data from its own files. External sources enter
through a conversion program once; every later program reads only typed
native files. This page describes each kind of input.

| Input | External source | Native artifact | Created by |
|---|---|---|---|
| Normal points | CRD v2, MINI | `normalPoint` | `NormalPointsConvert` |
| Earth orientation | IERS 20 C04, finals2000A | `earthOrientationParameter` | `IersC04EarthOrientationParameter`, `IersRapidEarthOrientationParameter`, `EarthOrientationParameterMerge` |
| Station coordinates | YAML list or inline | `stationCatalog` | `StationCatalogCreate` |
| Reflector coordinates | YAML list or inline | `reflectorCatalog` | `ReflectorCatalogCreate` |
| Body masses (dynamics) | YAML list | `massCatalog` | `MassCatalogCreate` |
| Ephemeris | SPICE `.bsp` + `.bpc` kernels | read directly | `ephemerides: {type: calceph}` |
| Ocean loading | BLQ coefficients | read directly | `iers2010OceanTidalLoading` |
| Ocean pole tide | IERS coefficient grid | read directly | `iers2010OceanPoleTide` |
| Gravity fields (dynamics) | ICGEM `.gfc` | read directly | `LunarOrbitPropagation.gravityFields` |

## Normal points

```text
CRD v2 or MINI files ──► NormalPointsConvert ──► normalPoint file
```

`NormalPointsConvert` accepts files and directories (searched recursively).
The format of each file is recognized from its first line; a known suffix
(`.crd`, `.frd`, `.npt`, …) must agree with the content. Native `normalPoint`
files are accepted too, so converted data sets can be combined. The output
holds, per normal point: the UTC transmit epoch as a two-part Julian date,
station and reflector names and source codes, the two-way light time, its
two-way uncertainty, pressure, temperature, humidity and laser wavelength.
An import report records the source checksums, record counts and every
rejected line.

The two-way uncertainty in the file is the only weight input of the
estimation; the one-way range sigma is `0.5 c uncertainty_two_way_s`. It is
never replaced by a station table.

### CRD

Only CRD version 2 normal-point data is imported. The importer requires:

- exactly one `H1` header declaring version 2;
- an `H2` station header with a supported station name and a matching pad ID,
  and a UTC realization code 3 (USNO), 4 (GPS) or 7 (BIPM);
- an `H3` target header naming a supported reflector (`APOLLO11`, `APOLLO14`,
  `APOLLO15`, `LUNOKHOD1`, `LUNOKHOD2`);
- `H4` sessions of normal-point data type 1, shorter than one day;
- record `11` with exactly 14 fields (including the SNR), epoch event 2
  (ground transmit time), a positive time of flight and a positive bin RMS;
- a `C0` configuration record giving the wavelength.

Structural errors (missing or misplaced headers, unsupported versions or
codes) stop the import with the file name and line. Individual invalid
normal-point rows are rejected and counted in the report.

The uncertainty must be chosen explicitly with `crdUncertaintyPolicy`:

`binRmsAsSigma`
:   Use the bin RMS of record 11 as the two-way sigma. No `√N` reduction is
    applied; the bin RMS, return count and window length stay available in the
    import metadata.

`fixed`
:   Assign `crdUncertaintyTwoWayS` (seconds) to every normal point.

Meteorological data come from the record `20` nearest in UTC to each normal
point, also across midnight, within `crdMeteorologyMaxAgeSeconds` (default
1800 s).

### MINI

MINI is a fixed-width, one-line-per-normal-point format with the launch epoch
(UTC, 100 ns resolution), the two-way light time and uncertainty (0.1 ps),
the reflector and station codes, the meteorological data and the wavelength.
The unit of the temperature field must be declared with `miniTemperatureUnit`
(`deciCelsius`, default, or `deciKelvin`).

### Station and reflector identity

Station and reflector identities are exact. The five-digit MINI station codes
and the CRD station names and pad IDs map to fixed station names (for example
`APOLLO`, `GRASSE`, `MATERA`, `MCDONALD`, `MLRS1`, `MLRS2`, `WETTZELL`,
`HALEAKALA`); MINI reflector codes 0–4 map to `APOLLO11`, `LUNOKHOD1`,
`APOLLO14`, `APOLLO15`, `LUNOKHOD2`. The station and reflector catalogs must use
the same keys. To process data under a different key, set `overrideStation` or
`overrideReflector` on the observation program.

## Earth orientation

```text
eopc04.1962-now.txt ──► IersC04EarthOrientationParameter ──┐
                                                           ├─► EarthOrientationParameterMerge ──► eop_merged.txt
finals2000A.all ─────► IersRapidEarthOrientationParameter ─┘
```

The C04 converter reads both the IERS 14 C04 and 20 C04 layouts (16 and 21
columns). The finals2000A converter takes the Bulletin A values with their
`I`/`P` flags. The native EOP file has one row per UTC day with the columns
`mjdUtc` (d), `xpArcsec`, `ypArcsec` (arcsec), `ut1MinusUtcSec`, `lodSec` (s),
`dxArcsec`, `dyArcsec` (arcsec) and three provenance columns: `source`,
`quality` and `sourceSha256`. The merge keeps C04 on every day where it
exists and fills later days from Bulletin A. The `quality` column marks each
row as `final` (C04), `rapid` or `prediction` (Bulletin A).

Observation programs use the file through
`earthRotation: {type: file, file: ...}`. With `allowPredicted: false`,
prediction rows are dropped, so an epoch that needs them fails.

## Station catalog

`StationCatalogCreate` writes ITRF positions and linear velocities. Each entry
has a `key` and exactly one position form:

```yaml
stationCoordinates:
  - {key: APOLLO, xyzM: [-1463998.9079, -5166632.7663, 3435012.8921],
     velocityMPerYear: [-0.0139, -0.0003, -0.0023], positionEpochUtc: "2000-01-01T00:00:00"}
  - {key: NEWSITE, longitudeDeg: 116.391, latitudeDeg: 39.907, heightM: 45.0}
```

`xyzM` is ITRF in metres; the geodetic form uses WGS84 longitude (east
positive) and latitude (north positive) in degrees and ellipsoidal height in
metres, and is converted to ITRF XYZ before writing. `velocityMPerYear`
defaults to zero and `positionEpochUtc` to 2010-01-01. The same list can be
read from a YAML file through `inputFileStationCoordinates`; the file holds a
mapping with a `stationCoordinates` key.

## Reflector catalog

`ReflectorCatalogCreate` writes reflector positions in the lunar principal-axis
frame of the ephemeris used for processing:

```yaml
reflectorCoordinates:
  - {key: APOLLO11, xyzM: [1591966.6407, 690699.4669, 21003.7578]}
```

Reflector coordinates are tied to an ephemeris. Coordinates from one
ephemeris' PA frame are not valid with another ephemeris' orientation.
`LlrProcessing` can publish an updated catalog after estimating the
reflector positions (`writeResults.outputFileReflectorCatalog`).

## Ephemerides

The `calceph` ephemeris reads a directory of SPICE kernels:

```yaml
ephemerides:
  type: calceph
  directory: "{dataDir}/kernels/de440"
  lunarRelativisticScaleConvention: tdbCompatibleLunarSurface
  longitudeLibrationCorrection: none
```

The directory must contain at least one `.bsp` position kernel and one `.bpc`
lunar-orientation kernel; other files are ignored. The lunar orientation
target is discovered in the `.bpc` kernel and must be referred to the ICRF
(SPICE frame code 1). Kernels of the DE, INPOP and EPM series can be used; set
`lunarRelativisticScaleConvention` to the convention of the kernel (see
[Time and reference frames](time-and-frames.md#bcrs-gcrs-and-bcrs-lcrs)).

Each process opens its own CALCEPH handle, so MPI workers read the kernels
locally.

## Displacement coefficients

`iers2010OceanTidalLoading`
:   Ocean-loading amplitudes and phases in the BLQ format, for example from the
    Onsala ocean-loading service. If `model` is given, the BLQ header must
    declare the same ocean-tide model (for example `FES2022b`). Every observed
    station must be present in the file.

`iers2010OceanPoleTide`
:   The ocean pole-tide loading coefficient grid of the IERS Conventions
    (2010), file `opoleloadcoefcmcor.txt`.

## Range-bias tables

`rangeBias: {type: inpop21a}` uses the built-in station and interval biases
published with INPOP21a. `rangeBias: {type: table}` takes a table inline
(`biases:`) or from a YAML file (`file:`):

```yaml
rangeBias:
  type: table
  source: local calibration 2024
  biases:
    - {station: MATERA, start: 2010-01-01, end: 2012-06-01, correctionTwoWayCm: 3.1}
```

Each row applies to the station's normal points with transmit date in
`[start, end)`. Overlapping rows add up. The correction, in centimetres of
two-way light distance, is subtracted from the computed round-trip time. These
are fixed forward-model corrections; to estimate biases, use the
`stationRangeBias` parametrization.

## Inputs for lunar dynamics

`MassCatalogCreate` turns a YAML list of bodies into a `massCatalog`. Each body
has a canonical NAIF-style ID (`SUN`, `MARS BARYCENTER`, `NAIF:2000001`), the
NAIF target number used to query the ephemeris, its GM in `m3/s2` or
`au3/day2`, and one or more group names used to select perturbers. With
`earthMoonCanonical`, the Earth and Moon GM are derived from the Earth–Moon
system GM and the mass ratio, so every model shares one consistent pair.
`configs/lunarops_mass_catalog.yml` holds the DE440 bodies and the 343
asteroids and 30 Kuiper-belt objects of the SB441-N373 set.

Gravity fields for `LunarOrbitPropagation` are read from ICGEM `.gfc` files
with fully normalized static coefficients. The field GM is replaced by the
mass-catalog value so that all models use the same GM. See
[Lunar orbit and rotation](lunar-dynamics.md).
