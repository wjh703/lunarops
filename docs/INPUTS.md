# Inputs and data contracts

## Normal points

External source data enters through one explicit program:

```text
MINI or CRD/FRD -> NormalPointsConvert -> NormalPointFile
```

The canonical file is typed ASCII (`.txt` or `.txt.gz`). It stores station and
reflector identity, a two-part UTC transmit epoch, two-way light time, two-way
uncertainty, pressure, temperature, humidity, wavelength, and source codes.

Every downstream LLR program accepts `inputFileNormalPoints` or
`inputFilesNormalPoints` according to its declared cardinality. It does not
auto-detect external formats.

`NptRecord.uncertainty_two_way_s` is the only observation uncertainty input.
Estimation uses `0.5 * c * uncertainty_two_way_s` as one-way range sigma.
For CRD record `11`, LunarOps uses the supplied `bin_rms` directly as that
two-way uncertainty. The normal-point window and number of returns are not
retained by the canonical LLR artifact. LLR record epochs are interpreted as
ground transmit times; the generic CRD epoch-event field is ignored.

## Time

`Epoch(jd1, jd2, scale)` is the only runtime scalar time type. Normal-point and
observation-equation files require UTC. Model boundaries explicitly convert to
TT or TDB through ERFA; ephemeris target 16 is not used for time conversion.

Configuration intervals use `[start, endExclusive)`. `null` means no bound.

## Ephemerides

The `calceph` backend accepts a SPICE kernel directory containing at least one
`.bsp` position kernel and one `.bpc` lunar-orientation kernel. Only those
binary kernels are loaded. LunarOps discovers the unique BPC orientation
target through CALCEPH and requires its records to use ICRF (SPICE frame code
1, historically labeled `J2000`).

```yaml
ephemerides:
  type: calceph
  directory: "{dataDir}/kernels/inpop21a"
  lunarRelativisticScaleConvention: alreadyScaled
  longitudeLibrationCorrection: inpop21a
```

## Catalogs

Use `stationCatalog: builtin` and `reflectorCatalog: builtin`, or paths to typed
station/reflector catalog text files. File headers fix ITRF and Moon PA frames,
respectively. `ReflectorCatalogCreate` imports PA coordinates from CSV into a
typed reflector catalog. `LlrProcessing.writeResults` can publish an updated
reflector catalog after estimation.

## Configuration

Run configuration remains YAML (`.yml` or `.yaml`). It is distinct from native
scientific artifacts, even where structured report payloads use YAML scalar
syntax after an LunarOps type header.
