# Configuration

A LunarOps run is described by one YAML file (`.yml` or `.yaml`). The file is
data only: it contains no code, and every key is checked against a declared
schema.

```yaml
variables:        # named values for placeholders
  dataDir: ~/llr/data
  outputDir: ./output

models:           # run-wide model classes
  ephemerides: {type: chebyshev, directory: "{dataDir}/kernels/de440", lunarRelativisticScaleConvention: tdbCompatibleLunarSurface}
  earthRotation: {type: file, file: "{outputDir}/eop_merged.txt"}
  troposphere: {type: mendesPavlis}
  relativity: {type: iersShapiro}
  stationDisplacement: [{type: iers2010SolidEarthTide}, {type: iers2010PoleTide}]
  reflectorDisplacement: {type: lunarSolidTide}
  rangeBias: {type: none}

programs:         # ordered program calls
  - program: LlrResiduals
    inputFilesNormalPoints: ["{outputDir}/normal_points.txt"]
    inputFileStationCatalog: "{outputDir}/stations.txt"
    inputFileReflectorCatalog: "{outputDir}/reflectors.txt"
    outputFileObservationResults: "{outputDir}/residuals.txt"
```

Only the top-level keys `variables`, `models`, `programs` and
[`extends`](#reusing-configuration-extends) are accepted. All of them are
optional, but a run without `programs` does nothing.

## Naming rules

- Configuration keys are lowerCamelCase and must be spelled exactly. Unknown or
  misspelled keys are errors, not warnings.
- Program names, class `type` values, enumeration choices and catalog keys are
  matched exactly, including case.
- A model class is always declared as a mapping with a `type` key, even when it
  has no other options: `troposphere: {type: mendesPavlis}`.
- Keys that carry a unit end with it: `stepSeconds`, `minElevationDeg`,
  `xyzM`, `velocityMPerYear`, `crdUncertaintyTwoWayS`.

## Variables and placeholders

`variables` maps identifier names to values. A string of the form `{name}`
anywhere in the file is replaced by the variable's value:

- If the whole string is a placeholder (`"{stations}"`), the variable's value
  is inserted with its type, so a list stays a list and a number stays a number.
- Inside a longer string (`"{outputDir}/eop.txt"`), the value is inserted as
  text.

Variables may refer to other variables; cycles and undefined names are errors.
`--set name=value` on the command line replaces a declared variable; the value
is parsed as YAML.

## Program entries

Each item of `programs` is a mapping with a `program` key and that program's
fields. The [program reference](../reference/programs.md) lists all fields;
`python -m lunarops describe-program NAME` prints them as well.

Two control keys are available on every entry:

`loop`
:   Run the entry once per value. `variable` names a loop variable that is
    available as a placeholder inside the entry, and `values` lists its values.

    ```yaml
    - program: LlrResiduals
      loop: {variable: station, values: [APOLLO, GRASSE, MATERA]}
      overrideStation: "{station}"
      outputFileObservationResults: "{outputDir}/residuals_{station}.txt"
      # ...
    ```

`when`
:   A condition, evaluated after placeholder substitution; the entry is skipped
    when it is false. A condition is either a boolean or a mapping with exactly
    one operator:

    | Operator | Operand | True when |
    |---|---|---|
    | `equals` | `[a, b]` | `a == b` |
    | `notEquals` | `[a, b]` | `a != b` |
    | `in` | `{value: v, values: [...]}` | `v` is in the list |
    | `all` | list of conditions | every condition is true |
    | `any` | list of conditions | at least one condition is true |
    | `not` | condition | the condition is false |

    ```yaml
    - program: LlrObservationPrediction
      when: {equals: ["{mode}", "predict"]}
      # ...
    ```

## Reusing configuration: `extends`

A file can start from one or more other files, so that a variant states only
what differs:

```yaml
extends: [base/parameter_estimation.yml, base/ephemeris_epm2021.yml]
variables:
  outputDir: ./output/epm_test
```

The named files are relative to the file that names them and may themselves
use `extends`. They are merged in order and the file itself comes last, so a
later entry wins:

- `variables` and `models` are merged key by key. A variable or a model
  category is replaced as a whole, never merged field by field.
- `programs` is replaced as a whole.

To let a variant change a value deep inside a program entry, make that value a
variable in the base file. A variable can hold a whole list or mapping. The
files in `configs/base/` work this way: `base/parameter_estimation.yml` holds
the processing plan and each `base/ephemeris_*.yml` sets the a priori reflector
coordinates and libration terms of one ephemeris.

`python -m lunarops validate FILE --resolve` prints the result of the merge.

## Model classes: `models` and program entries

Programs that evaluate the observation model need one class of each category:
`ephemerides`, `earthRotation`, `troposphere`, `relativity`,
`stationDisplacement` (a list), `reflectorDisplacement` and, for residuals and
estimation, `rangeBias`. A program uses the declaration on its own entry if
there is one and the one in `models` otherwise, so an entry overrides a
category only for that call. The categories and
their types are listed in the [model class reference](../reference/classes.md);
`{type: none}` switches a delay or displacement model off.

Identical class declarations share one instance for the whole run: an
ephemeris or an EOP table is opened once and reused by every program that
declares it the same way.

## Paths

Relative paths are resolved against the working directory (`--working-dir`,
default: the current directory), not against the location of the
configuration file; only the paths in `extends` are relative to the file. `~` is expanded. Native outputs use `.txt` or `.txt.gz`
for text artifacts and `.dat` or `.dat.gz` for binary tables; gzip
compression is selected by the `.gz` suffix. Normal-equation and covariance
outputs are directories.

## Times

Time fields accept ISO dates (`2024-09-18`) and timestamps
(`2024-09-18T08:00:00`, `2024-09-18T08:00:00Z`, `2024-09-18T08:00:00+08:00`).
Programs that take `utcOffsetHours` read naive timestamps as civil time with
that offset and convert them to UTC; all model evaluation runs in UTC/TT/TDB.
Intervals are half-open, `[start, end)`, unless the field description says
otherwise, and `null` means unbounded. See
[Time and reference frames](time-and-frames.md).

## Scientific fingerprint

`LlrProcessing` stores a fingerprint of its scientific inputs in the processing
state and report: the checksums of the normal-point, catalog, EOP and model
files, the ephemeris kernels, the resolved model configuration and the code.
Options that do not affect the result (progress reports, MPI chunk size, output
level) are excluded. A run that continues from a stored state
(`inputFileProcessingState`) must have the same fingerprint.

## Checking a configuration

```bash
python -m lunarops validate run.yml            # check everything, read input headers
python -m lunarops validate run.yml --resolve  # print the expanded configuration
```

See [Command line](command-line.md#validate).

## Editor support

`configs/lunarops.schema.json` is a JSON Schema of the configuration files,
generated from the program and class schemas. With the YAML extension of
VS Code (or another editor using yaml-language-server), it provides completion
of keys and model types, hover descriptions, enumeration choices and immediate
errors for unknown keys. A file connects to it with a first line such as:

```yaml
# yaml-language-server: $schema=lunarops.schema.json
```

The schema covers names, types, bounds and required keys. Rules that relate
several fields are checked by `validate`, which also runs at the start of every
`run`. `python -m lunarops schema -o FILE` writes the schema to another
location.
