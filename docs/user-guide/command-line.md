# Command line

LunarOps is run as a Python module (or through the `lunarops` console script
that the package installs):

```bash
python -m lunarops <command> [options]
```

| Command | Purpose |
|---|---|
| `run CONFIG` | Execute the program sequence of a configuration file. |
| `validate CONFIG` | Check a configuration without running it. |
| `list-programs` | List the registered programs. |
| `describe-program NAME` | Print the fields of one program with types, defaults and descriptions. |
| `list-classes [CATEGORY]` | List the registered model classes, optionally of one category. |
| `schema [-o FILE]` | Write the JSON Schema of configuration files for [YAML editors](configuration.md#editor-support). |

## `run`

```bash
python -m lunarops run CONFIG [--set NAME=VALUE ...] [--working-dir DIR] [--workers N | --mpi]
```

`run` expands the configuration (`extends`, variables, loops, conditions),
resolves the model defaults and checks every program call, as `validate` does,
before the first program starts. It then calls each program in order. Each call is
announced with a numbered banner, and the run ends with the number of calls
and the elapsed time.

`--set NAME=VALUE`
:   Replace the value of a variable declared in `variables:`. The value is
    parsed as YAML, so `--set stepSeconds=300` gives a number and
    `--set stations=[APOLLO,GRASSE]` a list. The variable must already exist in
    the file; repeat the option for several variables.

`--working-dir DIR`
:   Directory against which every relative path of the configuration is
    resolved, for inputs and outputs alike. Defaults to the current directory.

`--workers N`
:   Start N local worker processes (N ≥ 1) that evaluate observation or epoch
    tasks for the main process, which executes the program sequence. Needs no
    MPI; cannot be combined with `--mpi`. Without `--workers` or `--mpi` the
    run is serial. See [Parallel execution](parallel-execution.md).

`--mpi`
:   Run as an MPI job. Rank 0 executes the program sequence; the other ranks
    evaluate observation or epoch tasks for it. At least two ranks are
    required. See [Parallel execution](parallel-execution.md).

## `validate`

```bash
python -m lunarops validate CONFIG [--set NAME=VALUE ...] [--working-dir DIR] [--resolve]
```

`validate` performs every check that does not need scientific computation:

- the configuration structure, all field types, ranges and choices;
- every class declaration against its registered schema;
- that no two outputs of the run write the same path and no program reuses a
  path for two of its own fields;
- that every input file exists or is produced by an earlier program, and that
  its artifact type matches what the program expects (existing native files
  are identified from their header);
- the file extension of every native input and output (`.txt[.gz]` or
  `.dat[.gz]`).

It prints `valid: N program call(s)` on success and raises an error naming the
offending field otherwise.

With `--resolve`, input files need not exist. Instead, the command prints the
expanded configuration with all defaults filled in: the resolved `models`
section and one entry per program call. This is the exact
configuration the programs receive.

## `describe-program` and `list-classes`

```bash
python -m lunarops describe-program LlrProcessing
python -m lunarops list-classes stationDisplacement
```

`describe-program` prints the top-level fields of a program as YAML. The
[program reference](../reference/programs.md) shows the same information with
nested fields, and the [model class reference](../reference/classes.md) lists
the options of every class.

## Exit status and errors

A run stops at the first error and exits with a non-zero status. Configuration
errors name the full path of the offending key, for example
`LlrProcessing.processingSteps[2].robustWeighting.k1`. Outputs are written to
a temporary file and moved into place when complete, so a failed program
leaves no partial file behind.
