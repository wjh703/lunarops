# Extending LunarOps

New models and programs are added inside the package. They register
themselves with the class registry or the program registry, declare their
configuration as a `ConfigSchema`, and then appear in `list-classes`,
`describe-program`, `validate` and the generated reference pages.

## Configuration schemas

A schema is a tuple of field declarations built with the helpers of
`lunarops.config.schema`:

```python
from lunarops.config.schema import ConfigSchema, boolean, number, path, sequence, string

SCHEMA = ConfigSchema(
    fields=(
        path("coefficientFile", required=True, non_empty=True, description="Coefficient file in BLQ format."),
        number("scale", default=1.0, gt=0, description="Scale factor applied to the displacement."),
        string("mode", default="full", choices=("full", "radialOnly"), description="Components to apply."),
    ),
    type_name="myDisplacement",
)
```

Helpers exist for `string`, `boolean`, `integer`, `number`, `path`, `time`,
`mapping` (with `nested=` for a sub-schema), `sequence` (with `item_kind` or
`item_nested`), `class_config` and `class_list`. Constraints: `required`,
`default`, `allow_none`, `choices`, `gt`/`ge`/`lt`/`le`, `min_items`/`max_items`,
`unique`, `non_empty`. A `validator` function on the schema checks
relationships between fields and may normalize values. Always give a
`description` that states meaning, unit and range: it is shown by
`describe-program` and in the reference.

## Model classes

A model class provides a `SCHEMA` and a `from_config(cfg, ctx)` factory. The
first line of its docstring is used as its summary in the reference.

```python
class MyDisplacement:
    """Example station displacement proportional to a coefficient file."""

    SCHEMA = ConfigSchema(..., type_name="myDisplacement")

    @classmethod
    def from_config(cls, cfg, ctx):
        return cls(ctx.resolve_path(cfg["coefficientFile"]), scale=cfg["scale"])
```

Register it in `lunarops/classes/builtin_registry.py` by adding the category,
type name and import path to `BUILTINS`:

```python
("stationDisplacement", "myDisplacement"): "lunarops.classes.displacement.my_module:MyDisplacement",
```

Built-in classes are imported only when a configuration uses them.
Alternatively, `lunarops.config.registry.register_factory(category, type_name,
factory, schema=..., scope=...)` registers a factory at import time; the
`scope` `models` allows the class in `models:`,
`program` only on program entries.

The object must implement the interface its category expects:

| Category | Interface |
|---|---|
| `troposphere` | slant delay of one leg from meteorology and elevation; an elevation floor |
| `relativity` | one-way gravitational delay between two BCRS events |
| `stationDisplacement` | ITRF displacement of a station at a UTC epoch |
| `reflectorDisplacement` | LCRS displacement of a reflector at a TDB epoch |
| `rangeBias` | correction lookup by station and date |
| `parametrization` | the `Parametrization` interface below |

See the existing implementations in `lunarops/classes/*` and the
[API reference](../reference/api/index.md) for the exact signatures.

## Parametrizations

A parametrization subclasses `lunarops.classes.parametrization.base.Parametrization`
and owns a contiguous block of design-matrix columns:

| Method | Responsibility |
|---|---|
| `setup(batch)` | choose the parameters from the observed batch; return their `ParameterName`s |
| `design(batch, out)` | write the partial derivatives of the one-way residual into `out`, shape `(len(batch), n)` |
| `x0()` | current parameter values |
| `reduce(batch)` | contribution of the current values to the computed observable |
| `update(delta)` | apply a correction to the parameters and the model state |
| `max_update_norm(delta)` | the convergence measure of the block |
| `state()` | values stored in the processing state for a later restart |

Partial derivatives come from the observation model's columns in
`EquationBatch` (reflector PA position, station ITRF position, lunar
orientation). A parametrization that needs a new partial adds it to the
observation model and to `EquationBatch`. Check analytic partials against
central differences of the full model, as the existing tests do.

## Programs

A program is a function `(config, context)` decorated with `@program`:

```python
from lunarops.config.schema import number
from lunarops.programs.registry import ArtifactSlot, program, program_spec


@program(
    program_spec(
        name="MyProgram",
        summary="One-line description shown by list-programs and the reference.",
        inputs=(ArtifactSlot("inputFileStationCatalog", "stationCatalog", description="Native ITRF station catalog."),),
        outputs=(ArtifactSlot("outputFileResults", "observationResult", description="Output table."),),
        fields=(number("thresholdM", default=1.0, gt=0, description="Threshold, in metres."),),
    )
)
def my_program(config: dict, context):
    """Describe what the program reads, computes and writes."""
    catalog_path = context.resolve_path(config["inputFileStationCatalog"])
    ...
```

`ArtifactSlot` declares each file field with its artifact type, whether it is a
list (`many`), optional (`required=False`), binary or an external format. The
slots let `validate` check the file graph of a scenario. Add the module to
`_PROGRAM_MODULES` in `lunarops/programs/registry.py` so that it is imported
when programs are listed or run. Programs that use the observation model take
`observation_fields()` from `lunarops.programs.specs`.

## After a change

```bash
python scripts/generate_reference_docs.py
uv run ruff check lunarops
uv run -m pytest tests -q
```

The generator updates the program, class and API reference pages; CI fails if
they are stale.
