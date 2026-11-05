# Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/), the
Material theme and [mkdocstrings](https://mkdocstrings.github.io/) from the
Markdown files in `docs/` and the docstrings of the package.

```bash
uv sync --group docs
uv run mkdocs serve           # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict  # build into site/, failing on broken links
```

## Generated pages

Three groups of pages are generated from the code and must not be edited by
hand:

| Page | Source |
|---|---|
| `reference/programs.md` | program specifications: summaries, artifact slots, field schemas and descriptions |
| `reference/classes.md` | class registry: categories, types, schemas, class docstring summaries |
| `reference/api/*.md` | package layout: one `::: module` directive per public module |

Regenerate them after changing a schema, a field description, a class summary
or the module layout:

```bash
python scripts/generate_reference_docs.py
```

CI runs `generate_reference_docs.py --check` and fails when a generated page
is out of date.

## Docstrings

Docstrings follow the NumPy convention and are checked by Ruff's `pydocstyle`
rules (`ruff check lunarops`). Every public module, class, function and method
has one:

- a one-line summary (imperative for functions, a noun phrase for classes);
- `Parameters`, `Returns`, `Raises` and, for records, `Attributes`;
- the unit, frame, time scale and sign convention of every physical quantity,
  and the shape of every array;
- `Notes` or `References` where a published model is implemented, citing the
  [bibliography](../reference/references.md) by author and year.

Constructor parameters are documented in the class docstring. The summary line
of a registered model class is also its description in the model class
reference.

## Configuration field descriptions

Every field of a program or class schema has a `description=`. It states what
the field means, its unit and its range, and is shown by `describe-program`
and in the reference. Keep it to one or two sentences; longer explanations
belong in the user guide.

## Writing style

Pages describe the current behaviour in the present tense. Changes are
recorded in the version history, not in the manual.
