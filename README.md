# Dyana

Dyana is a molecular dynamics trajectory post-processing toolkit. It currently
provides an interactive command-line workflow for loading XYZ or LAMMPS dump
trajectories, recognizing molecular compounds, assigning stable atom labels, and
running structural, correlation, cluster, and proton/charge-transfer analyses.

## Current CLI

```bash
python main.py trajectory.xyz
python main.py trajectory.lammpstrj -i input.txt -l input.log
```

When installed as a package, the same entry point is exposed as:

```bash
dyana trajectory.lammpstrj
```

Interactive runs write:

- `input.log`: prompt/answer replay log
- `dyana.log`: full human-readable console transcript, including prompts and replies

Both default to the selected output directory.

## Development

Install the package in editable mode with development tools:

```bash
python -m pip install -e ".[dev]"
```

Run the lightweight phase-1 test suite:

```bash
python -m unittest discover -s tests
```

## Productionization Notes

The current interactive workflow is preserved. Phase-1 hardening adds package
metadata, shared periodic-boundary geometry helpers, shared atom-selection
helpers, typed configuration scaffolding for future non-interactive runs, and
unit tests around the most reusable behavior.

## Reference Docs

Current architecture/reference docs live in [docs](D:/python/dyana/docs):

- [runtime_topology_model.md](D:/python/dyana/docs/runtime_topology_model.md)
- [analysis_config_lifecycle.md](D:/python/dyana/docs/analysis_config_lifecycle.md)
- [output_table_format.md](D:/python/dyana/docs/output_table_format.md)
- [interactive_console_output.md](D:/python/dyana/docs/interactive_console_output.md)
- [documentation_comment_policy.md](D:/python/dyana/docs/documentation_comment_policy.md)
