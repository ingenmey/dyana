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

