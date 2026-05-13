# Dyana

Dyana is a molecular dynamics trajectory post-processing tool for interactive
analysis of molecular structure and local order in XYZ and LAMMPS trajectories.

## Implemented analyses

- radial distribution function (`rdf`)
- angular distribution function (`adf`)
- one-dimensional density profile (`dens`)
- neighbour-count probability (`ncount`)
- tetrahedral order parameters (`top`)
- Steinhardt `q6`, Lechner-Dellago `qbar6`, and global `Q6` (`q6`)
- local structure index (`lsi`)

## Install

```bash
python -m pip install .
```

Run:

```bash
dyana trajectory.xyz
```
