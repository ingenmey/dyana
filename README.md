# Dyana

Dyana is a molecular dynamics trajectory post-processing tool for interactive
analysis of molecular structure, local order, combined distributions, and
clustering in XYZ and LAMMPS trajectories.

## Implemented analyses

- radial distribution function (`rdf`)
- angular distribution function (`adf`)
- combined distribution function (`cdf`)
- proton coupling / transfer-chain correlation (`pccf`)
- dimer existence autocorrelation function (`dacf`, distance and/or nearest-neighbour selector)
- identity autocorrelation function (`idcf`)
- hydrogen-bond percolation pathway analysis (`perc`)
- one-dimensional density profile (`dens`)
- resolved / multi-dimensional neighbour-count analysis (`ncount`)
- cluster composition histogram (`cluster`)
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
