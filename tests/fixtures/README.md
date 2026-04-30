# Dyana Test Fixtures

These trajectories are intentionally small and belong in `tests/fixtures/`.
They are suitable for fast parser, topology, and smoke tests. Larger
demonstration trajectories live in `examples/`.

## Conventions

- Lengths are in Angstrom.
- Time spacing is the interval between stored trajectory frames.
- Fixtures should stay small enough to run in the default test suite.
- Expected scientific outputs should be tested with compact assertions whenever
  possible, not by comparing large generated files.

## Trajectories

| File | Format | System | Frames | Stored-frame spacing | Box | Intended use |
| --- | --- | --- | ---: | --- | --- | --- |
| `water128.xyz` | XYZ | 128 H2O | 10 | 1 fs | cubic, 15.67 | XYZ parsing, water topology, RDF smoke tests |
| `ca(bf4)2_thf.lmp` | LAMMPS dump | 200 THF + 2 Ca(BF4)2 | 10 | 2000 fs | cubic, 30.5247410713473 | LAMMPS parsing, excluded-ion/topology smoke tests |

## Fixture Placement

The files should remain in `tests/fixtures/`: they are deterministic test inputs,
not user-facing examples. If a fixture grows large or becomes primarily
documentation material, move it to `examples/` and keep only a reduced fixture
here.

