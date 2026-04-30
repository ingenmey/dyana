# Dyana Productionization Checklist

This checklist tracks the productionization outline for Dyana. Use the status
markers consistently:

- `[x]` done
- `[~]` partially done / started
- `[ ]` not started

Framework-specific architecture work now lives primarily in
[analysis_framework_checklist.md](D:/python/dyana/docs/analysis_framework_checklist.md).
This productionization checklist should stay aligned with that document and
avoid carrying competing framework guidance.

## 1. Stabilize Current Behavior

- [x] Add project/package metadata (`pyproject.toml`).
- [x] Declare core dependencies (`numpy`, `scipy`, `networkx`, `matplotlib`).
- [x] Declare optional dependencies for Voronoi support (`pyvoro`).
- [x] Declare development dependencies (`pytest`, `ruff`).
- [x] Add a console entry point (`dyana = "main:cli"`).
- [x] Keep at least one simple local entry path while packaging work is in progress.
- [x] Add package marker files for `analyses` and `core`.
- [x] Replace placeholder README with basic usage and development notes.
- [x] Add smoke/unit tests for reusable helpers.
- [x] Add compile verification as an explicit development command.
- [~] Fix encoding/text cleanliness.
  - Actual source text appears UTF-8 clean in Python-level checks.
  - Data-file headers still use Unicode symbols in several places; decide whether to keep UTF-8 or switch headers to ASCII.
- [x] Add example trajectories.
  - `examples/xyz/water128.xyz`: AIMD water trajectory, 1000 frames, 1 fs frame spacing.
  - `examples/lammps/naclo4_h2o.lmp`: classical MD NaClO4/water trajectory, 100 frames, 500 fs frame spacing.
  - Basic RDF input logs and reference outputs are present for both examples.

## 2. Package Structure

- [~] Start moving toward a package layout.
  - Current state still uses top-level modules (`main.py`, `utils.py`, `geometry.py`, `config_schema.py`).
  - `analyses` and `core` are now importable packages.
- [ ] Move to `src/dyana/` layout.
- [ ] Move `main.py` into `dyana.cli`.
- [ ] Replace top-level imports such as `from utils import ...` with explicit package imports.
- [ ] Split trajectory reading, topology detection, and data objects into smaller modules.
- [ ] Add `dyana.__version__`.
- [~] Add `docs/` reference pages for architecture and analyses.
  - Productionization checklist exists.
  - Architecture and per-analysis reference pages are still missing.

## 3. Separate CLI Interaction From Scientific Logic

- [x] Document the target framework split in the framework checklist.
- [~] Add typed config scaffolding in `config_schema.py`.
- [x] Add per-analysis config objects, starting with RDF.
- [~] Convert analyses so prompting only builds config objects, using the framework checklist plan.
- [x] Make analyses runnable directly from config, starting with RDF.
- [ ] Return result objects from analyses only after a shared result/output pattern is agreed.
- [ ] Rebuild interactive wrappers on top of the new framework where they still make sense.
- [ ] Add JSON/YAML non-interactive run mode.

## 4. Replace Global Prompt State

- [x] Introduce an `InputProvider` abstraction.
- [x] Add `InteractiveInputProvider`.
- [x] Add `FileInputProvider`.
- [x] Add `NullInputProvider` or config-only provider.
- [~] Remove global `input_file` and `log_file` state from core execution paths.
  - `utils.py` now delegates to provider objects.
  - A module-level default provider remains as compatibility state for existing prompt wrappers.
- [~] Pass input/log providers through CLI/config-building code only.
  - Provider classes exist and `utils.py` is a compatibility wrapper.
  - Existing analyses still import prompt wrappers directly.

## 5. Formalize Configuration

- [x] Add `FrameLoopConfig`.
- [x] Add `TopologyConfig`.
- [x] Add `AnalysisRunConfig`.
- [x] Validate positive topology/frame-loop parameters.
- [ ] Load topology config from `config.json` into `TopologyConfig`.
- [ ] Validate unknown elements in config against known atomic properties.
- [ ] Allow config path from CLI.
- [ ] Save resolved config into every output directory.
- [~] Include Dyana version, Python version, dependency versions, and git commit in run metadata.
  - `output_metadata.py` writes Python/dependency/git metadata.
  - Dyana package version is still missing.

## 6. Strengthen Trajectory I/O

- [x] Sort LAMMPS atom rows by `id` when the dump includes an `id` column.
- [~] Add trajectory parser tests.
  - Tiny LAMMPS row-order test exists.
  - Fixture-based XYZ and LAMMPS first-frame parser tests exist.
  - Missing-coordinate LAMMPS input is tested.
- [ ] Create a common `Frame` object.
- [ ] Make trajectory readers iterable.
- [ ] Use `StopIteration` for normal end-of-trajectory behavior.
- [~] Validate malformed XYZ/LAMMPS rows with clear errors.
  - Missing LAMMPS coordinate columns are tested.
  - Malformed XYZ and other LAMMPS malformed cases are still missing.
- [ ] Detect and report missing required LAMMPS columns.
- [ ] Record coordinate convention (`x/y/z`, `xu/yu/zu`, future scaled coordinates).
- [ ] Document orthorhombic-box assumption.
- [ ] Add explicit unsupported/triclinic box diagnostics.

## 7. Centralize Periodic Geometry

- [x] Add shared periodic geometry module (`geometry.py`).
- [x] Add `wrap`.
- [x] Add `minimum_image`.
- [x] Add `distance_squared`.
- [x] Add `unwrap_around_reference`.
- [x] Add `periodic_center`.
- [x] Wire shared helpers into core topology and common metrics.
- [x] Wire shared helpers into selected PBC-heavy analyses.
- [ ] Finish replacing repeated PBC math in remaining side/disabled modules.
- [ ] Add future design notes for triclinic support.

## 8. Harden Molecule Recognition

- [x] Add explicit topology tests for water.
  - Tiny water fixture recognizes one `H2O` compound with 128 members and stable `H1/H2/O1` labels.
- [ ] Add explicit topology tests for ions / excluded elements.
- [ ] Add explicit topology tests for molecules crossing periodic boundaries.
- [ ] Make bond rules configurable per element pair.
- [ ] Store topology/bond criteria in output metadata.
- [ ] Warn clearly on unknown elements or missing radii.
- [ ] Validate all members of a compound are isomorphic.
- [ ] Add deterministic molecule atom ordering.
- [ ] Decouple topology detection from mutable trajectory state.
- [ ] Support fixed/static topology loaded from file.
- [ ] Document static vs dynamic topology modes.

## 9. Improve Compound Identity

- [~] Current compound keys include formula, bond-type multiset, and graph hash.
- [ ] Expose stable compound IDs based on structural key.
- [ ] Separate internal compound ID from human display name.
- [ ] Include compound graph hash in logs/metadata.
- [ ] Prevent output-field collisions when compounds share `rep`.
- [ ] Update density and multi-field outputs to use unique internal field names plus display labels.

## 10. Standardize Output Handling

- [ ] Add output-directory option to CLI.
- [ ] Prevent accidental overwrite unless `--force`.
- [x] Add run metadata writer.
- [ ] Add resolved-config writer.
- [ ] Centralize plain-text table writing.
  - Deferred; analyses continue using existing writers such as `HistogramND.save_txt`.
- [ ] Add consistent naming conventions for output files.
- [ ] Put analysis outputs into timestamped or user-selected run directories.
- [~] Include units, frame range, stride, and normalization in headers/metadata.
  - Config-driven RDF writer uses ASCII units and can write metadata.
  - Other outputs still need metadata/header cleanup.

## 11. Normalize Analysis APIs

- [~] Current analyses share `BaseAnalysis` interactive frame loop.
- [x] Keep `BaseAnalysis` as the near-term canonical shared frame-loop/lifecycle base.
- [x] Introduce config-driven setup independent of prompts, following the framework checklist.
- [ ] Add `from_config` constructors only where they reduce boilerplate.
- [ ] Add result dataclasses after a shared result/output pattern is agreed.
- [ ] Move file writing out of calculation classes only after a central output layer exists.
- [x] Treat RDF as the reference analysis while the shared framework is rebuilt.
- [x] Allow temporary breakage of legacy analyses during framework migration.

## 12. Add Tests At Multiple Levels

- [x] Add geometry unit tests.
- [x] Add config validation tests.
- [x] Add framework config-builder tests.
- [x] Add dependency-gated LAMMPS atom-ordering parser test.
- [x] Add `label_matches` tests.
- [x] Add `HistogramND` simple/linear tests.
- [x] Add XYZ parser tests using tiny fixture.
- [x] Add LAMMPS malformed-input tests.
- [x] Add topology tests using tiny water fixture.
- [x] Add RDF counting tests on synthetic frames.
  - RDF config/result normalization tests exist.
  - `RDF.configure()` plus `process_frame()`/`postprocess()` is tested on a one-frame synthetic trajectory.
- [ ] Add CLI smoke tests with scripted input.
- [~] Add integration smoke tests using `examples/xyz/water128.xyz`.
  - Example trajectory, RDF input log, and reference RDF output are documented.
  - Automated opt-in slow test is still missing.
- [~] Add integration smoke tests using `examples/lammps/naclo4_h2o.lmp`.
  - Example trajectory, RDF input log, and reference RDF output are documented.
  - Automated opt-in slow test is still missing.

## 13. Add CI And Tooling

- [~] Add `ruff` to dev dependencies.
- [ ] Add formatter/linter configuration strict enough for CI.
- [ ] Add `pytest` or standardize on `unittest` in docs and CI.
- [ ] Add GitHub Actions or equivalent CI workflow.
- [ ] Run compile checks in CI.
- [ ] Run unit tests in CI.
- [ ] Run package build check in CI.
- [ ] Add optional slow/integration test marker for example trajectories.

## 14. Logging Instead Of Print

- [ ] Introduce module loggers.
- [ ] Add `--quiet` / `--verbose`.
- [ ] Route diagnostics to log file when requested.
- [ ] Keep interactive prompts/user-facing text separate from diagnostics.
- [ ] Replace progress `print` calls in frame loops with logging/progress helpers.

## 15. Error Handling

- [ ] Add `DyanaError`.
- [ ] Add `TrajectoryFormatError`.
- [ ] Add `TopologyError`.
- [ ] Add `SelectionError`.
- [ ] Add `AnalysisConfigError`.
- [ ] Replace generic `KeyError`, `IndexError`, and broad `ValueError` paths where user action is needed.
- [ ] Improve messages for unmatched labels and missing compounds.

## 16. Dependency Hygiene

- [x] Declare core dependencies in `pyproject.toml`.
- [x] Declare `pyvoro` as an optional extra.
- [ ] Move optional imports such as `pyvoro` inside optional analysis setup.
- [ ] Fail gracefully when optional dependencies are missing.
- [ ] Avoid heavy optional imports at package import time.

## 17. Performance Work

- [ ] Add profiling scripts.
- [ ] Add benchmark notes for example trajectories.
- [ ] Measure per-frame topology-recognition cost.
- [ ] Cache selectors when topology is static.
- [ ] Avoid recomputing topology unless needed.
- [ ] Review memory use in cluster/DACF/CMSD correlation trackers.
- [ ] Consider streaming autocorrelation implementations.

## 18. Data Model Typing And Invariants

- [ ] Convert suitable data containers to dataclasses.
- [ ] Document invariants for `Atom`, `Molecule`, `Compound`, and trajectory state.
- [ ] Add tests for label/global/local-index consistency.
- [ ] Add serialization-safe metadata representations separate from cyclic runtime objects.

## 19. Frame Indexing

- [ ] Document zero-based internal vs one-based user frame numbering.
- [ ] Replace `nframes=-1` sentinel with `None` in config-level APIs.
- [ ] Remove the `nframes=-1` sentinel once the framework path is ready to replace it cleanly.
- [ ] Store frame-indexing convention in output metadata.

## 20. Reduce Duplicate Analysis Code

- [x] Add `collect_atom_indices`.
- [x] Add `collect_indices_for_compounds`.
- [x] Add `build_atom_to_molecule`.
- [~] Use shared selection helpers in RDF, DACF, neighbor-count, and tetrahedral order analyses.
- [ ] Continue migrating ADF, ADF3B, percolation, PCCF, CMSD, and CDF selection paths.

## 21. Review Scientific Normalizations

- [ ] Audit RDF normalization for same-compound/self-pair selections.
- [ ] Audit ADF and ADF3B `1/sin(theta)` handling near 0 and 180 degrees.
- [ ] Audit three-body normalization against actual valid triplets under cutoffs.
- [ ] Document density normalization semantics.
- [ ] Audit DACF finite-size correction assumptions.
- [ ] Audit tetrahedral `S` formula.
- [ ] Audit charge MSD behavior for created/annihilated charges.
- [ ] Add references/definitions to analysis docs.

## 22. Handle Edge Cases

- [ ] Empty trajectory.
- [ ] One-frame trajectory.
- [ ] Zero matching atoms.
- [ ] Missing selected compound in some frames.
- [ ] Zero/invalid box dimension.
- [ ] Same reference and observed selections.
- [ ] Unknown element symbols.
- [ ] Dynamic topology changing labels.
- [ ] Multiple compounds with same formula.
- [ ] Invalid bin ranges or zero bin widths.
- [ ] Missing optional dependency.
- [ ] Output file already exists.

## 23. Documentation

- [x] Add basic README.
- [ ] Add installation docs.
- [ ] Add supported trajectory format docs.
- [ ] Add molecule-recognition docs.
- [ ] Add atom-labeling docs.
- [ ] Add non-interactive config examples.
- [ ] Add output file reference.
- [ ] Add per-analysis docs with purpose, assumptions, parameters, normalization, outputs, and examples.
- [x] Add examples README explaining the two new trajectories and expected smoke analyses.
- [x] Add fixtures README explaining test trajectories and placement.

## 24. Versioning And Reproducibility

- [ ] Add `dyana.__version__`.
- [ ] Add changelog.
- [~] Add output metadata with version/dependency/config information.
  - Metadata writer includes Python, dependency, git, and analysis parameter information.
  - Dyana version and full resolved config are still missing.
- [ ] Add reproducibility guidance for citing analysis settings.
- [ ] Preserve behavior with tests before changing scientific definitions.

## Immediate Next-Step Candidates

1. Continue the current focus in [analysis_framework_checklist.md](D:/python/dyana/docs/analysis_framework_checklist.md): settle the programmatic entry path and replace the `nframes=-1` sentinel in the framework path.
2. Use RDF as the only analysis that must stay functional during framework work; legacy analyses can be repaired later.
3. Keep the productionization checklist aligned with the framework checklist while framework work is in progress.
4. After the framework shape is stable, resume version/metadata/output-directory work here.
5. Then add opt-in slow smoke tests for the documented RDF examples.
