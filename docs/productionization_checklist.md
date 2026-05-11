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

- [x] Add project/package metadata (`setup.cfg` plus `pyproject.toml` build config).
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
  - `framework`, `io_support`, `workflow`, `analyses`, and `core` are now grouped as importable packages.
  - `main.py` and `utils.py` still remain at the top level for now.
- [ ] Move to `src/dyana/` layout.
- [ ] Move `main.py` into `dyana.cli`.
- [ ] Replace top-level imports such as `from utils import ...` with explicit package imports.
- [x] Do not split trajectory reading, topology detection, and data objects further unless a concrete readability or reuse problem appears.
- [ ] Add `dyana.__version__`.
- [~] Add `docs/` reference pages for architecture and analyses.
  - Architecture/reference pages now exist for the runtime topology model, analysis/config lifecycle, output-table format, and documentation/comment policy.
  - Per-analysis reference pages are still missing.

## 3. Separate CLI Interaction From Scientific Logic

- [x] Document the target framework split in the framework checklist.
- [~] Add typed config scaffolding in `framework/config_schema.py`.
- [x] Keep the current `CONFIG_SCHEMA` plus explicit per-analysis config-dataclass design for the supported path; do not collapse those layers as part of current productionization work.
- [x] Add per-analysis config objects, starting with RDF.
- [~] Convert analyses so prompting only builds config objects, using the framework checklist plan.
- [x] Make analyses runnable directly from config, starting with RDF.
- [x] Add a reusable prepared-setup artifact so interactive topology setup can be saved and later reused for compatible systems.
- [x] Make a real Python-driven end-to-end RDF run possible by combining prepared setup loading with analysis configuration.
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
  - Provider classes exist and drive the rebuilt RDF/workflow path directly.
  - Legacy analyses still need migration away from prompt-wrapper imports.

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
  - `io_support/output_metadata.py` writes Python/dependency/git metadata.
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

- [x] Add shared periodic geometry module (`core/geometry.py`).
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
- [x] Make topology rebuilding produce the authoritative runtime topology model used by the supported analyses.
- [x] Enforce canonical local ordering for runtime topology membership rows.
  - Equivalent molecules are canonicalized into template-local order.
  - Direct regression coverage now protects canonical paired-subgroup mapping across equivalent members.
  - For truly equivalent atoms, Dyana uses deterministic conventional numbering rather than pretending there is a chemically privileged distinction.
- [ ] Add explicit topology tests for ions / excluded elements.
- [ ] Add explicit topology tests for molecules crossing periodic boundaries.
- [ ] Make bond rules configurable per element pair.
- [ ] Store topology/bond criteria in output metadata.
- [ ] Warn clearly on unknown elements or missing radii.
- [ ] Validate all members of a compound are isomorphic.
- [x] Keep topology building inside the trajectory layer unless a concrete readability or reuse problem justifies separating it later.
- [ ] Support fixed/static topology loaded from file.
- [x] Document static vs dynamic topology modes.
  - Direct frame-loop tests protect the current static vs dynamic behavior.

## 9. Improve Compound Identity

- [~] Current compound keys include formula, bond-type multiset, and graph hash.
- [x] Use stable structural keys as the primary internal compound-type identity.
- [x] Separate internal compound-type identity from human display name / formula.
- [ ] Include compound graph hash in logs/metadata.
- [ ] Prevent output-field collisions when compounds share `formula`.
- [ ] Update density and multi-field outputs to use unique internal field names plus display labels.

## 10. Standardize Output Handling

- [x] Add output-directory option to CLI.
- [x] Prevent accidental overwrite unless `--force`.
- [x] Add run metadata writer.
- [ ] Add resolved-config writer.
- [x] Centralize plain-text table writing.
  - A shared output writer now exists for the migrated analyses.
  - The migrated analyses now share one documented text-table format.
- [ ] Add consistent naming conventions for output files.
- [x] Put analysis outputs into user-selected run directories.
- [x] Do not add timestamped run directories on top of the current explicit `--output-dir` support.
- [x] Mirror supported interactive console output into `<output-dir>/dyana.log`.
- [~] Include units, frame range, stride, and normalization in headers/metadata.
  - Migrated analyses now share a common output direction and documented table format.
  - Some analysis-specific metadata is still not written consistently.

## 11. Normalize Analysis APIs

- [x] RDF, density, neighbor count, ADF, and tetrahedral order now validate the new `BaseAnalysis` design; the remaining analyses are intentionally out of scope for the current cleanup phase.
- [x] Keep `BaseAnalysis` as the near-term canonical shared frame-loop/lifecycle base.
- [x] Introduce config-driven setup independent of prompts, following the framework checklist.
- [ ] Add `from_config` constructors only where they reduce boilerplate.
- [ ] Add result dataclasses after a shared result/output pattern is agreed.
- [~] Move file writing out of calculation classes once a central output layer exists.
  - RDF, density, neighbor count, ADF, and tetrahedral order now write through a shared output module.
  - Remaining analyses are intentionally left untouched for now.
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
- [x] Add CLI/end-to-end RDF test with scripted input.
- [~] Add integration smoke tests using `examples/xyz/water128.xyz`.
  - Example trajectory, RDF input log, and reference RDF output are documented.
  - A true end-to-end RDF regression test exists in `tests/fixtures/rdf`.
  - Example-level opt-in slow coverage is still missing.
- [~] Add integration smoke tests using `examples/lammps/naclo4_h2o.lmp`.
  - Example trajectory, RDF input log, and reference RDF output are documented.
  - Automated opt-in slow test is still missing.
- [x] Add a true Python-driven end-to-end RDF regression test using prepared setup loading plus `RDF.configure(...)`.

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

- [~] Introduce a shared interactive console helper for supported user-facing output.
- [ ] Introduce module loggers.
- [ ] Add `--quiet` / `--verbose`.
- [ ] Route diagnostics to log file when requested.
- [~] Keep interactive prompts/user-facing text separate from diagnostics.
  - Supported interactive output now goes through a shared console helper and mirrored `dyana.log`.
  - Prompt/answer replay remains separate in `input.log`.
- [~] Replace progress `print` calls in frame loops with logging/progress helpers.
  - Supported analyses now use shared console progress messages.

## 15. Error Handling

- [ ] Add `DyanaError`.
- [ ] Add `TrajectoryFormatError`.
- [ ] Add `TopologyError`.
- [ ] Add `SelectionError`.
- [ ] Add `AnalysisConfigError`.
- [ ] Replace generic `KeyError`, `IndexError`, and broad `ValueError` paths where user action is needed.
- [ ] Improve messages for unmatched labels and missing compounds.

## 16. Dependency Hygiene

- [x] Declare core dependencies in packaging config.
- [x] Declare `pyvoro` as an optional extra.
- [ ] Move optional imports such as `pyvoro` inside optional analysis setup.
- [ ] Fail gracefully when optional dependencies are missing.
- [ ] Avoid heavy optional imports at package import time.

## 17. Performance Work

- [ ] Add profiling scripts.
- [ ] Add benchmark notes for example trajectories.
- [ ] Measure per-frame topology-recognition cost.
- [x] Resolve supported-analysis selections once and reuse canonical local indices across topology rebuilds.
- [x] Keep static-topology mode as the no-rebuild path that avoids unnecessary topology rebuild work.
- [ ] Avoid recomputing topology unless needed.
- [ ] Review memory use in cluster/DACF/CMSD correlation trackers.
- [ ] Consider streaming autocorrelation implementations.

## 18. Data Model Typing And Invariants

- [~] Convert suitable runtime/data containers to dataclasses where that improves clarity.
- [~] Document invariants for `CompoundType`, `CompoundTypeRegistry`, `TopologyFrame`, `ResolvedSelection`, and trajectory state.
  - Legacy `Atom` / `Molecule` / `Compound` invariants are no longer the main supported runtime model.
- [x] Add tests for label/global/local-index consistency.
- [~] Add serialization-safe metadata representations separate from runtime topology objects.
  - Prepared setups already provide part of this path.

## 19. Frame Indexing

- [ ] Document zero-based internal vs one-based user frame numbering.
- [x] Keep `nframes=-1` as the accepted frame-count sentinel in the current frame-loop API.
- [x] Do not replace the `nframes=-1` sentinel unless the frame-loop API itself changes materially.
- [ ] Store frame-indexing convention in output metadata.

## 20. Reduce Duplicate Analysis Code

- [x] Add shared selection/helper utilities needed during the migration away from the legacy object-graph path.
- [x] Move the supported analyses onto the shared topology-frame selection path with resolved selections and canonical local indices.
- [~] Keep trimming shared selection/access helpers so the supported path does not grow parallel legacy-object and topology-frame helper stacks.
- [ ] Migrate additional analyses only when they are explicitly brought back into scope.

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
- [x] Output file already exists.
  - Managed analysis outputs are now rotated through `#1#<filename>`, `#2#<filename>`, and so on unless `--force` is used.

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
- [~] Add reusable prepared-setup metadata and validation.
  - Prepared setup JSON now stores recipe, compound-type signatures, and informational metadata.
  - Version tagging and broader docs are still missing.
- [ ] Add reproducibility guidance for citing analysis settings.
- [ ] Preserve behavior with tests before changing scientific definitions.

## Immediate Next-Step Candidates

1. Keep this checklist aligned with [analysis_framework_checklist.md](D:/python/dyana/docs/analysis_framework_checklist.md) and the current reference docs now that the runtime topology work is complete.
2. Version/metadata/output-directory work here means:
   - add a resolved-config writer
   - add consistent naming conventions for output files
   - add `dyana.__version__` and include it in output metadata
3. Add opt-in slow smoke tests for the documented example trajectories.
