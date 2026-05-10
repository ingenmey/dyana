# Dyana Analysis Framework Checklist

This checklist tracks the canonical analysis framework design for Dyana.
It is intentionally narrower than the broader productionization checklist and
should guide analysis-architecture work before further productionization.

Status markers:

- `[x]` done
- `[~]` partially done / started
- `[ ]` not started

## 1. Core Principle

- [x] Adopt one canonical analysis shape built around `BaseAnalysis`.
- [x] Treat RDF as the default/reference analysis during framework work.
- [x] Require interactive and programmatic paths to meet at `configure(config)`.
- [x] Keep the shared frame loop in `BaseAnalysis` for the near term.
- [x] Avoid duplicate configured/interactive analysis classes.
- [x] Prefer readability over framework layering unless duplication clearly falls.
- [x] Prefer clean, purpose-oriented framework code over temporary compatibility with legacy analyses.

## 2. BaseAnalysis Evolution

- [x] Keep `BaseAnalysis` as the shared lifecycle/frame-loop base.
- [x] Keep shared prompt helpers in `BaseAnalysis`:
  - `compound_selection`
  - `atom_selection`
  - `prompt_frame_loop_config`
- [x] Keep generic typed prompt primitives out of `BaseAnalysis`.
- [x] Add `CONFIG_CLASS` support to `BaseAnalysis`.
- [x] Add `CONFIG_SCHEMA` support to `BaseAnalysis`.
- [x] Add a default `prompt_config()` path driven by schema.
- [x] Support `configure(config)` in migrated analyses.
- [x] Add a small `configure_frame_loop(frame_loop)` helper to `BaseAnalysis`.
- [x] Add `bind_config(config)` in `BaseAnalysis` so analyses can bind config fields automatically instead of copying them one by one in `configure()`.
- [x] Use a single public `run()` entry point for both interactive and programmatic execution.
- [ ] Add `from_config(...)` only where it reduces boilerplate.
- [x] Allow temporary breakage of non-RDF analyses while the framework is being cleaned up.

## 3. Declarative Config Schema

- [x] Add shared parameter spec primitives:
  - `Param`
  - `CompoundParam`
  - `AtomLabelsParam`
  - `IntParam`
  - `FloatParam`
  - `BoolParam`
  - `ChoiceParam`
- [x] Extend the schema language with repeated and conditional steps:
  - `ForEach`
  - `When`
- [x] Keep parameter specs focused on gathering/validating config, not science.
- [~] Reassess the duplicated field definitions between `CONFIG_SCHEMA` and explicit config dataclasses such as `RDFConfig` / `ADFConfig`.
- [x] Keep explicit config dataclasses for now as the typed programmatic API and validation layer, even when `CONFIG_SCHEMA` duplicates the field list.
- [x] Treat deeper collapse of `CONFIG_SCHEMA` and explicit config classes as later work only if the schema becomes expressive enough to cover field defaults, prompt defaults, and validation semantics cleanly.
- [ ] Add room for later param types without overdesign:
  - `StringParam`
  - `PathParam`
  - `ListParam`
  - `PerCompoundParam`
  - `PerLabelParam`
  - `CutoffMatrixParam`

## 4. Prompt Dispatcher

- [x] Add `prompt_config_from_schema(owner, schema, config_class, provider=None)`.
- [x] Add a lightweight `PromptContext`.
- [x] Dispatch shared prompt types through modular handlers:
  - compound selection
  - atom label selection
  - int
  - float
  - bool
  - choice
- [x] Dispatch repeated prompt groups through `ForEach`.
- [x] Dispatch conditional prompt groups through `When`.
- [x] Support dependencies between parameters through context.
- [x] Support loop-local scoped values inside schema-driven prompts.
- [x] Avoid speculative config-builder hooks; only add special prompt-to-config assembly paths when a concrete migrated analysis truly needs one.
- [x] Keep prompt builders easy to extend without editing each analysis.
- [x] Reuse the same `InputProvider` abstraction across workflow and analysis code.
- [x] Keep the schema engine focused on migrated analyses rather than forcing it into the workflow layer.

## 5. Compound Selection Representation

- [x] Use compound indices in configs near-term.
- [ ] Revisit stable compound keys later when dynamic-topology use cases demand it.
- [ ] If needed later, add a richer `CompoundSelection` object without forcing it now.

## 6. Frame Loop Configuration

- [x] Keep `FrameLoopConfig` as the shared frame-loop config object.
- [x] Let `BaseAnalysis` own prompting for frame-loop config.
- [x] Let programmatic mode inject `FrameLoopConfig` without prompts.
- [~] Keep the current `nframes=-1` sentinel consistently through the frame-loop path until the loop itself is redesigned.
- [ ] Replace `nframes=-1` with a cleaner frame-count API as part of a later frame-loop redesign.
- [x] Mirror the analysis-config flow: prompt a frame-loop config object, then apply it with `configure_frame_loop(...)`.

## 7. Analysis File Readability

- [x] Keep science logic local to the dedicated analysis file.
- [x] Allow generic reusable math imports such as `DistanceMetric`.
- [x] Keep each migrated simple analysis roughly self-contained and readable.
- [x] Prefer one analysis file with:
  - config dataclass
  - config schema
  - `configure`
  - per-frame logic
  - `postprocess`
- [ ] Avoid scattering basic setup logic across many files unless reuse is substantial.

## 8. Handling Different Analysis Types

### Simple Analyses

- [x] Treat RDF as the first simple-analysis migration target.
- [x] Treat RDF as the canonical reference implementation for the framework.
- [x] Migrate density after RDF.
- [x] Migrate neighbor count after density.
- [ ] Migrate tetrahedral order after neighbor count.

### Medium-Complex Analyses

- [x] Migrate ADF with schema plus dependent parameters.
- [ ] Migrate ADF3B.
- [ ] Migrate DACF.
- [ ] Migrate percolation.

### Complex Analyses

- [ ] Avoid falling back to custom `prompt_config()` by strengthening the schema language first.
- [ ] Leave cluster on custom builder until last.
- [ ] Leave PCCF on custom builder until last.
- [ ] Leave CMSD on custom builder until last.

Rule:

- [x] Use declarative schema when clean.
- [x] Extend declarative schema to cover dynamic prompting patterns such as loops and conditionals.
- [x] Still end at `configure(config)`.

## 9. Programmatic Access

- [x] Programmatic mode should call `configure(config)`.
- [x] Add a consistent programmatic frame-loop setup path.
- [x] Unify execution under `run()` rather than keeping a separate `run_configured()` method.
- [x] Provide a practical programmatic entry path by loading a saved prepared setup before analysis configuration.
- [x] Treat prepared setups as the near-term bridge between interactive topology review and Python-driven analyses.
- [ ] Delay general result-return APIs until output design is agreed.

## 10. Output Handling In Framework Work

- [x] Add a shared output-writing layer for migrated analyses.
- [x] Keep `HistogramND` as a data/binning container rather than a text-output API.
- [x] Avoid per-analysis formatting code when a shared writer can handle it.
- [x] Standardize one clear text-table format across migrated analyses instead of reproducing legacy spacing per analysis.
- [x] Apply the shared output layer only to the currently migrated analyses until more migrations are requested.

## 11. Proposed Minimal Framework API

- [x] Add `analysis_params.py` or equivalent shared parameter-spec module.
- [x] Add `config_builder.py` or equivalent shared prompt-dispatch module.
- [x] Extend `BaseAnalysis` minimally before migrating more analyses.
- [x] Avoid introducing a separate runner right now.

## 12. Migration Plan

- [x] Step 1: Add schema primitives without changing analysis behavior.
- [x] Step 2: Extend `BaseAnalysis` minimally with config/schema hooks.
- [x] Step 3: Refactor RDF to the target design.
  - `RDFConfig` exists.
  - `RDF.configure(config)` exists.
  - RDF uses schema-driven prompting through shared framework code.
- [x] Step 4: Add tests for schema-driven config building.
- [x] Step 5: Migrate density.
- [x] Step 6: Migrate the currently supported simple analyses.
  - RDF, density, and neighbor count are on the canonical framework path.
  - Other simple legacy analyses remain intentionally out of scope.
- [ ] Step 7: Leave complex analyses custom until last.

Notes:

- [x] During framework work, RDF remains the reference analysis, but RDF, density, neighbor count, and ADF now define the supported migrated set that should remain functional.
- [x] Other analyses may break temporarily and be reintroduced after the framework is clean.
- [x] Do not preserve legacy code paths solely for compatibility if they make the framework harder to read.
- [x] Treat the workflow layer and the analysis layer as separate design problems.
- [x] Keep the workflow layer imperative and provider-driven.
- [x] Keep schema-driven prompting as an analysis-layer tool, not a universal rule.

## 13. Checklist Guidance

- [x] Document the canonical analysis design.
- [x] Document the rule against duplicate configured analysis classes.
- [x] Document the rule against per-analysis custom output writers unless shared.
- [x] Keep the productionization checklist aligned with this framework checklist.
- [x] Treat this checklist as the current focus before wider productionization resumes.

## Later Cleanup Notes

- [ ] Reassess remaining shared schema/config scaffolding that is not part of the main migrated-analysis path.
- [ ] Reassess whether `PromptContext` in `config_builder.py` is more machinery than the current size needs.
- [ ] Split responsibilities inside `workflow_prompts.py` if the current single class becomes harder to read.
- [ ] Revisit the non-XYZ `prompt_cell_vectors(...)` behavior so the method name and return semantics line up more cleanly.
- [ ] Tighten validation and error handling in `BaseAnalysis.compound_selection(..., multi=True)`.

## Current Focus

1. Keep RDF, density, neighbor count, and ADF as the clean migrated analyses and preserve the working Python programmatic RDF path.
2. Keep the workflow/session layer imperative and provider-driven.
3. Use prepared setups as the practical bridge from interactive topology review to Python-driven analysis runs.
4. Use the strengthened schema system, not ad hoc prompt flows, as the default path for further migrated analyses.
5. Revisit compound-selection stability once multiple migrated analyses put real pressure on index-based configs.
6. Only then resume wider productionization tasks in depth.
