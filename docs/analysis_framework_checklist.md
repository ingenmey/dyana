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
- [x] Keep parameter specs focused on gathering/validating config, not science.
- [ ] Add room for later param types without overdesign:
  - `StringParam`
  - `PathParam`
  - `ListParam`
  - `PerCompoundParam`
  - `PerLabelParam`
  - `CutoffMatrixParam`

## 4. Prompt Dispatcher

- [x] Add `prompt_config_from_schema(analysis, schema, config_class, provider=None)`.
- [x] Add a lightweight `PromptContext`.
- [x] Dispatch shared prompt types through modular handlers:
  - compound selection
  - atom label selection
  - int
  - float
  - bool
  - choice
- [x] Support dependencies between parameters through context.
- [x] Keep prompt builders easy to extend without editing each analysis.

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

- [~] Treat RDF as the first simple-analysis migration target.
- [x] Treat RDF as the canonical reference implementation for the framework.
- [ ] Migrate density after RDF.
- [ ] Migrate neighbor count after density.
- [ ] Migrate tetrahedral order after neighbor count.

### Medium-Complex Analyses

- [ ] Migrate ADF with schema plus dependent parameters.
- [ ] Migrate ADF3B.
- [ ] Migrate DACF.
- [ ] Migrate percolation.

### Complex Analyses

- [x] Allow custom `prompt_config()` when schema would be awkward.
- [ ] Leave cluster on custom builder until last.
- [ ] Leave PCCF on custom builder until last.
- [ ] Leave CMSD on custom builder until last.

Rule:

- [x] Use declarative schema when clean.
- [x] Use custom `prompt_config()` when the interaction is genuinely dynamic.
- [x] Still end at `configure(config)`.

## 9. Programmatic Access

- [x] Programmatic mode should call `configure(config)`.
- [x] Add a consistent programmatic frame-loop setup path.
- [x] Unify execution under `run()` rather than keeping a separate `run_configured()` method.
- [ ] Delay general result-return APIs until output design is agreed.

## 10. Output Handling In Framework Work

- [x] Keep current output behavior short-term.
- [x] Keep using `HistogramND.save_txt` where already used.
- [x] Avoid per-analysis custom writers unless a shared output layer exists.
- [ ] Revisit shared output helpers after multiple analyses are migrated.

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
- [ ] Step 5: Migrate density.
- [ ] Step 6: Migrate simple analyses.
- [ ] Step 7: Leave complex analyses custom until last.

Notes:

- [x] During framework work, RDF is the only analysis that must remain fully functional.
- [x] Other analyses may break temporarily and be reintroduced after the framework is clean.
- [x] Do not preserve legacy code paths solely for compatibility if they make the framework harder to read.

## 13. Checklist Guidance

- [x] Document the canonical analysis design.
- [x] Document the rule against duplicate configured analysis classes.
- [x] Document the rule against per-analysis custom output writers unless shared.
- [x] Keep the productionization checklist aligned with this framework checklist.
- [x] Treat this checklist as the current focus before wider productionization resumes.

## Current Focus

1. Add config-schema parameter primitives.
2. Add the shared prompt dispatcher.
3. Teach `BaseAnalysis` to use `CONFIG_CLASS` and `CONFIG_SCHEMA`.
4. Finish RDF as the canonical simple-analysis example.
5. Allow non-RDF analyses to lag or break temporarily while RDF and the shared framework are clarified.
6. Decide how top-level pre-analysis prompts should be separated from analysis config prompts.
7. Only then resume wider productionization tasks.
