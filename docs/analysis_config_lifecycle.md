# Dyana Analysis And Configuration Lifecycle

This document describes the current supported analysis lifecycle in Dyana.
It focuses on the migrated analysis framework used by:

- RDF
- density
- ADF
- neighbor count

It is a reference for the current path, not a migration plan.

## 1. Design Summary

Supported analyses in Dyana share one canonical shape built around
[BaseAnalysis](D:/python/dyana/analyses/common/base_analysis.py).

The central ideas are:

- interactive and programmatic paths meet at `configure(config)`
- prompting builds typed config objects
- runtime setup binds that config to the current trajectory/topology state
- execution happens through one shared `run()` flow

## 2. Main Pieces

### `CONFIG_CLASS`

Each migrated analysis defines a typed config dataclass, for example:

- `RDFConfig`
- `ADFConfig`

The config dataclass is responsible for:

- field definitions
- defaults
- validation in `__post_init__`

It should not perform live trajectory-dependent work.

### `CONFIG_SCHEMA`

Each migrated analysis also defines a schema describing the interactive prompt
flow.

The schema is built from shared parameter specs in
[analysis_params.py](D:/python/dyana/framework/analysis_params.py), such as:

- `CompoundParam`
- `AtomLabelsParam`
- `IntParam`
- `FloatParam`
- `BoolParam`
- `ChoiceParam`
- `ForEach`
- `When`

The schema is responsible for:

- prompt order
- prompt text
- prompt-time dependencies and conditionals

It should not perform analysis runtime work.

### `configure(config)`

This is the point where interactive and programmatic execution meet.

`configure(config)` is responsible for:

- binding config fields to the analysis instance
- resolving selected compound types against the current topology
- resolving label selections into canonical local indices
- building analysis runtime state
- initializing metrics, histograms, counters, and other live objects

This is the supported entry point for analysis setup.

## 3. `BaseAnalysis` Lifecycle

The shared lifecycle lives in
[analyses/common/base_analysis.py](D:/python/dyana/analyses/common/base_analysis.py).

The high-level flow is:

1. `analysis.run()`
2. if not already configured:
   - `setup()`
   - `prompt_config()`
   - `configure(config)`
3. if frame-loop config is not already set:
   - `prompt_frame_loop_config()`
   - `configure_frame_loop(frame_loop)`
4. skip to the configured start frame if needed
5. execute the shared frame loop
6. call `postprocess()`

## 4. Interactive Path

The interactive analysis path is:

1. workflow prepares the trajectory/topology state
2. user chooses an analysis
3. `BaseAnalysis.prompt_config()` calls
   `prompt_config_from_schema(...)`
4. the schema-driven prompt builder returns a config dataclass instance
5. `configure(config)` binds that config to the live trajectory/topology
6. the frame loop runs

This keeps prompting separate from analysis runtime logic.

## 5. Programmatic Path

The supported programmatic path is:

1. construct or load a prepared setup
2. load a trajectory
3. rebuild/validate topology as needed
4. construct an analysis config object directly in Python
5. call `analysis.configure(config)`
6. optionally call `analysis.configure_frame_loop(frame_loop)`
7. call `analysis.run()`

The important point is that programmatic use does not need prompt wrappers.

## 6. `bind_config(config)`

`BaseAnalysis.bind_config(config)` exists to reduce repetitive boilerplate in
`configure()`.

It:

- stores `self.config`
- copies config fields onto the analysis instance
- shallow-copies mutable containers such as `list`, `dict`, and `set`

The purpose is convenience and readability in analysis code. It does not
replace the typed config dataclass or `configure(config)` as a design layer.

## 7. Frame-Loop Configuration

Frame-loop settings are represented by
[FrameLoopConfig](D:/python/dyana/framework/config_schema.py).

The main fields are:

- `start_frame`
- `nframes`
- `frame_stride`
- `update_compounds`

`BaseAnalysis.configure_frame_loop(...)` applies these settings to the live
analysis instance.

## 8. Topology Interaction During Analysis

The supported analyses operate against:

- `traj.topology_registry`
- `traj.topology_frame`

Typical setup flow inside `configure(config)` is:

1. resolve selected compound-type indices to:
   - a `CompoundType`
   - its stable key
2. resolve label patterns once using
   `TopologyFrame.resolve_selection(...)`
3. build runtime selectors/indices using canonical local positions

Typical dynamic-topology update flow inside `post_compound_update()` is:

1. check whether the selected compound-type keys still exist
2. reattach current compound types by key
3. rebuild runtime indices/selectors from stored local indices

The supported path should not repeatedly rematch label strings during runtime.

## 9. Analysis File Shape

The intended shape of a migrated analysis file is:

- config dataclass
- config schema
- `configure(config)`
- `rebuild_runtime_state()` if needed
- `post_compound_update()`
- `process_frame()`
- `postprocess()`

This keeps:

- prompting declarative
- runtime setup explicit
- science logic local to the analysis file

## 10. Scope Of This Lifecycle

This lifecycle is the supported framework path for the currently migrated
analyses only:

- RDF
- density
- ADF
- neighbor count

Older analyses that still use legacy patterns should not be treated as the
reference design for future framework work.
