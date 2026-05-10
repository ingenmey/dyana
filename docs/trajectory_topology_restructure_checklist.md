# Dyana Trajectory/Topology Restructure Checklist

This checklist tracks the intended redesign of Dyana's trajectory and topology
handling. It complements, but does not replace:

- [analysis_framework_checklist.md](D:/python/dyana/docs/analysis_framework_checklist.md)
- [productionization_checklist.md](D:/python/dyana/docs/productionization_checklist.md)

This document should guide work on the trajectory/topology model before broad
re-expansion to additional analyses.

Status markers:

- `[x]` done
- `[~]` partially done / started
- `[ ]` not started

## 1. Core Principle

- [x] Prefer efficient and effective code, readability, and clarity over preserving legacy topology code intact.
- [x] Allow temporary breakage while restructuring the trajectory/topology layer.
- [x] Treat temporary breakage of analyses other than RDF, density, ADF, and neighbor count as acceptable during this work.
- [x] Treat RDF as the default/reference analysis during trajectory/topology redesign.
- [x] Treat density as the next simplest validation target after RDF.
- [x] Treat ADF and neighbor count as the next topology-pressure analyses after density.
- [x] Do not currently consider other analyses as migration or compatibility targets during this restructuring stage.
- [x] If a unit test breaks, first decide whether the break reflects an intentional model change; update the test when the old expectation is no longer the intended behavior.
- [x] Do not keep outdated behavior solely to satisfy an outdated unit test.

## 2. The Main Design Shift

- [x] Move away from making analyses consume a nested mutable object graph as the primary topology interface.
- [x] Move toward a topology model centered on:
  - compound-type templates
  - frame-level membership arrays
  - direct lookup tables for atom/molecule/type relationships
- [x] Preserve the information analyses need:
  - molecule identity
  - molecule center of mass
  - molecule topology
  - label-to-global-index resolution
  - direct access to atoms belonging to selected compound/type/label combinations
- [x] Require canonical template-local atom ordering for every molecule/member stored in the runtime topology representation.
- [x] Treat raw detected per-molecule atom ordering as an internal detection detail, not as the analysis-facing representation.
- [x] Keep compact integer ids as internal runtime handles, but anchor persistent or external references to stable structural/template keys rather than transient numeric ids.

## 3. Target Layering

### Trajectory Reader Layer

- [~] Keep raw trajectory reading focused on:
  - reading frames
  - symbols
  - coordinates
  - box dimensions
- [~] Reduce trajectory readers' responsibility for interpreted topology state over time.

### Topology Builder Layer

- [~] Separate topology detection/building from frame reading.
- [x] Centralize:
  - connectivity detection
  - molecule identification
  - compound-type classification
  - canonical label assignment
  - canonical local ordering
- [x] Make topology building return a compact topology representation rather than mutating an ad hoc analysis-facing object web.

### Runtime Topology State Layer

- [x] Introduce a runtime topology frame/snapshot concept that analyses consume.
- [x] Keep long-lived type/template information separate from per-frame topology realization.
- [x] Make runtime topology state explicitly own or provide direct access to:
  - frame-level molecule membership
  - atom-to-molecule mappings
  - atom-to-compound-type mappings
  - local-index mappings
  - per-molecule centers of mass
- [x] Make runtime topology state the natural home for selection helpers used by analyses.
- [x] Keep global coordinates as the primary coordinate source of truth; do not duplicate per-member coordinate arrays as the main runtime representation unless a clear benefit justifies it.

### Type/Template Registry Layer

- [x] Introduce an explicit type/template registry concept separate from the frame snapshot.
- [x] Let the registry own all known structural templates used by the current session/run.
- [x] Make prepared setups and static-topology reuse align naturally with the registry/template concept.

### Analysis-Facing Access Layer

- [x] Define a minimal analysis-facing topology access API before porting RDF to the new model.
- [x] Keep analyses from depending directly on raw internal array layout wherever a small, clear access layer can preserve readability.
- [x] Make the access layer support the common analysis queries directly:
  - selected global atom ids by compound type and canonical local labels
  - per-type member atom rows
  - per-type centers of mass
  - reverse lookup for a global atom

## 4. Compound-Type Template Model

- [x] Introduce an explicit `CompoundType`-like concept.
- [x] Store one template per structural compound type.
- [x] Let the template own:
  - stable structural key
  - display name / representative formula
  - canonical labels
  - canonical local bond topology
  - local label-to-index mapping
  - any template-local bond metadata that is independent of frame geometry
- [x] Keep frame-specific membership separate from the template.
- [x] Keep the template/type registry as the home for these definitions rather than rebuilding type metadata ad hoc per frame.

## 5. Frame-Level Membership Representation

- [x] Represent per-frame molecule membership primarily as arrays rather than nested Python objects.
- [x] For each compound type, store member/global-atom mappings in canonical local order.
- [x] Introduce structures and direct-access operations equivalent to:
  - `molecule_atom_ids`
  - per-molecule COM access
  - `atom_to_molecule`
  - `atom_to_compound_type`
  - `atom_to_local_index`
- [x] Make direct global-index retrieval for selected labels/types a first-class operation.
- [x] Treat a flat global molecule-id map as optional/derived convenience data, not as a required primary identity layer if `(type_id, member_index)` already suffices.

## 6. Canonical Local Ordering

- [x] Make canonical local ordering a hard invariant of the new topology model.
- [x] Use graph-isomorphism/template mapping to place every molecule/member into template-local order.
- [x] Ensure that equivalent molecules with different raw detected atom orderings still produce identical canonical local ordering.
- [x] Define deterministic behavior for symmetry-equivalent atoms.
  - For truly equivalent atoms, Dyana uses a deterministic convention because chemistry does not provide a privileged distinction.
  - The important invariants are reproducibility within one trajectory/build and consistent subgroup pairing within one compound type; direct tests now protect those invariants.
- [x] Decide explicitly whether symmetry-equivalent atoms should:
  - receive a deterministic tie-broken order, or
  - be treated as stable equivalence classes where a unique label is not fundamentally meaningful
- [x] Treat canonical local order as the basis for:
  - label lookup
  - per-label selections
  - member-array column meaning
  - stable per-type analysis selections

## 7. Static vs Dynamic Topology

- [x] Make the distinction between static-topology and dynamic-topology operation explicit.
- [x] In static-topology mode:
  - build the template/type structure once
  - keep that topology while coordinates change frame to frame
  - avoid unnecessary topology rebuild work by not calling `rebuild_topology()` each frame
- [x] In dynamic-topology mode:
  - rebuild the current topology frame/snapshot when requested
  - preserve the same analysis-facing access patterns
- [x] Keep the analysis-side API as similar as possible across static and dynamic topology modes.

## 8. Analysis-Facing Access Goals

- [x] Make the common analysis-side queries easy and direct:
  - all atoms matching label(s) in a selected compound type
  - all atoms for a specific molecule/member
  - molecule identity for a global atom
  - local label position for a global atom
  - all COMs for a selected compound type
  - per-type topology information
- [x] Reduce dependence on walking `Compound -> Molecule -> label dict` structures during normal analysis work.
- [x] Make selection retrieval anchored to canonical template-local positions wherever possible.
- [x] Keep the access API thin and high-value rather than mirroring every raw array with a separate accessor.

## 9. Transition Strategy

- [x] First redesign the topology/runtime representation, even if it temporarily breaks analyses.
- [x] Keep `Molecule` only as an internal builder/debug/view abstraction during transition if that makes migration easier.
- [x] Stop treating `Molecule` as the primary analysis-facing runtime abstraction.
- [x] Keep RDF working as the first reference analysis as early as practical.
- [x] Then restore density.
- [x] Then restore ADF.
- [x] Then restore neighbor count.
- [x] Do not spend time preserving or restoring other analyses during this phase.
- [x] Be willing to remove or bypass legacy topology representations once the new one clearly supersedes them.

## 10. Relationship To Prepared Setups

- [x] Keep prepared setups aligned with the new topology model.
- [x] Move prepared-setup validation toward compound-type/template expectations rather than transient object layouts.
- [x] Preserve the ability to reuse one prepared setup across compatible systems with different counts/concentrations.
- [x] Keep prepared setups focused on compatible type/signature validation rather than exact molecule counts.

## 11. Relationship To Selection Handling

- [x] Revisit analysis selection helpers after the new topology representation exists.
- [x] Prefer selections expressed in terms of:
  - compound type
  - canonical local labels / local indices
- [x] Resolve user label selections once into canonical template-local positions wherever possible, rather than repeatedly rematching labels during analysis execution.
- [x] Add a small reusable `ResolvedSelection` structure so migrated analyses do not each reinvent selection bookkeeping.
- [x] Reduce dependence on transient per-frame compound object identity where possible.
- [x] Keep label-matching behavior compatible with Dyana's intended atom-label semantics, even if the internal representation changes substantially.

## 12. Testing Guidance

- [x] Use unit tests to protect intended behavior, not accidental structure.
- [x] When restructuring breaks a test, decide whether the old expectation is still correct before changing code just to satisfy it.
- [x] Add tests for canonical local ordering across equivalent molecules with different raw atom orderings.
- [x] Add tests for direct label-to-global-index lookup in the new model.
- [x] Add tests for atom-to-molecule / atom-to-type reverse lookup.
- [x] Add tests for static-topology refresh behavior.
- [x] Add tests for dynamic-topology rebuild behavior.
- [x] Keep RDF as the first end-to-end correctness check during topology restructuring.

## 13. What Not To Optimize For

- [x] Do not optimize for preserving the current object graph if it obscures cleaner design.
- [x] Do not optimize for keeping all analyses working at every intermediate step.
- [x] Do not optimize for minimizing short-term breakage if that prevents a clearer long-term topology model.
- [x] Do not accept abstraction churn that merely moves code between files without improving the runtime representation.

## 14. Immediate Next-Step Candidates

1. Finish harmonizing the shared output writer and migrated-analysis table format.
2. Reassess `CONFIG_SCHEMA` versus explicit config-dataclass duplication only after the output path settles.
3. Revisit moving topology-builder code out of `trajectory_loader.py` only if a concrete readability or reuse problem appears.
