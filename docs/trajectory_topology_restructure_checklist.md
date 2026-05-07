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

- [ ] Keep raw trajectory reading focused on:
  - reading frames
  - symbols
  - coordinates
  - box dimensions
- [ ] Reduce trajectory readers' responsibility for interpreted topology state over time.

### Topology Builder Layer

- [ ] Separate topology detection/building from frame reading.
- [ ] Centralize:
  - connectivity detection
  - molecule identification
  - compound-type classification
  - canonical label assignment
  - canonical local ordering
- [ ] Make topology building return a compact topology representation rather than mutating an ad hoc analysis-facing object web.

### Runtime Topology State Layer

- [ ] Introduce a runtime topology frame/snapshot concept that analyses consume.
- [ ] Keep long-lived type/template information separate from per-frame topology realization.
- [ ] Make runtime topology state explicitly own:
  - frame-level molecule membership
  - atom-to-molecule mappings
  - atom-to-compound-type mappings
  - local-index mappings
  - per-molecule centers of mass
- [ ] Make runtime topology state the natural home for selection helpers used by analyses.
- [ ] Keep global coordinates as the primary coordinate source of truth; do not duplicate per-member coordinate arrays as the main runtime representation unless a clear benefit justifies it.

### Type/Template Registry Layer

- [ ] Introduce an explicit type/template registry concept separate from the frame snapshot.
- [ ] Let the registry own all known structural templates used by the current session/run.
- [ ] Make prepared setups and static-topology reuse align naturally with the registry/template concept.

### Analysis-Facing Access Layer

- [ ] Define a minimal analysis-facing topology access API before porting RDF to the new model.
- [ ] Keep analyses from depending directly on raw internal array layout wherever a small, clear access layer can preserve readability.
- [ ] Make the access layer support the common analysis queries directly:
  - selected global atom ids by compound type and canonical local labels
  - per-type member atom rows
  - per-type centers of mass
  - reverse lookup for a global atom

## 4. Compound-Type Template Model

- [ ] Introduce an explicit `CompoundType`-like concept.
- [ ] Store one template per structural compound type.
- [ ] Let the template own:
  - stable structural key
  - display name / representative formula
  - canonical labels
  - canonical local bond topology
  - local label-to-index mapping
  - any template-local bond metadata that is independent of frame geometry
- [ ] Keep frame-specific membership separate from the template.
- [ ] Keep the template/type registry as the home for these definitions rather than rebuilding type metadata ad hoc per frame.

## 5. Frame-Level Membership Representation

- [ ] Represent per-frame molecule membership primarily as arrays rather than nested Python objects.
- [ ] For each compound type, store member/global-atom mappings in canonical local order.
- [ ] Introduce structures equivalent to:
  - `member_atom_ids`
  - `member_coms`
  - `atom_to_molecule`
  - `atom_to_compound_type`
  - `atom_to_local_index`
- [ ] Make direct global-index retrieval for selected labels/types a first-class operation.
- [ ] Treat a flat global molecule-id map as optional/derived convenience data, not as a required primary identity layer if `(type_id, member_index)` already suffices.

## 6. Canonical Local Ordering

- [ ] Make canonical local ordering a hard invariant of the new topology model.
- [ ] Use graph-isomorphism/template mapping to place every molecule/member into template-local order.
- [ ] Ensure that equivalent molecules with different raw detected atom orderings still produce identical canonical local ordering.
- [ ] Define deterministic behavior for symmetry-equivalent atoms rather than leaving tie-breaking implicit.
- [ ] Decide explicitly whether symmetry-equivalent atoms should:
  - receive a deterministic tie-broken order, or
  - be treated as stable equivalence classes where a unique label is not fundamentally meaningful
- [ ] Treat canonical local order as the basis for:
  - label lookup
  - per-label selections
  - member-array column meaning
  - stable per-type analysis selections

## 7. Static vs Dynamic Topology

- [ ] Make the distinction between static-topology and dynamic-topology operation more explicit in the topology model.
- [ ] In static-topology mode:
  - build the template/type structure once
  - refresh geometry and derived per-frame arrays cheaply
- [ ] In dynamic-topology mode:
  - rebuild the current topology frame/snapshot when requested
  - preserve the same analysis-facing access patterns
- [ ] Keep the analysis-side API as similar as possible across static and dynamic topology modes.

## 8. Analysis-Facing Access Goals

- [ ] Make the common analysis-side queries easy and direct:
  - all atoms matching label(s) in a selected compound type
  - all atoms for a specific molecule/member
  - molecule identity for a global atom
  - local label position for a global atom
  - all COMs for a selected compound type
  - per-type topology information
- [ ] Reduce dependence on walking `Compound -> Molecule -> label dict` structures during normal analysis work.
- [ ] Make selection retrieval anchored to canonical template-local positions wherever possible.
- [ ] Keep the access API thin and high-value rather than mirroring every raw array with a separate accessor.

## 9. Transition Strategy

- [ ] First redesign the topology/runtime representation, even if it temporarily breaks analyses.
- [ ] Keep `Molecule` only as an internal builder/debug/view abstraction during transition if that makes migration easier.
- [ ] Stop treating `Molecule` as the primary analysis-facing runtime abstraction.
- [ ] Keep RDF working as the first reference analysis as early as practical.
- [ ] Then restore density.
- [ ] Then restore ADF.
- [ ] Then restore neighbor count.
- [ ] Do not spend time preserving or restoring other analyses during this phase.
- [ ] Be willing to remove or bypass legacy topology representations once the new one clearly supersedes them.

## 10. Relationship To Prepared Setups

- [ ] Keep prepared setups aligned with the new topology model.
- [ ] Move prepared-setup validation toward compound-type/template expectations rather than transient object layouts.
- [ ] Preserve the ability to reuse one prepared setup across compatible systems with different counts/concentrations.
- [ ] Keep prepared setups focused on compatible type/signature validation rather than exact molecule counts.

## 11. Relationship To Selection Handling

- [ ] Revisit analysis selection helpers after the new topology representation exists.
- [ ] Prefer selections expressed in terms of:
  - compound type
  - canonical local labels / local indices
- [ ] Resolve user label selections once into canonical template-local positions wherever possible, rather than repeatedly rematching labels during analysis execution.
- [ ] Consider a small reusable `SelectionSpec`-like structure once the new topology access API is clear, so migrated analyses do not each reinvent selection bookkeeping.
- [ ] Reduce dependence on transient per-frame compound object identity where possible.
- [ ] Keep label-matching behavior compatible with Dyana's intended atom-label semantics, even if the internal representation changes substantially.

## 12. Testing Guidance

- [x] Use unit tests to protect intended behavior, not accidental structure.
- [x] When restructuring breaks a test, decide whether the old expectation is still correct before changing code just to satisfy it.
- [ ] Add tests for canonical local ordering across equivalent molecules with different raw atom orderings.
- [ ] Add tests for direct label-to-global-index lookup in the new model.
- [ ] Add tests for atom-to-molecule / atom-to-type reverse lookup.
- [ ] Add tests for static-topology refresh behavior.
- [ ] Add tests for dynamic-topology rebuild behavior.
- [ ] Keep RDF as the first end-to-end correctness check during topology restructuring.

## 13. What Not To Optimize For

- [x] Do not optimize for preserving the current object graph if it obscures cleaner design.
- [x] Do not optimize for keeping all analyses working at every intermediate step.
- [x] Do not optimize for minimizing short-term breakage if that prevents a clearer long-term topology model.
- [x] Do not accept abstraction churn that merely moves code between files without improving the runtime representation.

## 14. Immediate Next-Step Candidates

1. Define the concrete runtime topology data structures before changing more code.
2. Decide the minimum fields and invariants for the new compound-type template representation and the type/template registry that owns it.
3. Decide the minimum frame-level membership arrays and reverse lookup arrays the analyses will rely on.
4. Define the minimal analysis-facing topology access API before porting RDF.
5. Decide and document how symmetry-equivalent atoms are handled in canonical ordering.
6. Decide whether a small reusable `SelectionSpec` is warranted once the access API shape is clearer.
7. Rework RDF first against that representation.
8. Then restore density, ADF, and neighbor count in that order.
