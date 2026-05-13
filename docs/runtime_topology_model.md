# Dyana Runtime Topology Model

This document describes the current supported runtime topology model in Dyana.
It is a reference for the active analysis path, not a planning document.

The supported analyses for this model are:

- RDF
- density
- ADF
- neighbor count
- tetrahedral order
- q6 / Q6
- LSI

Other analyses may still use older patterns and are not the reference path.

## 1. Design Summary

Dyana's supported analysis path no longer treats a nested mutable
`Compound -> Molecule -> Atom` object graph as the primary runtime topology
interface.

Instead, the runtime model is centered on:

- compound-type templates
- a type/template registry
- a per-frame topology snapshot
- canonical local atom ordering
- direct lookup arrays for atom/type/molecule relationships

The goal is to make analysis-side access:

- more consistent
- easier to reason about
- less dependent on object walking

## 2. Main Runtime Objects

### `CompoundType`

Defined in [core/topology.py](D:/python/dyana/core/topology.py).

One `CompoundType` represents one structural compound type. It stores:

- `type_id`
- `key`
- `formula`
- `canonical_labels`
- `label_to_local_index`
- `local_bonds`
- `local_elements`
- `atomic_masses`

Important meaning:

- `canonical_labels` define the template-local labeling for this type
- `label_to_local_index` maps those labels to canonical local positions
- `local_bonds` describe template-local connectivity, not frame-global atom ids

### `CompoundTypeRegistry`

Also defined in [core/topology.py](D:/python/dyana/core/topology.py).

This is the session/runtime registry of known compound types for the current
trajectory state. It provides lookup by:

- index
- stable structural key

Persistent references should use the stable compound-type key, not transient
numeric indices.

### `TopologyFrame`

Also defined in [core/topology.py](D:/python/dyana/core/topology.py).

This is the per-frame runtime topology snapshot that analyses consume. It owns
or directly provides access to:

- molecule membership rows per compound type
- reverse lookup arrays:
  - `atom_to_type_id`
  - `atom_to_molecule_index`
  - `atom_to_local_index`
- accessors for:
  - global atom ids for selected local indices
  - molecule atom rows
  - molecule centers of mass
  - bond-length summaries
  - atom topology location

The main membership structure is `molecule_atom_ids_by_key`, where each row is
one molecule/member of a compound type and each column is one canonical
template-local atom position.

### `ResolvedSelection`

Also defined in [core/topology.py](D:/python/dyana/core/topology.py).

This is the runtime payload for a resolved atom selection. It stores:

- `compound_type_key`
- `local_indices`

The supported analyses resolve user label patterns once into
`ResolvedSelection`, then reuse the canonical local indices during runtime and
topology rebuilds.

## 3. Topology Construction

The builder currently lives in [core/trajectory_loader.py](D:/python/dyana/core/trajectory_loader.py).

For the current frame:

1. `BaseTrajectory.read_frame()` populates:
   - `n_atoms`
   - `symbols`
   - `coords`
   - `box_size`
2. `BaseTrajectory.rebuild_topology()`:
   - identifies detected molecules
   - groups equivalent molecules into compound types
   - assigns canonical labels on one template per type
   - maps every member molecule onto that template by graph isomorphism
   - stores each member row in canonical local order
3. `rebuild_topology()` produces:
   - `traj.topology_registry`
   - `traj.topology_frame`

This runtime topology model is the authoritative supported path for the
migrated analyses.

Connectivity detection uses:

- pair-specific absolute cutoffs from `BOND_DISTANCE_OVERRIDES` when present
- otherwise the fallback heuristic
  `(covalent_radius_A + covalent_radius_B) * BOND_DISTANCE_SCALE`

Override keys use canonical sorted element-pair strings such as `H-O` or
`Cl-H`, and override values are absolute distances in Angstrom.

## 4. Canonical Local Ordering

Canonical local ordering is a hard invariant of the supported runtime model.

For each compound type:

- one template member is labeled
- `canonical_labels` are established from that template
- every other member of the same type is mapped onto the template by graph
  isomorphism
- stored molecule rows follow the same canonical local ordering

This means:

- local column `i` has the same meaning across all members of a compound type
- label-based selections can be resolved once and reused safely
- per-member raw detection order is not analysis-facing state

## 5. Symmetry-Equivalent Atoms

For truly symmetry-equivalent atoms or equivalent subgroups:

- chemistry does not provide a unique physically meaningful numbering
- Dyana therefore uses a deterministic convention rather than a chemically
  privileged one

The current practical policy is:

- numbering is deterministic enough for reproducible work within a given
  trajectory/build
- canonical ordering is preserved consistently within one compound type
- template-to-member mapping preserves bonded/topological relationships under
  the chosen labeling

Important consequence:

- labels such as `H1` and `H2` may distinguish symmetry-equivalent atoms by
  convention only
- analyses should not assign deeper physical meaning to such numbering when the
  atoms are chemically equivalent

## 6. Analysis-Facing Access Pattern

The supported analyses should use the topology model like this:

### Setup/config time

- select compound types by index
- resolve them to compound-type keys
- resolve label patterns once via `TopologyFrame.resolve_selection(...)`

### Runtime

- reuse `ResolvedSelection.local_indices`
- retrieve current global atom ids via
  `TopologyFrame.get_atom_ids_for_local_indices(...)`
- reattach compound types by stable key after topology rebuilds

The supported path should not repeatedly rematch label strings during runtime.

## 7. Static vs Dynamic Topology

The code currently supports two operational modes in the frame loop:

### Static topology mode

- `update_compounds = False`
- topology is built once before the analysis frame loop
- the same topology structure is reused while coordinates change by frame

### Dynamic topology mode

- `update_compounds = True`
- `traj.rebuild_topology()` is called each processed frame
- analyses reattach their selected compound types by stable key
- analyses rebuild runtime selections/indices against the new frame topology

The analysis-facing access pattern is intended to stay the same in both modes.

## 8. Prepared Setups

Prepared setups align with the runtime topology model by storing:

- the topology recipe/configuration used for reconstruction
- compound-type signatures
- canonical labels per compound type
- informational metadata

Prepared setups validate compatible compound-type structure, not exact molecule
counts.

## 9. Current Boundaries

The current runtime topology model is the reference path for:

- RDF
- density
- ADF
- neighbor count
- tetrahedral order
- q6 / Q6
- LSI

Other analyses may still depend on older structures or helper paths and should
not be treated as the reference for topology-model behavior.
