# Trajectory and Topology Naming Remap Plan

This note captures the intended naming cleanup for the reworked trajectory/topology path.

Guiding rule: prefer clear, consistent, convenient names over preserving legacy terminology.

## Planned Renames

- `guess_molecules` -> `rebuild_topology` - the method now rebuilds the full runtime topology state, not just a molecule guess.
- `TypeRegistry` -> `CompoundTypeRegistry` - `TypeRegistry` is too generic for a compound-type-specific registry.
- `SelectionSpec` -> `ResolvedSelection` - the object now holds an already-resolved runtime selection rather than a generic "spec".
- `SelectionSpec.type_key` -> `ResolvedSelection.compound_type_key` - `type_key` is too vague for persisted/runtime topology references.
- `MoleculeGroup` -> `CompoundTypeGroup` - the grouped builder object represents one future compound type, not an arbitrary molecule collection.
- `rep` -> `formula` - the current value is a formula string, and `rep` is too short and opaque.
- `formula_str` -> `formula` - use one consistent formula name instead of mixing `rep` and `formula_str`.
- `member_atom_ids_by_key` -> `molecule_atom_ids_by_key` - the runtime rows represent molecules, and that should be explicit.
- `get_member_atom_ids` -> `get_molecule_atom_ids` - analyses reason about molecule atom rows, not generic members.
- `get_member_count` -> `get_molecule_count` - the count is a molecule count, and the name should say so directly.
- `get_member_coms` -> `get_molecule_coms` - these are molecule centers of mass, not generic member COMs.
- `get_member_coords` -> `get_molecule_coords` - the returned coordinates are molecule coordinates, and the name should match that.
- `atom_to_member_index` -> `atom_to_molecule_index` - the reverse map points from atom to molecule row, and `member` is less clear than `molecule`.
- local `member_index` variables -> `molecule_index` - keep loop/index variable names aligned with the runtime model vocabulary.
- `natoms` -> `n_atoms` - align with the existing `n_local_atoms` style and improve readability.
- `dimx` -> `box_x` - the current alias is terse and older-style; the value is the x box length.
- `dimy` -> `box_y` - the current alias is terse and older-style; the value is the y box length.
- `dimz` -> `box_z` - the current alias is terse and older-style; the value is the z box length.
- `are_connected` -> `is_bonded` - the current name sounds boolean already, so the implementation should match the name instead of returning a distance/false hybrid.
- `reset_frame_idx` -> `rewind_to_first_frame` - the method rewinds the trajectory and reloads frame one, not just an index.
- `_group_molecules` -> `_group_compound_types` - the helper groups detected molecules into future compound-type groups, not just generic molecule bins.
- `_build_topology_state` -> `_build_registry_and_frame` - the helper returns two concrete objects, and the name should say which ones.
- `compound_type_or_key` -> `type_or_key` - the longer parameter name is repetitive at call sites and can be shortened without losing meaning.
- `PreparedSetup.compound_types` -> `PreparedSetup.compound_type_entries` - the property returns serialized entries, not live `CompoundType` objects.

## Notes

- Where a rename implies a behavior cleanup too, the behavior should be cleaned up at the same time rather than leaving a misleading name in place.
- The `member` -> `molecule` renames should be treated as one systematic sweep, not as one-off patches.
- If a future user-facing label diverges from `formula`, introduce a separate `display_name` field instead of reviving `rep`.
