import unittest
from dataclasses import dataclass

import numpy as np

from analyses.common.pair_selectors import (
    ObservedAtomGroupSpec,
    PairSelector,
    PairSelectorSpec,
    pair_selector_schema,
)
from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from framework.analysis_params import CompoundParam
from framework.config_builder import prompt_config_from_schema
from io_support.input_providers import FileInputProvider, NullInputProvider


@dataclass(frozen=True)
class SelectorWrapperConfig:
    ref_compound_index: int
    selector: PairSelectorSpec


class PromptCompoundType:
    def __init__(self, formula, type_id, canonical_labels):
        self.formula = formula
        self.type_id = type_id
        self.canonical_labels = canonical_labels


class PromptTopologyFrame:
    def __init__(self, compound_types):
        self.compound_types = compound_types

    def get_compound_type_by_index(self, index):
        return self.compound_types[index]


class PromptTrajectory:
    def __init__(self, compound_types):
        self.topology_frame = PromptTopologyFrame(compound_types)


class PromptAnalysis:
    def __init__(self, provider=None):
        self.input_provider = provider
        self.compound_types = [
            PromptCompoundType("A", 0, ("A1",)),
            PromptCompoundType("B", 1, ("B1", "B2")),
        ]
        self.traj = PromptTrajectory(self.compound_types)

    def compound_selection(self, role="reference", multi=False, prompt_text=None, provider=None):
        input_provider = provider or self.input_provider
        if multi:
            prompt = prompt_text or f"Choose the {role} compounds (comma-separated numbers): "
            answer = input_provider.ask_str(prompt)
            indices = [int(item.strip()) - 1 for item in answer.split(",") if item.strip()]
            return [(index, self.compound_types[index]) for index in indices]
        prompt = prompt_text or f"Choose the {role} compound (number): "
        index = input_provider.ask_int(prompt, 1, minval=1) - 1
        return index, self.compound_types[index]

    def atom_selection(self, role="reference", compound=None, prompt_text=None, allow_empty=False, provider=None):
        input_provider = provider or self.input_provider
        prompt = prompt_text or f"Which atom(s) in {role} compound? (comma-separated) "
        answer = input_provider.ask_str(prompt, default="" if allow_empty else None)
        return [item.strip() for item in answer.split(",") if item.strip()]


class PairSelectorTrajectory:
    def __init__(self):
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self.coords = np.array(
            [
                [0.0, 0.0, 0.0],   # A1
                [2.0, 0.0, 0.0],   # B1
                [4.0, 0.0, 0.0],   # B2
                [7.0, 0.0, 0.0],   # B1 second molecule
            ],
            dtype=float,
        )
        ref_type = CompoundType(
            type_id=0,
            key=("A", (), "ref"),
            formula="A",
            canonical_labels=("A1",),
            label_to_local_index={"A1": 0},
            local_bonds=tuple(),
            local_elements=("A",),
            atomic_masses=(1.0,),
        )
        obs_type = CompoundType(
            type_id=1,
            key=("B2", (), "obs"),
            formula="B2",
            canonical_labels=("B1", "B2"),
            label_to_local_index={"B1": 0, "B2": 1},
            local_bonds=tuple(),
            local_elements=("B", "B"),
            atomic_masses=(1.0, 1.0),
        )
        self.topology_registry = CompoundTypeRegistry([ref_type, obs_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                ref_type.key: np.array([[0]], dtype=np.int32),
                obs_type.key: np.array([[1, 2], [3, 3]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 1, 1, 1], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 0, 1], dtype=np.int32),
            atom_to_local_index=np.array([0, 0, 1, 0], dtype=np.int32),
        )


class SelfPairSelectorTrajectory:
    def __init__(self):
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self.coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        compound_type = CompoundType(
            type_id=0,
            key=("A", (), "same"),
            formula="A",
            canonical_labels=("A1",),
            label_to_local_index={"A1": 0},
            local_bonds=tuple(),
            local_elements=("A",),
            atomic_masses=(1.0,),
        )
        self.topology_registry = CompoundTypeRegistry([compound_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={compound_type.key: np.array([[0], [1]], dtype=np.int32)},
            atom_to_type_id=np.array([0, 0], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 1], dtype=np.int32),
            atom_to_local_index=np.array([0, 0], dtype=np.int32),
        )


class RuntimeOwner:
    def __init__(self, traj):
        self.traj = traj

    def resolve_compound_types(self, indices):
        resolved = []
        for index in indices:
            compound_type = self.traj.topology_frame.get_compound_type_by_index(index)
            resolved.append((compound_type, compound_type.key))
        return resolved


class PairSelectorTests(unittest.TestCase):
    def test_pair_selector_spec_validates_criteria(self):
        PairSelectorSpec(
            ref_labels=["A"],
            observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
            min_distance=0.0,
            max_distance=3.5,
        )

        with self.assertRaises(ValueError):
            PairSelectorSpec(
                ref_labels=["A"],
                observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
            )
        with self.assertRaises(ValueError):
            PairSelectorSpec(
                ref_labels=["A"],
                observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
                min_distance=1.0,
            )
        with self.assertRaises(ValueError):
            PairSelectorSpec(
                ref_labels=["A"],
                observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
                min_rank=0,
                max_rank=1,
            )

    def test_pair_selector_schema_builds_multi_observed_selection_and_conditions(self):
        provider = FileInputProvider(
            lines=["1", "A", "2", "B", "y", "1.0", "4.5", "y", "1", "2"],
            fallback=NullInputProvider(),
        )
        analysis = PromptAnalysis(provider=provider)
        schema = [
            CompoundParam(name="ref_compound_index", role="reference"),
            pair_selector_schema(
                name="selector",
                label="the test selector",
                ref_compound_field="ref_compound_index",
            ),
        ]

        config = prompt_config_from_schema(analysis, schema, SelectorWrapperConfig, provider=provider)

        self.assertEqual(
            config,
            SelectorWrapperConfig(
                ref_compound_index=0,
                selector=PairSelectorSpec(
                    ref_labels=["A"],
                    observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
                    min_distance=1.0,
                    max_distance=4.5,
                    min_rank=1,
                    max_rank=2,
                ),
            ),
        )

    def test_distance_only_selector_returns_distance_window(self):
        traj = PairSelectorTrajectory()
        selector = PairSelector(
            RuntimeOwner(traj),
            traj.topology_frame.get_compound_type_by_index(0),
            ("A", (), "ref"),
            PairSelectorSpec(
                ref_labels=["A"],
                observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
                min_distance=3.0,
                max_distance=7.5,
            ),
        )

        obs_ids_by_ref, _, distances_by_ref = selector.select_frame()

        self.assertEqual(obs_ids_by_ref[0].tolist(), [2, 3])
        np.testing.assert_allclose(distances_by_ref[0], [4.0, 7.0])

    def test_rank_only_selector_returns_requested_neighbour_ranks(self):
        traj = PairSelectorTrajectory()
        selector = PairSelector(
            RuntimeOwner(traj),
            traj.topology_frame.get_compound_type_by_index(0),
            ("A", (), "ref"),
            PairSelectorSpec(
                ref_labels=["A"],
                observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
                min_rank=2,
                max_rank=3,
            ),
        )

        obs_ids_by_ref, _, distances_by_ref = selector.select_frame()

        self.assertEqual(obs_ids_by_ref[0].tolist(), [2, 3])
        np.testing.assert_allclose(distances_by_ref[0], [4.0, 7.0])

    def test_rank_is_assigned_before_distance_filtering(self):
        traj = PairSelectorTrajectory()
        selector = PairSelector(
            RuntimeOwner(traj),
            traj.topology_frame.get_compound_type_by_index(0),
            ("A", (), "ref"),
            PairSelectorSpec(
                ref_labels=["A"],
                observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
                min_distance=3.0,
                max_distance=5.0,
                min_rank=1,
                max_rank=1,
            ),
        )

        obs_ids_by_ref, _, distances_by_ref = selector.select_frame()

        self.assertEqual(obs_ids_by_ref[0].tolist(), [])
        np.testing.assert_allclose(distances_by_ref[0], [])

    def test_repeated_observed_groups_are_combined(self):
        traj = PairSelectorTrajectory()
        selector = PairSelector(
            RuntimeOwner(traj),
            traj.topology_frame.get_compound_type_by_index(0),
            ("A", (), "ref"),
            PairSelectorSpec(
                ref_labels=["A"],
                observed_groups=[
                    ObservedAtomGroupSpec(compound_index=1, labels=["B1"]),
                    ObservedAtomGroupSpec(compound_index=1, labels=["B2"]),
                ],
                min_rank=1,
                max_rank=3,
            ),
        )

        obs_ids_by_ref, _, distances_by_ref = selector.select_frame()

        self.assertEqual(obs_ids_by_ref[0].tolist(), [1, 2, 3])
        np.testing.assert_allclose(distances_by_ref[0], [2.0, 4.0, 7.0])

    def test_selector_excludes_exact_self_pairs_before_ranking(self):
        traj = SelfPairSelectorTrajectory()
        selector = PairSelector(
            RuntimeOwner(traj),
            traj.topology_frame.get_compound_type_by_index(0),
            ("A", (), "same"),
            PairSelectorSpec(
                ref_labels=["A"],
                observed_groups=[ObservedAtomGroupSpec(compound_index=0, labels=["A"])],
                min_rank=1,
                max_rank=1,
            ),
        )

        obs_ids_by_ref, _, distances_by_ref = selector.select_frame()

        self.assertEqual(obs_ids_by_ref[0].tolist(), [1])
        self.assertEqual(obs_ids_by_ref[1].tolist(), [0])
        np.testing.assert_allclose(distances_by_ref[0], [3.0])
        np.testing.assert_allclose(distances_by_ref[1], [3.0])


if __name__ == "__main__":
    unittest.main()
