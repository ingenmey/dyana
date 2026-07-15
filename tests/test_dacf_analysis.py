import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyses.common.pair_selectors import ObservedAtomGroupSpec, PairSelectorSpec
from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from framework.config_schema import FrameLoopConfig
from io_support.input_providers import FileInputProvider, NullInputProvider

if importlib.util.find_spec("scipy") is None:
    DACFAnalysis = None
    DACFConfig = None
else:
    from analyses.dacf_analysis import DACFAnalysis, DACFConfig


def build_topology(include_observed=True):
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
        key=("B", (), "obs"),
        formula="B",
        canonical_labels=("B1", "B2"),
        label_to_local_index={"B1": 0, "B2": 1},
        local_bonds=tuple(),
        local_elements=("B", "B"),
        atomic_masses=(1.0, 1.0),
    )
    registry = CompoundTypeRegistry([ref_type, obs_type])
    molecule_atom_ids_by_key = {
        ref_type.key: np.array([[0]], dtype=np.int32),
    }
    atom_to_type_id = np.array([0], dtype=np.int32)
    atom_to_molecule_index = np.array([0], dtype=np.int32)
    atom_to_local_index = np.array([0], dtype=np.int32)

    if include_observed:
        molecule_atom_ids_by_key[obs_type.key] = np.array([[1, 2]], dtype=np.int32)
        atom_to_type_id = np.array([0, 1, 1], dtype=np.int32)
        atom_to_molecule_index = np.array([0, 0, 0], dtype=np.int32)
        atom_to_local_index = np.array([0, 0, 1], dtype=np.int32)

    return TopologyFrame(
        registry=registry,
        molecule_atom_ids_by_key=molecule_atom_ids_by_key,
        atom_to_type_id=atom_to_type_id,
        atom_to_molecule_index=atom_to_molecule_index,
        atom_to_local_index=atom_to_local_index,
    )


class PairTrajectory:
    def __init__(self, frames):
        self.frames = [np.array(frame, dtype=float) for frame in frames]
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self.coords = self.frames[0].copy()
        self.topology_frame = build_topology()
        self.topology_registry = self.topology_frame.registry
        self._frame_index = 0

    def rebuild_topology(self):
        return None

    def read_frame(self):
        self._frame_index += 1
        if self._frame_index >= len(self.frames):
            raise ValueError("End of trajectory")
        self.coords = self.frames[self._frame_index].copy()


def parse_numeric_table(text):
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(value) for value in stripped.split()])
    return np.array(rows, dtype=float)


@unittest.skipIf(DACFConfig is None, "scipy is not installed")
class DACFAnalysisTests(unittest.TestCase):
    def build_config(self, **overrides):
        config = dict(
            ref_compound_index=0,
            selector=PairSelectorSpec(
                ref_labels=["A"],
                observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B1"])],
                min_distance=0.0,
                max_distance=1.5,
            ),
            use_continuous=False,
            corr_depth=4,
            apply_correction=False,
            frame_time_fs=1000.0,
            count_missing_as_zero=True,
        )
        config.update(overrides)
        return DACFConfig(**config)

    def test_dacf_config_validates_inputs(self):
        self.build_config()

        with self.assertRaises(ValueError):
            self.build_config(ref_compound_index=-1)
        with self.assertRaises(ValueError):
            self.build_config(corr_depth=0)
        with self.assertRaises(ValueError):
            self.build_config(frame_time_fs=0.0)

    def test_prompt_config_uses_shared_pair_selector_schema(self):
        provider = FileInputProvider(
            lines=["1", "A", "2", "B", "y", "0.0", "3.0", "n", "n", "4", "n", "2.0", "y"],
            fallback=NullInputProvider(),
        )
        analysis = DACFAnalysis(
            PairTrajectory([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]]]),
            input_provider=provider,
        )

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            DACFConfig(
                ref_compound_index=0,
                selector=PairSelectorSpec(
                    ref_labels=["A"],
                    observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
                    min_distance=0.0,
                    max_distance=3.0,
                ),
                use_continuous=False,
                corr_depth=4,
                apply_correction=False,
                frame_time_fs=2.0,
                count_missing_as_zero=True,
            ),
        )

    def test_configure_resolves_runtime_indices(self):
        analysis = DACFAnalysis(PairTrajectory([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]]]))

        analysis.configure(self.build_config())

        self.assertEqual(analysis.ref_indices.tolist(), [0])
        self.assertEqual(analysis.obs_indices.tolist(), [1])
        self.assertEqual(analysis.dimer_selector.ref_selection.local_indices, (0,))
        self.assertEqual(analysis.dimer_selector.obs_selections[0].local_indices, (0,))

    def test_post_compound_update_can_treat_missing_compounds_as_zero(self):
        analysis = DACFAnalysis(PairTrajectory([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]]]))
        analysis.configure(self.build_config(count_missing_as_zero=True))
        analysis.traj.topology_frame = build_topology(include_observed=False)

        self.assertTrue(analysis.post_compound_update())
        self.assertFalse(analysis.frame_active)

        strict = DACFAnalysis(PairTrajectory([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]]]))
        strict.configure(self.build_config(count_missing_as_zero=False))
        strict.traj.topology_frame = build_topology(include_observed=False)

        self.assertFalse(strict.post_compound_update())
        self.assertFalse(strict.frame_active)

    def test_process_frame_rank_selector_keeps_only_nearest_neighbour(self):
        frames = [[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]]]
        analysis = DACFAnalysis(PairTrajectory(frames))
        analysis.configure(
            self.build_config(
                selector=PairSelectorSpec(
                    ref_labels=["A"],
                    observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
                    min_rank=1,
                    max_rank=1,
                )
            )
        )

        analysis.process_frame()

        self.assertEqual(analysis.total_pair_occupancy, 1)
        self.assertEqual(dict(analysis.open_intervals), {(0, 1): 0})

    def test_process_frame_rank_selector_can_also_apply_distance_filter(self):
        frames = [[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.6, 0.0, 0.0]]]
        analysis = DACFAnalysis(PairTrajectory(frames))
        analysis.configure(
            self.build_config(
                selector=PairSelectorSpec(
                    ref_labels=["A"],
                    observed_groups=[ObservedAtomGroupSpec(compound_index=1, labels=["B"])],
                    min_distance=0.0,
                    max_distance=1.0,
                    min_rank=1,
                    max_rank=1,
                )
            )
        )

        analysis.process_frame()

        self.assertEqual(analysis.total_pair_occupancy, 0)
        self.assertEqual(dict(analysis.open_intervals), {})

    def test_run_writes_intermittent_output(self):
        frames = [
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ]
        analysis = DACFAnalysis(PairTrajectory(frames), input_provider=NullInputProvider())
        analysis.configure(self.build_config(use_continuous=False, apply_correction=False))
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=4, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                output = Path("dacf_A-A_B1-B.dat")
                self.assertTrue(output.exists())
                data = parse_numeric_table(output.read_text(encoding="utf-8"))
            finally:
                os.chdir(cwd)

        np.testing.assert_allclose(
            data,
            [
                [0.0, 1.0],
                [1.0, 2.0 / 3.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            atol=1e-6,
        )

    def test_run_writes_continuous_output(self):
        frames = [
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ]
        analysis = DACFAnalysis(PairTrajectory(frames), input_provider=NullInputProvider())
        analysis.configure(self.build_config(use_continuous=True))
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=4, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                output = Path("dacf_A-A_B1-B.dat")
                self.assertTrue(output.exists())
                data = parse_numeric_table(output.read_text(encoding="utf-8"))
            finally:
                os.chdir(cwd)

        np.testing.assert_allclose(
            data,
            [
                [0.0, 1.0],
                [1.0, 0.5],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
