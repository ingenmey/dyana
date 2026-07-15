import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from framework.config_schema import FrameLoopConfig
from io_support.console import console
from io_support.input_providers import FileInputProvider, NullInputProvider

if importlib.util.find_spec("scipy") is None:
    PercolationAnalysis = None
    PercolationCompoundSelection = None
    PercolationConfig = None
    _fit_power_law = None
else:
    from analyses.percolation_analysis import (
        PercolationAnalysis,
        PercolationCompoundSelection,
        PercolationConfig,
        _fit_power_law,
    )


class PercolationTrajectory:
    def __init__(self):
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self.coords = np.array(
            [
                [0.9, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [1.6, 0.0, 0.0],
                [4.1, 0.0, 0.0],
                [3.2, 0.0, 0.0],
            ],
            dtype=float,
        )

        compound_type = CompoundType(
            type_id=0,
            key=("HO", (), "percolation"),
            formula="HO",
            canonical_labels=("H1", "O1"),
            label_to_local_index={"H1": 0, "O1": 1},
            local_bonds=((0, 1),),
            local_elements=("H", "O"),
            atomic_masses=(1.0, 16.0),
        )
        self.topology_registry = CompoundTypeRegistry([compound_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                compound_type.key: np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 0, 0, 0, 0, 0], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
            atom_to_local_index=np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        )

    def rebuild_topology(self):
        return None

    def read_frame(self):
        raise ValueError("End of trajectory")


class PeriodicPercolationTrajectory:
    def __init__(self):
        self.box_size = np.array([10.0, 10.0, 10.0], dtype=float)
        self.coords = np.array(
            [
                [9.5, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [9.8, 0.0, 0.0],
            ],
            dtype=float,
        )

        compound_type = CompoundType(
            type_id=0,
            key=("HO", (), "periodic"),
            formula="HO",
            canonical_labels=("H1", "O1"),
            label_to_local_index={"H1": 0, "O1": 1},
            local_bonds=((0, 1),),
            local_elements=("H", "O"),
            atomic_masses=(1.0, 16.0),
        )
        self.topology_registry = CompoundTypeRegistry([compound_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                compound_type.key: np.array([[0, 1], [2, 3]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 0, 0, 0], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 1, 1], dtype=np.int32),
            atom_to_local_index=np.array([0, 1, 0, 1], dtype=np.int32),
        )

    def rebuild_topology(self):
        return None

    def read_frame(self):
        raise ValueError("End of trajectory")


def parse_numeric_table(text):
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(value) for value in stripped.split()])
    return np.array(rows, dtype=float)


@unittest.skipIf(PercolationConfig is None, "scipy is not installed")
class PercolationAnalysisTests(unittest.TestCase):
    def build_config(self, **overrides):
        config = PercolationConfig(
            compounds=[PercolationCompoundSelection(compound_index=0, acceptor_labels=["O"], proton_labels=["H"])],
            cutoff=1.0,
            max_depth=2,
            use_alternating=False,
            fit_min_depth=1,
            fit_max_depth=-1,
        )
        values = vars(config).copy()
        values.update(overrides)
        return PercolationConfig(**values)

    def test_percolation_config_validates_inputs(self):
        self.build_config()
        PercolationConfig(
            compounds=[PercolationCompoundSelection(compound_index=0, acceptor_labels=["O"], proton_labels=[])],
            cutoff=1.0,
            max_depth=2,
            fit_min_depth=1,
            fit_max_depth=-1,
        )
        PercolationConfig(
            compounds=[PercolationCompoundSelection(compound_index=0, acceptor_labels=[], proton_labels=["H"])],
            cutoff=1.0,
            max_depth=2,
            fit_min_depth=1,
            fit_max_depth=-1,
        )

        with self.assertRaises(ValueError):
            PercolationConfig(compounds=[], cutoff=1.0, max_depth=2)
        with self.assertRaises(ValueError):
            PercolationCompoundSelection(compound_index=0, acceptor_labels=[], proton_labels=[])
        with self.assertRaises(ValueError):
            self.build_config(cutoff=0.0)
        with self.assertRaises(ValueError):
            self.build_config(max_depth=0)
        with self.assertRaises(ValueError):
            self.build_config(fit_min_depth=0)
        with self.assertRaises(ValueError):
            self.build_config(fit_min_depth=2, fit_max_depth=1)

    def test_prompt_config_builds_supported_config(self):
        provider = FileInputProvider(
            lines=["1", "O", "", "1.0", "4", "y", "1", "-1"],
            fallback=NullInputProvider(),
        )
        analysis = PercolationAnalysis(PercolationTrajectory(), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            PercolationConfig(
                compounds=[PercolationCompoundSelection(compound_index=0, acceptor_labels=["O"], proton_labels=[])],
                cutoff=1.0,
                max_depth=4,
                use_alternating=True,
                fit_min_depth=1,
                fit_max_depth=-1,
            ),
        )

    def test_configure_sets_up_runtime_state(self):
        analysis = PercolationAnalysis(PercolationTrajectory())
        analysis.configure(self.build_config())

        self.assertEqual(analysis.n_nodes, 3)
        self.assertEqual(analysis.acceptor_indices.tolist(), [1, 3, 5])
        self.assertEqual(analysis.acceptor_owner_ids.tolist(), [0, 1, 2])
        self.assertEqual(analysis.proton_indices.tolist(), [0, 2, 4])
        self.assertEqual(analysis.proton_owner_ids.tolist(), [0, 1, 2])
        self.assertEqual(analysis.representative_atom_ids.tolist(), [1, 3, 5])

    def test_fit_power_law_returns_nan_when_too_few_points_survive(self):
        slope, intercept = _fit_power_law([1.0, 2.0], [0.0, 3.0])
        self.assertTrue(np.isnan(slope))
        self.assertTrue(np.isnan(intercept))

    def test_run_writes_directed_reachability_metrics(self):
        analysis = PercolationAnalysis(PercolationTrajectory(), input_provider=NullInputProvider())
        analysis.configure(self.build_config(use_alternating=True, fit_min_depth=1))
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))
        output = io.StringIO()
        previous_console_state = console.capture_state()

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                console.configure(stream=output, log_path=None, use_color=False)
                analysis.run()
                result = Path("percolation_alternating.dat")
                self.assertTrue(result.exists())
                text = result.read_text(encoding="utf-8")
                data = parse_numeric_table(text)
            finally:
                console.close()
                console.restore_state(previous_console_state)
                os.chdir(cwd)

        self.assertIn("ShellPopulation", text)
        self.assertIn("A_l", text)
        self.assertIn("R_l", text)
        np.testing.assert_allclose(
            data,
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 2.0 / 3.0, 5.0 / 3.0, 1.6],
                [2.0, 1.0 / 3.0, 2.0, 3.2],
            ],
            atol=1e-6,
        )
        console_text = output.getvalue()
        self.assertIn("S(l)=0.666667, A(l)=1.666667, R(l)=1.600000", console_text)
        self.assertIn("d_l: 0.263034", console_text)
        self.assertIn("d_min: 1.000000", console_text)
        self.assertIn("d_f: 0.263034", console_text)
        self.assertIn("mean out-degree: 0.666667", console_text)
        self.assertIn("mean in-degree: 0.666667", console_text)
        self.assertIn("f_out_degree2: 0.000000", console_text)
        self.assertIn("f_in_degree2: 0.000000", console_text)
        self.assertIn("f_out_branch: 0.000000", console_text)
        self.assertIn("f_in_branch: 0.000000", console_text)
        self.assertNotIn("loop density:", console_text)

    def test_run_writes_undirected_network_metrics(self):
        analysis = PercolationAnalysis(PercolationTrajectory(), input_provider=NullInputProvider())
        analysis.configure(self.build_config(use_alternating=False, fit_min_depth=1))
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))
        output = io.StringIO()
        previous_console_state = console.capture_state()

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                console.configure(stream=output, log_path=None, use_color=False)
                analysis.run()
                result = Path("percolation_network.dat")
                self.assertTrue(result.exists())
                text = result.read_text(encoding="utf-8")
                data = parse_numeric_table(text)
            finally:
                console.close()
                console.restore_state(previous_console_state)
                os.chdir(cwd)

        self.assertIn("ShellPopulation", text)
        self.assertIn("A_l", text)
        self.assertIn("R_l", text)
        np.testing.assert_allclose(
            data,
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 4.0 / 3.0, 7.0 / 3.0, 1.6],
                [2.0, 2.0 / 3.0, 3.0, 3.2],
            ],
            atol=1e-6,
        )
        console_text = output.getvalue()
        self.assertIn("d_l: 0.362570", console_text)
        self.assertIn("d_min: 1.000000", console_text)
        self.assertIn("d_f: 0.362570", console_text)
        self.assertIn("mean degree: 1.333333", console_text)
        self.assertIn("f_degree2: 0.333333", console_text)
        self.assertIn("f_branch: 0.000000", console_text)
        self.assertIn("loop density: 0.000000", console_text)

    def test_r_l_respects_minimum_image_in_undirected_mode(self):
        analysis = PercolationAnalysis(PeriodicPercolationTrajectory(), input_provider=NullInputProvider())
        analysis.configure(self.build_config(max_depth=1, fit_min_depth=1, cutoff=0.8))
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                result = Path("percolation_network.dat")
                self.assertTrue(result.exists())
                data = parse_numeric_table(result.read_text(encoding="utf-8"))
            finally:
                os.chdir(cwd)

        np.testing.assert_allclose(
            data,
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 2.0, 0.4],
            ],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
