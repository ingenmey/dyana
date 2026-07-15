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
    IDCFAnalysis = None
    IDCFCompoundSelection = None
    IDCFConfig = None
else:
    from analyses.idcf_analysis import IDCFAnalysis, IDCFCompoundSelection, IDCFConfig


class IdentityTrajectory:
    def __init__(self, proton_x_positions):
        self.box_size = np.array([40.0, 40.0, 40.0], dtype=float)
        self._frames = [self._frame(*positions) for positions in proton_x_positions]
        self.coords = self._frames[0].copy()
        self._frame_pointer = 0

        proton_type = CompoundType(
            type_id=0,
            key=("HP", (), "protons"),
            formula="HP",
            canonical_labels=("H1", "H2"),
            label_to_local_index={"H1": 0, "H2": 1},
            local_bonds=((0, 1),),
            local_elements=("H", "H"),
            atomic_masses=(1.0, 1.0),
        )
        acceptor_type = CompoundType(
            type_id=1,
            key=("A", (), "acceptors"),
            formula="A",
            canonical_labels=("O1",),
            label_to_local_index={"O1": 0},
            local_bonds=tuple(),
            local_elements=("O",),
            atomic_masses=(16.0,),
        )
        self.topology_registry = CompoundTypeRegistry([proton_type, acceptor_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                proton_type.key: np.array([[0, 1]], dtype=np.int32),
                acceptor_type.key: np.array([[2], [3]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 0, 1, 1], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 0, 1], dtype=np.int32),
            atom_to_local_index=np.array([0, 1, 0, 0], dtype=np.int32),
        )

    def _frame(self, proton1_x, proton2_x):
        return np.array(
            [
                [proton1_x, 0.0, 0.0],
                [proton2_x, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def read_frame(self):
        self._frame_pointer += 1
        if self._frame_pointer >= len(self._frames):
            raise ValueError("End of trajectory")
        self.coords = self._frames[self._frame_pointer].copy()


def parse_numeric_table(text):
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(value) for value in stripped.split()])
    return np.array(rows, dtype=float)


@unittest.skipIf(IDCFConfig is None, "scipy is not installed")
class IDCFAnalysisTests(unittest.TestCase):
    def build_config(self, **overrides):
        config = IDCFConfig(
            proton_sites=[IDCFCompoundSelection(compound_index=0, labels=["H1", "H2"])],
            acceptor_sites=[IDCFCompoundSelection(compound_index=1, labels=["O1"])],
            acceptor_identity="molecule",
            bond_cutoff=None,
            use_continuous=False,
            corr_depth=4,
            frame_time_fs=1000.0,
        )
        values = vars(config).copy()
        values.update(overrides)
        return IDCFConfig(**values)

    def test_idcf_config_validates_inputs(self):
        self.build_config()

        with self.assertRaises(ValueError):
            IDCFConfig(proton_sites=[], acceptor_sites=[IDCFCompoundSelection(compound_index=1, labels=["O1"])])
        with self.assertRaises(ValueError):
            self.build_config(acceptor_identity="cluster")
        with self.assertRaises(ValueError):
            self.build_config(bond_cutoff=0.0)
        with self.assertRaises(ValueError):
            self.build_config(corr_depth=0)
        with self.assertRaises(ValueError):
            self.build_config(frame_time_fs=0.0)

    def test_prompt_config_builds_supported_config(self):
        provider = FileInputProvider(
            lines=["1", "H1,H2", "2", "O1", "site", "", "n", "4", "2.0"],
            fallback=NullInputProvider(),
        )
        analysis = IDCFAnalysis(IdentityTrajectory([(0.1, 0.2)]), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            IDCFConfig(
                proton_sites=[IDCFCompoundSelection(compound_index=0, labels=["H1", "H2"])],
                acceptor_sites=[IDCFCompoundSelection(compound_index=1, labels=["O1"])],
                acceptor_identity="site",
                bond_cutoff=None,
                use_continuous=False,
                corr_depth=4,
                frame_time_fs=2.0,
            ),
        )

    def test_optional_cutoff_can_leave_protons_unassigned(self):
        analysis = IDCFAnalysis(IdentityTrajectory([(1.6, 1.7)]))
        analysis.configure(self.build_config(bond_cutoff=1.0))

        self.assertEqual(
            set(analysis._collect_frame_entities()),
            {(0, ()), (1, ())},
        )

        no_cutoff = IDCFAnalysis(IdentityTrajectory([(1.6, 1.7)]))
        no_cutoff.configure(self.build_config(bond_cutoff=None))

        self.assertEqual(
            set(no_cutoff._collect_frame_entities()),
            {(0, (0, 1)), (1, ())},
        )

    def test_run_writes_state_resolved_intermittent_output(self):
        analysis = IDCFAnalysis(
            IdentityTrajectory([(0.1, 0.2), (0.1, 5.1), (0.1, 0.2)]),
            input_provider=NullInputProvider(),
        )
        analysis.configure(self.build_config(use_continuous=False, corr_depth=3))
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=3, frame_stride=1, update_compounds=False))
        output = io.StringIO()
        previous_console_state = console.capture_state()

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                console.configure(stream=output, log_path=None, use_color=False)
                analysis.run()
                summary = Path("idcf_molecule_H1+H2-HP_O1-A.dat")
                n0 = Path("idcf_molecule_n0_H1+H2-HP_O1-A.dat")
                n1 = Path("idcf_molecule_n1_H1+H2-HP_O1-A.dat")
                n2 = Path("idcf_molecule_n2_H1+H2-HP_O1-A.dat")
                self.assertTrue(summary.exists())
                self.assertTrue(n0.exists())
                self.assertTrue(n1.exists())
                self.assertTrue(n2.exists())
                summary_data = parse_numeric_table(summary.read_text(encoding="utf-8"))
                n0_data = parse_numeric_table(n0.read_text(encoding="utf-8"))
                n1_data = parse_numeric_table(n1.read_text(encoding="utf-8"))
                n2_data = parse_numeric_table(n2.read_text(encoding="utf-8"))
            finally:
                console.close()
                console.restore_state(previous_console_state)
                os.chdir(cwd)

        np.testing.assert_allclose(
            summary_data,
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            n0_data,
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            n1_data,
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            n2_data,
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
            ],
            atol=1e-6,
        )
        console_text = output.getvalue()
        self.assertIn("IDCF state labels:", console_text)
        self.assertIn("n0: acceptor context with 0 selected protons bound (unprotonated)", console_text)
        self.assertIn("n1: acceptor context with 1 selected proton bound", console_text)
        self.assertIn("n2: acceptor context with 2 selected protons bound", console_text)
        self.assertIn("Saved summarized IDCF results to idcf_molecule_H1+H2-HP_O1-A.dat", console_text)

    def test_run_writes_state_resolved_continuous_output(self):
        analysis = IDCFAnalysis(
            IdentityTrajectory([(0.1, 0.2), (0.1, 5.1), (0.1, 0.2)]),
            input_provider=NullInputProvider(),
        )
        analysis.configure(self.build_config(use_continuous=True, corr_depth=3))
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=3, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                summary = Path("idcf_molecule_H1+H2-HP_O1-A.dat")
                n0 = Path("idcf_molecule_n0_H1+H2-HP_O1-A.dat")
                n1 = Path("idcf_molecule_n1_H1+H2-HP_O1-A.dat")
                n2 = Path("idcf_molecule_n2_H1+H2-HP_O1-A.dat")
                self.assertTrue(summary.exists())
                self.assertTrue(n0.exists())
                self.assertTrue(n1.exists())
                self.assertTrue(n2.exists())
                summary_data = parse_numeric_table(summary.read_text(encoding="utf-8"))
                n0_data = parse_numeric_table(n0.read_text(encoding="utf-8"))
                n1_data = parse_numeric_table(n1.read_text(encoding="utf-8"))
                n2_data = parse_numeric_table(n2.read_text(encoding="utf-8"))
            finally:
                os.chdir(cwd)

        np.testing.assert_allclose(
            summary_data,
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            n0_data,
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            n1_data,
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            n2_data,
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
