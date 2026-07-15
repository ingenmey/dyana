import importlib.util
import io
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from framework.config_schema import FrameLoopConfig
from io_support.console import console
from io_support.input_providers import FileInputProvider, NullInputProvider

if importlib.util.find_spec("scipy") is None:
    PCCFAnalysis = None
    PCCFCompoundSelection = None
    PCCFConfig = None
    TransferEvent = None
    _count_transfer_chains = None
else:
    from analyses.pccf_analysis import (
        PCCFAnalysis,
        PCCFCompoundSelection,
        PCCFConfig,
        TransferEvent,
        _count_transfer_chains,
    )


class ProtonTransferTrajectory:
    def __init__(self, proton_x_positions):
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self._frames = [self._frame(proton_x) for proton_x in proton_x_positions]
        self.coords = self._frames[0].copy()
        self._frame_pointer = 0

        proton_type = CompoundType(
            type_id=0,
            key=("H", (), "proton"),
            formula="H",
            canonical_labels=("H1",),
            label_to_local_index={"H1": 0},
            local_bonds=tuple(),
            local_elements=("H",),
            atomic_masses=(1.0,),
        )
        acceptor_type = CompoundType(
            type_id=1,
            key=("A2", (), "acceptor"),
            formula="A2",
            canonical_labels=("O1", "O2"),
            label_to_local_index={"O1": 0, "O2": 1},
            local_bonds=((0, 1),),
            local_elements=("O", "O"),
            atomic_masses=(16.0, 16.0),
        )
        self.topology_registry = CompoundTypeRegistry([proton_type, acceptor_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                ("H", (), "proton"): np.array([[0]], dtype=np.int32),
                ("A2", (), "acceptor"): np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int32),
            atom_to_local_index=np.array([0, 0, 1, 0, 1, 0, 1], dtype=np.int32),
        )

    def _frame(self, proton_x):
        return np.array(
            [
                [proton_x, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.2, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.2, 0.0, 0.0],
            ],
            dtype=float,
        )

    def read_frame(self):
        self._frame_pointer += 1
        if self._frame_pointer >= len(self._frames):
            raise ValueError("End of trajectory")
        self.coords = self._frames[self._frame_pointer].copy()


@unittest.skipIf(PCCFConfig is None, "scipy is not installed")
class PCCFAnalysisTests(unittest.TestCase):
    def build_config(self, **overrides):
        config = PCCFConfig(
            proton_sites=[PCCFCompoundSelection(compound_index=0, labels=["H"])],
            acceptor_sites=[PCCFCompoundSelection(compound_index=1, labels=["O1", "O2"])],
            acceptor_identity="molecule",
            bond_cutoff=1.0,
            dwell_threshold=1,
            max_unassigned_gap=2,
            max_chain_gaps=[1, 3],
        )
        values = vars(config).copy()
        values.update(overrides)
        return PCCFConfig(**values)

    def test_pccf_config_validates_inputs(self):
        self.build_config()

        with self.assertRaises(ValueError):
            PCCFConfig(
                proton_sites=[],
                acceptor_sites=[PCCFCompoundSelection(compound_index=1, labels=["O1"])],
            )
        with self.assertRaises(ValueError):
            PCCFCompoundSelection(compound_index=0, labels=[])
        with self.assertRaises(ValueError):
            self.build_config(acceptor_identity="cluster")
        with self.assertRaises(ValueError):
            self.build_config(bond_cutoff=0.0)
        with self.assertRaises(ValueError):
            self.build_config(max_unassigned_gap=-1)
        with self.assertRaises(ValueError):
            self.build_config(max_chain_gaps=[])

    def test_prompt_config_builds_supported_config(self):
        provider = FileInputProvider(
            lines=["1", "H", "2", "O1,O2", "site", "1.0", "2", "3", "1, 3"],
            fallback=NullInputProvider(),
        )
        analysis = PCCFAnalysis(ProtonTransferTrajectory([0.1]), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            PCCFConfig(
                proton_sites=[PCCFCompoundSelection(compound_index=0, labels=["H"])],
                acceptor_sites=[PCCFCompoundSelection(compound_index=1, labels=["O1", "O2"])],
                acceptor_identity="site",
                bond_cutoff=1.0,
                dwell_threshold=2,
                max_unassigned_gap=3,
                max_chain_gaps=[1, 3],
            ),
        )

    def test_configure_sets_up_runtime_state_for_molecule_and_site_identity(self):
        analysis = PCCFAnalysis(ProtonTransferTrajectory([0.1]))
        analysis.configure(self.build_config())

        self.assertFalse(analysis.allow_compound_update)
        self.assertEqual(analysis.proton_indices.tolist(), [0])
        self.assertEqual(analysis.acceptor_indices.tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(analysis.acceptor_context_ids.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(analysis.n_protons, 1)
        self.assertEqual(analysis.n_acceptors, 3)

        site_mode = PCCFAnalysis(ProtonTransferTrajectory([0.1]))
        site_mode.configure(self.build_config(acceptor_identity="site"))
        self.assertEqual(site_mode.acceptor_context_ids.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(site_mode.n_acceptors, 6)

    def test_short_unassigned_gap_is_bridged_between_stable_residences(self):
        analysis = PCCFAnalysis(
            ProtonTransferTrajectory([0.1, 0.1, 2.5, 2.5, 5.1, 5.1]),
            input_provider=NullInputProvider(),
        )
        analysis.configure(self.build_config(dwell_threshold=2, max_unassigned_gap=2))

        _process_all_frames(analysis, 6)

        self.assertEqual(
            analysis.transfer_events,
            [TransferEvent(frame=4, proton_id=0, donor_context=0, acceptor_context=1)],
        )

    def test_unstable_excursion_back_to_original_acceptor_emits_no_transfer(self):
        analysis = PCCFAnalysis(
            ProtonTransferTrajectory([0.1, 0.1, 5.1, 0.1, 0.1]),
            input_provider=NullInputProvider(),
        )
        analysis.configure(self.build_config(dwell_threshold=2, max_unassigned_gap=2))

        _process_all_frames(analysis, 5)

        self.assertEqual(analysis.transfer_events, [])

    def test_long_unassigned_gap_breaks_residence_carry_over(self):
        analysis = PCCFAnalysis(
            ProtonTransferTrajectory([0.1, 0.1, 2.5, 2.5, 2.5, 5.1, 5.1]),
            input_provider=NullInputProvider(),
        )
        analysis.configure(self.build_config(dwell_threshold=2, max_unassigned_gap=2))

        _process_all_frames(analysis, 7)

        self.assertEqual(analysis.transfer_events, [])

    def test_count_transfer_chains_links_relays_by_accept_then_donate_order(self):
        chain_counts = _count_transfer_chains(
            [
                TransferEvent(frame=1, proton_id=0, donor_context=0, acceptor_context=1),
                TransferEvent(frame=3, proton_id=1, donor_context=1, acceptor_context=2),
                TransferEvent(frame=20, proton_id=2, donor_context=4, acceptor_context=1),
                TransferEvent(frame=21, proton_id=3, donor_context=1, acceptor_context=3),
            ],
            max_chain_gap=30,
        )

        self.assertEqual(chain_counts, Counter({1: 2, 2: 2}))

    def test_run_writes_gap_specific_outputs_with_stable_events(self):
        analysis = PCCFAnalysis(ProtonTransferTrajectory([0.1, 5.1, 5.1, 10.1]), input_provider=NullInputProvider())
        analysis.configure(self.build_config())
        analysis.configure_frame_loop(
            FrameLoopConfig(start_frame=1, nframes=4, frame_stride=1, update_compounds=False)
        )
        output = io.StringIO()
        previous_console_state = console.capture_state()

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                console.configure(stream=output, log_path=None, use_color=False)
                analysis.run()
                gap1 = Path("pccf_molecule_gap1.dat")
                gap3 = Path("pccf_molecule_gap3.dat")
                self.assertTrue(gap1.exists())
                self.assertTrue(gap3.exists())
                gap1_text = gap1.read_text(encoding="utf-8")
                gap3_text = gap3.read_text(encoding="utf-8")
            finally:
                console.close()
                console.restore_state(previous_console_state)
                os.chdir(cwd)

        self.assertEqual(_read_rows(gap1_text), [[1.0, 2.0, 1.0, 1.0]])
        self.assertEqual(_read_rows(gap3_text), [[1.0, 1.0, 0.5, 1.0], [2.0, 1.0, 0.5, 0.5]])
        self.assertIn("Stable transfer events: 2", output.getvalue())
        self.assertNotIn("squashing", output.getvalue())


def _process_all_frames(analysis, nframes):
    for frame_number in range(nframes):
        analysis.process_frame()
        analysis.processed_frames += 1
        if frame_number == nframes - 1:
            break
        analysis.frame_idx += 1
        analysis.traj.read_frame()


def _read_rows(text: str) -> list[list[float]]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(value) for value in stripped.split()])
    return rows


if __name__ == "__main__":
    unittest.main()
