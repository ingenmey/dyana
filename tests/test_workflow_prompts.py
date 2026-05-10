import unittest
import tempfile
from pathlib import Path
import importlib.util

import numpy as np

from io_support.input_providers import FileInputProvider, NullInputProvider
from workflow.workflow_prompts import WorkflowPrompts
from workflow.prepared_setup import build_prepared_setup, save_prepared_setup

if importlib.util.find_spec("networkx") is None:
    load_trajectory = None
else:
    from core.trajectory_loader import load_trajectory


TWO_FRAME_WATER = """3
frame 1
O 0.000 0.000 0.000
H 0.958 0.000 0.000
H -0.239 0.927 0.000
3
frame 2
O 0.100 0.000 0.000
H 1.058 0.000 0.000
H -0.139 0.927 0.000
"""


class NonInteractiveWorkflowPrompts(WorkflowPrompts):
    def process_compounds(self, traj, provider=None):
        traj.rebuild_topology()


class WorkflowPromptsTests(unittest.TestCase):
    def test_prompt_cell_vectors_uses_shared_schema_engine(self):
        provider = FileInputProvider(lines=["10.0", "11.0", "12.0"], fallback=NullInputProvider())
        prompts = WorkflowPrompts(input_provider=provider)

        cell_vectors = prompts.prompt_cell_vectors("xyz")

        np.testing.assert_allclose(cell_vectors, [10.0, 11.0, 12.0])

    def test_prompt_cell_vectors_for_lammps_returns_zero_box_placeholder(self):
        prompts = WorkflowPrompts(input_provider=NullInputProvider())

        cell_vectors = prompts.prompt_cell_vectors("lammps")

        np.testing.assert_allclose(cell_vectors, [0.0, 0.0, 0.0])

    @unittest.skipIf(load_trajectory is None, "networkx is not installed")
    def test_prepare_trajectory_keeps_file_open_for_next_frame(self):
        provider = FileInputProvider(lines=["10.0", "10.0", "10.0"], fallback=NullInputProvider())
        prompts = NonInteractiveWorkflowPrompts(input_provider=provider)

        with tempfile.TemporaryDirectory() as tmp:
            traj_path = Path(tmp) / "water2.xyz"
            traj_path.write_text(TWO_FRAME_WATER, encoding="utf-8")

            traj = prompts.prepare_trajectory(str(traj_path), "xyz")
            self.assertFalse(traj.fin.closed)

            traj.read_frame()
            np.testing.assert_allclose(traj.coords[0], [0.1, 0.0, 0.0])
            traj.fin.close()

    @unittest.skipIf(load_trajectory is None, "networkx is not installed")
    def test_prepare_trajectory_from_setup_keeps_file_open_for_next_frame(self):
        prompts = NonInteractiveWorkflowPrompts(input_provider=NullInputProvider())

        with tempfile.TemporaryDirectory() as tmp:
            traj_path = Path(tmp) / "water2.xyz"
            setup_path = Path(tmp) / "prepared_setup.json"
            traj_path.write_text(TWO_FRAME_WATER, encoding="utf-8")

            with open(traj_path, "r", encoding="utf-8") as fin:
                traj = load_trajectory(fin, "xyz", np.array([10.0, 10.0, 10.0]))
                traj.read_frame()
                traj.rebuild_topology()
                prepared = build_prepared_setup(traj, str(traj_path), "xyz", [10.0, 10.0, 10.0])
                save_prepared_setup(setup_path, prepared)

            loaded_traj = prompts.prepare_trajectory_from_setup(str(traj_path), str(setup_path))
            self.assertFalse(loaded_traj.fin.closed)

            loaded_traj.read_frame()
            np.testing.assert_allclose(loaded_traj.coords[0], [0.1, 0.0, 0.0])
            loaded_traj.fin.close()


if __name__ == "__main__":
    unittest.main()
