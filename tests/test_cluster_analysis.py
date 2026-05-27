import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from framework.config_schema import FrameLoopConfig
from io_support.input_providers import FileInputProvider, NullInputProvider

if importlib.util.find_spec("scipy") is None or importlib.util.find_spec("networkx") is None:
    ClusterAnalysis = None
    ClusterCompoundSpec = None
    ClusterConfig = None
    ClusterCutoffSpec = None
else:
    from analyses.cluster_analysis import ClusterAnalysis, ClusterCompoundSpec, ClusterConfig, ClusterCutoffSpec


class DummyTrajectory:
    def __init__(self):
        self.box_size = np.array([10.0, 10.0, 10.0], dtype=float)
        self.coords = np.array(
            [
                [0.0, 0.0, 0.0],   # Na molecule 0
                [5.0, 0.0, 0.0],   # Na molecule 1
                [0.9, 0.0, 0.0],   # Cl molecule 0
                [5.9, 0.0, 0.0],   # Cl molecule 1
            ],
            dtype=float,
        )

        na_type = CompoundType(
            type_id=0,
            key=("Na", (), "na"),
            formula="Na",
            canonical_labels=("Na1",),
            label_to_local_index={"Na1": 0},
            local_bonds=(),
            local_elements=("Na",),
            atomic_masses=(22.99,),
        )
        cl_type = CompoundType(
            type_id=1,
            key=("Cl", (), "cl"),
            formula="Cl",
            canonical_labels=("Cl1",),
            label_to_local_index={"Cl1": 0},
            local_bonds=(),
            local_elements=("Cl",),
            atomic_masses=(35.45,),
        )
        self.topology_registry = CompoundTypeRegistry([na_type, cl_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                ("Na", (), "na"): np.array([[0], [1]], dtype=np.int32),
                ("Cl", (), "cl"): np.array([[2], [3]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 0, 1, 1], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 1, 0, 1], dtype=np.int32),
            atom_to_local_index=np.array([0, 0, 0, 0], dtype=np.int32),
        )

    def read_frame(self):
        raise ValueError("End of trajectory")


@unittest.skipIf(ClusterConfig is None, "scipy or networkx is not installed")
class ClusterAnalysisTests(unittest.TestCase):
    def build_config(self, **overrides):
        config = ClusterConfig(
            selected_compounds=[
                ClusterCompoundSpec(compound_index=0, labels=["Na"]),
                ClusterCompoundSpec(compound_index=1, labels=["Cl"]),
            ],
            cutoffs=[
                ClusterCutoffSpec(
                    left_compound_index=0,
                    left_label="Na",
                    right_compound_index=1,
                    right_label="Cl",
                    cutoff=1.5,
                )
            ],
            hash_graphs=False,
            graph_format=None,
            save_xyz=False,
            save_whole_molecules=False,
            compute_cacf=False,
            corr_depth=100,
            compute_errors=True,
        )
        values = vars(config).copy()
        values.update(overrides)
        return ClusterConfig(**values)

    def test_cluster_config_validates_inputs(self):
        self.build_config()

        with self.assertRaises(ValueError):
            ClusterConfig(
                selected_compounds=[],
                cutoffs=[],
            )

        with self.assertRaises(ValueError):
            ClusterConfig(
                selected_compounds=[
                    ClusterCompoundSpec(compound_index=0, labels=["Na"]),
                    ClusterCompoundSpec(compound_index=1, labels=["Cl"]),
                ],
                cutoffs=[],
            )

    def test_prompt_config_builds_supported_cluster_config(self):
        provider = FileInputProvider(
            lines=["Na", "Cl", "1.5", "y", "n", "n", "n", "y"],
            fallback=NullInputProvider(),
        )
        analysis = ClusterAnalysis(DummyTrajectory(), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            ClusterConfig(
                selected_compounds=[
                    ClusterCompoundSpec(compound_index=0, labels=["Na"]),
                    ClusterCompoundSpec(compound_index=1, labels=["Cl"]),
                ],
                cutoffs=[
                    ClusterCutoffSpec(
                        left_compound_index=0,
                        left_label="Na",
                        right_compound_index=1,
                        right_label="Cl",
                        cutoff=1.5,
                    )
                ],
                hash_graphs=True,
                graph_format=None,
                save_xyz=False,
                save_whole_molecules=False,
                compute_cacf=False,
                corr_depth=100,
                compute_errors=True,
            ),
        )

    def test_configure_and_process_frame_count_clusters(self):
        analysis = ClusterAnalysis(DummyTrajectory())
        analysis.configure(self.build_config())

        analysis.process_frame()

        self.assertEqual(analysis.cluster_histogram[("1-Na-1_2-Cl-1", 0)], 2)
        self.assertEqual(analysis.frame_cluster_counts[("1-Na-1_2-Cl-1", 0)], [2])

    def test_postprocess_writes_supported_outputs(self):
        analysis = ClusterAnalysis(DummyTrajectory())
        analysis.configure(self.build_config())
        analysis.process_frame()
        analysis.processed_frames = 1

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.postprocess()
                occurrences = Path("cluster_occurrences.dat")
                populations = Path("cluster_populations.dat")
                size = Path("cluster_size.dat")
                self.assertTrue(occurrences.exists())
                self.assertTrue(populations.exists())
                self.assertTrue(size.exists())
                text = occurrences.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("1-Na-1_2-Cl-1", text)
        self.assertIn("GraphID", text)

    def test_run_uses_programmatic_configuration_without_prompting(self):
        analysis = ClusterAnalysis(DummyTrajectory(), input_provider=NullInputProvider())
        analysis.configure(self.build_config())
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                self.assertTrue(Path("cluster_occurrences.dat").exists())
                self.assertTrue(Path("cluster_populations.dat").exists())
                self.assertTrue(Path("cluster_size.dat").exists())
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
