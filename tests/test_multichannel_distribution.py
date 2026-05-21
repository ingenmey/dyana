import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyses.common.multichannel_distribution import (
    MultichannelDistributionEngine,
    combine_channel_samples,
)
from analyses.common.reference_batch import ReferenceBatch
from analyses.common.reference_channels import ChannelSamples
from io_support.output_writer import configure_output, restore_output


class DummyChannel:
    def __init__(self, output_name, bin_edges, samples_by_reference, axis_factors=None):
        self.output_name = output_name
        self.bin_edges = np.asarray(bin_edges, dtype=np.float64)
        self.samples_by_reference = list(samples_by_reference)
        self._axis_factors = axis_factors

    def prepare(self, traj, ref_compound_type):
        return None

    def rebuild_runtime_state(self, traj, ref_compound_type):
        return None

    def samples_for_reference(self, batch, ref_molecule_index):
        return self.samples_by_reference[ref_molecule_index]

    def axis_normalization_factors(self):
        return self._axis_factors


class MultichannelDistributionTests(unittest.TestCase):
    def setUp(self):
        self.previous_output = configure_output(".", False)

    def tearDown(self):
        restore_output(self.previous_output)

    def test_channel_samples_rejects_mismatched_sample_ids(self):
        with self.assertRaisesRegex(ValueError, "sample_ids must match"):
            ChannelSamples(values=[1.0, 2.0], sample_ids=[10])

    def test_combine_channel_samples_cartesian_builds_all_combinations(self):
        sample_sets = [
            ChannelSamples(values=[1.0, 2.0]),
            ChannelSamples(values=[10.0, 20.0, 30.0]),
        ]

        combined = combine_channel_samples(sample_sets, mode="cartesian")

        np.testing.assert_allclose(
            combined,
            np.array(
                [
                    [1.0, 10.0],
                    [1.0, 20.0],
                    [1.0, 30.0],
                    [2.0, 10.0],
                    [2.0, 20.0],
                    [2.0, 30.0],
                ]
            ),
        )

    def test_combine_channel_samples_matched_uses_shared_sample_ids(self):
        sample_sets = [
            ChannelSamples(values=[1.0, 2.0, 3.0], sample_ids=[10, 20, 30]),
            ChannelSamples(values=[100.0, 200.0], sample_ids=[20, 30]),
        ]

        combined = combine_channel_samples(sample_sets, mode="matched")

        np.testing.assert_allclose(combined, np.array([[2.0, 100.0], [3.0, 200.0]]))

    def test_combine_channel_samples_matched_rejects_duplicate_ids(self):
        sample_sets = [
            ChannelSamples(values=[1.0, 2.0], sample_ids=[10, 10]),
            ChannelSamples(values=[3.0, 4.0], sample_ids=[10, 20]),
        ]

        with self.assertRaisesRegex(ValueError, "unique sample_ids"):
            combine_channel_samples(sample_sets, mode="matched")

    def test_engine_process_batch_accumulates_joint_histogram_and_marginals(self):
        batch = ReferenceBatch(
            ref_compound_key=("H2O", 0),
            ref_compound_type=object(),
            molecule_atom_ids=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
            coords=np.zeros((6, 3), dtype=np.float64),
            box=np.array([10.0, 10.0, 10.0]),
            topology_frame=object(),
        )
        channels = [
            DummyChannel(
                "r",
                [0.0, 1.0, 2.0, 3.0],
                [
                    ChannelSamples(values=[0.2, 1.2]),
                    ChannelSamples(values=[2.2]),
                ],
            ),
            DummyChannel(
                "a",
                [0.0, 1.0, 2.0, 3.0],
                [
                    ChannelSamples(values=[0.4]),
                    ChannelSamples(values=[1.6, 2.6]),
                ],
            ),
        ]
        engine = MultichannelDistributionEngine(channels, combination_mode="cartesian")

        engine.process_batch(batch)

        self.assertEqual(engine.reference_count, 2)
        self.assertEqual(engine.tuple_count, 4)
        np.testing.assert_allclose(
            engine.hist.counts,
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
        )
        self.assertEqual(
            engine.joint_rows(normalize=True),
            [
                [0.5, 0.5, 0.25],
                [1.5, 0.5, 0.25],
                [2.5, 1.5, 0.25],
                [2.5, 2.5, 0.25],
            ],
        )
        self.assertEqual(
            engine.marginal_rows(axis=0, normalize=True),
            [
                [0.5, 0.25],
                [1.5, 0.25],
                [2.5, 0.5],
            ],
        )

    def test_engine_applies_axis_normalization(self):
        channels = [
            DummyChannel("x", [0.0, 1.0, 2.0], [ChannelSamples(values=[])], axis_factors=np.array([2.0, 4.0])),
            DummyChannel("y", [0.0, 1.0, 2.0], [ChannelSamples(values=[])], axis_factors=np.array([5.0, 10.0])),
        ]
        engine = MultichannelDistributionEngine(channels)
        engine.hist.counts = np.array([[20.0, 40.0], [40.0, 80.0]])

        engine.apply_channel_axis_normalization()

        np.testing.assert_allclose(engine.hist.counts, np.array([[2.0, 2.0], [2.0, 2.0]]))

    def test_engine_can_write_joint_and_marginal_tables(self):
        channels = [
            DummyChannel("r", [0.0, 1.0, 2.0], [ChannelSamples(values=[])]),
            DummyChannel("a", [0.0, 1.0, 2.0], [ChannelSamples(values=[])]),
        ]
        engine = MultichannelDistributionEngine(channels)
        engine.hist.counts = np.array([[1.0, 0.0], [0.0, 3.0]])
        engine.tuple_count = 4

        with tempfile.TemporaryDirectory() as tmp:
            configure_output(Path(tmp), False)

            engine.write_joint_table("joint.dat", normalize=True)
            engine.write_marginal_table(0, "marginal.dat", normalize=True)

            joint_text = (Path(tmp) / "joint.dat").read_text(encoding="utf-8")
            marginal_text = (Path(tmp) / "marginal.dat").read_text(encoding="utf-8")

        self.assertIn("# r", joint_text)
        self.assertIn("0.500000", joint_text)
        self.assertIn("0.250000", joint_text)
        self.assertIn("# r", marginal_text)
        self.assertIn("1.500000", marginal_text)
        self.assertIn("0.750000", marginal_text)


if __name__ == "__main__":
    unittest.main()
