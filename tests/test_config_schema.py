import unittest

from config_schema import FrameLoopConfig, TopologyConfig


class ConfigSchemaTests(unittest.TestCase):
    def test_frame_loop_accepts_defaults(self):
        cfg = FrameLoopConfig()

        self.assertEqual(cfg.start_frame, 1)
        self.assertEqual(cfg.nframes, -1)
        self.assertEqual(cfg.frame_stride, 1)
        self.assertFalse(cfg.update_compounds)

    def test_frame_loop_rejects_invalid_stride(self):
        with self.assertRaises(ValueError):
            FrameLoopConfig(frame_stride=0)

    def test_frame_loop_accepts_all_frames_sentinel(self):
        cfg = FrameLoopConfig(nframes=-1)

        self.assertEqual(cfg.nframes, -1)

    def test_frame_loop_rejects_invalid_nframes(self):
        with self.assertRaises(ValueError):
            FrameLoopConfig(nframes=0)

    def test_topology_config_requires_positive_scales(self):
        with self.assertRaises(ValueError):
            TopologyConfig(neighbor_search_scale=0)
        with self.assertRaises(ValueError):
            TopologyConfig(bond_distance_scale=-1)


if __name__ == "__main__":
    unittest.main()
