import unittest

from analyses.common.base_analysis import BaseAnalysis
from framework.config_schema import FrameLoopConfig
from io_support.input_providers import NullInputProvider


class RecordingTrajectory:
    def __init__(self):
        self.read_calls = 0
        self.rebuild_calls = 0

    def read_frame(self):
        self.read_calls += 1

    def rebuild_topology(self):
        self.rebuild_calls += 1


class RecordingAnalysis(BaseAnalysis):
    def __init__(self, traj, post_update_results=None):
        super().__init__(traj, input_provider=NullInputProvider())
        self.process_calls = 0
        self.post_update_calls = 0
        self.postprocess_calls = 0
        self.post_update_results = list(post_update_results or [])

    def configure(self, config=None):
        return self.mark_configured()

    def post_compound_update(self):
        self.post_update_calls += 1
        if self.post_update_results:
            return self.post_update_results.pop(0)
        return True

    def process_frame(self):
        self.process_calls += 1

    def postprocess(self):
        self.postprocess_calls += 1


class BaseAnalysisFrameLoopTests(unittest.TestCase):
    def test_static_topology_mode_does_not_rebuild_or_post_update(self):
        traj = RecordingTrajectory()
        analysis = RecordingAnalysis(traj)

        analysis.configure()
        analysis.configure_frame_loop(
            FrameLoopConfig(start_frame=1, nframes=2, frame_stride=1, update_compounds=False)
        )
        analysis.run()

        self.assertEqual(traj.rebuild_calls, 0)
        self.assertEqual(analysis.post_update_calls, 0)
        self.assertEqual(analysis.process_calls, 2)
        self.assertEqual(analysis.processed_frames, 2)
        self.assertEqual(traj.read_calls, 2)
        self.assertEqual(analysis.postprocess_calls, 1)

    def test_dynamic_topology_mode_rebuilds_before_each_processed_frame(self):
        traj = RecordingTrajectory()
        analysis = RecordingAnalysis(traj)

        analysis.configure()
        analysis.configure_frame_loop(
            FrameLoopConfig(start_frame=1, nframes=2, frame_stride=1, update_compounds=True)
        )
        analysis.run()

        self.assertEqual(traj.rebuild_calls, 2)
        self.assertEqual(analysis.post_update_calls, 2)
        self.assertEqual(analysis.process_calls, 2)
        self.assertEqual(analysis.processed_frames, 2)
        self.assertEqual(traj.read_calls, 2)
        self.assertEqual(analysis.postprocess_calls, 1)

    def test_dynamic_topology_mode_skips_frames_that_fail_post_update(self):
        traj = RecordingTrajectory()
        analysis = RecordingAnalysis(traj, post_update_results=[False, True])

        analysis.configure()
        analysis.configure_frame_loop(
            FrameLoopConfig(start_frame=1, nframes=2, frame_stride=1, update_compounds=True)
        )
        analysis.run()

        self.assertEqual(traj.rebuild_calls, 2)
        self.assertEqual(analysis.post_update_calls, 2)
        self.assertEqual(analysis.process_calls, 1)
        self.assertEqual(analysis.processed_frames, 1)
        self.assertEqual(traj.read_calls, 2)
        self.assertEqual(analysis.postprocess_calls, 1)


if __name__ == "__main__":
    unittest.main()
