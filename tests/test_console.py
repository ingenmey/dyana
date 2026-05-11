import io
import tempfile
import unittest
from pathlib import Path

from io_support.console import Console


class ConsoleTests(unittest.TestCase):
    def test_plain_and_status_messages_are_mirrored_to_log(self):
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dyana.log"
            console = Console(stream=stream, log_path=log_path, use_color=False)

            console.header("Dyana", lines=["Trajectory: test.xyz", "Output dir: results"])
            console.info("Preparing topology")
            console.success("Saved rdf.dat")
            console.close()

            self.assertEqual(
                stream.getvalue(),
                "Dyana\n=====\nTrajectory: test.xyz\nOutput dir: results\n> Preparing topology\n+ Saved rdf.dat\n",
            )
            self.assertEqual(log_path.read_text(encoding="utf-8"), stream.getvalue())

    def test_colored_output_is_stripped_in_log(self):
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dyana.log"
            console = Console(stream=stream, log_path=log_path, use_color=True)

            console.warn("Rotated existing output file")
            console.close()

            self.assertIn("\x1b[33m", stream.getvalue())
            self.assertEqual(log_path.read_text(encoding="utf-8"), "! Rotated existing output file\n")

    def test_log_file_is_created_on_first_write(self):
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "dyana.log"
            console = Console(stream=stream, log_path=log_path, use_color=False)

            self.assertFalse(log_path.exists())
            console.plain("Hello")
            console.close()

            self.assertTrue(log_path.exists())
            self.assertEqual(log_path.read_text(encoding="utf-8"), "Hello\n")


if __name__ == "__main__":
    unittest.main()
