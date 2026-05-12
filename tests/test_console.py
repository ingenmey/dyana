import io
import tempfile
import unittest
from pathlib import Path

from io_support.console import Console
from io_support.run_header import RunHeader, render_run_header


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

    def test_prompt_and_reply_are_mirrored_to_log(self):
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dyana.log"
            console = Console(stream=stream, log_path=log_path, use_color=False)

            console.prompt("Choose an analysis:")
            console.log_reply("rdf")
            console.close()

            self.assertEqual(stream.getvalue(), "Choose an analysis: ")
            self.assertEqual(log_path.read_text(encoding="utf-8"), "Choose an analysis: rdf\n")

    def test_render_run_header_keeps_plain_banner_in_log(self):
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dyana.log"
            console = Console(stream=stream, log_path=log_path, use_color=True)

            render_run_header(
                console,
                RunHeader(version="0.1.0", title="Dyana 0.1.0", lines=["Started: now"]),
            )
            console.close()

            self.assertIn("\x1b[34m", stream.getvalue())
            self.assertIn("\x1b[36m", stream.getvalue())
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("Đ Y A N A", log_text)
            self.assertIn("Dynamics Analyzer", log_text)
            self.assertIn("ver. 0.1.0", log_text)
            self.assertIn("Started: now", log_text)


if __name__ == "__main__":
    unittest.main()
