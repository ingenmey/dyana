import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from io_support.console import console
from io_support.output_writer import configure_output, restore_output, write_table


class OutputWriterTests(unittest.TestCase):
    def setUp(self):
        self._previous_policy = configure_output(".", False)

    def tearDown(self):
        restore_output(self._previous_policy)

    def test_write_table_writes_into_configured_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "results"
            configure_output(output_dir, False)

            write_table("sample.dat", headers=["x", "y"], data=[[1.0, 2.0]])

            self.assertTrue((output_dir / "sample.dat").exists())

    def test_write_table_rotates_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            configure_output(output_dir, False)

            base = output_dir / "sample.dat"
            backup_1 = output_dir / "#1#sample.dat"
            base.write_text("old-current\n", encoding="utf-8")
            backup_1.write_text("old-backup\n", encoding="utf-8")

            stdout = io.StringIO()
            previous_console_state = console.capture_state()
            console.configure(stream=stdout, log_path=None, use_color=False)
            try:
                with redirect_stdout(stdout):
                    write_table("sample.dat", headers=["x", "y"], data=[[1.0, 2.0]])
            finally:
                console.close()
                console.restore_state(previous_console_state)

            new_text = base.read_text(encoding="utf-8")
            self.assertIn("# x", new_text)
            self.assertIn("1.000000", new_text)
            self.assertIn("2.000000", new_text)
            self.assertEqual((output_dir / "#1#sample.dat").read_text(encoding="utf-8"), "old-current\n")
            self.assertEqual((output_dir / "#2#sample.dat").read_text(encoding="utf-8"), "old-backup\n")
            self.assertIn("moved the previous file to #1#sample.dat", stdout.getvalue())
            self.assertIn("shifted 1 older backup(s)", stdout.getvalue())

    def test_write_table_force_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            configure_output(output_dir, True)

            base = output_dir / "sample.dat"
            base.write_text("old-current\n", encoding="utf-8")

            write_table("sample.dat", headers=["x", "y"], data=[[1.0, 2.0]])

            new_text = base.read_text(encoding="utf-8")
            self.assertIn("# x", new_text)
            self.assertIn("1.000000", new_text)
            self.assertIn("2.000000", new_text)
            self.assertFalse((output_dir / "#1#sample.dat").exists())


if __name__ == "__main__":
    unittest.main()
