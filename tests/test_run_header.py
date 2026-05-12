import io
import unittest
from pathlib import Path
from unittest.mock import patch

from io_support import run_header
from io_support.console import Console


class RunHeaderTests(unittest.TestCase):
    def test_resolve_dyana_version_falls_back_to_setup_cfg_metadata(self):
        with patch(
            "io_support.run_header.importlib.metadata.version",
            side_effect=run_header.importlib.metadata.PackageNotFoundError,
        ):
            self.assertEqual(run_header.resolve_dyana_version(), "0.1.0")

    def test_build_run_header_uses_absolute_paths(self):
        with patch("io_support.run_header.resolve_dyana_version", return_value="0.1.0"):
            with patch("io_support.run_header.format_started_at", return_value="2026-05-11 14:32:10 CEST"):
                header = run_header.build_run_header(
                    "tests/fixtures/water128.xyz",
                    traj_format="xyz",
                    output_dir="results",
                    console_log_path="results/dyana.log",
                    input_log_path="results/input.log",
                    prepared_setup="setup.json",
                )

        self.assertEqual(header.version, "0.1.0")
        self.assertEqual(header.title, "Dyana 0.1.0")
        self.assertEqual(header.lines[0], "Started: 2026-05-11 14:32:10 CEST")
        self.assertEqual(
            header.lines[1],
            f"Trajectory: {(Path.cwd() / 'tests' / 'fixtures' / 'water128.xyz').resolve()} (xyz)",
        )
        self.assertEqual(header.lines[2], f"Output dir: {(Path.cwd() / 'results').resolve()}")
        self.assertEqual(header.lines[3], f"Console log: {(Path.cwd() / 'results' / 'dyana.log').resolve()}")
        self.assertEqual(header.lines[4], f"Input log: {(Path.cwd() / 'results' / 'input.log').resolve()}")
        self.assertEqual(header.lines[5], f"Prepared setup: {(Path.cwd() / 'setup.json').resolve()}")

    def test_build_run_header_art_includes_versioned_banner_text(self):
        art_lines = run_header.build_run_header_art("0.1.0")
        plain_lines = ["".join(text for text, _ in segments) for segments in art_lines]

        self.assertEqual(plain_lines[1], "  ╔═════════════════════════════════════╗")
        self.assertIn("Đ Y A N A", plain_lines[4])
        self.assertIn("Dynamics Analyzer", plain_lines[5])
        self.assertIn("ver. 0.1.0", plain_lines[6])
        self.assertEqual(plain_lines[9], "  ╚═════════════════════════════════════╝")

    def test_render_run_header_falls_back_for_non_unicode_streams(self):
        class Cp1252Stream(io.StringIO):
            encoding = "cp1252"

        stream = Cp1252Stream()
        console = Console(stream=stream, log_path=None, use_color=False)
        header = run_header.RunHeader(
            version="0.1.0",
            title="Dyana 0.1.0",
            lines=["Started: now"],
        )

        run_header.render_run_header(console, header)

        self.assertEqual(stream.getvalue(), "Dyana 0.1.0\n===========\nStarted: now\n")


if __name__ == "__main__":
    unittest.main()
