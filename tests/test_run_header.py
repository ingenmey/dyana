import unittest
from pathlib import Path
from unittest.mock import patch

from io_support import run_header


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
                title, lines = run_header.build_run_header(
                    "tests/fixtures/water128.xyz",
                    traj_format="xyz",
                    output_dir="results",
                    console_log_path="results/dyana.log",
                    input_log_path="results/input.log",
                    prepared_setup="setup.json",
                )

        self.assertEqual(title, "Dyana 0.1.0")
        self.assertEqual(lines[0], "Started: 2026-05-11 14:32:10 CEST")
        self.assertEqual(
            lines[1],
            f"Trajectory: {(Path.cwd() / 'tests' / 'fixtures' / 'water128.xyz').resolve()} (xyz)",
        )
        self.assertEqual(lines[2], f"Output dir: {(Path.cwd() / 'results').resolve()}")
        self.assertEqual(lines[3], f"Console log: {(Path.cwd() / 'results' / 'dyana.log').resolve()}")
        self.assertEqual(lines[4], f"Input log: {(Path.cwd() / 'results' / 'input.log').resolve()}")
        self.assertEqual(lines[5], f"Prepared setup: {(Path.cwd() / 'setup.json').resolve()}")


if __name__ == "__main__":
    unittest.main()
