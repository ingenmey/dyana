import json
import tempfile
import unittest
from pathlib import Path

import main


class MainTests(unittest.TestCase):
    def test_available_analyses_include_dacf(self):
        self.assertTrue(any(entry[0] == "a" and entry[1] == "dacf" for entry in main.AVAILABLE_ANALYSES))

    def test_available_analyses_include_idcf(self):
        self.assertTrue(any(entry[0] == "a" and entry[1] == "idcf" for entry in main.AVAILABLE_ANALYSES))

    def test_available_analyses_include_perc(self):
        self.assertTrue(any(entry[0] == "a" and entry[1] == "perc" for entry in main.AVAILABLE_ANALYSES))

    def test_available_analyses_do_not_include_percolation(self):
        self.assertFalse(any(entry[0] == "a" and entry[1] == "percolation" for entry in main.AVAILABLE_ANALYSES))

    def test_available_analyses_do_not_include_dacf_nn(self):
        self.assertFalse(any(entry[0] == "a" and entry[1] == "dacf_nn" for entry in main.AVAILABLE_ANALYSES))

    def test_load_output_defaults_reads_force_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"OUTPUT_FORCE_DEFAULT": True}), encoding="utf-8")

            defaults = main._load_output_defaults(config_path)

        self.assertEqual(defaults, {"force_overwrite": True})

    def test_load_output_defaults_falls_back_when_config_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "missing.json"

            defaults = main._load_output_defaults(config_path)

        self.assertEqual(defaults, {"force_overwrite": False})


if __name__ == "__main__":
    unittest.main()
