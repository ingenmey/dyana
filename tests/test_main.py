import json
import tempfile
import unittest
from pathlib import Path

import main


class MainTests(unittest.TestCase):
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
