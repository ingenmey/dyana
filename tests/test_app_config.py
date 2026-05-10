import json
import tempfile
import unittest
from pathlib import Path

from core.app_config import DEFAULT_APP_CONFIG, load_app_config


class AppConfigTests(unittest.TestCase):
    def test_load_app_config_uses_defaults_when_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "missing.json"

            config = load_app_config(config_path)

        self.assertEqual(config, DEFAULT_APP_CONFIG)

    def test_load_app_config_merges_json_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "OUTPUT_FORCE_DEFAULT": True,
                        "BOND_DISTANCE_SCALE": 1.25,
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(config_path)

        self.assertTrue(config["OUTPUT_FORCE_DEFAULT"])
        self.assertEqual(config["BOND_DISTANCE_SCALE"], 1.25)
        self.assertEqual(config["EXCLUDED_ELEMENTS"], DEFAULT_APP_CONFIG["EXCLUDED_ELEMENTS"])


if __name__ == "__main__":
    unittest.main()
