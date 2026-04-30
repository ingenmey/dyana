import json
import tempfile
import unittest
from pathlib import Path

from output_metadata import build_run_metadata, write_metadata


class OutputMetadataTests(unittest.TestCase):
    def test_build_run_metadata_contains_reproducibility_fields(self):
        metadata = build_run_metadata("rdf", parameters={"bin_count": 10})

        self.assertEqual(metadata["analysis"], "rdf")
        self.assertEqual(metadata["parameters"]["bin_count"], 10)
        self.assertIn("python_version", metadata)
        self.assertIn("numpy", metadata["dependencies"])

    def test_write_metadata_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            write_metadata({"analysis": "rdf"}, path)

            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["analysis"], "rdf")


if __name__ == "__main__":
    unittest.main()

