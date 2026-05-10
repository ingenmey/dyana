import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from input_providers import FileInputProvider, InteractiveInputProvider, NullInputProvider


class InputProviderTests(unittest.TestCase):
    def test_file_provider_reads_non_comment_lines_and_defaults(self):
        provider = FileInputProvider(lines=["# comment", "", "42"], fallback=NullInputProvider())

        self.assertEqual(provider.ask_str("Question?", default="fallback"), "fallback")
        self.assertEqual(provider.ask_int("Number?"), 42)

    def test_null_provider_raises_without_default(self):
        provider = NullInputProvider()

        with self.assertRaises(RuntimeError):
            provider.ask_str("No prompt allowed")

    def test_file_provider_writes_prompt_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "input.log"
            provider = FileInputProvider(lines=["answer"], fallback=NullInputProvider(), log_path=log_path)

            self.assertEqual(provider.ask_str("Question?"), "answer")
            provider.close()

            text = log_path.read_text(encoding="utf-8")

        self.assertIn("# Question?", text)
        self.assertIn("answer", text)

    def test_interactive_provider_writes_prompt_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "input.log"
            provider = InteractiveInputProvider(log_path=log_path)

            with patch("builtins.input", return_value="answer"):
                self.assertEqual(provider.ask_str("Question?"), "answer")
            provider.close()

            text = log_path.read_text(encoding="utf-8")

        self.assertIn("# Question?", text)
        self.assertIn("answer", text)

    def test_interactive_provider_logs_blank_when_accepting_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "input.log"
            provider = InteractiveInputProvider(log_path=log_path)

            with patch("builtins.input", return_value=""):
                self.assertEqual(provider.ask_str("Question?", default="fallback"), "fallback")
            provider.close()

            text = log_path.read_text(encoding="utf-8")

        self.assertEqual(text, "# Question? [fallback]\n\n")

    def test_interactive_provider_creates_log_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "nested" / "logs" / "input.log"
            provider = InteractiveInputProvider(log_path=log_path)

            with patch("builtins.input", return_value="answer"):
                self.assertEqual(provider.ask_str("Question?"), "answer")
            provider.close()

            self.assertTrue(log_path.exists())

    def test_file_provider_creates_log_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "nested" / "logs" / "input.log"
            provider = FileInputProvider(lines=["answer"], fallback=NullInputProvider(), log_path=log_path)

            self.assertEqual(provider.ask_str("Question?"), "answer")
            provider.close()

            self.assertTrue(log_path.exists())


if __name__ == "__main__":
    unittest.main()
