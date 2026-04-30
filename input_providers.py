"""Input providers for interactive and scripted CLI prompts.

This module starts replacing the old global prompt state in `utils.py`.
The public `utils.prompt*` functions remain as compatibility wrappers while
new code can depend on provider objects directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class InputProvider(ABC):
    """Abstract prompt/input provider."""

    @abstractmethod
    def ask_str(self, question: str, default=None, display_default=None) -> str:
        """Ask for a string value."""

    def ask_int(self, question: str, default=None, display_default=None, minval=None, maxval=None) -> int:
        while True:
            answer = self.ask_str(
                question,
                default=str(default) if default is not None else None,
                display_default=display_default,
            )
            try:
                value = int(answer)
            except ValueError:
                print("Please enter a valid integer.")
                continue

            if (minval is not None and value < minval) or (maxval is not None and value > maxval):
                print(_range_error_message("integer", minval, maxval))
            else:
                return value

    def ask_float(self, question: str, default=None, display_default=None, minval=None, maxval=None) -> float:
        while True:
            answer = self.ask_str(
                question,
                default=str(default) if default is not None else None,
                display_default=display_default,
            )
            try:
                value = float(answer)
            except ValueError:
                print("Please enter a valid number.")
                continue

            if (minval is not None and value < minval) or (maxval is not None and value > maxval):
                print(_range_error_message("number", minval, maxval))
            else:
                return value

    def ask_bool(self, question: str, default=False) -> bool:
        display_default = "Yes" if default else "No"
        default_letter = "y" if default else "n"

        while True:
            answer = self.ask_str(question, default=default_letter, display_default=display_default).strip().lower()
            if answer in ["y", "yes"]:
                return True
            if answer in ["n", "no"]:
                return False
            print("Please answer with 'y' or 'n'.")

    def ask_choice(self, question: str, choices, default=None) -> str:
        choices_str = "/".join(choices)
        choices_lower = [choice.lower() for choice in choices]

        while True:
            answer = self.ask_str(f"{question} ({choices_str})", default=default).strip().lower()
            if answer in choices_lower:
                return answer
            print(f"Please choose one of {choices_str}.")


class InteractiveInputProvider(InputProvider):
    """Prompt through stdin/stdout."""

    def ask_str(self, question: str, default=None, display_default=None) -> str:
        question = _format_question(question, default, display_default)
        while True:
            try:
                answer = input(question + " ").strip()
            except EOFError:
                raise SystemExit("\nInput interrupted. Exiting.")

            if answer:
                return answer
            if default is not None:
                return default
            print("Invalid input. Try again.")


class FileInputProvider(InputProvider):
    """Read prompt answers from a scripted input file, falling back to another provider."""

    def __init__(self, file_path=None, fallback: InputProvider | None = None, log_path=None, lines=None):
        if lines is not None:
            self.lines = list(lines)
        elif file_path is not None:
            with open(file_path, "r", encoding="utf-8") as fin:
                self.lines = fin.read().splitlines()
        else:
            self.lines = []
        self.fallback = fallback or InteractiveInputProvider()
        self.log_file = None
        if log_path:
            self.set_log_file(log_path)

    def set_log_file(self, log_path):
        self.close()
        self.log_file = open(log_path, "w", buffering=1, encoding="utf-8")

    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def ask_str(self, question: str, default=None, display_default=None) -> str:
        question = _format_question(question, default, display_default)
        answer = self._read_next_input_line()
        if answer is None:
            fallback_answer = self.fallback.ask_str(question, default=default, display_default="")
            self._write_log(question, "" if fallback_answer == default else fallback_answer)
            return fallback_answer

        print(question, answer)
        self._write_log(question, answer)
        if answer:
            return answer
        if default is not None:
            return default
        return self.fallback.ask_str(question, default=default, display_default="")

    def _read_next_input_line(self):
        while self.lines and self.lines[0].startswith("#"):
            self.lines.pop(0)
        if self.lines:
            return self.lines.pop(0).strip()
        return None

    def _write_log(self, question, answer):
        if self.log_file:
            self.log_file.write(f"# {question.strip()}\n{answer}\n")
            self.log_file.flush()


class NullInputProvider(InputProvider):
    """Provider for code paths that must not prompt."""

    def ask_str(self, question: str, default=None, display_default=None) -> str:
        if default is not None:
            return default
        raise RuntimeError(f"Input requested without a default: {question}")


def _format_question(question, default=None, display_default=None):
    if display_default is None:
        display_default = default
    if display_default is not None and display_default != "":
        return f"{question} [{display_default}]"
    return question


def _range_error_message(kind, minval=None, maxval=None):
    if minval is not None and maxval is not None:
        return f"Please enter a {kind} between {minval} and {maxval}."
    if minval is not None:
        return f"Please enter a {kind} >= {minval}."
    if maxval is not None:
        return f"Please enter a {kind} <= {maxval}."
    return f"Please enter a valid {kind}."
