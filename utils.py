# utils.py

import re
from input_providers import FileInputProvider, InteractiveInputProvider

_input_provider = InteractiveInputProvider()

def set_input_file(file_path):
    global _input_provider
    log_file = getattr(_input_provider, "log_file", None)
    log_path = log_file.name if log_file else None
    _input_provider = FileInputProvider(file_path, log_path=log_path)

def set_log_file(file_path):
    global _input_provider
    if not isinstance(_input_provider, FileInputProvider):
        _input_provider = FileInputProvider(lines=[], fallback=_input_provider)
    _input_provider.set_log_file(file_path)

def close_log_file():
    close = getattr(_input_provider, "close", None)
    if close:
        close()

def label_matches(user_label, label):
    if bool(re.match(r"^[A-Za-z]+\d+$", user_label)):
        return user_label == label
    else:
        return label.startswith(user_label)

def prompt(question, default=None, display_default=None):
    """Generic prompt function handling input file, logging, and default values."""
    return _input_provider.ask_str(question, default=default, display_default=display_default)

def prompt_int(question, default=None, display_default=None, minval=None, maxval=None):
    """Prompt the user for an integer input with optional min/max limits."""
    return _input_provider.ask_int(
        question,
        default=default,
        display_default=display_default,
        minval=minval,
        maxval=maxval,
    )

def prompt_float(question, default=None, display_default=None, minval=None, maxval=None):
    """Prompt the user for a float input with optional min/max limits."""
    return _input_provider.ask_float(
        question,
        default=default,
        display_default=display_default,
        minval=minval,
        maxval=maxval,
    )

def prompt_yn(question, default=False):
    """Prompt the user for a yes/no (boolean) answer."""
    return _input_provider.ask_bool(question, default=default)

def prompt_choice(question, choices, default=None):
    """Prompt the user to select from a list of choices."""
    return _input_provider.ask_choice(question, choices, default=default)
