# utils.py

import re

input_file = None
log_file = None

def set_input_file(file_path):
    global input_file
    with open(file_path, 'r') as fin:
        input_file = fin.read().splitlines()

def set_log_file(file_path):
    global log_file
    log_file = open(file_path, 'w')

def close_log_file():
    global log_file
    if log_file:
        log_file.close()
        log_file = None

def label_matches(user_label, label):
    if bool(re.match(r"^[A-Za-z]+\d+$", user_label)):
        return user_label == label
    else:
        return label.startswith(user_label)

def _read_next_input_line():
    """Read the next non-comment line from input file, or return None."""
    global input_file
    while input_file and input_file[0].startswith('#'):
        input_file.pop(0)
    if input_file:
        return input_file.pop(0).strip()
    return None

def prompt(question, default=None, display_default=None):
    """Generic prompt function handling input file, logging, and default values."""
    global input_file, log_file

    # Allow different internal and display defaults
    if display_default is None:
        display_default = default

    if display_default is not None and display_default != "":
        question = f"{question} [{display_default}]"

    answer = None

    if input_file:
        answer = _read_next_input_line()
        if answer is not None:
            print(question, answer)

    if answer is None:
        while True:
            try:
                answer = input(question + " ").strip()
            except EOFError:
                print("\nInput interrupted. Exiting.")
                exit(1)

            if answer:
                break
            elif default is not None:
                answer = default
                break
            else:
                print("Invalid input. Try again.")

    if log_file:
        log_file.write(f"# {question.strip()}\n{answer}\n")

    return answer

def prompt_int(question, default=None, display_default=None, minval=None, maxval=None):
    """Prompt the user for an integer input with optional min/max limits."""
    while True:
        answer = prompt(question, default=str(default) if default is not None else None, display_default=display_default)
        try:
            value = int(answer)
            if (minval is not None and value < minval) or (maxval is not None and value > maxval):
                print(f"Please enter an integer between {minval} and {maxval}.")
            else:
                return value
        except ValueError:
            print("Please enter a valid integer.")

def prompt_float(question, default=None, display_default=None, minval=None, maxval=None):
    """Prompt the user for a float input with optional min/max limits."""
    while True:
        answer = prompt(question, default=str(default) if default is not None else None, display_default=display_default)
        try:
            value = float(answer)
            if (minval is not None and value < minval) or (maxval is not None and value > maxval):
                print(f"Please enter a number between {minval} and {maxval}.")
            else:
                return value
        except ValueError:
            print("Please enter a valid number.")

def prompt_yn(question, default=False):
    """Prompt the user for a yes/no (boolean) answer."""
    display_default = "Yes" if default else "No"
    default_letter = "y" if default else "n"

    while True:
        answer = prompt(question, default=default_letter, display_default=display_default).strip().lower()

        if answer in ["y", "yes"]:
            return True
        elif answer in ["n", "no"]:
            return False
        else:
            print("Please answer with 'y' or 'n'.")

def prompt_choice(question, choices, default=None):
    """Prompt the user to select from a list of choices."""
    choices_str = "/".join(choices)
    choices_lower = [c.lower() for c in choices]

    while True:
        answer = prompt(f"{question} ({choices_str})", default=default).strip().lower()
        if answer in choices_lower:
            return answer
        else:
            print(f"Please choose one of {choices_str}.")

