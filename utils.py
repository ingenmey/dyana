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

def prompt(question, default=None):
    global input_file, log_file

    if default is not None and default != "":
        question = f"{question} [{default}]"

    answer = None

    if input_file:
        try:
            while input_file and input_file[0].startswith('#'):
                input_file.pop(0)
            if input_file:
                answer = input_file.pop(0).strip()
                print(question, answer)
        except IndexError:
            pass

    if answer is None:
        while True:
            answer = input(question)
            if answer.strip():
                break
            elif default is not None:
                answer = default
                break
            else:
                print("Invalid input. Try again.")

    if log_file:
        log_file.write(f"# {question.strip()}\n{answer}\n")

    return answer

