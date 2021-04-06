from datetime import datetime


def log(message):
    _log("Info", message)


def warning(message):
    _log("WARNING", message)


def _log(error_type, message):
    t_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{t_string}    {error_type}    {message}")
