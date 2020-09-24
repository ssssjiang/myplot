# Interact with the user.

import os
import logging

logger = logging.getLogger(__name__)

try:
    input = raw_input
except NameError:
    pass

def prompt_val(msg="enter a value:"):
    return input(msg + "\n")


def confirm(msg="enter 'y' to confirm or any other key to cancel", key='y'):
    if input(msg + "\n") != key:
        return False
    else:
        return True


def check_and_confirm_overwrite(file_path):
    if os.path.isfile(file_path):
        logger.warning(file_path + " exists, overwrite?")
        return confirm("enter 'y' to overwrite or any other key to cancel")
    else:
        return True