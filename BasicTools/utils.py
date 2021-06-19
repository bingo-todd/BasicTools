import os
import re


def update_path(path):
    """ update Work_Space
    """
    Work_Space = os.genenv('Work_Space')
    new_path = re.sub('[/a-zA-Z0-9]*Work_Space', Work_Space, path)
    return new_path
