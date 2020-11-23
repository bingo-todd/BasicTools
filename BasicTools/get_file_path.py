import os


def get_realpath(file_path):
    """ if file_path is a link, the real path of this link rather than the
    file the link refers is returned
    """
    if os.path.islink(file_path):
        file_name = os.path.basename(file_path)
        dir_path = os.path.dirname(file_path)
        realpath = file_name
        while os.path.islink(dir_path):
            dir_name = os.path.basename(dir_path)
            realpath = f'{dir_name}/{realpath}'
            dir_path = os.path.dirname(dir_path)
        realpath = f'{dir_path}/{realpath}'
    else:
        realpath = os.path.realpath(file_path)
    return realpath


def get_file_path(dir_path, suffix=None, filter_func=None, is_absolute=False):
    """ return a list of file paths
    """
    if filter_func is None:
        filter_func = lambda x: True  # noqa E731

    file_relpath_all = []
    for file_dir, _, file_names in os.walk(dir_path):
        for file_name in file_names:
            if (suffix is None
                    or file_name.split('.')[-1] == suffix.split('.')[-1]):
                file_relpath = f'{file_dir}/{file_name}'
                if filter_func(file_relpath):
                    file_relpath_all.append(file_relpath)

    if is_absolute:
        file_realpath_all = [get_realpath(file_relpath)
                             for file_relpath in file_relpath_all]
        return file_realpath_all
    else:
        return file_relpath_all


if __name__ == '__main__':
    import sys
    dir_path = sys.argv[1]

    if len(sys.argv) > 2:
        args = sys.argv[2:]
    else:
        args = []

    file_path_all = get_file_path(dir_path, *args)
    print('\n'.join(file_path_all))
