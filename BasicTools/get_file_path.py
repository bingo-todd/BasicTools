import os


def get_realpath(file_path, root_dir):
    realpath = os.path.realpath(
        os.path.expanduser(file_path))
    if root_dir is not None:
        root_dir_tmp = os.path.expanduser(root_dir)
        if os.path.islink(root_dir_tmp):
            root_dir_tmp = os.readlink(root_dir_tmp)
            root_dir_tmp = os.path.expanduser(root_dir_tmp)

        len_tmp = len(root_dir_tmp)
        if realpath[:len_tmp] == root_dir_tmp[:len_tmp]:
            realpath = f'{root_dir}/{realpath[len_tmp:]}'
        else:
            print(f'{realpath} do not in {root_dir}, realpath is returned')
    return realpath.replace('//', '/')


def get_file_path(dir_path, suffix=None, filter_func=None,
                  is_absolute=False, root_dir=None):
    """
    """
    if filter_func is None:
        filter_func = lambda x: True

    file_relpath_all = []
    for file_dir, _, file_names in os.walk(dir_path):
        for file_name in file_names:
            if (suffix is None
                    or file_name.split('.')[-1] == suffix.split('.')[-1]):
                file_relpath = f'{file_dir}/{file_name}'
                if filter_func(file_relpath):
                    file_relpath_all.append(file_relpath)

    if is_absolute:
        file_realpath_all = [get_realpath(file_relpath, root_dir)
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
