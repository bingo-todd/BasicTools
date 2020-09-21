import os
import re


def file_filter(file_name, suffix):
    if suffix is None:
        return True

    suffix_len = len(suffix)
    return (len(file_name) > suffix_len
            and file_name[-suffix_len:] == suffix and
            file_name[0] != '.')


def get_realpath(file_path, root_dir=None):
    realpath = os.path.realpath(file_path)
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


def get_file_path(dir_path, suffix=None, pattern=None, is_exclude=False,
              is_absolute=False, root_dir=None):
    """
    """
    file_path_rel_all = []
    for file_dir, _, file_names in os.walk(dir_path):
        for file_name in file_names:
            if not file_filter(file_name, suffix):
                continue
 
            file_path_rel = f'{file_dir}/{file_name}'

            if pattern is not None:
                n_match = len(re.findall(pattern, file_path_rel))
                if is_exclude and (n_match > 0):
                    continue
                if (not is_exclude) and (n_match < 1):
                    continue
            file_path_rel_all.append(file_path_rel)

    if is_absolute:
        file_path_real_all = [get_realpath(file_path_rel, root_dir)
                              for file_path_rel in file_path_rel_all]
        return file_path_real_all
    else:
        return file_path_rel_all


if __name__ == '__main__':
    import sys
    dir_path = sys.argv[1]

    if len(sys.argv) > 2:
        args = sys.argv[2:]
    else:
        args = []

    file_path_all = get_file_path(dir_path, *args)
    print('\n'.join(file_path_all))
