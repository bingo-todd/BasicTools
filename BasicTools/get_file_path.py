import os
import re


def file_filter(fname, suffix):
    if suffix is None:
        return True

    suffix_len = len(suffix)
    return (len(fname) > suffix_len
            and fname[-suffix_len:] == suffix and
            fname[0] != '.')


def get_file_path(dir_path, suffix=None, pattern=None, is_exclude=False,
              is_absolute=False):
    """
    """
    fpath_relative_all = []
    for root, dirs, files in os.walk(dir_path):
        for fname in files:
            if not file_filter(fname, suffix):
                continue

            fpath_absolute = os.path.join(root, fname)
            fpath_relative = os.path.relpath(fpath_absolute, dir_path)

            if pattern is not None:
                n_match = len(re.findall(pattern, fpath_relative))
                if is_exclude and (n_match > 0):
                    continue
                if (not is_exclude) and (n_match < 1):
                    continue
            fpath_relative_all.append(fpath_relative)

    if is_absolute:
        fpath_absolute_all = [os.path.join(dir_path, fpath_relative)
                              for fpath_relative in fpath_relative_all]
        return fpath_absolute_all
    else:
        return fpath_relative_all


if __name__ == '__main__':
    import sys
    dir_path = sys.argv[1]

    if len(sys.argv) > 2:
        args = sys.argv[2:]
    else:
        args = []

    fpath_all = get_file_path(dir_path, *args)
    print('\n'.join(fpath_all))
