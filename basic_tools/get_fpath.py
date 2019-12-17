import os
import re

def get_fpath(dir,suffix,pattern=None,is_exclude=False,is_absolute=False):
    """
    """
    suffix_len = len(suffix)
    file_filter = lambda fname:(len(fname)>suffix_len and
                                     fname[-suffix_len:] == suffix and
                                     fname[0]!='.')
    fpath_relative_all = []
    for root,dirs,files in os.walk(dir):
        for fname in files:
            if not file_filter(fname):
                continue

            fpath_relative = os.path.relpath(fpath_absolute,dir)
            if pattern is not None:
                n_match = len(re.findall(pattern,fpath_relative))
                if is_exclude and n_match>0:
                    continue
                if (not is_exclude) and n_match<1:
                    continue
            fpath_relative_all.append(fpath_relative)

    if is_absolute:
        fpath_absolute = [os.path.join(dir,fpath_relative)
                             for fpath_relative in fpath_relative_all]
        return fpath_absolute
    else:
        return fpath_relative_all


if __name__ == '__main__':
    import sys
    dir,suffix = sys.argv[1:3]
    if len(sys.argv) >=4:
        pattern = sys.argv[3]
    else:
        pattern = ''
    if len(sys.argv) >=5:
        is_exclude = sys.argv[3]
    else:
        is_exclude = False
    # print()
    fpath_all = get_fpath(dir,suffix,pattern,is_exclude)
    print('\n'.join(fpath_all))
